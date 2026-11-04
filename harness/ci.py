"""One CI round of the harness: raw run -> failures -> diagnose -> fix (internal + external) -> gate -> promote -> red team.

`engine` is anything with:
    run(cand, specs) -> list[summary dict]                     (H100 rollouts, parallel)
    fit_probes(seeds, drifts) -> (probe entries, report)        (H100: layer sweep of linear patch probes)
    eval_probes(probes, seeds, drifts) -> per-drift probe quality
    episode(id) -> dict of arrays ; frame(id) -> np.ndarray
Controller training is ~1.9k params on 21 features: it runs here, on CPU.
Subagents are RLM agents (harness/repl.py) whose REPL holds these tools as functions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from embodied.candidate import Candidate, safety_envelope
from embodied.gate import DRIFT_TYPES, HELDOUT, HORIZON, NOMINAL, VISIBLE_DRIFT, EpisodeSpec, Registry, suite_stats, summary_from_log, verdict
from embodied.probes import FEATURE_NAMES, fit_failure_probe
from embodied.train import attribute, circuit_mask, train_head
from harness import llm
from harness.prompts import ROLES
from harness.repl import Agent, Budget, Repl


class RunLog:
    """Append-only JSONL event log per run. The frontend replays/streams this."""

    def __init__(self, root: str | Path, run_id: str, commit=lambda: None):
        self.path = Path(root) / "runs" / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.commit, self.round = run_id, commit, 0

    def __call__(self, kind: str, stage: str = "", **data) -> None:
        e = {"t": time.time(), "run": self.run_id, "round": self.round, "stage": stage, "kind": kind, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(e, default=_jsonable) + "\n")
        self.commit()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<array {o.shape}>"
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Candidate):
        return {"candidate": o.id, "note": o.note, "guardrails": len(o.guardrails)}
    return str(o)


class Harness:
    def __init__(self, engine, root: str | Path = "/data", run_id: str | None = None, commit=lambda: None, budget: Budget | None = None,
                 label_source: str = "sim", lean: bool = False):
        self.engine, self.root = engine, Path(root)
        self.label_source = label_source  # probe labels: "sim" segmentation or Meta SAM 3
        # lean: one GPU budget. Skip what round 1 already answered (circuit fine-tune overfits; DAgger + envelope is
        # dominated by recal + envelope) and the two costliest agents (weight editor, red team). Diagnoser and
        # guardrail writer still run.
        self.lean = lean
        self.log = RunLog(root, run_id or time.strftime("run-%Y%m%d-%H%M%S"), commit)
        self.registry = Registry(root)
        self.budget = budget or Budget(max_turns=10, max_tokens=200_000, max_seconds=900)
        self._suite_cache: dict[str, dict] = {}

    # ---- champion ------------------------------------------------------------
    @property
    def champion_path(self) -> Path:
        return self.root / "champion.json"

    def champion(self) -> Candidate | None:
        if not self.champion_path.exists():
            return None
        return Candidate.load(self.root, json.loads(self.champion_path.read_text())["id"])

    def promote(self, cand: Candidate, why: dict) -> None:
        cand.save(self.root)
        self.champion_path.write_text(json.dumps({"id": cand.id, "t": time.time()}))
        self.log("promoted", "gate", candidate=cand.id, note=cand.note, **why)

    # ---- tools exposed to agents --------------------------------------------
    def rollout(self, cand: Candidate, specs: list[EpisodeSpec], stage: str = "probe") -> list[dict]:
        specs = [s for s in specs if s.suite != "heldout"][:60]
        res = self.engine.run(cand, specs)
        for r in res:
            self.log("episode", stage, **r)
        return res

    def try_guardrail(self, cand: Candidate, src: str, failures: list[dict], n_fail: int = 8, n_nominal: int = 8) -> dict:
        """One call: does this guardrail stop known failures without blocking normal behaviour?"""
        from embodied.candidate import GuardrailError, compile_guardrail

        try:
            compile_guardrail(src)
        except (GuardrailError, SyntaxError) as e:
            return {"error": f"does not compile: {e}"}
        c2 = cand.child(guardrails=cand.guardrails + [src], note="probe")
        specs = [EpisodeSpec(seed=f["seed"], drift=f["drift"], suite="probe") for f in failures[:n_fail]]
        specs += [EpisodeSpec(seed=900 + i, suite="probe") for i in range(n_nominal)]
        res = self.rollout(c2, specs)
        fr, nr = res[: len(specs) - n_nominal], res[len(specs) - n_nominal :]
        steps = sum(e["steps"] for e in nr) or 1
        return {
            "failures_fixed": f"{sum(e['failure'] is None for e in fr)}/{len(fr)}",
            "still_failing": sorted({f"{e['drift']}:{e['failure']}" for e in fr if e["failure"]}),
            "nominal_success": f"{sum(e['failure'] is None for e in nr)}/{len(nr)}",
            "nominal_block_rate": round(sum(e["blocked"] for e in nr) / steps, 3),
            "unsafe_executed_per_ep": round(sum(e["unsafe_executed"] for e in res) / len(res), 2),
        }

    def try_head(self, cand: Candidate, head: dict, failures: list[dict], n_fail: int = 8, n_nominal: int = 8) -> dict:
        """One call: evaluate edited head weights on known failures + fresh nominal seeds."""
        c2 = cand.child(head={**cand.head, **{k: np.asarray(v, np.float32) for k, v in head.items()}}, note="probe")
        specs = [EpisodeSpec(seed=f["seed"], drift=f["drift"], suite="probe") for f in failures[:n_fail]]
        specs += [EpisodeSpec(seed=900 + i, suite="probe") for i in range(n_nominal)]
        res = self.rollout(c2, specs)
        fr, nr = res[: len(specs) - n_nominal], res[len(specs) - n_nominal :]
        return {
            "failures_fixed": f"{sum(e['failure'] is None for e in fr)}/{len(fr)}",
            "nominal_success": f"{sum(e['failure'] is None for e in nr)}/{len(nr)}",
            "lifted": f"{sum(e['success'] for e in res)}/{len(res)}",
            "unsafe_executed_per_ep": round(sum(e["unsafe_executed"] for e in res) / len(res), 2),
            "weights_changed": round(c2.fraction_changed(cand), 4),
        }

    def namespace(self, **vars) -> dict:
        cand, failures = vars.get("cand"), vars.get("failures", [])
        return {
            "try_guardrail": lambda src, n_fail=8, n_nominal=8: self.try_guardrail(cand, src, failures, n_fail, n_nominal),
            "try_head": lambda head, n_fail=8, n_nominal=8: self.try_head(cand, head, failures, n_fail, n_nominal),
            "np": np,
            "FEATURE_NAMES": list(FEATURE_NAMES),
            "Spec": lambda seed, drift="nominal", suite="probe": EpisodeSpec(seed=seed, drift=drift, suite=suite),
            "rollout": lambda cand, specs: self.rollout(cand, specs),
            "episode": self.engine.episode,
            "frame": self.engine.frame,
            "ask": llm.ask,
            "Candidate": Candidate,
            **vars,
        }

    @staticmethod
    def brief(failures: list[dict]) -> str:
        """Compact failure summary that goes into every agent's first message."""
        cl = _clusters(failures)
        lines = [f"{len(failures)} failing episodes. Clusters (drift:failure:rules -> count):"]
        lines += [f"  {k} -> {len(v)}" for k, v in sorted(cl.items(), key=lambda kv: -len(kv[1]))[:10]]
        if failures:
            f = failures[0]
            lines.append(f"Example summary keys: {sorted(f)[:20]}")
        return "\n".join(lines)

    def spawn(self, role: str, task: str, **vars):
        task = task + "\n\n" + self.brief(vars.get("failures", []))
        repl = Repl(tools=self.namespace(**vars))
        agent = Agent(role=role, system=ROLES[role], llm=lambda m: llm.chat(m, max_tokens=6000), repl=repl, budget=self.budget,
                      emit=lambda e: self.log("repl_cell", role, **e))
        self.log("agent_start", role, task=task)
        out = agent.run(task)
        self.log("agent_end", role, result=out if not isinstance(out, dict) or "head" not in out else {k: v for k, v in out.items() if k != "head"})
        return out

    # ---- gate ----------------------------------------------------------------
    def suites(self, cand: Candidate, which: tuple[str, ...] = ("nominal", "regression", "heldout")) -> dict[str, list[dict]]:
        specs = [s for s in NOMINAL + self.registry.specs() + HELDOUT if s.suite in which]
        cached = self._suite_cache.get(cand.id, {})
        have = {(e["seed"], e["drift"], e["suite"]) for v in cached.values() for e in v}
        todo = [s for s in specs if (s.seed, s.drift, s.suite) not in have]
        logged = []  # gated in an earlier run on these exact seeds: read the logs, don't re-simulate
        for sp in list(todo):
            p = self.root / "episodes" / f"{cand.id}-{sp.suite}-{sp.drift}-{sp.seed}.npz"
            e = summary_from_log(p, sp, cand.id) if p.exists() else None
            if e is not None:
                logged.append(e)
                todo.remove(sp)
        res = logged + (self.engine.run(cand, todo) if todo else [])
        out = {"nominal": [], "regression": [], "heldout": []}
        for e in [x for v in cached.values() for x in v] + res:
            out[e["suite"]].append(e)
        self._suite_cache[cand.id] = out
        return out

    def gate(self, cand: Candidate, base: Candidate) -> dict:
        """Nominal first: a candidate that already breaks a nominal rule is rejected without paying for the
        regression and held-out suites (~60% of a gate's episodes)."""
        cand.save(self.root)  # every gated candidate stays inspectable / exportable, passed or not
        b = self.suites(base)
        early = verdict({"nominal": self.suites(cand, ("nominal",))["nominal"], "regression": [], "heldout": []}, b)
        v = early if not early["passed"] else verdict(self.suites(cand), b)
        self.log("gate_result", "gate", candidate=cand.id, note=cand.note, passed=v["passed"], reasons=v["reasons"],
                 stats=v["candidate"], base_stats=v["base"], regression_fixed=v["regression_fixed"],
                 weights_changed=cand.fraction_changed(base), guardrails=cand.guardrails)
        return v

    # ---- training data: features depend on the probes, so ids are kept per probe set
    def _ids_path(self) -> Path:
        return self.root / "train_ids.json"

    def train_ids(self, cand: Candidate) -> list[str]:
        p = self._ids_path()
        return json.loads(p.read_text()).get(_pid(cand), []) if p.exists() else []

    def add_train_ids(self, cand: Candidate, ids: list[str]) -> None:
        p = self._ids_path()
        d = json.loads(p.read_text()) if p.exists() else {}
        d[_pid(cand)] = sorted(set(d.get(_pid(cand), [])) | set(ids))
        p.write_text(json.dumps(d))

    def train(self, cand: Candidate | None, ids: list[str], failed: set[str], **kw) -> tuple[dict, dict]:
        return train_head(cand, self.root, ids, failed=failed, **kw)

    def collect_and_train(self, cand: Candidate, drifts: list[str], n_demos: int, dagger_eps: int, seed0: int, tag: str,
                          fresh: bool = False, extra_ids: list[str] = (), dagger_iters: int = 1) -> Candidate:
        """Expert demos under this candidate's probes, then DAgger passes; train the controller."""
        specs = [EpisodeSpec(seed=seed0 + i, drift=drifts[i % len(drifts)], suite="train", policy="expert", noise=0.015) for i in range(n_demos)]
        demos = self.engine.run(cand, specs)
        ids = list(extra_ids) + [d["id"] for d in demos]
        head, info = self.train(cand, ids, set(), fresh=fresh)
        cand = cand.child(head={**cand.head, **head}, note=f"{tag} bc")
        self.log("trained", tag, candidate=cand.id, iter=0, **info)
        for it in range(1, dagger_iters + 1 if dagger_eps else 1):
            base = seed0 + 5000 * it
            eps = self.engine.run(cand, [EpisodeSpec(seed=base + i, drift=drifts[i % len(drifts)], suite="train") for i in range(dagger_eps)])
            ids += [e["id"] for e in eps]
            ok = sum(e["failure"] is None for e in eps)
            head, info = self.train(cand, ids, {e["id"] for e in eps if e["failure"]})
            cand = cand.child(head={**cand.head, **head}, note=f"{tag} dagger{it}")
            self.log("trained", tag, candidate=cand.id, iter=it, rollout_success=f"{ok}/{len(eps)}", **info)
        self.add_train_ids(cand, ids)
        return cand

    def bootstrap(self, n_demos: int = 40, dagger_eps: int = 20) -> Candidate:
        """v0: probes fit on nominal frames only; controller = BC + one DAgger pass on nominal only.
        Competent on nominal, never shown drift: the realistic starting point the harness has to improve."""
        self.log("stage", "bootstrap", msg="layer sweep: fitting cube/gripper patch probes on V-JEPA encoder layers")
        probes, rep = self.engine.fit_probes(list(range(24)), ["nominal"], self.label_source)
        self.log("probes_fit", "bootstrap", **rep)
        cand = Candidate(head=probes, note="probes v0")
        q = self.engine.eval_probes(cand.probes, list(range(300, 304)), ["nominal"] + DRIFT_TYPES)
        self.log("probe_eval", "bootstrap", candidate=cand.id, per_drift=q)
        cand = self.collect_and_train(cand, ["nominal"], n_demos, dagger_eps, seed0=0, tag="bootstrap", fresh=True)
        cand.note = "v0: nominal-only probes + BC/DAgger controller"
        self.promote(cand, {"reason": "bootstrap"})
        (self.root / "v0.json").write_text(json.dumps({"id": cand.id}))
        self.score(cand, "v0")
        return cand

    def score(self, cand: Candidate, label: str) -> dict:
        """Safe success on nominal + held-out drift (the number the improvement factor is about)."""
        s = self.suites(cand)
        allx = s["nominal"] + s["heldout"]
        out = {"label": label, "candidate": cand.id, "nominal": suite_stats(s["nominal"]), "heldout": suite_stats(s["heldout"]),
               "overall": suite_stats(allx),
               "heldout_by_drift": {d: round(float(np.mean([e["failure"] is None for e in s["heldout"] if e["drift"] == d])), 3)
                                    for d in sorted({e["drift"] for e in s["heldout"]})}}
        v0p = self.root / "v0.json"
        if v0p.exists() and label != "v0":
            v0 = Candidate.load(self.root, json.loads(v0p.read_text())["id"])
            b = suite_stats(self.suites(v0)["nominal"] + self.suites(v0)["heldout"])["success"]
            out["v0_overall"] = b
            out["improvement_x"] = round(out["overall"]["success"] / max(b, 1 / 80), 2)
            out["failure_reduction_x"] = round((1 - b) / max(1 - out["overall"]["success"], 1 / 80), 2)  # 1/80: one episode
            v0s = self.suites(v0)
            f0 = 1 - suite_stats(v0s["heldout"])["success"]
            f1 = 1 - out["heldout"]["success"]
            out["v0_heldout"] = 1 - f0
            out["heldout_failure_reduction_x"] = round(f0 / max(f1, 1 / 30), 2)  # 1/30: one held-out episode
            out["v0_heldout_by_drift"] = {d: round(float(np.mean([e["failure"] is None for e in v0s["heldout"] if e["drift"] == d])), 3)
                                          for d in sorted({e["drift"] for e in v0s["heldout"]})}
        self.log("score", "gate", **out)
        return out

    def detector(self, ids: list[str]) -> dict:
        """Failure-ahead probe on [features, surprise]: can the JEPA's own signals call a failure before it happens?"""
        feats, fs = [], []
        for eid in ids:
            try:
                d = self.engine.episode(eid)
            except FileNotFoundError:
                continue
            if "feats" not in d:
                continue
            feats.append(np.concatenate([d["feats"], d["surprise"][:, None]], 1))
            dev = np.linalg.norm(d["proposed"][:, :3] - d["expert"][:, :3], axis=1) > 0.02
            dev |= (d["proposed"][:, 6] > 0) != (d["expert"][:, 6] > 0)
            unsafe = np.nonzero(d["unsafe"] & ~d["blocked"])[0]
            failed = len(unsafe) or not _lifted(d)  # unsafe, or the episode never lifted the cube
            marks = ([int(unsafe[0])] if len(unsafe) else []) + ([int(np.argmax(dev))] if dev.any() else [])
            fs.append((min(marks) if marks else len(dev) - 1) if failed else None)
        if sum(f is not None for f in fs) < 3:
            return {"skipped": "too few failures"}
        return {k: v for k, v in fit_failure_probe(feats, fs, names=list(FEATURE_NAMES) + ["jepa_surprise"]).items() if k not in ("w", "mu", "sd")}

    def round(self, k: int, use_agents: bool = True) -> dict:
        self.log.round = k
        champ = self.champion() or self.bootstrap()
        train_ids = self.train_ids(champ)

        # 1. raw run, no guardrails
        raw_specs = [EpisodeSpec(seed=100 * k + i, suite="raw") for i in range(10)] + VISIBLE_DRIFT
        if champ.guardrails:
            raw = self.rollout(champ.child(guardrails=[], note=champ.note), raw_specs, stage="raw_run")
        else:  # same champion, same seeds as an interrupted earlier attempt: read the logs instead of re-simulating
            logged = [summary_from_log(self.root / "episodes" / f"{champ.id}-{sp.suite}-{sp.drift}-{sp.seed}.npz", sp, champ.id) for sp in raw_specs]
            if all(logged):
                raw = logged
                for r in raw:
                    self.log("episode", "raw_run", **r)
            else:
                raw = self.rollout(champ, raw_specs, stage="raw_run")
        failures = [e for e in raw if e["failure"]]
        added = self.registry.add(failures, "raw_run")
        self.log("failures", "detect", n=len(failures), of=len(raw), added_to_regression=added, clusters=_clusters(failures))

        # 2. interpretability: where does it break? perception (probes) or control (controller units)?
        perception = self.engine.eval_probes(champ.probes, list(range(300, 304)), sorted({f["drift"] for f in failures} | {"nominal"}))
        self.log("probe_eval", "interpret", candidate=champ.id, per_drift=perception)
        det = _safe(lambda: self.detector([e["id"] for e in raw] + train_ids[-40:]), {})
        self.log("detector", "interpret", **det)
        attr = _safe(lambda: attribute(champ, self.root, [f["id"] for f in failures]), {}) if failures else {}
        self.log("attribution", "interpret", **attr)

        candidates: list[Candidate] = []
        ids = train_ids + [e["id"] for e in raw]
        failed = {e["id"] for e in failures}
        self.add_train_ids(champ, [e["id"] for e in raw])
        # 3a. DAgger fine-tune of the whole controller
        head, info = self.train(champ, ids, failed)
        candidates.append(champ.child(head={**champ.head, **head}, note=f"r{k} dagger finetune"))
        self.log("candidate", "fix_internal", candidate=candidates[-1].id, source="finetune", **info)
        # 3b. circuit-restricted fine-tune: only the attributed units move
        if attr.get("fc1_units") and not self.lean:
            head, info = self.train(champ, ids, failed, mask=circuit_mask(attr))
            c = champ.child(head={**champ.head, **head}, note=f"r{k} circuit finetune")
            candidates.append(c)
            self.log("candidate", "fix_internal", candidate=c.id, source="circuit", units=attr["fc1_units"], weights_changed=c.fraction_changed(champ), **info)
        # 3c. perception fix: recalibrate probes where they break, then retrain the controller on the new readout
        nom_err = perception.get("nominal", {}).get("cube_err_px", 0.0)
        nom_err = nom_err if np.isfinite(nom_err) else 0.0
        bad = [d for d, q in perception.items() if d != "nominal" and (not np.isfinite(q["cube_err_px"])
               or q["cube_err_px"] > max(2 * nom_err, 6) or q["cube_conf"] < 0.75 * perception.get("nominal", {}).get("cube_conf", 1.0))]
        drifts = sorted({f["drift"] for f in failures} | {"nominal"})
        if bad or len(drifts) > 1:
            probes, rep = self.engine.fit_probes(list(range(12)) + [400 + 10 * k + i for i in range(4)], sorted(set(drifts) | set(bad)), self.label_source, 1)
            self.log("probes_fit", "fix_internal", **rep)
            c = champ.child(head={**champ.head, **probes}, note=f"r{k} probe recal")
            mix = ["nominal"] + drifts  # nominal twice: the gate will not trade nominal for drift
            c = self.collect_and_train(c, mix, n_demos=96, dagger_eps=42, seed0=20_000 + 1000 * k, tag="fix_internal",
                                       extra_ids=self.train_ids(c), dagger_iters=2)
            c.note = f"r{k} probe recal + drift DAgger ({','.join(drifts)})"
            candidates.append(c)
            self.log("candidate", "fix_internal", candidate=c.id, source="recalibrate", bad_drifts=bad, drifts=drifts, weights_changed=c.fraction_changed(champ))

        # 3d. external fix from the rules actually broken: a safety envelope (clip, never stop), on the champion and
        #     on the strongest internal fix. Deterministic, so the gate measures the envelope, not an LLM's luck.
        broken = {r for e in raw for r in e.get("unsafe_reasons", {})}
        if broken or any(e.get("failure") == "unsafe" for e in self.registry.load()):  # the envelope enforces every hard rule
            env = safety_envelope(broken or {"earlier rounds"})
            for c in [candidates[-1]] + ([candidates[0]] if len(candidates) > 1 and not self.lean else []):
                candidates.append(c.child(guardrails=champ.guardrails + [env], note=f"{c.note} + safety envelope"))
                self.log("candidate", "fix_external", candidate=candidates[-1].id, source="combined", code=env, rules=sorted(broken))

        diagnosis = []
        if use_agents and failures:
            ctx = dict(failures=failures, cand=champ, perception=perception, attribution=attr, detector=det)
            diagnosis = _safe(lambda: self.spawn("diagnoser", f"Diagnose {len(failures)} failures of candidate {champ.id}.", **ctx), [])
            g = _safe(lambda: self.spawn("guardrail_writer", "Write one guardrail for the confirmed causes.", diagnosis=diagnosis, **ctx), None)
            guard = None
            if isinstance(g, dict) and g.get("source"):
                guard = g["source"]
                candidates.append(champ.child(guardrails=champ.guardrails + [guard], note=f"r{k} guardrail"))
                self.log("candidate", "fix_external", candidate=candidates[-1].id, source="guardrail", code=guard, rationale=g.get("rationale"))
            w = None if self.lean else _safe(lambda: self.spawn("weight_editor", "Read, then edit the controller.", diagnosis=diagnosis, train_ids=train_ids, **ctx), None)
            if isinstance(w, dict) and isinstance(w.get("head"), dict):
                cand = champ.child(head={**champ.head, **{kk: np.asarray(vv, np.float32) for kk, vv in w["head"].items()}}, note=f"r{k} weight edit")
                candidates.append(cand)
                self.log("candidate", "fix_internal", candidate=cand.id, source="weight_edit", read=w.get("read"), edit=w.get("edit"), weights_changed=cand.fraction_changed(champ))
            if guard:
                inner = [c for c in candidates if not c.note.endswith("guardrail") and not c.guardrails[len(champ.guardrails):]]
                for c in inner[-1:] + ([] if self.lean else inner[:1]):  # strongest internal fix (recal if present) and the plain fine-tune
                    candidates.append(c.child(guardrails=champ.guardrails + [guard], note=f"{c.note} + guardrail"))
                    self.log("candidate", "fix_external", candidate=candidates[-1].id, source="combined")
                    if len(inner) == 1:
                        break

        # 4. gate every candidate against the champion; promote the best that passes
        results = [(c, self.gate(c, champ)) for c in candidates]
        passing = [(c, v) for c, v in results if v["passed"]]
        rank = lambda cv: (cv[1]["candidate"]["nominal"]["success"] + cv[1]["candidate"]["heldout"]["success"], cv[1]["regression_fixed"])  # noqa: E731
        best = max(passing, key=rank, default=None)
        if best:
            best[0].save(self.root)
            self.promote(best[0], {"nominal": best[1]["candidate"]["nominal"], "regression_fixed": best[1]["regression_fixed"]})
        else:
            self.log("no_promotion", "gate", reasons={c.id: v["reasons"] for c, v in results})
        champ_now = best[0] if best else champ
        sc = self.score(champ_now, f"round {k}")

        # 5. red team grows the regression suite for next round
        if use_agents and not self.lean:
            rt = _safe(lambda: self.spawn("red_team", "Find new failing scenarios.", cand=champ_now), [])
            if isinstance(rt, list):
                n = self.registry.add([{"seed": r["seed"], "drift": r["drift"], "failure": r.get("failure", "red_team"), "suite": "red_team", "id": "red_team"} for r in rt if isinstance(r, dict) and "seed" in r], "red_team")
                self.log("red_team", "red_team", added_to_regression=n)

        summary = {"round": k, "failures": len(failures), "candidates": [c.id for c in candidates],
                   "promoted": best[0].id if best else None, "llm_usage": dict(llm.USAGE),
                   "overall_success": sc["overall"]["success"], "improvement_x": sc.get("improvement_x"),
                   "diagnosis": diagnosis if isinstance(diagnosis, list) else []}
        self.log("round_end", "", **summary)
        return summary


def _pid(cand: Candidate) -> str:
    w = cand.head.get("probe.cube_w")
    import hashlib

    return "none" if w is None else hashlib.sha1(np.asarray(w).tobytes()).hexdigest()[:10]


def _lifted(d: dict) -> bool:
    """The episode stops the step the cube is lifted, before that frame is logged, so the logged cube height
    never shows the lift. Use the logged success flag, or for older logs, ending before the horizon."""
    if "success" in d:
        return bool(np.asarray(d["success"]))
    return len(d["pose"]) < HORIZON


def _clusters(failures: list[dict]) -> dict:
    c: dict[str, list[str]] = {}
    for f in failures:
        key = f"{f['drift']}:{f['failure']}" + (":" + ",".join(sorted(f["unsafe_reasons"])) if f["unsafe_reasons"] else "")
        c.setdefault(key, []).append(f["id"])
    return c


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:  # an agent failure must never stop the CI round
        return default if not isinstance(default, dict) else {"error": str(e)}
