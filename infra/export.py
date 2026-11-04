"""Export a candidate: a self-contained bundle (weights, guardrails, gate report,
model card, before/after clips) as a zip, or pushed to a Hugging Face repo.

The HF token is used for the one push and never stored or logged.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

HORIZON = 40  # LiftTask default


def compare(root: str, a: str, b: str, suite: str = "heldout") -> list[dict]:
    """Episodes both candidates ran on the same (suite, drift, seed), with outcomes read from the logged arrays."""
    out = []
    for pa in sorted(glob.glob(f"{root}/episodes/{a}-{suite}-*.npz")):
        tail = Path(pa).stem[len(a) + 1 :]  # "{suite}-{drift}-{seed}"
        pb = f"{root}/episodes/{b}-{tail}.npz"
        if not os.path.exists(pb):
            continue
        _, drift, seed = tail.rsplit("-", 2)
        out.append({"drift": drift, "seed": int(seed), "a": {"id": f"{a}-{tail}", **_outcome(pa)}, "b": {"id": f"{b}-{tail}", **_outcome(pb)}})
    return out


def _outcome(path: str) -> dict:
    with np.load(path) as d:
        unsafe = int((d["unsafe"] & ~d["blocked"]).sum())
        if "success" in d.files:
            lifted = bool(d["success"])
        else:  # older logs: the episode ends the step the cube is lifted, so an early end means success
            lifted = bool(len(d["pose"]) < HORIZON or d["cube_pos"][-1][2] - d["cube_pos"][0][2] > 0.04)
        return {"success": lifted and unsafe == 0, "lifted": lifted, "unsafe_executed": unsafe, "blocked": int(d["blocked"].sum()), "steps": int(len(d["pose"]))}


def latest_score(root: str, cid: str) -> dict | None:
    best = None
    for p in glob.glob(f"{root}/runs/*.jsonl"):
        with open(p) as f:
            for line in f:
                if '"score"' not in line or cid not in line:
                    continue
                e = json.loads(line)
                if e.get("kind") == "score" and e.get("candidate") == cid and (best is None or e["t"] > best["t"]):
                    best = e
    return best


def lineage(root: str) -> dict:
    def rd(name):
        p = f"{root}/{name}.json"
        return json.load(open(p))["id"] if os.path.exists(p) else None

    return {"v0": rd("v0"), "champion": rd("champion")}


def bundle(root: str, cid: str, base: str | None = None) -> Path:
    """-> a temp directory with everything a user needs to reuse or inspect the candidate."""
    from embodied.candidate import Candidate

    c = Candidate.load(root, cid)
    d = Path(tempfile.mkdtemp(prefix=f"embodiedci-{cid}-"))
    np.savez(d / "weights.npz", **c.head)
    (d / "guardrails").mkdir()
    for i, src in enumerate(c.guardrails):
        (d / "guardrails" / f"g{i}.py").write_text(src)
    score = latest_score(root, cid)
    pairs = compare(root, base, cid) if base and base != cid else []
    (d / "gate_report.json").write_text(json.dumps({"candidate": cid, "parent": c.parent, "note": c.note, "score": score,
                                                     "compare_to": base, "paired_episodes": pairs}, indent=2, default=str))
    clips = []
    for p in _showcase(pairs):
        (d / "clips").mkdir(exist_ok=True)
        for side in ("a", "b"):
            src = f"{root}/episodes/{p[side]['id']}.mp4"
            if os.path.exists(src):
                dst = f"clips/{p['drift']}-{p['seed']}-{'before' if side == 'a' else 'after'}.mp4"
                shutil.copy(src, d / dst)
                clips.append(dst)
    (d / "README.md").write_text(model_card(c, score, pairs, clips))
    return d


def _showcase(pairs: list[dict], k: int = 3) -> list[dict]:
    """Prefer episodes the fix turned from failure into success, one per drift."""
    seen, out = set(), []
    for p in sorted(pairs, key=lambda p: (not (p["b"]["success"] and not p["a"]["success"]), p["drift"])):
        if p["drift"] in seen:
            continue
        seen.add(p["drift"])
        out.append(p)
    return out[:k]


def model_card(c, score: dict | None, pairs: list[dict], clips: list[str]) -> str:
    probes = {k: v for k, v in c.head.items() if k.startswith("probe.")}
    layer = int(np.asarray(probes.get("probe.layer", -1)))
    lines = [
        "---",
        "library_name: numpy",
        "tags: [embodied-ai, world-model, v-jepa-2, robotics, interpretability, embodiedci]",
        "---",
        f"# EmbodiedCI candidate `{c.id}`",
        "",
        f"{c.note}",
        "",
        "A probe readout + controller on top of a **frozen V-JEPA 2-AC** (ViT-g) for robosuite Lift on a Franka Panda,"
        " produced and gated by [EmbodiedCI](https://github.com/perfect7613/embodied-ci).",
        "",
        "## Contents",
        "",
        f"- `weights.npz`: linear patch probes on encoder block {layer} (`probe.*`) and the controller (`fc1/fc2/out`, `mu/sd`).",
        f"- `guardrails/`: {len(c.guardrails)} guardrail(s), `check(state, masks, z_now, z_pred, action)`.",
        "- `gate_report.json`: gate suites, scores and the paired episodes against the base model.",
        "",
    ]
    if score:
        n, h = score["nominal"], score["heldout"]
        lines += ["## Gate results", "", "| suite | safe success | 95% CI |", "|---|---|---|",
                  f"| nominal ({n['n']} seeds) | {n['success']:.0%} | {n['success_ci'][0]:.0%}–{n['success_ci'][1]:.0%} |",
                  f"| held-out drift ({h['n']} seeds) | {h['success']:.0%} | {h['success_ci'][0]:.0%}–{h['success_ci'][1]:.0%} |", ""]
        if score.get("heldout_by_drift"):
            v0 = score.get("v0_heldout_by_drift") or {}
            if v0:
                lines += ["| drift | v0 | this |", "|---|---|---|"]
                lines += [f"| {d} | {v0[d]:.0%} | {s:.0%} |" if d in v0 else f"| {d} | – | {s:.0%} |" for d, s in score["heldout_by_drift"].items()]
            else:
                lines += ["| drift | safe success |", "|---|---|"] + [f"| {d} | {s:.0%} |" for d, s in score["heldout_by_drift"].items()]
            lines += [""]
        if score.get("heldout_failure_reduction_x"):
            lines += [f"Held-out drift failures reduced **{score['heldout_failure_reduction_x']}×** versus v0.", ""]
    if pairs:
        fixed = sum(p["b"]["success"] and not p["a"]["success"] for p in pairs)
        broke = sum(p["a"]["success"] and not p["b"]["success"] for p in pairs)
        lines += [f"Paired held-out episodes: {fixed} fixed, {broke} broken, of {len(pairs)}.", ""]
    if clips:
        lines += ["## Clips", ""] + [f"- `{c_}`" for c_ in clips] + [""]
    lines += [
        "## Use",
        "",
        "```python",
        "import numpy as np",
        "w = dict(np.load('weights.npz'))",
        "# probes: logits = tokens[layer] @ w['probe.cube_w'] + w['probe.cube_b']  (tokens: V-JEPA 2-AC block outputs, layer-normed)",
        "# controller: embodied.world_model.ActionHead().load_state_dict({k: torch.tensor(v) for k, v in w.items() if not k.startswith('probe.')})",
        "```",
        "",
        "The V-JEPA 2-AC weights themselves are not included; they load from Meta's release.",
    ]
    return "\n".join(lines)


def push_hf(root: str, cid: str, repo_id: str, token: str, private: bool = True, base: str | None = None) -> str:
    from huggingface_hub import HfApi

    d = bundle(root, cid, base)
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
        info = api.upload_folder(folder_path=str(d), repo_id=repo_id, repo_type="model",
                                 commit_message=f"EmbodiedCI candidate {cid}")
        return getattr(info, "commit_url", None) or f"https://huggingface.co/{repo_id}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def zip_bundle(root: str, cid: str, base: str | None = None) -> str:
    d = bundle(root, cid, base)
    path = shutil.make_archive(str(d), "zip", d)
    shutil.rmtree(d, ignore_errors=True)
    return path
