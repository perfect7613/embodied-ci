"use client";

import { ArrowSquareOut, DownloadSimple, Play } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type GateResult, type Pair } from "@/lib/api";
import { StepHead } from "./Studio";

const DRIFT_LABEL: Record<string, string> = {
  low_friction: "Low friction", heavy_cube: "Heavy cube", dim_light: "Dim light", cube_color: "Blue cube", camera_shift: "Camera moved", nominal: "Nominal",
};

/** Before / after on the held-out drift seeds, then export. v0 vs the champion by default, or any gated candidate of this run. */
export function Ship({ gated = [] }: { gated?: GateResult[] }) {
  const [lin, setLin] = useState<{ v0: string | null; champion: string | null } | null>(null);
  const [pick, setPick] = useState<string | null>(null);
  const [pairs, setPairs] = useState<Pair[] | null>(null);

  useEffect(() => {
    api.lineage().then(setLin).catch(() => setLin(null));
  }, []);
  const after = pick ?? lin?.champion ?? null;
  useEffect(() => {
    setPairs(null);
    if (lin?.v0 && after && lin.v0 !== after) api.compare(lin.v0, after).then(setPairs).catch(() => setPairs([]));
  }, [lin, after]);

  if (!lin?.champion) return null;
  const options = [
    ...(gated.some((g) => g.candidate === lin.champion) ? [] : [{ id: lin.champion, label: "Champion", passed: true, why: "" }]),
    ...gated.map((g) => ({ id: g.candidate, label: niceNote(g.note), passed: g.passed, why: g.reasons[0] ?? "" })),
  ].filter((o, i, all) => o.id !== lin.v0 && all.findIndex((x) => x.id === o.id) === i);
  const sel = options.find((o) => o.id === after);

  return (
    <>
      {lin.v0 && options.length > 0 && (
        <section id="step-6" className="rise mt-16">
          <StepHead n={7} title="Before and after" sub="Same unseen seeds. v0 on the left, the fix on the right." />
          <div className="mt-5 mb-4 flex flex-wrap gap-2">
            {options.map((o) => (
              <button key={o.id} onClick={() => setPick(o.id)}
                className={`rounded-[8px] border px-3 py-1.5 text-left text-[12.5px] ${o.id === after ? "border-ink bg-raised shadow-[var(--shadow)]" : "border-line bg-surface text-ink-2 hover:border-line-strong"}`}>
                <span className="font-medium">{o.label}</span>{" "}
                <span className={o.passed ? "text-pass" : "text-fail"}>{o.id === lin.champion ? "· champion" : o.passed ? "· passed" : "· rejected"}</span>
              </button>
            ))}
          </div>
          {sel && !sel.passed && <p className="mb-3 text-[12.5px] text-ink-3">Rejected by the gate, shown for comparison only.</p>}
          {pairs === null ? <div className="card p-6 text-[13px] text-ink-3">Loading paired episodes…</div>
            : pairs.length ? <BeforeAfter key={after!} v0={lin.v0} champ={after!} pairs={pairs} />
            : <div className="card p-6 text-[13px] text-ink-3">No paired held-out episodes for this candidate yet.</div>}
        </section>
      )}
      <Export champ={lin.champion} base={lin.v0 && lin.v0 !== lin.champion ? lin.v0 : undefined} />
    </>
  );
}

function niceNote(n: string): string {
  const t = n.replace(/^r\d+ /, "").replace(/\s*\(.*\)$/, "");
  if (/recal/i.test(t)) return "Probe recalibration" + (/guardrail/i.test(t) ? " + guardrail" : "");
  if (/circuit/i.test(t)) return "Circuit-restricted fine-tune";
  if (/finetune|fine-tune/i.test(t)) return "Fine-tune (DAgger)" + (/guardrail/i.test(t) ? " + guardrail" : "");
  return t.charAt(0).toUpperCase() + t.slice(1);
}

function BeforeAfter({ v0, champ, pairs }: { v0: string; champ: string; pairs: Pair[] }) {
  const drifts = useMemo(() => [...new Set(pairs.map((p) => p.drift))].sort(), [pairs]);
  const by = (d: string) => pairs.filter((p) => p.drift === d);
  const firstFixed = (d: string) => by(d).find((p) => p.b.success && !p.a.success) ?? by(d)[0];
  const best = drifts.find((d) => by(d).some((p) => p.b.success && !p.a.success)) ?? drifts[0];
  const [drift, setDrift] = useState(best);
  const [pair, setPair] = useState<Pair | undefined>(firstFixed(best));
  const fixed = pairs.filter((p) => p.b.success && !p.a.success).length;
  const broke = pairs.filter((p) => p.a.success && !p.b.success).length;
  const ok = (side: "a" | "b") => pairs.filter((p) => p[side].success).length;

  return (
    <div>
      <div className="card mb-4 flex flex-wrap items-baseline gap-x-6 gap-y-1 p-4">
        <span className="num text-[13px] text-ink-3">v0 {ok("a")}/{pairs.length}</span>
        <span className="num text-[22px]">{ok("b")}/{pairs.length}</span>
        <span className="text-[12.5px] text-ink-3">held-out episodes safely lifted</span>
        <span className="num ml-auto text-[12px]"><span className="text-pass">{fixed} fixed</span> · <span className={broke ? "text-fail" : "text-ink-3"}>{broke} broken</span></span>
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        {drifts.map((d) => {
          const a = by(d).filter((p) => p.a.success).length, b = by(d).filter((p) => p.b.success).length, n = by(d).length;
          return (
            <button key={d} onClick={() => { setDrift(d); setPair(firstFixed(d)); }}
              className={`rounded-full border px-3 py-1 text-[12px] ${d === drift ? "border-ink bg-ink text-surface" : "border-line bg-raised text-ink-2 hover:border-line-strong"}`}>
              {DRIFT_LABEL[d] ?? d} <span className="num opacity-70">{a}→{b}/{n}</span>
            </button>
          );
        })}
      </div>

      {pair && <Duel key={`${pair.drift}-${pair.seed}`} pair={pair} v0={v0} champ={champ} />}

      <div className="mt-3 flex flex-wrap gap-1.5">
        {by(drift).map((p) => (
          <button key={p.seed} onClick={() => setPair(p)}
            className={`num rounded-md border px-2 py-1 text-[11.5px] ${p.seed === pair?.seed ? "border-accent text-accent" : "border-line text-ink-3"}`}>
            seed {p.seed} <span className={p.a.success ? "text-pass" : "text-fail"}>●</span><span className={p.b.success ? "text-pass" : "text-fail"}>●</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Duel({ pair, v0, champ }: { pair: Pair; v0: string; champ: string }) {
  const refs = [useRef<HTMLVideoElement>(null), useRef<HTMLVideoElement>(null)];
  const play = () => refs.forEach((r) => { if (r.current) { r.current.currentTime = 0; void r.current.play(); } });
  useEffect(() => { const t = setTimeout(play, 250); return () => clearTimeout(t); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="card overflow-hidden">
      <div className="grid sm:grid-cols-2">
        {(["a", "b"] as const).map((side, i) => {
          const o = pair[side];
          return (
            <figure key={side} className={`relative ${i === 0 ? "sm:border-r" : ""} border-line`}>
              <video ref={refs[i]} src={api.video(o.id)} muted playsInline className="aspect-square w-full bg-ground object-cover" />
              <figcaption className="flex items-center gap-2 border-t border-line px-3 py-2 text-[12px]">
                <span className="font-medium">{side === "a" ? "v0" : "after"}</span>
                <span className="num text-ink-4">{side === "a" ? v0 : champ}</span>
                <span className={`ml-auto rounded-full px-2 py-0.5 text-[11px] ${o.success ? "bg-pass-soft text-pass" : "bg-fail-soft text-fail"}`}>
                  {o.success ? "lifted safely" : o.lifted ? "lifted, unsafe" : "no lift"}
                </span>
                {o.blocked > 0 && <span className="num text-[11px] text-warn">{o.blocked} blocked</span>}
              </figcaption>
            </figure>
          );
        })}
      </div>
      <div className="flex items-center gap-3 border-t border-line px-3 py-2 text-[12px] text-ink-3">
        <button onClick={play} className="inline-flex items-center gap-1.5 rounded-md bg-ink px-2.5 py-1 text-surface"><Play size={12} weight="fill" /> Replay both</button>
        <span>{DRIFT_LABEL[pair.drift] ?? pair.drift} · seed {pair.seed} · played in sync</span>
      </div>
    </div>
  );
}

function Export({ champ, base }: { champ: string; base?: string }) {
  const [repo, setRepo] = useState("");
  const [token, setToken] = useState("");
  const [priv, setPriv] = useState(true);
  const [st, setSt] = useState<{ kind: "idle" | "busy" | "ok" | "err"; msg?: string; url?: string }>({ kind: "idle" });

  const push = async () => {
    if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) return setSt({ kind: "err", msg: "Repo id looks like your-name/embodiedci-lift" });
    if (!token) return setSt({ kind: "err", msg: "Paste a Hugging Face token with write access" });
    setSt({ kind: "busy" });
    try {
      const r = await api.exportHF(champ, { repo_id: repo, token, private: priv }, base);
      setToken("");
      setSt({ kind: "ok", url: r.url });
    } catch (e) {
      setSt({ kind: "err", msg: (e as Error).message });
    }
  };

  return (
    <section id="step-7" className="rise mt-16">
      <StepHead n={8} title="Ship the champion" sub="Download it, or push it to Hugging Face." />
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="card p-5">
          <div className="text-[14px] font-medium">Download bundle</div>
          <p className="mt-1 text-[12.5px] leading-[1.55] text-ink-3">A zip you can load anywhere: <span className="num">weights.npz</span>, <span className="num">guardrails/</span>, <span className="num">gate_report.json</span>, <span className="num">README.md</span>.</p>
          <a href={api.bundleUrl(champ, base)} className="mt-4 inline-flex items-center gap-2 rounded-[8px] bg-ink px-3.5 py-2 text-[13px] font-medium text-surface hover:bg-ink-2">
            <DownloadSimple size={14} /> embodiedci-{champ}.zip
          </a>
        </div>
        <div className="card p-5">
          <div className="text-[14px] font-medium">Push to Hugging Face</div>
          <p className="mt-1 text-[12.5px] leading-[1.55] text-ink-3">Goes straight from your deployment to your repo. The token is used for this one push and never stored.</p>
          <div className="mt-3 grid gap-2">
            <input value={repo} onChange={(e) => setRepo(e.target.value.trim())} placeholder="your-name/embodiedci-lift"
              className="num rounded-[8px] border border-line-strong bg-raised px-3 py-2 text-[13px] outline-none focus:border-accent" />
            <input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value.trim())} placeholder="hf_… (write token)"
              className="num rounded-[8px] border border-line-strong bg-raised px-3 py-2 text-[13px] outline-none focus:border-accent" />
            <label className="flex items-center gap-2 text-[12.5px] text-ink-2">
              <input type="checkbox" checked={priv} onChange={(e) => setPriv(e.target.checked)} /> private repo
            </label>
            <button onClick={push} disabled={st.kind === "busy"} className="mt-1 rounded-[8px] bg-accent px-3.5 py-2 text-[13px] font-medium text-white disabled:opacity-60">
              {st.kind === "busy" ? "Pushing…" : "Push champion"}
            </button>
          </div>
          {st.kind === "ok" && st.url && (
            <a href={st.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-[12.5px] text-pass hover:underline">
              Pushed <ArrowSquareOut size={12} />
            </a>
          )}
          {st.kind === "err" && <p className="mt-3 text-[12.5px] text-fail">{st.msg}</p>}
        </div>
      </div>
    </section>
  );
}
