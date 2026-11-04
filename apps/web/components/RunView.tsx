"use client";

import { CaretDown, CheckCircle, CircleNotch, Code, Prohibit, ShieldCheck, Sliders, X, XCircle } from "@phosphor-icons/react";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { api, type Episode } from "@/lib/api";
import { candidateSource, diagnosis, history, pct, sourceLabel, type Candidate, type Round, type RunState } from "@/lib/run";
import { StepHead } from "./Studio";

export function RunView({ runId, state }: { runId: string; state: RunState }) {
  const [sel, setSel] = useState<number | null>(null);
  const latest = state.rounds.at(-1)?.n ?? 1;
  const n = sel ?? latest;
  useEffect(() => setSel(null), [runId]);
  const r = state.rounds.find((x) => x.n === n);
  const live = !state.ended;

  const tabs = (
    <div className="inline-flex rounded-lg border border-line bg-surface p-0.5">
      {state.rounds.map((x) => (
        <button key={x.n} onClick={() => setSel(x.n)} className={`num rounded-md px-3 py-1 text-[12px] ${x.n === n ? "bg-raised shadow-[var(--shadow)]" : "text-ink-3"}`}>
          Round {x.n}
        </button>
      ))}
    </div>
  );

  const score = [...state.scores].reverse().find((x) => x.improvement_x !== undefined);
  return (
    <>
      <div className="mt-14 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line pt-5 text-[12px] text-ink-3">
        <span className="flex items-center gap-2">
          <span className={`size-2 rounded-full ${live ? "pulse-dot bg-accent" : "bg-pass"}`} />
          <span className="num text-ink-2">{runId.replace("run-", "")}</span>
          <span>{live ? "running" : "finished"}</span>
        </span>
        {score && (
          <span className="num ml-auto flex items-baseline gap-2">
            <span className="text-ink-4">safe success</span>
            <span>{pct(score.v0_overall)}</span>
            <span className="text-ink-4">→</span>
            <span className="text-[15px] text-ink">{pct(score.overall.success)}</span>
            {score.failure_reduction_x !== undefined && (
              <span className="rounded-full bg-accent-soft px-2 py-0.5 text-[12px] text-accent">{score.failure_reduction_x}× fewer failures</span>
            )}
          </span>
        )}
      </div>

      {!r && <Bootstrap state={state} />}

      {r && (
        <>
          <RawRun r={r} right={state.rounds.length > 1 ? tabs : undefined} />
          <Failures r={r} />
          <Inside r={r} state={state} />
          <FixGate r={r} state={state} />
        </>
      )}
    </>
  );
}

function Bootstrap({ state }: { state: RunState }) {
  const last = state.bootstrap.at(-1);
  return (
    <section className="rise mt-6">
      <div className="card flex items-center gap-3 p-4 text-[13px]">
        <CircleNotch size={16} className="animate-spin text-accent" />
        <div>
          <div className="font-medium">Preparing the model</div>
          <div className="text-ink-3">
            {last ? (last.kind === "trained" ? `Controller trained, iteration ${String(last.iter ?? 0)}${last.rollout_success ? `, rollouts ${String(last.rollout_success)}` : ""}` : String(last.msg)) : "Loading V-JEPA 2-AC on an H100"}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------------- inside the JEPA ---------------- */

type Probe = { auroc: number; median_err_px: number };

function Inside({ r, state }: { r: Round; state: RunState }) {
  const fit = (r.inside.probesFit ?? state.probesFit) as { layers?: Record<string, Record<string, Probe>>; best_layer?: number } | undefined;
  const per = r.inside.perception;
  const det = r.inside.detector as { auroc?: number; median_lead_steps?: number | null; caught?: string; top_features?: [string, number][]; skipped?: string } | undefined;
  const at = r.inside.attribution as { features?: [string, number][]; unit_reads?: Record<string, string[]>; concentration?: number } | undefined;
  if (!fit && !per && !det && !at) return null;
  const layers = fit?.layers ? Object.entries(fit.layers) : [];
  const maxErr = Math.max(1, ...layers.map(([, v]) => v.gripper?.median_err_px ?? 0));
  const best = fit?.best_layer;
  const perRows = per ? Object.entries(per) : [];
  const maxCube = Math.max(1, ...perRows.map(([, q]) => Number(q.cube_err_px) || 0));
  const attr = at?.features?.slice(0, 4) ?? [];
  const maxAttr = Math.max(1e-9, ...attr.map(([, v]) => v));
  return (
    <section id="step-4" className="rise mt-16">
      <StepHead n={5} title="Inside the JEPA" sub="Did perception break, or control?" />
      <div className="mt-5 grid items-stretch gap-4 lg:grid-cols-3">
        <Panel title="Where it sees" note={best !== undefined ? `Probes read block ${best === -1 ? "out" : best}` : undefined}>
          <BarList rows={layers.map(([l, v]) => ({ k: l === "-1" ? "Output" : `Block ${l}`, v: v.gripper?.median_err_px ?? 0, label: `${v.gripper?.median_err_px}px`, hi: String(best) === l }))} max={maxErr} />
          <Foot>Gripper position error. Lower is sharper.</Foot>
          {perRows.length > 0 && (
            <>
              <div className="mt-5 mb-2 text-[12px] text-ink-3">Cube error under drift</div>
              <BarList rows={perRows.map(([d, q]) => ({ k: DRIFT_NAME[d] ?? d, v: Number(q.cube_err_px) || 0, label: `${q.cube_err_px}px`, bad: Number(q.cube_conf) < 0.7 }))} max={maxCube} />
            </>
          )}
        </Panel>
        <Panel title="What drives the failure" note={at?.concentration ? `Top units carry ${pct(at.concentration)}` : undefined}>
          {attr.length ? <BarList rows={attr.map(([f, v]) => ({ k: human(f), v, label: "" }))} max={maxAttr} /> : <Empty>No failures to attribute</Empty>}
          {attr.length > 0 && <Foot>Inputs ranked by how much they push the failing actions.</Foot>}
        </Panel>
        <Panel title="Seeing it coming" note="Failure within 5 steps">
          {det && !det.skipped ? (
            <>
              <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line">
                <Big k="AUROC" v={String(det.auroc)} />
                <Big k="Caught" v={det.caught ?? "–"} />
              </div>
              <div className="mt-4 mb-2 text-[12px] text-ink-3">Strongest signals</div>
              <ul className="space-y-1.5 text-[12.5px]">
                {(det.top_features ?? []).slice(0, 3).map(([f]) => <li key={f} className="text-ink-2">{human(f)}</li>)}
              </ul>
            </>
          ) : <Empty>{det?.skipped ?? "Waiting"}</Empty>}
        </Panel>
      </div>
    </section>
  );
}

function Panel({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className="card flex flex-col p-5">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <div className="text-[14px] font-medium">{title}</div>
        {note && <div className="text-[11.5px] text-ink-4">{note}</div>}
      </div>
      {children}
    </div>
  );
}

function BarList({ rows, max }: { rows: { k: string; v: number; label: string; hi?: boolean; bad?: boolean }[]; max: number }) {
  return (
    <div className="space-y-2.5">
      {rows.map((r, i) => (
        <div key={`${r.k}-${i}`} className="grid grid-cols-[minmax(0,1fr)_42%_auto] items-center gap-3 text-[12.5px]">
          <span className={`truncate ${r.hi ? "font-medium text-ink" : "text-ink-2"}`}>{r.k}</span>
          <span className="h-[6px] rounded-full bg-ground">
            <span className={`block h-full rounded-full ${r.bad ? "bg-fail" : "bg-accent"}`} style={{ width: `${Math.max(4, (r.v / max) * 100)}%`, opacity: r.hi || r.bad || rows.every((x) => !x.hi) ? 1 : 0.3 }} />
          </span>
          <span className="num w-[44px] text-right text-[11.5px] text-ink-3">{r.label}</span>
        </div>
      ))}
    </div>
  );
}

function Big({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-raised px-3 py-3">
      <div className="text-[11.5px] text-ink-4">{k}</div>
      <div className="num mt-1 text-[22px] leading-none tracking-tight">{v}</div>
    </div>
  );
}

const Foot = ({ children }: { children: React.ReactNode }) => <div className="mt-3 text-[11.5px] text-ink-4">{children}</div>;
const Empty = ({ children }: { children: React.ReactNode }) => <div className="py-6 text-[13px] text-ink-4">{children}</div>;

/* ---------------- 03 raw run ---------------- */

function RawRun({ r, right }: { r: Round; right?: React.ReactNode }) {
  const eps = r.raw;
  const ok = eps.filter((e) => e.failure === null).length;
  const byDrift = useMemo(() => {
    const m: Record<string, Episode[]> = {};
    for (const e of eps) (m[e.drift] ??= []).push(e);
    return m;
  }, [eps]);
  const [open, setOpen] = useState<Episode | null>(null);

  return (
    <section id="step-2" className="rise mt-6">
      <StepHead n={3} title="Raw run" sub="The model alone, no guardrails. Click any episode to watch it." right={right} />
      <div className="mt-4 grid gap-3 lg:grid-cols-[260px_1fr]">
        <div className="card flex flex-col p-4">
          <div className="text-[12px] text-ink-3">Succeeded safely</div>
          <div className="num mt-1 text-[44px] leading-none font-medium tracking-tight">
            {ok}
            <span className="text-ink-4">/{eps.length || "–"}</span>
          </div>
          <div className="mt-5 space-y-2 border-t border-line pt-4 text-[12px]">
            {Object.entries(byDrift).map(([d, es]) => {
              const k = es.filter((e) => e.failure === null).length;
              return (
                <div key={d} className="flex items-center gap-2">
                  <span className="w-[96px] text-ink-2">{DRIFT_NAME[d] ?? d}</span>
                  <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-fail-soft">
                    <span className="absolute inset-y-0 left-0 rounded-full bg-pass" style={{ width: `${(100 * k) / es.length}%` }} />
                  </span>
                  <span className="num w-10 text-right text-ink-3">
                    {k}/{es.length}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
        <div className="card p-3">
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 xl:grid-cols-10">
            {eps.map((e) => (
              <button key={e.id} onClick={() => setOpen(e)} className="group relative aspect-square overflow-hidden rounded-md bg-surface" title={`${e.drift} seed ${e.seed}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={api.still(e.id)} alt="" loading="lazy" className="size-full object-cover transition group-hover:scale-105" />
                <span className={`absolute inset-x-0 bottom-0 h-1 ${e.failure ? "bg-fail" : "bg-pass"}`} />
              </button>
            ))}
            {eps.length === 0 && Array.from({ length: 20 }).map((_, i) => <div key={i} className="aspect-square animate-pulse rounded-md bg-surface" />)}
          </div>
        </div>
      </div>
      {open && <EpisodeModal e={open} onClose={() => setOpen(null)} />}
    </section>
  );
}

/** Rendered into <body>: the sections use transform animations, which would otherwise trap position:fixed inside them. */
function EpisodeModal({ e, onClose }: { e: Episode; onClose: () => void }) {
  useEffect(() => {
    const key = (k: KeyboardEvent) => k.key === "Escape" && onClose();
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("keydown", key);
      document.body.style.overflow = prev;
    };
  }, [onClose]);
  const rules = Object.keys(e.unsafe_reasons);
  return createPortal(
    <div className="modal-backdrop fixed inset-0 z-50 grid place-items-center bg-ink/40 p-4 backdrop-blur-[3px]" onClick={onClose} role="dialog" aria-modal="true" aria-label={`Episode ${e.drift} seed ${e.seed}`}>
      <div className="modal-panel card flex max-h-[calc(100dvh-32px)] w-full max-w-[560px] flex-col overflow-hidden" onClick={(x) => x.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <Status ok={e.failure === null} label={e.failure === null ? "lifted safely" : e.failure === "no_lift" ? "no lift" : "unsafe"} />
          <span className="text-[13px] font-medium">{DRIFT_NAME[e.drift] ?? e.drift}</span>
          <span className="num text-[12px] text-ink-4">seed {e.seed}</span>
          <button onClick={onClose} aria-label="Close" className="press ml-auto grid size-7 place-items-center rounded-md text-ink-3 hover:bg-surface hover:text-ink">
            <X size={15} weight="bold" />
          </button>
        </div>
        <video src={api.video(e.id)} autoPlay loop muted playsInline className="min-h-0 w-full flex-1 bg-ground object-contain" />
        <div className="grid grid-cols-3 border-t border-line text-[12px]">
          <Cell k="Unsafe proposals" v={String(e.unsafe_proposals)} />
          <Cell k="Blocked" v={String(e.blocked)} />
          <Cell k="Peak surprise" v={e.max_surprise.toFixed(2)} />
        </div>
        {rules.length > 0 && <div className="border-t border-line px-4 py-2.5 text-[12px] text-fail">Broke: {rules.join(", ").replaceAll("_", " ")}</div>}
      </div>
    </div>,
    document.body,
  );
}

function Cell({ k, v }: { k: string; v: string }) {
  return (
    <div className="border-line px-4 py-2.5 [&:not(:first-child)]:border-l">
      <div className="text-ink-4">{k}</div>
      <div className="num mt-0.5 text-[14px] text-ink">{v}</div>
    </div>
  );
}

const RULE: Record<string, string> = { speed: "Moved too fast", table_penetration: "Hit the table", workspace: "Left the workspace", contact_force: "Pushed too hard" };

const DRIFT_NAME: Record<string, string> = { nominal: "Nominal", low_friction: "Low friction", heavy_cube: "Heavy cube", dim_light: "Dim light", cube_color: "Blue cube", camera_shift: "Camera moved" };

/** Probe feature ids -> words. cube_u_agent -> "Cube x · front cam". */
function human(f: string): string {
  const fixed: Record<string, string> = { eef_x: "Hand x", eef_y: "Hand y", eef_z: "Hand height", gripper_closed: "Gripper closed", grip_settled: "Grip settled" };
  if (fixed[f]) return fixed[f];
  const m = f.match(/^(cube_minus_gripper|cube|gripper)_(u|v|conf)_(side|agent)$/);
  if (!m) return f.replaceAll("_", " ");
  const obj = m[1] === "cube_minus_gripper" ? "Cube–gripper gap" : m[1] === "cube" ? "Cube" : "Gripper";
  const ax = m[2] === "conf" ? "confidence" : m[2] === "u" ? "x" : "y";
  return `${obj} ${ax} · ${m[3] === "side" ? "side" : "front"} cam`;
}

/* ---------------- 04 failures ---------------- */

function Failures({ r }: { r: Round }) {
  const f = r.failures;
  const hyp = diagnosis(r);
  const clusters = Object.entries(f?.clusters ?? {}).sort((a, b) => b[1].length - a[1].length);
  return (
    <section id="step-3" className="rise mt-16">
      <StepHead n={4} title="Failures" sub="Grouped by cause. Each one becomes a regression test." />
      <div className="mt-4 grid items-start gap-3 lg:grid-cols-[1fr_1.2fr]">
        <div className="card p-4">
          <div className="flex items-baseline justify-between">
            <div className="text-[12px] text-ink-3">Failure clusters</div>
            {f && <div className="num text-[12px] text-ink-3">+{f.added_to_regression} regression tests</div>}
          </div>
          <div className="mt-3 divide-y divide-line">
            {clusters.slice(0, 8).map(([k, ids]) => {
              const [drift, kind, rules] = k.split(":");
              return (
                <div key={k} className="flex items-center gap-3 py-2.5">
                  <span className="num w-7 text-right text-[13px] font-medium">{ids.length}</span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13px]">{kind === "no_lift" ? "Never lifted the cube" : (rules ?? "unsafe").split(",").map((x) => RULE[x] ?? x.replaceAll("_", " ")).join(" · ")}</div>
                    <div className="text-[12px] text-ink-3">{DRIFT_NAME[drift] ?? drift}</div>
                  </div>
                  <span className={`rounded-full px-2 py-0.5 text-[11px] ${kind === "unsafe" ? "bg-fail-soft text-fail" : "bg-warn-soft text-warn"}`}>{kind === "unsafe" ? "unsafe" : "no lift"}</span>
                </div>
              );
            })}
            {!f && <div className="py-6 text-[13px] text-ink-4">Waiting for the raw run</div>}
          </div>
        </div>

        <div className="card p-4">
          <div className="text-[12px] text-ink-3">Hypotheses, checked in sim</div>
          <div className="mt-3 space-y-2.5">
            {hyp.map((h, i) => (
              <div key={i} className="rounded-lg border border-line bg-surface p-3">
                <div className="flex items-start gap-2">
                  {h.verdict === "confirmed" ? <CheckCircle size={16} weight="fill" className="mt-0.5 flex-none text-pass" /> : <XCircle size={16} weight="fill" className="mt-0.5 flex-none text-ink-4" />}
                  <div className="min-w-0">
                    <div className="text-[13px] leading-snug">{h.hypothesis}</div>
                    {h.evidence && <div className="mt-1 text-[12px] text-ink-3">{h.evidence}</div>}
                  </div>
                  <span className={`ml-auto flex-none text-[11px] ${h.verdict === "confirmed" ? "text-pass" : "text-ink-3"}`}>{h.verdict}</span>
                </div>
              </div>
            ))}
            {hyp.length === 0 && <AgentStatus r={r} role="diagnoser" />}
          </div>
          <AgentTrace r={r} />
        </div>
      </div>
    </section>
  );
}

function AgentStatus({ r, role }: { r: Round; role: string }) {
  const a = r.agents.find((x) => x.role === role);
  if (!a) return <div className="py-4 text-[13px] text-ink-4">{r.done ? "Agents were off for this round" : "Waiting"}</div>;
  return (
    <div className="flex items-center gap-2 py-4 text-[13px] text-ink-3">
      {a.status === "running" ? <CircleNotch size={14} className="animate-spin text-accent" /> : null}
      {a.status === "running" ? `${role} is working · ${a.cells.length} REPL cells` : `${role} returned no confirmed hypotheses`}
    </div>
  );
}

function AgentTrace({ r }: { r: Round }) {
  const [open, setOpen] = useState(false);
  const cells = r.agents.flatMap((a) => a.cells.map((c) => ({ code: String(c.code ?? ""), stdout: String(c.stdout ?? ""), role: a.role })));
  if (!cells.length) return null;
  return (
    <div className="mt-4 border-t border-line pt-3">
      <button onClick={() => setOpen(!open)} className="flex items-center gap-1.5 text-[12px] text-ink-3 hover:text-ink">
        <CaretDown size={12} className={open ? "" : "-rotate-90"} />
        REPL trace · {cells.length} cells across {r.agents.length} agents
      </button>
      {open && (
        <div className="mt-2 max-h-[420px] space-y-2 overflow-auto">
          {cells.map((c, i) => (
            <div key={i} className="overflow-hidden rounded-lg border border-line">
              <div className="flex items-center gap-2 bg-surface px-3 py-1.5 text-[11px] text-ink-3">
                <Code size={12} weight="bold" /> {c.role}
              </div>
              <pre className="num overflow-auto bg-raised px-3 py-2 text-[11.5px] leading-[1.5] text-ink-2">{String(c.code)}</pre>
              {String(c.stdout ?? "").trim() && <pre className="num max-h-40 overflow-auto border-t border-line bg-surface px-3 py-2 text-[11.5px] leading-[1.5] text-ink-3">{String(c.stdout)}</pre>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------------- 05 fix + gate ---------------- */

const SOURCE_ICON = { finetune: Sliders, circuit: Sliders, recalibrate: Sliders, weight_edit: Sliders, guardrail: ShieldCheck, combined: ShieldCheck } as const;

function FixGate({ r, state }: { r: Round; state: RunState }) {
  const hist = history(state);
  return (
    <section id="step-5" className="rise mt-16">
      <StepHead n={6} title="Fix and gate" sub="Each fix is scored on seeds it never saw. Only a pass ships." />
      <div className="mt-5 grid items-stretch gap-3 lg:grid-cols-2 2xl:grid-cols-3">
        {r.candidates.map((c) => (
          <CandidateCard key={c.id} c={c} promoted={r.promoted?.candidate === c.id} />
        ))}
        {r.candidates.length === 0 && <div className="card p-6 text-[13px] text-ink-4">No candidates yet</div>}
      </div>
      <div className="mt-3 grid items-start gap-3 lg:grid-cols-[1.4fr_1fr]">
        <GateTable r={r} />
        <div className="card p-4">
          <div className="flex items-center justify-between text-[12px] text-ink-3">
            <span>Champion across rounds</span>
            <span className="flex items-center gap-3 text-ink-2">
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-accent" /> Nominal
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full border-[1.5px] border-ink-4" /> Held-out drift
              </span>
            </span>
          </div>
          <HistoryChart pts={hist} />
        </div>
      </div>
    </section>
  );
}

function CandidateCard({ c, promoted }: { c: Candidate; promoted: boolean }) {
  const src = candidateSource(c);
  const I = SOURCE_ICON[src as keyof typeof SOURCE_ICON] ?? Sliders;
  const g = c.gate;
  const code = (c.info.code as string | undefined) ?? (g?.guardrails?.at(-1) && src !== "finetune" && src !== "weight_edit" ? g.guardrails.at(-1) : undefined);
  return (
    <div className={`card flex flex-col p-5 ${promoted ? "outline-2 outline-pass" : ""}`}>
      <div className="flex items-center gap-2">
        <span className="grid size-7 place-items-center rounded-md bg-surface text-ink-2">
          <I size={14} weight="fill" />
        </span>
        <span className="font-medium">{sourceLabel[src] ?? src}</span>

        <span className="ml-auto">{g ? <Status ok={g.passed} label={promoted ? "promoted" : g.passed ? "passed" : "rejected"} /> : <span className="text-[12px] text-ink-4">gating…</span>}</span>
      </div>
      {g && (
        <div className="mt-4 grid grid-cols-3 gap-px overflow-hidden rounded-lg border border-line bg-line">
          <Big k="Nominal" v={pct(g.stats.nominal.success)} />
          <Big k="Held-out drift" v={pct(g.stats.heldout.success)} />
          <Big k="Tests fixed" v={String(g.regression_fixed)} />
        </div>
      )}
      {typeof c.info.edit === "string" && <p className="mt-3 text-[13px] leading-snug text-ink-2">{c.info.edit}</p>}
      {code && (
        <pre className="num mt-3 max-h-56 overflow-auto rounded-lg border border-line bg-surface px-3 py-2 text-[11.5px] leading-[1.5] text-ink-2">{code}</pre>
      )}
      {g && !g.passed && g.reasons.length > 0 && (
        <div className="mt-3 flex items-start gap-2 text-[12.5px] leading-snug text-fail">
          <Prohibit size={14} weight="bold" className="mt-0.5 flex-none" />
          {reason(g.reasons[0])}
        </div>
      )}
    </div>
  );
}

function GateTable({ r }: { r: Round }) {
  const rows = r.candidates.filter((c) => c.gate);
  const base = rows[0]?.gate?.base_stats;
  const cols = ["", "Nominal", "Held-out", "Unsafe / 100 steps", ""];
  return (
    <div className="card overflow-x-auto p-4">
      <div className="text-[14px] font-medium">The gate</div>
      <table className="mt-3 w-full min-w-[440px] text-[12.5px]">
        <thead>
          <tr className="text-left text-ink-4">
            {cols.map((h, i) => (
              <th key={i} className="pb-2 font-normal">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="num">
          {base && (
            <tr className="border-t border-line text-ink-3">
              <td className="py-2.5 font-sans">Current champion</td>
              <td>{pct(base.nominal.success)}</td>
              <td>{pct(base.heldout.success)}</td>
              <td>{base.nominal.unsafe_exec_per_100?.toFixed(1)}</td>
              <td />
            </tr>
          )}
          {rows.map((c) => {
            const s = c.gate!.stats;
            return (
              <tr key={c.id} className="border-t border-line">
                <td className="py-2.5 font-sans">{sourceLabel[candidateSource(c)] ?? c.source}</td>
                <td>{pct(s.nominal.success)}</td>
                <td>{pct(s.heldout.success)}</td>
                <td>{s.nominal.unsafe_exec_per_100?.toFixed(1)}</td>
                <td className="text-right font-sans">{c.gate!.passed ? <span className="text-pass">pass</span> : <span className="text-fail">fail</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length === 0 && <div className="py-4 text-[13px] text-ink-4">Candidates are being evaluated on all three suites</div>}
    </div>
  );
}

/** Gate reasons are written for logs; say them plainly. */
function reason(r?: string): string {
  if (!r) return "";
  const m = r.match(/held-out drift success fell ([\d.]+) -> ([\d.]+)/);
  if (m) return `Held-out drift fell from ${pct(+m[1])} to ${pct(+m[2])}: it overfit the failures it saw.`;
  if (/unsafe/i.test(r) && /nominal/i.test(r)) return "More unsafe actions on nominal than the current champion.";
  return r.charAt(0).toUpperCase() + r.slice(1);
}

function HistoryChart({ pts }: { pts: { label: string; nominal?: number; heldout?: number }[] }) {
  const W = 360,
    H = 180,
    L = 34,
    B = 24,
    T = 12;
  if (pts.length < 1) return <div className="grid h-[180px] place-items-center text-[13px] text-ink-4">Appears after the first gate</div>;
  const x = (i: number) => L + (pts.length === 1 ? (W - L - 12) / 2 : (i * (W - L - 12)) / (pts.length - 1));
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const line = (k: "nominal" | "heldout") =>
    pts
      .map((p, i) => (p[k] === undefined ? null : `${x(i)},${y(p[k]!)}`))
      .filter(Boolean)
      .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 w-full" role="img" aria-label="Champion nominal and held-out success by round">
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - 8} y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text x={L - 8} y={y(v) + 4} textAnchor="end" fontSize="10.5" fill="var(--ink-4)" className="num">
            {v * 100}%
          </text>
        </g>
      ))}
      {pts.map((p, i) => (
        <text key={`${p.label}-${i}`} x={x(i)} y={H - 6} textAnchor="middle" fontSize="10.5" fill="var(--ink-3)" className="num">
          {p.label}
        </text>
      ))}
      <polyline points={line("heldout")} fill="none" stroke="var(--ink-4)" strokeWidth="1.5" strokeDasharray="4 3" />
      <polyline points={line("nominal")} fill="none" stroke="var(--accent)" strokeWidth="2" />
      {pts.map((p, i) => (
        <g key={i}>
          {p.heldout !== undefined && <circle cx={x(i)} cy={y(p.heldout)} r="3" fill="var(--raised)" stroke="var(--ink-4)" strokeWidth="1.5" />}
          {p.nominal !== undefined && <circle cx={x(i)} cy={y(p.nominal)} r="3.5" fill="var(--accent)" />}
        </g>
      ))}
    </svg>
  );
}

/* ---------------- small parts ---------------- */

function Status({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${ok ? "bg-pass-soft text-pass" : "bg-fail-soft text-fail"}`}>
      {ok ? <CheckCircle size={12} weight="fill" /> : <XCircle size={12} weight="fill" />}
      {label.replace("_", " ")}
    </span>
  );
}
