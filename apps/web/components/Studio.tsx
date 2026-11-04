"use client";

import { ArrowRight, Check, CircleNotch, Cube, Lightning, Robot, Sun, Feather, Camera, Palette } from "@phosphor-icons/react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type Model, type Task } from "@/lib/api";
import { empty, reduce, type RunState } from "@/lib/run";
import { RunView } from "./RunView";
import { Ship } from "./Ship";

const STEPS = ["Model", "Task", "Raw run", "Failures", "Inside the JEPA", "Fix and gate", "Before / after", "Ship"] as const;
const DRIFT_ICON: Record<string, typeof Cube> = { nominal: Cube, low_friction: Feather, heavy_cube: Lightning, dim_light: Sun, cube_color: Palette, camera_shift: Camera };
const DRIFT_LABEL: Record<string, string> = { nominal: "Nominal", low_friction: "Low friction", heavy_cube: "Heavy cube", dim_light: "Dim light", cube_color: "Blue cube", camera_shift: "Camera moved" };

export function Studio() {
  const [models, setModels] = useState<Model[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [model, setModel] = useState<string>("vjepa2-ac-vitg");
  const [task, setTask] = useState<string>("lift");
  const [rounds, setRounds] = useState(2);
  const [agents, setAgents] = useState(true);
  const [runs, setRuns] = useState<{ run_id: string; updated: number }[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [state, setState] = useState<RunState>(empty());
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const seq = useRef(0);

  useEffect(() => {
    api.models().then(setModels).catch((e) => setErr(String(e)));
    api.tasks().then(setTasks).catch(() => {});
    const q = new URLSearchParams(window.location.search).get("run");
    if (q) setRunId(q);
    api.runs().then((rs) => {
      setRuns(rs);
      if (!q && rs[0]) setRunId(rs[0].run_id); // a new visitor lands on the latest run, so every step has something to show
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!runId) return;
    seq.current = 0;
    setState(empty());
    let live = true;
    const tick = async () => {
      try {
        const ev = await api.events(runId, seq.current);
        if (!live) return;
        if (ev.length) {
          seq.current = ev[ev.length - 1].seq + 1;
          setState((s) => reduce(s, ev));
        }
      } catch (e) {
        setErr(String(e));
      }
    };
    tick();
    const h = setInterval(tick, 2500);
    return () => {
      live = false;
      clearInterval(h);
    };
  }, [runId]);

  const open = useCallback((id: string) => {
    setRunId(id);
    window.history.replaceState(null, "", `?run=${id}`);
  }, []);

  const start = async () => {
    setStarting(true);
    setErr(null);
    try {
      const r = await api.start({ model, task, rounds, use_agents: agents, reset: true });
      open(r.run_id);
      setRuns((x) => [{ run_id: r.run_id, updated: Date.now() / 1000 }, ...x]);
    } catch (e) {
      setErr(String(e));
    } finally {
      setStarting(false);
    }
  };

  /** Index of the step in progress; steps after it are not on the page yet, so the rail must not link to them. */
  const reached = useMemo(() => {
    const r = state.rounds.at(-1);
    if (!runId) return 1;
    if (state.ended) return STEPS.length;
    if (r?.candidates.some((c) => c.gate)) return 5;
    if (r?.inside.attribution || r?.inside.detector) return 4;
    if (r?.failures) return 3;
    if (r?.raw.length || state.bootstrap.length) return 2;
    return 2;
  }, [runId, state]);

  const chosen = models.find((m) => m.id === model);
  const chosenTask = tasks.find((t) => t.id === task);

  return (
    <div className="mx-auto grid min-h-screen max-w-[1440px] grid-cols-1 lg:grid-cols-[248px_1fr]">
      {/* rail */}
      <aside className="border-line px-5 pt-6 pb-8 lg:sticky lg:top-0 lg:h-screen lg:border-r">
        <Link href="/" className="text-[15px] font-semibold tracking-tight">EmbodiedCI</Link>
        <div className="mt-0.5 text-[12px] text-ink-3">Fix world models, then gate the fix</div>
        <Link href="/deploy" className="mt-3 inline-block text-[12px] text-accent hover:underline">Run on your own GPU →</Link>

        <ol className="mt-8 space-y-1">
          {STEPS.map((s, i) => {
            const done = i < reached;
            const now = i === reached;
            const shown = i <= reached;
            const Row = shown ? "a" : "span";
            return (
              <li key={s}>
                <Row {...(shown ? { href: `#step-${i}` } : { "aria-disabled": true })} className={`flex items-center gap-3 rounded-lg px-2.5 py-2 ${now ? "bg-raised shadow-[var(--shadow)]" : shown ? "hover:bg-surface" : "cursor-default opacity-60"}`}>
                  <span
                    className={`num grid size-6 flex-none place-items-center rounded-full text-[11px] ${
                      done ? "bg-ink text-raised" : now ? "bg-accent text-raised" : "border border-line-strong text-ink-3"
                    }`}
                  >
                    {done ? <Check size={12} weight="bold" /> : i + 1}
                  </span>
                  <span className={now ? "font-medium" : done ? "text-ink-2" : "text-ink-3"}>{s}</span>
                </Row>
              </li>
            );
          })}
        </ol>

        <div className="mt-10 text-[12px] font-medium text-ink-3">Runs</div>
        <div className="mt-2 max-h-[34vh] space-y-0.5 overflow-auto">
          {runs.length === 0 && <div className="text-[12px] text-ink-4">No runs yet</div>}
          {runs.map((r) => (
            <button
              key={r.run_id}
              onClick={() => open(r.run_id)}
              className={`num block w-full truncate rounded-md px-2 py-1 text-left text-[12px] ${r.run_id === runId ? "bg-raised text-ink" : "text-ink-3 hover:text-ink"}`}
            >
              {runLabel(r.run_id)}
            </button>
          ))}
        </div>
      </aside>

      <main className="px-5 pt-6 pb-24 lg:px-10">
        {err && <div className="mb-4 rounded-lg border border-fail/30 bg-fail-soft px-3 py-2 text-[13px] text-fail">{err}</div>}

        {/* 1. model */}
        <section id="step-0" className="rise">
          <StepHead n={1} title="Pick a JEPA model" sub="The world model under test." />
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            {models.map((m) => (
              <button
                key={m.id}
                disabled={!m.available}
                onClick={() => setModel(m.id)}
                className={`card group relative overflow-hidden p-4 text-left transition ${model === m.id ? "outline-2 outline-accent" : ""}`}
              >
                <div className="flex items-start gap-3">
                  <span className="grid size-9 flex-none place-items-center rounded-lg bg-surface text-ink-2">
                    <Robot size={18} weight="fill" />
                  </span>
                  <div className="min-w-0">
                    <div className="font-medium">{m.name}</div>
                    <div className="mt-0.5 text-[12px] text-ink-3">
                      {m.org} · {m.params}
                    </div>
                  </div>
                  {model === m.id && <Check size={16} weight="bold" className="ml-auto text-accent" />}
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 border-t border-line pt-3 text-[12px]">
                  <Spec k="Runs on" v={`Modal ${m.gpu}`} />
                  <Spec k="Editable surface" v={m.editable} />
                </div>
              </button>
            ))}
            {models.length === 0 && !err && <div className="card h-[132px] animate-pulse" />}
          </div>
        </section>

        {/* 2. task */}
        <section id="step-1" className="rise mt-16" style={{ animationDelay: "60ms" }}>
          <StepHead n={2} title="Pick a task" sub="The raw run covers clean seeds and every drift below." />
          <div className="mt-4 grid gap-3 md:grid-cols-[1fr_320px]">
            {tasks.map((t) => (
              <button key={t.id} onClick={() => setTask(t.id)} className={`card p-4 text-left ${task === t.id ? "outline-2 outline-accent" : ""}`}>
                <div className="flex items-center gap-2">
                  <span className="font-medium">{t.name}</span>
                  <span className="text-[12px] text-ink-3">· {t.sim}</span>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {t.drifts.map((d) => {
                    const I = DRIFT_ICON[d] ?? Cube;
                    return (
                      <span key={d} className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[12px] text-ink-2">
                        <I size={12} weight="fill" />
                        {DRIFT_LABEL[d] ?? d}
                      </span>
                    );
                  })}
                </div>
              </button>
            ))}
            <div className="card flex flex-col gap-3 p-4">
              <label className="flex items-center justify-between text-[13px]">
                <span className="text-ink-2">CI rounds</span>
                <span className="inline-flex rounded-lg border border-line bg-surface p-0.5">
                  {[1, 2, 3].map((n) => (
                    <button key={n} onClick={() => setRounds(n)} className={`num rounded-md px-3 py-1 text-[12px] ${rounds === n ? "bg-raised shadow-[var(--shadow)]" : "text-ink-3"}`}>
                      {n}
                    </button>
                  ))}
                </span>
              </label>
              <label className="flex items-center justify-between text-[13px]">
                <span className="text-ink-2">Muse Spark agents</span>
                <button onClick={() => setAgents(!agents)} className={`relative h-5 w-9 rounded-full transition ${agents ? "bg-accent" : "bg-line-strong"}`} aria-pressed={agents}>
                  <span className={`absolute top-0.5 size-4 rounded-full bg-raised transition-all ${agents ? "left-[18px]" : "left-0.5"}`} />
                </button>
              </label>
              <button
                onClick={start}
                disabled={starting || !chosen || !chosenTask}
                className="mt-auto inline-flex items-center justify-center gap-2 rounded-lg bg-ink px-4 py-2.5 text-[13px] font-medium text-raised hover:bg-ink-2 disabled:opacity-50"
              >
                {starting ? <CircleNotch size={14} className="animate-spin" /> : <ArrowRight size={14} weight="bold" />}
                Run without guardrails
              </button>
            </div>
          </div>
        </section>

        {runId && <RunView runId={runId} state={state} />}
        {!runId && (
          <div className="card mt-14 p-6 text-[13.5px] text-ink-3">
            No runs yet. Start one above, or <Link href="/deploy" className="text-accent hover:underline">connect your own deployment</Link>.
          </div>
        )}
        {runId && state.ended && <Ship key={`${runId}-${state.ended}`} gated={state.rounds.flatMap((r) => r.candidates.map((c) => c.gate).filter((g): g is NonNullable<typeof g> => !!g))} />}
      </main>
    </div>
  );
}

/** run-20261104-235115 -> "Nov 4 · 23:51" (the id is the start time). */
function runLabel(id: string): string {
  const m = id.match(/^run-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  if (!m) return id;
  const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][+m[2] - 1];
  return `${mon} ${+m[3]} · ${m[4]}:${m[5]}`;
}

export function StepHead({ n, title, sub, right }: { n: number; title: string; sub?: string; right?: React.ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4">
      <div>
        <div className="num text-[11px] text-ink-4">0{n}</div>
        <div className="mt-1.5 text-[22px] font-medium leading-tight tracking-[-0.01em]">{title}</div>
        {sub && <div className="mt-1.5 max-w-[56ch] text-[13.5px] text-ink-3">{sub}</div>}
      </div>
      {right}
    </div>
  );
}

function Spec({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="text-ink-4">{k}</div>
      <div className="mt-0.5 text-ink-2">{v}</div>
    </div>
  );
}
