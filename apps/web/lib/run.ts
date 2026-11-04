import type { Episode, Ev, GateResult } from "./api";

export type Hypothesis = { cluster: string; episodes?: string[]; hypothesis: string; test?: string; verdict: "confirmed" | "refuted" | string; evidence?: string };

export type Candidate = {
  id: string;
  source: "finetune" | "guardrail" | "weight_edit" | "combined" | string;
  info: Record<string, unknown>;
  gate?: GateResult;
};

export type Round = {
  n: number;
  raw: Episode[];
  failures?: { n: number; of: number; added_to_regression: number; clusters: Record<string, string[]> };
  agents: { role: string; status: "running" | "done"; cells: Ev[]; result?: unknown }[];
  candidates: Candidate[];
  promoted?: { candidate: string; note: string };
  noPromotion?: boolean;
  /** interpretability: probe quality per drift, failure-ahead detector, controller attribution */
  inside: { perception?: Record<string, { cube_err_px: number; cube_conf: number; auroc: number }>; detector?: Ev; attribution?: Ev; probesFit?: Ev };
  done: boolean;
};

export type Score = { label: string; candidate: string; overall: { success: number }; nominal: { success: number }; heldout: { success: number };
  heldout_by_drift: Record<string, number>; improvement_x?: number; failure_reduction_x?: number; v0_overall?: number };

export type RunState = {
  started?: Ev;
  ended: boolean;
  bootstrap: Ev[];
  champion?: string;
  probesFit?: Ev;
  scores: Score[];
  rounds: Round[];
  lastSeq: number;
};

export const empty = (): RunState => ({ ended: false, bootstrap: [], scores: [], rounds: [], lastSeq: -1 });

function round(s: RunState, n: number): Round {
  let r = s.rounds.find((x) => x.n === n);
  if (!r) {
    r = { n, raw: [], agents: [], candidates: [], inside: {}, done: false };
    s.rounds.push(r);
    s.rounds.sort((a, b) => a.n - b.n);
  }
  return r;
}

export function reduce(prev: RunState, events: Ev[]): RunState {
  const s: RunState = structuredClone(prev);
  for (const e of events) {
    if (e.seq <= s.lastSeq) continue;
    s.lastSeq = e.seq;
    const r = () => round(s, e.round);
    switch (e.kind) {
      case "run_start":
        s.started = e;
        break;
      case "run_end":
        s.ended = true;
        break;
      case "stage":
      case "trained":
        if (e.stage === "bootstrap") s.bootstrap.push(e);
        break;
      case "episode":
        if (e.stage === "raw_run") r().raw.push(e as Episode);
        break;
      case "failures":
        r().failures = e as unknown as Round["failures"];
        break;
      case "agent_start":
        r().agents.push({ role: e.stage, status: "running", cells: [] });
        break;
      case "repl_cell": {
        const a = [...r().agents].reverse().find((x) => x.role === e.stage);
        a?.cells.push(e);
        break;
      }
      case "agent_end": {
        const a = [...r().agents].reverse().find((x) => x.role === e.stage);
        if (a) {
          a.status = "done";
          a.result = e.result;
        }
        break;
      }
      case "candidate":
        r().candidates.push({ id: e.candidate as string, source: e.source as string, info: e });
        break;
      case "gate_result": {
        const c = r().candidates.find((x) => x.id === e.candidate);
        if (c) c.gate = e as GateResult;
        break;
      }
      case "promoted":
        s.champion = e.candidate as string;
        if (e.stage === "gate" && e.reason !== "bootstrap") r().promoted = { candidate: e.candidate as string, note: e.note as string };
        break;
      case "no_promotion":
        r().noPromotion = true;
        break;
      case "probes_fit":
        if (e.stage === "bootstrap") s.probesFit = e;
        else r().inside.probesFit = e;
        break;
      case "probe_eval":
        if (e.stage === "interpret") r().inside.perception = e.per_drift as Round["inside"]["perception"];
        break;
      case "detector":
        r().inside.detector = e;
        break;
      case "attribution":
        r().inside.attribution = e;
        break;
      case "score":
        s.scores.push(e as unknown as Score);
        break;
      case "round_end":
        r().done = true;
        break;
    }
  }
  return s;
}

export const diagnosis = (r: Round): Hypothesis[] => {
  const d = r.agents.find((a) => a.role === "diagnoser")?.result;
  return Array.isArray(d) ? (d as Hypothesis[]) : [];
};

export const pct = (x?: number) => (x === undefined || Number.isNaN(x) ? "–" : `${Math.round(x * 100)}%`);

export const sourceLabel: Record<string, string> = {
  finetune: "Fine-tune (DAgger)",
  guardrail: "Guardrail",
  weight_edit: "Weight edit",
  combined: "Internal fix + guardrail",
  circuit: "Circuit-restricted fine-tune",
  recalibrate: "Probe recal + drift DAgger",
};

export function candidateSource(c: Candidate): string {
  if (c.source === "finetune" && c.gate?.note?.includes("+ guardrail")) return "combined";
  return c.source;
}

/** Champion quality per round: base (before round 1) then whatever was promoted. */
export function history(s: RunState): { label: string; nominal?: number; heldout?: number }[] {
  const pts: { label: string; nominal?: number; heldout?: number }[] = [];
  const first = s.rounds.find((r) => r.candidates.some((c) => c.gate));
  const base = first?.candidates.find((c) => c.gate)?.gate?.base_stats;
  if (base) pts.push({ label: "v0", nominal: base.nominal.success, heldout: base.heldout.success });
  let cur = base;
  for (const r of s.rounds) {
    if (!r.candidates.some((c) => c.gate)) continue;
    const p = r.candidates.find((c) => c.id === r.promoted?.candidate)?.gate?.stats;
    if (p) cur = p;
    pts.push({ label: `R${r.n}`, nominal: cur?.nominal.success, heldout: cur?.heldout.success });
  }
  return pts;
}
