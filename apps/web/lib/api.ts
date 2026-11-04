const DEFAULT_API = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const KEY = "embodiedci.api";

/** The engine API the studio talks to: the hosted demo, or your own deployment (Deploy -> Connect). */
export function apiBase(): string {
  if (typeof window === "undefined") return DEFAULT_API;
  try {
    return (localStorage.getItem(KEY) || DEFAULT_API).replace(/\/$/, "");
  } catch {
    return DEFAULT_API;
  }
}

export function setApiBase(url: string | null): void {
  try {
    if (url) localStorage.setItem(KEY, url.replace(/\/$/, ""));
    else localStorage.removeItem(KEY);
  } catch {}
}

export const DEFAULT_API_URL = DEFAULT_API;

const OWNER = "embodiedci.owner";
let memOwner = "";
/** A random id for this browser. The API stores only its hash next to each run you start, and lists only your runs (plus the demo runs). */
export function ownerId(): string {
  try {
    let id = localStorage.getItem(OWNER);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(OWNER, id);
    }
    return id;
  } catch {
    return (memOwner ||= crypto.randomUUID());
  }
}

export type Model = { id: string; name: string; org: string; params: string; gpu: string; editable: string; available: boolean };
export type Task = { id: string; name: string; sim: string; drifts: string[]; available: boolean };

export type Ev = {
  seq: number;
  t: number;
  run: string;
  round: number;
  stage: string;
  kind: string;
  [k: string]: unknown;
};

export type Episode = Ev & {
  id: string;
  candidate: string;
  seed: number;
  drift: string;
  suite: string;
  success: boolean;
  failure: null | "unsafe" | "no_lift";
  steps: number;
  unsafe_proposals: number;
  unsafe_executed: number;
  unsafe_reasons: Record<string, number>;
  blocked: number;
  block_reasons: Record<string, number>;
  max_surprise: number;
};

export type SuiteStats = {
  n: number;
  success?: number;
  success_ci?: [number, number];
  unsafe_per_100?: number;
  unsafe_exec_per_100?: number;
  blocked_per_100?: number;
};

export type GateResult = Ev & {
  candidate: string;
  note: string;
  passed: boolean;
  reasons: string[];
  stats: Record<"nominal" | "regression" | "heldout", SuiteStats>;
  base_stats: Record<"nominal" | "regression" | "heldout", SuiteStats>;
  regression_fixed: number;
  weights_changed: number;
  guardrails: string[];
};

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${apiBase()}${path}`, { cache: "no-store", ...init, headers: { ...init?.headers, "x-owner": ownerId() } });
  if (!r.ok) {
    const detail = await r.json().then((b) => (typeof b?.detail === "string" ? b.detail : ""), () => "");
    throw new Error(detail || `${r.status} ${path}`);
  }
  return r.json();
}

export type Outcome = { id: string; success: boolean; lifted: boolean; unsafe_executed: number; blocked: number; steps: number };
export type Pair = { drift: string; seed: number; a: Outcome; b: Outcome };

export const api = {
  lineage: () => j<{ v0: string | null; champion: string | null }>("/lineage"),
  compare: (a: string, b: string) => j<Pair[]>(`/compare/${a}/${b}`),
  bundleUrl: (cid: string, base?: string) => `${apiBase()}/candidates/${cid}/bundle.zip${base ? `?base=${base}` : ""}`,
  exportHF: (cid: string, body: { repo_id: string; token: string; private: boolean }, base?: string) =>
    j<{ url: string }>(`/candidates/${cid}/export/hf${base ? `?base=${base}` : ""}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  health: (base: string) => fetch(`${base.replace(/\/$/, "")}/health`, { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status))))) as Promise<{ ok: boolean; backend: string }>,
  models: () => j<Model[]>("/models"),
  tasks: () => j<Task[]>("/tasks"),
  runs: () => j<{ run_id: string; updated: number }[]>("/runs"),
  events: (run: string, after: number) => j<Ev[]>(`/runs/${run}/events?after=${after}`),
  start: (body: { model: string; task: string; rounds: number; use_agents: boolean; reset: boolean }) =>
    j<{ run_id: string }>("/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  video: (id: string) => `${apiBase()}/episodes/${id}.mp4`,
  still: (id: string) => `${apiBase()}/episodes/${id}.png`,
};
