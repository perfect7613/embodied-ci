"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { DEFAULT_API_URL, api, apiBase, setApiBase } from "@/lib/api";

const REPO = "https://github.com/perfect7613/embodied-ci";
type Target = "modal" | "ssh";

export default function Deploy() {
  const [target, setTarget] = useState<Target>("modal");
  const [key, setKey] = useState("");
  const [llm, setLlm] = useState("meta-models/muse-spark-open-1026");
  const [host, setHost] = useState("you@gpu-box");
  const [reveal, setReveal] = useState(false);

  const shownKey = key ? (reveal ? key : `${key.slice(0, 4)}${"•".repeat(Math.max(0, Math.min(key.length - 4, 20)))}`) : "<your-baseten-key>";
  const commands = useMemo(() => {
    const k = reveal || !key ? shownKey : key; // the copied text always carries the real key
    return target === "modal"
      ? { shown: modalCmd(shownKey, llm), copy: modalCmd(k, llm) }
      : { shown: sshCmd(shownKey, llm, host), copy: sshCmd(k, llm, host) };
  }, [target, key, reveal, shownKey, llm, host]);

  return (
    <div className="mx-auto max-w-[860px] px-4 pb-24 sm:px-6">
      <nav className="flex items-center gap-4 py-5">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="grid h-[22px] w-[22px] place-items-center rounded-[6px] border-2 border-ink">
            <span className="h-[6px] w-[6px] rounded-[1px] bg-accent" />
          </span>
          <span className="text-[15px] font-semibold tracking-tight">EmbodiedCI</span>
        </Link>
        <span className="text-ink-4">/</span>
        <span className="text-[14px] text-ink-2">Run on your own GPU</span>
        <Link href="/studio" className="ml-auto text-[13px] text-ink-3 hover:text-ink">Studio →</Link>
      </nav>

      <header className="pt-10">
        <h1 className="font-serif text-[40px] leading-[1.05] tracking-[-0.015em] sm:text-[52px]">Run it on your own GPU.</h1>
        <p className="mt-4 max-w-[600px] text-[16px] leading-[1.6] text-ink-2">
          The studio runs on our demo server by default. Deploy your own copy in about ten minutes, then point the studio at it. Your runs, models and keys stay in your account.
        </p>
        <div className="mt-7 grid gap-px overflow-hidden rounded-[12px] border border-line bg-line sm:grid-cols-3">
          <Need k="You need" v="A Modal account, or a CUDA GPU with 40 GB+" />
          <Need k="For the agents" v="A Baseten API key (optional)" />
          <Need k="Cost" v="~$4 per H100-hour while a run is going. Idle is free." />
        </div>
      </header>

      <Step n={1} title="Choose where it runs">
        <div className="grid gap-3 sm:grid-cols-2">
          <Choice on={target === "modal"} onClick={() => setTarget("modal")} title="Modal" badge="Easiest" sub="One deploy command. Modal starts an H100 only while a run is going." />
          <Choice on={target === "ssh"} onClick={() => setTarget("ssh")} title="Your own GPU" sub="Any CUDA machine you can SSH into. A script installs and starts the engine." />
        </div>
      </Step>

      <Step n={2} title="Add your Baseten key" sub="Muse Spark agents diagnose failures and write guardrails through it. Skip it to run without agents. The key only fills in the command below and never leaves this tab.">
        <div className="relative max-w-[460px]">
          <input type="password" autoComplete="off" spellCheck={false} value={key} onChange={(e) => setKey(e.target.value.trim())} placeholder="Baseten API key"
            className="num w-full rounded-[8px] border border-line-strong bg-raised px-3 py-2 text-[13px] outline-none focus:border-accent" />
        </div>
        {target === "ssh" && (
          <label className="mt-3 block max-w-[460px]">
            <span className="text-[12px] text-ink-3">How you SSH into the GPU machine</span>
            <input value={host} onChange={(e) => setHost(e.target.value.trim())}
              className="num mt-1 w-full rounded-[8px] border border-line-strong bg-raised px-3 py-2 text-[13px] outline-none focus:border-accent" />
          </label>
        )}
      </Step>

      <Step n={3} title="Run these commands" sub={target === "modal"
        ? "In a terminal on your laptop. The last line prints your API address, ending in --embodied-ci-api.modal.run. Copy it for step 4."
        : "In a terminal on your laptop. Leave the last command running: it makes your GPU reachable at http://localhost:8700."}>
        <Terminal text={commands.shown} copy={commands.copy} onReveal={key ? () => setReveal((r) => !r) : undefined} revealed={reveal} />
      </Step>

      <Step n={4} title="Connect the studio" sub="Paste the address. The studio checks it answers, then uses it in this browser.">
        <Connect />
      </Step>

      <details className="group mt-14 border-t border-line pt-6">
        <summary className="cursor-pointer list-none text-[13.5px] text-ink-3 hover:text-ink">
          <span className="inline-block transition-transform group-open:rotate-90">›</span> Advanced
        </summary>
        <div className="mt-5 space-y-6 pl-4">
          <label className="block max-w-[460px]">
            <span className="text-[12px] text-ink-3">Agent model on Baseten Model APIs</span>
            <input value={llm} onChange={(e) => setLlm(e.target.value.trim())}
              className="num mt-1 w-full rounded-[8px] border border-line-strong bg-raised px-3 py-2 text-[13px] outline-none focus:border-accent" />
          </label>
          <div>
            <div className="text-[13.5px] font-medium">Post-train a guardrail writer</div>
            <p className="mt-1 mb-3 max-w-[600px] text-[13px] leading-[1.6] text-ink-3">
              A GRPO job (Qwen3-4B, LoRA) on Baseten Training, rewarded by the gate&apos;s own rules on your logged episodes. Needs a key with Training access.
            </p>
            <Terminal text={GRPO} copy={GRPO} />
          </div>
        </div>
      </details>
    </div>
  );
}

function Need({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-raised px-4 py-3.5">
      <div className="text-[11.5px] text-ink-4">{k}</div>
      <div className="mt-1 text-[13.5px] leading-snug text-ink-2">{v}</div>
    </div>
  );
}

function modalCmd(key: string, llm: string) {
  return `git clone ${REPO} && cd embodied-ci
uv sync                                   # install
uv run modal setup                        # sign in to Modal (opens your browser)
umask 077 && printf 'BASETEN_API_KEY=%s\\nBASETEN_MODEL=%s\\n' '${key}' '${llm}' > .env
uv run modal deploy infra/modal_app.py    # prints your API address`;
}

function sshCmd(key: string, llm: string, host: string) {
  return `git clone ${REPO} && cd embodied-ci
ssh ${host} "BASETEN_API_KEY='${key}' BASETEN_MODEL='${llm}' bash -s" < deploy/ssh/bootstrap.sh
ssh -N -L 8700:localhost:8700 ${host}   # keep open: the studio talks to http://localhost:8700`;
}

const GRPO = `uv run modal run -m posttrain.build_dataset   # failure clusters + replay steps -> data/train.jsonl
cd posttrain/baseten_grpo
uv run truss login                              # a key with Baseten Training access
uv run truss train push config.py`;

function Step({ n, title, sub, children }: { n: number; title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="mt-14 grid gap-4 sm:grid-cols-[40px_1fr]">
      <span className="num grid size-7 place-items-center rounded-full border border-line-strong text-[12px] text-ink-3">{n}</span>
      <div className="min-w-0">
        <h2 className="text-[17px] font-semibold tracking-tight">{title}</h2>
        {sub && <p className="mt-1 max-w-[600px] text-[13.5px] leading-[1.6] text-ink-3">{sub}</p>}
        <div className="mt-4">{children}</div>
      </div>
    </section>
  );
}

function Choice({ on, onClick, title, sub, badge }: { on: boolean; onClick: () => void; title: string; sub: string; badge?: string }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={on}
      className={`rounded-[10px] border bg-raised p-4 text-left transition-colors ${on ? "border-accent shadow-[0_0_0_3px_var(--accent-soft)]" : "border-line hover:border-line-strong"}`}>
      <div className="flex items-center gap-2">
        <span className={`h-[9px] w-[9px] rounded-full border ${on ? "border-accent bg-accent" : "border-line-strong"}`} />
        <span className="text-[14px] font-medium">{title}</span>
        {badge && <span className="ml-auto rounded-full bg-accent-soft px-2 py-0.5 text-[11px] text-accent">{badge}</span>}
      </div>
      <p className="mt-1.5 pl-[17px] text-[12.5px] leading-[1.5] text-ink-3">{sub}</p>
    </button>
  );
}

function Terminal({ text, copy, onReveal, revealed }: { text: string; copy: string; onReveal?: () => void; revealed?: boolean }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="overflow-hidden rounded-[10px] bg-[#1b1a17]">
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2">
        <span className="num text-[11px] text-[#a29e94]">shell</span>
        {onReveal && (
          <button type="button" onClick={onReveal} className="ml-auto text-[11.5px] text-[#a29e94] hover:text-white">{revealed ? "Hide key" : "Show key"}</button>
        )}
        <button type="button" className={`${onReveal ? "" : "ml-auto"} rounded-[6px] bg-white/10 px-2.5 py-1 text-[11.5px] text-white hover:bg-white/20`}
          onClick={() => navigator.clipboard.writeText(copy).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1400); })}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="num overflow-x-auto p-4 text-[12.5px] leading-[1.7] text-[#e9e6df]">{text}</pre>
    </div>
  );
}

function Connect() {
  const [url, setUrl] = useState("");
  const [current, setCurrent] = useState("");
  const [state, setState] = useState<{ kind: "idle" | "checking" | "ok" | "err"; msg?: string }>({ kind: "idle" });
  useEffect(() => setCurrent(apiBase()), []);

  const check = async () => {
    const u = url.trim().replace(/\/$/, "");
    if (!/^https?:\/\//.test(u)) return setState({ kind: "err", msg: "Enter a full URL, starting with https:// or http://localhost" });
    setState({ kind: "checking" });
    try {
      const h = await api.health(u);
      setApiBase(u);
      setCurrent(u);
      setState({ kind: "ok", msg: `Connected: ${h.backend === "ssh" ? "your GPU" : "your Modal"} deployment` });
    } catch (e) {
      setState({ kind: "err", msg: `No EmbodiedCI API answered at ${u}/health (${(e as Error).message}).` });
    }
  };

  return (
    <div>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://<you>--embodied-ci-api.modal.run  or  http://localhost:8700"
          className="num min-w-0 flex-1 rounded-[8px] border border-line-strong bg-raised px-3 py-2 text-[13px] outline-none focus:border-accent" />
        <button type="button" onClick={check} className="rounded-[8px] bg-ink px-4 py-2 text-[13px] font-medium text-surface hover:bg-ink-2">
          {state.kind === "checking" ? "Checking…" : "Test and connect"}
        </button>
      </div>
      {state.msg && <p className={`mt-2 text-[12.5px] ${state.kind === "ok" ? "text-pass" : "text-fail"}`}>{state.msg}</p>}
      <div className="mt-4 flex flex-wrap items-center gap-3 rounded-[8px] bg-surface px-3 py-2.5 text-[12.5px] text-ink-3">
        <span>{current && current !== DEFAULT_API_URL ? "Studio is connected to" : "Studio is using the demo server"}</span>
        {current && current !== DEFAULT_API_URL && <span className="num text-ink-2">{current}</span>}
        {current && current !== DEFAULT_API_URL && (
          <button type="button" className="text-accent hover:underline" onClick={() => { setApiBase(null); setCurrent(DEFAULT_API_URL); setState({ kind: "idle" }); }}>
            Switch back to the demo
          </button>
        )}
        <Link href="/studio" className="ml-auto font-medium text-ink hover:underline">Open studio →</Link>
      </div>
    </div>
  );
}
