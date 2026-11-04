import Link from "next/link";
import { HeroDemo } from "@/components/landing/HeroDemo";
import { RESULTS } from "@/components/landing/results";

export const metadata = {
  title: "EmbodiedCI: continuous integration for JEPA world models",
  description:
    "Run a JEPA world model on a robot task, find its failures, look inside it, fix it in the weights and with guardrails, and promote only what a held-out gate passes.",
};

const REPO = "https://github.com/perfect7613/embodied-ci";

export default function Landing() {
  return (
    <div className="landing">
      <Nav />
      <Hero />
      <Loop />
      <Inside />
      <Fixes />
      <Gate />
      <Results />
      <Compute />
      <Footer />
    </div>
  );
}

function Nav() {
  return (
    <nav className="mx-auto flex max-w-[1120px] items-center gap-6 px-4 py-5 sm:px-6">
      <Link href="/" className="flex items-center gap-2.5">
        <span className="grid h-[22px] w-[22px] place-items-center rounded-[6px] border-2 border-ink">
          <span className="h-[6px] w-[6px] rounded-[1px] bg-accent" />
        </span>
        <span className="text-[15px] font-semibold tracking-tight">EmbodiedCI</span>
      </Link>
      <div className="ml-6 hidden items-center gap-6 text-[13px] text-ink-3 md:flex">
        <a href="#loop" className="hover:text-ink">How it works</a>
        <a href="#inside" className="hover:text-ink">Inside the JEPA</a>
        <a href="#gate" className="hover:text-ink">The gate</a>
        <Link href="/deploy" className="hover:text-ink">Self-host</Link>
        <a href={REPO} className="hover:text-ink">GitHub</a>
      </div>
      <Link href="/studio" className="ml-auto rounded-[8px] bg-ink px-3.5 py-2 text-[13px] font-medium text-surface hover:bg-ink-2">
        Open studio
      </Link>
    </nav>
  );
}

function Hero() {
  return (
    <header className="mx-auto max-w-[1120px] px-4 pt-12 sm:px-6 sm:pt-20">
      <h1 className="max-w-[880px] font-serif text-[44px] leading-[1.02] tracking-[-0.02em] sm:text-[72px]">
        Continuous integration <em className="text-accent">for world models.</em>
      </h1>
      <p className="mt-6 max-w-[620px] text-[17px] leading-[1.55] text-ink-2">
        Find where your robot model fails, see why inside it, and ship only the fixes that pass a held-out gate.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link href="/studio" className="rounded-[9px] bg-accent px-4 py-2.5 text-[14px] font-medium text-white hover:brightness-110">
          Open the studio
        </Link>
        <Link href="/deploy" className="rounded-[9px] border border-line-strong bg-raised px-4 py-2.5 text-[14px] font-medium hover:border-ink-3">
          Run on your own GPU
        </Link>
      </div>
      <div className="mt-14 sm:mt-16">
        <HeroDemo />
      </div>
    </header>
  );
}

function Section({ id, eyebrow, title, children, lede }: { id?: string; eyebrow: string; title: React.ReactNode; lede?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mx-auto max-w-[1120px] scroll-mt-8 px-4 pt-28 sm:px-6">
      <p className="num text-[12px] uppercase tracking-[0.12em] text-accent">{eyebrow}</p>
      <h2 className="mt-3 max-w-[760px] font-serif text-[32px] leading-[1.1] tracking-[-0.015em] sm:text-[44px]">{title}</h2>
      {lede && <p className="mt-4 max-w-[640px] text-[16px] leading-[1.6] text-ink-2">{lede}</p>}
      <div className="mt-10">{children}</div>
    </section>
  );
}

const STEPS = [
  ["Raw run", "No guardrails, nominal and drift."],
  ["Failures → tests", "Every failure joins the regression suite."],
  ["Look inside", "Probes, attribution, a failure detector."],
  ["Fix", "Weights inside, guardrails outside."],
  ["Gate", "Held-out seeds decide what ships."],
];

function Loop() {
  return (
    <Section id="loop" eyebrow="How it works" title="One round, end to end.">
      <ol className="grid gap-px overflow-hidden rounded-[12px] border border-line bg-line sm:grid-cols-5">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="bg-raised p-5">
            <span className="num text-[12px] text-ink-4">0{i + 1}</span>
            <h3 className="mt-3 text-[15px] font-semibold tracking-tight">{t}</h3>
            <p className="mt-2 text-[13px] leading-[1.55] text-ink-3">{d}</p>
          </li>
        ))}
      </ol>
    </Section>
  );
}

function Inside() {
  const layers = RESULTS.layers;
  const max = Math.max(...layers.map((l) => l.gripper));
  return (
    <Section id="inside" eyebrow="Mechanistic interpretability" title={<>Look inside the JEPA <em>before</em> you touch it.</>}>
      <div className="grid gap-4 md:grid-cols-3">
        <article className="card p-6">
          <h3 className="text-[15px] font-semibold">Patch probes, swept across depth</h3>
          <p className="mt-2 text-[13px] leading-[1.55] text-ink-3">
            Where in the encoder can the model see the cube and gripper?
          </p>
          <div className="mt-5 space-y-2.5">
            {layers.map((l) => (
              <div key={l.layer} className="grid grid-cols-[52px_1fr_44px] items-center gap-2 text-[12px]">
                <span className="num text-ink-3">L{l.layer}</span>
                <div className="relative h-[8px] rounded-full bg-ground">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${(l.gripper / max) * 100}%`, opacity: l.best ? 1 : 0.35 }} />
                </div>
                <span className="num text-right text-ink-2">{l.gripper.toFixed(1)}px</span>
              </div>
            ))}
            <p className="num pt-1 text-[11px] text-ink-4">gripper localisation error by layer · cube ±{RESULTS.cubeErr}px everywhere</p>
          </div>
        </article>
        <article className="card p-6">
          <h3 className="text-[15px] font-semibold">Attribution over the controller</h3>
          <p className="mt-2 text-[13px] leading-[1.55] text-ink-3">
            Which units drive the failing actions. Fine-tune only those.
          </p>
          <pre className="num mt-5 overflow-x-auto rounded-[8px] bg-ground p-3 text-[11.5px] leading-[1.6] text-ink-2">
{RESULTS.attribution}
          </pre>
        </article>
        <article className="card p-6">
          <h3 className="text-[15px] font-semibold">A failure-ahead detector</h3>
          <p className="mt-2 text-[13px] leading-[1.55] text-ink-3">
            Probe features plus JEPA surprise flag a failure before it lands.
          </p>
          <div className="mt-5 grid grid-cols-2 gap-3">
            <Stat k="AUROC" v={RESULTS.detector.auroc} />
            <Stat k="failures caught" v={RESULTS.detector.lead} />
          </div>
          <p className="num mt-3 text-[11px] text-ink-4">top signal · {RESULTS.detector.top}</p>
        </article>
      </div>
    </Section>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-[8px] bg-ground px-3 py-2.5">
      <div className="text-[11px] text-ink-3">{k}</div>
      <div className="num mt-0.5 text-[20px] text-ink">{v}</div>
    </div>
  );
}

const GUARD = `def check(state, masks, z_now, z_pred, action):
    # predicted contact with the table: clip, don't stop
    nxt = state["eef_pos"][2] + action[2]
    if nxt < state["table_z"] + 0.012:
        a = action.copy(); a[2] = max(a[2], 0.0)
        return block("near_table", a)
    if state["perception_conf"] < 0.3:
        return block("cant_see_cube", retreat_up())
    return ALLOW`;

function Fixes() {
  return (
    <Section eyebrow="Fix inside and outside" title="Weights for the cause, guardrails for the consequence.">
      <div className="grid gap-4 md:grid-cols-2">
        <article className="card p-6">
          <span className="num text-[11px] uppercase tracking-[0.08em] text-ink-3">internal · weights on Modal</span>
          <h3 className="mt-2 font-serif text-[24px] leading-tight">Fix the cause in the weights.</h3>
          <p className="mt-3 text-[13.5px] leading-[1.55] text-ink-2">Probe recalibration, DAgger, circuit-restricted fine-tunes and Muse Spark weight edits.</p>
        </article>
        <article className="card overflow-hidden">
          <div className="p-6 pb-4">
            <span className="num text-[11px] uppercase tracking-[0.08em] text-warn">external · written by Muse Spark</span>
            <h3 className="mt-2 font-serif text-[24px] leading-tight">Guard the consequence.</h3>
            <p className="mt-3 text-[13.5px] leading-[1.55] text-ink-2">Sandboxed, tested in sim, fails closed.</p>
          </div>
          <pre className="num overflow-x-auto border-t border-line bg-[#1b1a17] p-5 text-[12px] leading-[1.65] text-[#e9e6df]">{GUARD}</pre>
        </article>
      </div>
    </Section>
  );
}

const RULES = [
  "Nominal success may not fall",
  "Unsafe actions may not rise",
  "Guardrails block ≤ 5% of steps",
  "No passing test may break",
  "Held-out drift may not fall",
  "Must fix at least one test",
];

function Gate() {
  return (
    <Section id="gate" eyebrow="The gate" title="A fix is a hypothesis until the gate passes it.">
      <div className="grid gap-4 md:grid-cols-[1fr_1.2fr]">
        <div className="grid gap-3">
          {[
            ["Nominal", "50 seeds", "Still works?"],
            ["Regression", "grows each round", "Past failures fixed?"],
            ["Held-out drift", "30 unseen seeds", "Generalises?"],
          ].map(([n, c, d]) => (
            <div key={n} className="card flex items-baseline gap-4 p-5">
              <span className="w-[120px] text-[15px] font-semibold">{n}</span>
              <span className="num w-[150px] text-[12px] text-ink-3">{c}</span>
              <span className="text-[13px] text-ink-2">{d}</span>
            </div>
          ))}
        </div>
        <ul className="card divide-y divide-line">
          {RULES.map((r) => (
            <li key={r} className="flex items-center gap-3 px-5 py-3 text-[13.5px] text-ink-2">
              <span className="h-[6px] w-[6px] shrink-0 rounded-full bg-pass" />
              {r}
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}

function Results() {
  const r = RESULTS;
  return (
    <Section eyebrow="Results so far" title={<>From 18 failures to <em className="text-accent">zero</em>, on 80 seeds it never saw.</>} lede="v0 failed 18 of 80 gate episodes. After two rounds the champion fails none: 50/50 nominal, 30/30 held-out drift.">
      <div className="card overflow-x-auto">
        <table className="w-full min-w-[560px] text-[13px]">
          <thead className="text-left text-[11.5px] text-ink-3">
            <tr className="border-b border-line">
              <th className="px-5 py-3 font-normal">round 2 candidate (vs. champion: 98% nominal, 63% held-out)</th>
              <th className="px-5 py-3 text-right font-normal">nominal</th>
              <th className="px-5 py-3 text-right font-normal">held-out drift</th>
              <th className="px-5 py-3 text-right font-normal">tests fixed</th>
              <th className="px-5 py-3 font-normal">gate</th>
            </tr>
          </thead>
          <tbody>
            {r.gate.map((g) => (
              <tr key={g.cand} className="border-b border-line last:border-0">
                <td className="px-5 py-3 font-medium">{g.cand}</td>
                <td className="num px-5 py-3 text-right">{pct(g.nominal)}</td>
                <td className="num px-5 py-3 text-right">{pct(g.heldout)}</td>
                <td className="num px-5 py-3 text-right">{Number.isNaN(g.fixed) ? "–" : g.fixed}</td>
                <td className={`px-5 py-3 ${g.ok ? "text-pass" : "text-fail"}`}>{g.verdict}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[12.5px] text-ink-4">– : rejected on nominal, so the gate never ran its held-out and regression suites.</p>
    </Section>
  );
}

const pct = (x: number) => (Number.isNaN(x) ? "–" : `${Math.round(x * 100)}%`);

function Compute() {
  return (
    <Section id="deploy" eyebrow="Bring your own compute" title="Your keys, your GPUs." lede="Your Baseten key and GPUs never touch our servers.">
      <div className="grid gap-4 md:grid-cols-3">
        {[
          ["Modal", "One command deploys to your workspace."],
          ["Your GPU over SSH", "One script, served over a tunnel."],
          ["Baseten", "Agents run on Muse Spark."],
        ].map(([t, d]) => (
          <div key={t} className="card p-6">
            <h3 className="text-[15px] font-semibold">{t}</h3>
            <p className="mt-2 text-[13px] leading-[1.55] text-ink-3">{d}</p>
          </div>
        ))}
      </div>
      <Link href="/deploy" className="mt-6 inline-flex rounded-[9px] bg-ink px-4 py-2.5 text-[14px] font-medium text-surface hover:bg-ink-2">
        Run on your own GPU →
      </Link>
    </Section>
  );
}

function Footer() {
  return (
    <footer className="mx-auto mt-28 flex max-w-[1120px] flex-wrap items-center gap-4 border-t border-line px-4 py-8 text-[12.5px] text-ink-3 sm:px-6">
      <span>EmbodiedCI · built for the Meta Global AI Developer Hackathon 2026</span>
      <span className="ml-auto">V-JEPA 2-AC and Muse Spark are Meta models.</span>
      <a href={REPO} className="hover:text-ink">Source</a>
    </footer>
  );
}
