"use client";

import { Bar, Check, Stage, Text, Window, type Act } from "@/components/stage/kit";
import "./hero.css";

// One CI round as it plays in the studio: raw episodes fail, the probe finds the
// cube inside the JEPA, fixes are gated, the champion's safe success rolls up.

type S = { step: number };
const START: S = { step: 0 };
const POSTER: S = { step: 6 };

// 12 raw episodes; which ones fail (drift seeds)
const FAILS = new Set([2, 3, 6, 7, 10, 11]);
// a 16x16 patch probe heat map: cube near (10, 9), a tail for the gripper above it
const HEAT = Array.from({ length: 256 }, (_, i) => {
  const x = i % 16, y = Math.floor(i / 16);
  const cube = Math.exp(-((x - 10) ** 2 + (y - 10) ** 2) / 1.6);
  const grip = 0.55 * Math.exp(-((x - 7) ** 2 + (y - 5) ** 2) / 2.4);
  return Math.max(cube, grip);
});

const FIXES = [
  { name: "Probe recal + drift DAgger", kind: "internal", pass: true },
  { name: "Circuit-restricted fine-tune", kind: "internal", pass: true },
  { name: "Guardrail · clip descent near table", kind: "external", pass: false, why: "blocks 7% of nominal" },
];

async function script(a: Act<S>) {
  await a.wait(400);
  a.put({ step: 1 }); // raw episodes play in
  await a.wait(1500);
  a.put({ step: 2 }); // failures marked
  await a.wait(1200);
  a.put({ step: 3 }); // probe heat map
  await a.wait(1600);
  a.put({ step: 4 }); // candidates arrive
  await a.wait(1400);
  a.put({ step: 5 }); // gate verdicts
  await a.wait(1300);
  a.put({ step: 6 }); // payoff
  await a.wait(3600);
  a.put({ step: 0 });
  await a.wait(700);
}

export function HeroDemo() {
  return (
    <Stage<S>
      width={980}
      height={540}
      compact={{ below: 720, width: 358, height: 600 }}
      initial={START}
      poster={POSTER}
      script={script}
      accent="#2747d6"
      className="eh"
    >
      {(s, _rect, layout) => {
        const compact = layout === "compact";
        return (
          <>
            <Window url="embodied-ci · studio · round 1" className="eh-window">
              <div className="eh-app">
                {!compact && (
                  <aside className="eh-rail">
                    <span className="eh-mark" />
                    {["Model", "Task", "Raw run", "Failures", "Fix + gate"].map((l, i) => (
                      <div key={l} className="eh-step" data-on={(i < 2 || (i === 2 && s.step >= 1) || (i === 3 && s.step >= 2) || (i === 4 && s.step >= 4)) || undefined}>
                        <span className="eh-dot" />
                        <Text className="eh-step-label">{l}</Text>
                      </div>
                    ))}
                    <Bar w={64} h={5} tone={2} style={{ marginTop: "auto" }} />
                  </aside>
                )}

                <main className="eh-main">
                  <header className="eh-head">
                    <span className="eh-title">V-JEPA 2-AC · Lift</span>
                    <span className="eh-chip num">H100</span>
                    <Bar w={52} h={5} tone={2} style={{ marginLeft: "auto" }} />
                  </header>

                  <div className="eh-grid">
                    <section className="eh-panel">
                      <div className="eh-label">Raw run, no guardrails</div>
                      <div className="eh-tiles">
                        {Array.from({ length: 12 }, (_, i) => (
                          <div
                            key={i}
                            className="eh-tile"
                            data-in={s.step >= 1 || undefined}
                            data-fail={(s.step >= 2 && FAILS.has(i)) || undefined}
                            style={{ transitionDelay: s.step === 1 ? `${i * 70}ms` : s.step === 2 ? `${[...FAILS].indexOf(i) * 90}ms` : "0ms" }}
                          >
                            <span className="eh-arm" />
                            <span className="eh-cube" />
                          </div>
                        ))}
                      </div>
                      <div className="eh-foot num" data-in={s.step >= 2 || undefined}>
                        <span className="eh-fail-dot" /> 17 of 40 fail · cube_color, camera_shift
                      </div>
                    </section>

                    <section className="eh-panel eh-probe" data-in={s.step >= 3 || undefined}>
                      <div className="eh-label">Inside the JEPA · cube probe, encoder layer 19</div>
                      <div className="eh-heat">
                        {HEAT.map((h, i) => (
                          <span key={i} style={{ opacity: s.step >= 3 ? 0.08 + 0.92 * h : 0.04, transitionDelay: s.step >= 3 ? `${(i % 16) * 18 + Math.floor(i / 16) * 10}ms` : "0ms" }} />
                        ))}
                        <span className="eh-reticle" data-in={s.step >= 3 || undefined} />
                      </div>
                      <div className="eh-foot num" data-in={s.step >= 3 || undefined}>
                        cube located ±2 px · AUROC 0.9999
                      </div>
                    </section>
                  </div>

                  <section className="eh-fixes">
                    {FIXES.map((f, i) => (
                      <div key={f.name} className="eh-fix" data-in={s.step >= 4 || undefined} style={{ transitionDelay: s.step >= 4 ? `${i * 120}ms` : "0ms" }}>
                        <span className="eh-kind" data-kind={f.kind}>{f.kind}</span>
                        <span className="eh-fix-name">{f.name}</span>
                        <span className="eh-verdict" data-show={s.step >= 5 || undefined} data-pass={f.pass || undefined} style={{ transitionDelay: s.step >= 5 ? `${i * 160}ms` : "0ms" }}>
                          {f.pass ? <Check on={s.step >= 5} size={14} /> : <span className="eh-x">×</span>}
                          <span>{f.pass ? "passes gate" : f.why}</span>
                        </span>
                      </div>
                    ))}
                  </section>
                </main>
              </div>
            </Window>

            <div className="eh-payoff" data-in={s.step >= 6 || undefined}>
              <div className="eh-label">diagnoser · cube_color:no_lift · confirmed</div>
              <p className="eh-diag">The side-view cube probe misreads a blue cube by 5–9 px, so the controller hovers and never closes the gripper. JEPA surprise does not notice.</p>
              <div className="eh-x-factor num">confidence 0.54 vs 0.78 nominal · 5 of 5 episodes</div>
            </div>
          </>
        );
      }}
    </Stage>
  );
}
