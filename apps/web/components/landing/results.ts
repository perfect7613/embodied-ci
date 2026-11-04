// Numbers on the landing page, copied from the run logs (see README "Results").
// Baseline and probes: run-20261104-110352. Gate: run-20261104-235115 (round 2).
export const RESULTS = {
  caption:
    "Measured on the gate's fixed seeds: 50 nominal, plus 30 held-out drift seeds that no agent or training run sees.",
  // run-20261104-235115, round 2: gated against the round 1 champion (98% nominal, 63% held-out)
  gate: [
    { cand: "Probe recalibration + safety envelope", nominal: 1.0, heldout: 1.0, fixed: 11, verdict: "promoted", ok: true },
    { cand: "Probe recalibration", nominal: 0.96, heldout: NaN, fixed: NaN, verdict: "rejected: unsafe actions on nominal rose", ok: false },
    { cand: "DAgger fine-tune", nominal: 0.94, heldout: NaN, fixed: NaN, verdict: "rejected: nominal success fell", ok: false },
    { cand: "Muse Spark guardrail", nominal: 1.0, heldout: NaN, fixed: NaN, verdict: "rejected: blocks 36% of steps (max 5%)", ok: false },
  ],
  cubeErr: 2,
  layers: [
    { layer: 9, gripper: 11.6, best: false },
    { layer: 19, gripper: 4.2, best: true },
    { layer: 29, gripper: 4.8, best: false },
    { layer: 39, gripper: 3.9, best: false },
  ],
  attribution: `round 1 · 637 failing steps
fc1[24] ← gripper_closed, cube_v_agent
fc1[15] ← cube_v_agent, gripper_closed
fc1[31] ← eef_x, cube_v_agent, cube_v_side
top inputs: eef_y, cube_u_agent, gripper_u_agent
6 of 32 units carry 40% of the attribution`,
  detector: { auroc: "0.931", lead: "8/8", top: "eef_x, gripper_conf_agent, cube_conf_agent" },
  baseline: [
    { cond: "nominal", success: 0.86, found: "Competent: BC + DAgger on nominal demos only" },
    { cond: "low_friction ×0.05", success: 1.0, found: "Robust" },
    { cond: "heavy_cube ×20", success: 0.833, found: "Descends and lifts too fast: speed and contact-force violations (control)" },
    { cond: "dim_light ×0.25", success: 0.667, found: "Probes still fine (2.2 px); depth control overshoots into the table (control)" },
    { cond: "cube_color (blue)", success: 0.0, found: "Side probe misreads the cube by 5–9 px; controller hovers, never grasps (perception)" },
    { cond: "camera_shift 9 cm", success: 0.0, found: "Probes still localise in the image; the learned image→world mapping is wrong (control). The perception hypothesis was refuted" },
  ],
};
