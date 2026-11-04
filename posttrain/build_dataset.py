"""Build the GRPO dataset for the Baseten guardrail-writer job from logged episodes.

    uv run modal run -m posttrain.build_dataset    # -> posttrain/baseten_grpo/data/train.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from infra.modal_app import app, image, vol

PROMPT = """You write safety guardrails for a robot arm (Franka Panda lifting a cube) driven by a learned policy.
Write ONE Python function, no imports:

def check(state, masks, z_now, z_pred, action):
    # return ALLOW to let the action through, or block("reason", safe_action) to replace it
    ...

state: eef_pos (3,) world metres, cube_pos (3,), gripper_closed (0 or 1), table_z (float), contact_force (newtons).
action: (7,) [dx, dy, dz, droll, dpitch, dyaw, dgripper] deltas in metres / radians. masks, z_now, z_pred are unavailable (None); do not use them.
Names available: np, ALLOW, block, HOLD (7 zeros), retreat_up(dz). Prefer clipping a bad action to blocking with HOLD.
Unsafe means: next eef outside x,y in [-0.35, 0.35], z in [0.80, 1.30]; step norm > 0.05 m; next eef z below table_z + 0.005; contact_force > 60.
It must catch the failures below and block fewer than 5% of normal steps.

Failure cluster: {cluster}  ({n_unsafe} unsafe steps across {n_eps} episodes)
Example unsafe steps (state -> proposed action -> violated rule):
{examples}

Reply with the function in one ```python block."""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=4)
def build(max_prompts: int = 96, seed: int = 0) -> list[dict]:
    import glob
    import os

    import numpy as np

    from embodied.sim import MAX_CONTACT_FORCE, MAX_STEP_M, TABLE_Z_MARGIN, WORKSPACE

    rng = np.random.default_rng(seed)

    def why(st, a):
        nxt = st["eef_pos"] + a[:3]
        r = []
        if np.any(nxt < WORKSPACE[:, 0]) or np.any(nxt > WORKSPACE[:, 1]):
            r.append("workspace")
        if np.linalg.norm(a[:3]) > MAX_STEP_M + 1e-6:
            r.append("speed")
        if nxt[2] < st["table_z"] + TABLE_Z_MARGIN:
            r.append("table_penetration")
        if st["contact_force"] > MAX_CONTACT_FORCE:
            r.append("contact_force")
        return r

    clusters: dict[str, list[dict]] = {}
    normal: list[dict] = []
    files = [f for f in glob.glob("/data/episodes/*.npz") if "-train-" not in f and "-heldout-" not in f]
    for f in files:
        eid = os.path.basename(f)[:-4]
        drift = eid.split("-")[2]
        with np.load(f) as d:
            if "eef_pos" not in d.files:
                continue
            tz = float(d["table_z"])
            for t in range(len(d["pose"])):
                st = {"eef_pos": d["eef_pos"][t].tolist(), "cube_pos": d["cube_pos"][t].tolist(),
                      "gripper_closed": float(d["gripper_closed"][t]), "table_z": tz, "contact_force": float(d["contact_force"][t])}
                a = d["proposed"][t].astype(float)
                reasons = why({k: np.asarray(v) if isinstance(v, list) else v for k, v in st.items()}, a)
                rec = {"state": st, "action": a.tolist(), "unsafe": bool(reasons), "ep": eid, "why": reasons}
                if reasons:
                    clusters.setdefault(f"{drift}:{'+'.join(sorted(reasons))}", []).append(rec)
                else:
                    normal.append(rec)
    rows = []
    keys = sorted(clusters)
    for i in range(max_prompts):
        if not keys:
            break
        k = keys[i % len(keys)]
        bad = clusters[k]
        ex = [bad[j] for j in rng.choice(len(bad), size=min(4, len(bad)), replace=False)]
        examples = "\n".join(f"- eef {np.round(e['state']['eef_pos'], 3).tolist()} cube {np.round(e['state']['cube_pos'], 3).tolist()} grip {e['state']['gripper_closed']} -> {np.round(e['action'], 3).tolist()} : {','.join(e['why'])}" for e in ex)
        reward_bad = [bad[j] for j in rng.choice(len(bad), size=min(40, len(bad)), replace=False)]
        reward_ok = [normal[j] for j in rng.choice(len(normal), size=min(160, len(normal)), replace=False)] if normal else []
        rows.append({"id": f"{k}#{i}", "prompt": PROMPT.format(cluster=k, n_unsafe=len(bad), n_eps=len({b["ep"] for b in bad}), examples=examples),
                     "steps": [{kk: r[kk] for kk in ("state", "action", "unsafe")} for r in reward_bad + reward_ok]})
    return rows


@app.local_entrypoint()
def main(max_prompts: int = 96):
    rows = build.remote(max_prompts)
    out = Path(__file__).parent / "baseten_grpo" / "data" / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{len(rows)} prompts -> {out}")
