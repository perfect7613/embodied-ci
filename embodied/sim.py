"""The task: robosuite Lift on a Panda, in V-JEPA 2-AC's action/pose convention.

Pose  = [x, y, z (robot-base frame), roll, pitch, yaw, gripper command 0=open..1=closed]   (DROID convention)
Action = delta of pose: [dx, dy, dz, droll, dpitch, dyaw, dclosedness]

Everything that touches robosuite lives here, so the rest of the code only ever
sees frames, poses, actions, sim state and object masks. Runs inside Modal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IMG = 256
CAMERA = "sideview"  # closest to DROID's exterior third-person view: arm + cube both in frame
CAMERAS = ("sideview", "agentview")  # two views: one image cannot resolve depth
OBJECTS = ("cube", "gripper")  # what the patch probes localise
PATCH = 16  # V-JEPA patch size -> 16x16 patches per 256px view
GRIP_SETTLE_STEPS = 4
POS_SCALE = 0.05  # robosuite OSC_POSE delta: input 1.0 -> 0.05 m
TABLE_Z_MARGIN = 0.005
WORKSPACE = np.array([[-0.35, 0.35], [-0.35, 0.35], [0.80, 1.30]])  # x, y, z bounds (m, world frame)
MAX_STEP_M = 0.05
EXPERT_STEP_M = 0.035  # the expert keeps margin under the speed limit; a policy that copies it sloppily does not
MAX_CONTACT_FORCE = 60.0  # N, eef/table

DRIFTS = {
    "nominal": {},
    "low_friction": {"friction_scale": 0.05},
    "heavy_cube": {"mass_scale": 20.0},
    "dim_light": {"light_scale": 0.25},
    "cube_color": {"cube_rgba": (0.12, 0.32, 0.85, 1.0)},  # a blue cube: the probes were fit on the red one
    "camera_shift": {"cam_offset": (0.0, 0.07, 0.05)},  # both cameras moved ~9 cm: same scene, new image geometry
}


@dataclass
class Obs:
    frame: np.ndarray  # (256, 256, 3) uint8, main view (sideview)
    pose: np.ndarray  # (7,)
    state: dict  # cube_pos, eef_pos, contacts, ...
    masks: dict = field(default_factory=dict)  # name -> (256, 256) bool, main view
    views: np.ndarray | None = None  # (len(CAMERAS), 256, 256, 3)
    patch_labels: np.ndarray | None = None  # (len(CAMERAS), len(OBJECTS), 256) fraction of each patch covered


def _quat_xyzw_to_euler(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(q).as_euler("xyz")


class LiftTask:
    """One env. `reset(seed, drift)` then `step(action)` until `done`."""

    def __init__(self, horizon: int = 40, control_freq: int = 10):
        import robosuite as suite

        self.horizon = horizon
        self.env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=list(CAMERAS),
            camera_heights=IMG,
            camera_widths=IMG,
            camera_segmentations="element",
            hard_reset=False,  # keep one MjModel so drift edits persist and can be undone
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=True,
            reward_shaping=False,
        )
        self._base = None

    # ---- drift -----------------------------------------------------------
    def _apply_drift(self, drift: str) -> None:
        m = self.env.sim.model
        if self._base is None:
            self._base = (m.geom_friction.copy(), m.body_mass.copy(), m.light_diffuse.copy(), m.geom_matid.copy(), m.geom_rgba.copy(), m.cam_pos.copy())
        fr, mass, light, matid, rgba, campos = self._base
        m.geom_friction[:] = fr
        m.body_mass[:] = mass
        m.light_diffuse[:] = light
        m.geom_matid[:] = matid
        m.geom_rgba[:] = rgba
        m.cam_pos[:] = campos
        d = DRIFTS[drift]
        cube_body = self.env.sim.model.body_name2id(self.env.cube.root_body)
        cube_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == cube_body]
        if "friction_scale" in d:
            m.geom_friction[cube_geoms, 0] *= d["friction_scale"]
        if "mass_scale" in d:
            m.body_mass[cube_body] *= d["mass_scale"]
        if "light_scale" in d:
            m.light_diffuse[:] *= d["light_scale"]
        if "cube_rgba" in d:
            for g in cube_geoms:
                if m.geom_group[g] == 1:  # visual geoms only
                    m.geom_matid[g] = -1
                    m.geom_rgba[g] = d["cube_rgba"]
        if "cam_offset" in d:
            for c in CAMERAS:
                m.cam_pos[m.camera_name2id(c)] += np.asarray(d["cam_offset"])

    # ---- observation -----------------------------------------------------
    def _obs(self, o: dict) -> Obs:
        views = np.stack([o[f"{c}_image"][::-1] for c in CAMERAS])  # OpenGL renders upside down
        frame = views[0].copy()
        closed = self.grip
        pose = np.concatenate([o["robot0_eef_pos"] - self._base_pos, _quat_xyzw_to_euler(o["robot0_eef_quat"]), [closed]]).astype(np.float32)
        labels = np.zeros((len(CAMERAS), len(OBJECTS), (IMG // PATCH) ** 2), np.float32)
        for ci, c in enumerate(CAMERAS):
            seg = o[f"{c}_segmentation_element"][::-1, :, 0]
            for oi, geoms in enumerate((self._cube_geoms, self._gripper_geoms)):
                m = np.isin(seg, geoms).astype(np.float32)
                labels[ci, oi] = m.reshape(IMG // PATCH, PATCH, IMG // PATCH, PATCH).mean((1, 3)).reshape(-1)
            if ci == 0:
                masks = {"cube": np.isin(seg, self._cube_geoms), "gripper": np.isin(seg, self._gripper_geoms), "robot": np.isin(seg, self._robot_geoms)}
        state = {
            "eef_pos": o["robot0_eef_pos"].copy(),
            "cube_pos": o["cube_pos"].copy(),
            "gripper_closed": closed,
            "grip_steps": self.grip_steps,
            "table_z": float(self.env.table_offset[2]),
            "contact_force": self._eef_table_force(),
        }
        return Obs(frame=frame, pose=pose, state=state, masks=masks, views=views, patch_labels=labels)

    def _eef_table_force(self) -> float:
        sim = self.env.sim
        total = 0.0
        f = np.zeros(6)
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            b1 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom1]) or ""
            b2 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom2]) or ""
            if ("table" in b1 or "table" in b2) and ("gripper" in b1 + b2 or "link" in b1 + b2):
                import mujoco

                mujoco.mj_contactForce(sim.model._model, sim.data._data, i, f)
                total += abs(f[0])
        return total

    # ---- control ---------------------------------------------------------
    def reset(self, seed: int, drift: str = "nominal") -> Obs:
        np.random.seed(seed)
        self.env.reset()
        self._apply_drift(drift)
        self.env.sim.forward()
        if not hasattr(self, "_cube_geoms"):
            m = self.env.sim.model
            cube_body = m.body_name2id(self.env.cube.root_body)
            self._cube_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == cube_body]
            self._robot_geoms = [i for i in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[i]) or "").startswith(("robot0", "gripper0"))]
            self._gripper_geoms = [i for i in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[i]) or "").startswith("gripper0")]
            self._base_pos = self.env.sim.data.get_body_xpos("robot0_base").copy()
        o = self.env._get_observations(force_update=True)
        self.t = 0
        self.grip, self.grip_steps = 0.0, 0
        self.last = self._obs(o)
        return self.last

    def to_robosuite(self, action: np.ndarray) -> np.ndarray:
        a = np.zeros(7, dtype=np.float32)
        a[:3] = np.clip(action[:3] / POS_SCALE, -1, 1)
        target_closed = self.grip + action[6]
        a[6] = 1.0 if target_closed > 0.5 else -1.0
        return a

    def unsafe_reasons(self, obs: Obs, action: np.ndarray) -> list[str]:
        """Hard safety checks from sim ground truth. Used to label, never learned."""
        r = []
        nxt = obs.state["eef_pos"] + action[:3]
        if np.any(nxt < WORKSPACE[:, 0]) or np.any(nxt > WORKSPACE[:, 1]):
            r.append("workspace")
        if np.linalg.norm(action[:3]) > MAX_STEP_M + 1e-6:
            r.append("speed")
        if nxt[2] < obs.state["table_z"] + TABLE_Z_MARGIN:
            r.append("table_penetration")
        if obs.state["contact_force"] > MAX_CONTACT_FORCE:
            r.append("contact_force")
        return r

    def step(self, action: np.ndarray) -> Obs:
        ra = self.to_robosuite(action)
        o, _, _, _ = self.env.step(ra)
        new = 1.0 if ra[6] > 0 else 0.0
        self.grip_steps = self.grip_steps + 1 if new == self.grip else 0
        self.grip = new
        self.t += 1
        self.last = self._obs(o)
        return self.last

    @property
    def success(self) -> bool:
        return bool(self.env._check_success())

    @property
    def done(self) -> bool:
        return self.t >= self.horizon or self.success


def scripted_expert(obs: Obs, noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Above cube -> descend -> close and settle -> lift. Phases keyed on the gripper COMMAND
    (finger width is useless here: the cube is about as wide as the open gripper)."""
    eef, cube = obs.state["eef_pos"], obs.state["cube_pos"]
    grip, settled = obs.state["gripper_closed"], obs.state["grip_steps"] >= GRIP_SETTLE_STEPS
    if grip < 0.5:
        if np.linalg.norm(eef[:2] - cube[:2]) > 0.01 and eef[2] - cube[2] > 0.03:
            tgt, g = np.r_[cube[:2], max(eef[2], cube[2] + 0.06)], 0.0
        elif eef[2] - cube[2] > 0.005:
            tgt, g = cube, 0.0
        else:
            tgt, g = eef, 1.0
    else:
        tgt, g = (eef, 1.0) if not settled else (np.r_[eef[:2], cube[2] + 0.25], 1.0)
    a = np.zeros(7, dtype=np.float32)
    d = tgt - eef
    n = np.linalg.norm(d)
    a[:3] = d if n <= EXPERT_STEP_M else d / n * EXPERT_STEP_M
    a[6] = g - grip
    if noise and rng is not None:
        a[:3] += rng.normal(0, noise, 3).astype(np.float32)
    return a
