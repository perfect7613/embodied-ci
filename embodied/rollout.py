"""Run episodes of (JEPA world model + action head + guardrails) on the task.

Per step: both camera views go through the V-JEPA encoder once; the candidate's
linear probes read cube/gripper positions off the probe layer (named features);
the controller acts on those; the action-conditioned predictor forecasts the
next latent (surprise + guardrail input). Logged per step: features, pose,
proposed / executed action, the scripted expert's label for the same state
(free DAgger supervision), surprise, unsafe reasons, guardrail blocks, and the
probes' error against sim ground truth (so perception failures are visible).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from embodied.candidate import Candidate
from embodied.gate import EpisodeSpec
from embodied.probes import Readout, perception_confidence
from embodied.sim import LiftTask, scripted_expert
from embodied.world_model import ActionHead, WorldModel, grid_pool


@dataclass
class EpisodeSummary:
    id: str
    candidate: str
    seed: int
    drift: str
    suite: str
    success: bool
    steps: int
    unsafe_proposals: int
    unsafe_executed: int = 0
    unsafe_reasons: dict = field(default_factory=dict)
    blocked: int = 0
    block_reasons: dict = field(default_factory=dict)
    max_surprise: float = 0.0
    min_perception_conf: float = 1.0
    probe_err_px: float = 0.0  # median cube localisation error of the probes this episode
    first_unsafe_step: int | None = None
    failure: str | None = None  # None | "unsafe" | "no_lift"

    def dict(self) -> dict:
        return asdict(self)


def load_head(cand: Candidate, device: str = "cuda") -> ActionHead:
    head = ActionHead().to(device)
    head.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in cand.controller.items()})
    return head.eval()


def _true_uv(labels: np.ndarray) -> np.ndarray:
    """(V, O, 256) coverage -> (V, O, 2) centroid in [0,1], nan where absent."""
    g = 16
    ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    grid = np.stack([(xs.ravel() + 0.5) / g, (ys.ravel() + 0.5) / g], -1)
    m = labels.sum(-1, keepdims=True)
    return np.where(m > 0.3, labels @ grid / np.maximum(m, 1e-6), np.nan)


class Runner:
    """Holds the loaded world model + sim. One per GPU container."""

    def __init__(self, wm: WorldModel, data_root: str = "/data"):
        self.wm = wm
        self.task = LiftTask()
        self.root = Path(data_root)

    def episode(self, cand: Candidate, spec: EpisodeSpec, head: ActionHead | None, readout: Readout) -> EpisodeSummary:
        wm, task = self.wm, self.task
        guard = cand.guard() if cand.guardrails else None
        rng = np.random.default_rng(spec.seed)
        o = task.reset(spec.seed, spec.drift)
        readout.reset()
        keys = ("feats", "pose", "proposed", "executed", "expert", "surprise", "unsafe", "blocked", "eef_pos", "cube_pos", "contact_force", "gripper_closed", "true_uv")
        log = {k: [] for k in keys}
        s = EpisodeSummary(id=f"{cand.id}-{spec.suite}-{spec.drift}-{spec.seed}", candidate=cand.id, seed=spec.seed, drift=spec.drift, suite=spec.suite, success=False, steps=0, unsafe_proposals=0)
        z_prev_pred = None
        frames = [o.frame]
        errs = []
        while not task.done:
            z_all, acts = wm.encode_layers(o.views, [readout.layer])
            z = z_all[:1]  # (1, N, D) sideview: the predictor's view
            zg = grid_pool(z)[0]
            feats = readout.features(acts[readout.layer], o.pose, o.state["grip_steps"])
            tuv = _true_uv(o.patch_labels)
            pred_cube = feats[[0, 1]]  # cube_u_side, cube_v_side
            if not np.isnan(tuv[0, 0, 0]):
                errs.append(float(np.linalg.norm(pred_cube - tuv[0, 0]) * 256))
            surprise = float(wm.residual(z_prev_pred, z[0]).mean()) if z_prev_pred is not None else 0.0
            expert = scripted_expert(o)
            if spec.policy == "expert":
                proposed = scripted_expert(o, noise=spec.noise, rng=rng)
            else:
                proposed = head.act(feats)
            reasons = task.unsafe_reasons(o, proposed)
            executed, blocked = proposed, None
            pose_t = torch.from_numpy(o.pose).cuda().view(1, 1, 7)
            z_pred = wm.predict(z.unsqueeze(1), torch.from_numpy(proposed).cuda().view(1, 1, 7), pose_t)[0]
            if guard is not None:
                st = dict(o.state, perception_conf=perception_confidence(feats), surprise=surprise)
                g = guard(st, o.masks, zg.cpu().numpy(), grid_pool(z_pred[None])[0].cpu().numpy(), proposed)
                if g is not None:
                    blocked, executed = g[0], g[1]
                    z_pred = wm.predict(z.unsqueeze(1), torch.from_numpy(executed).cuda().view(1, 1, 7), pose_t)[0]
            if reasons and not blocked:
                s.first_unsafe_step = s.first_unsafe_step if s.first_unsafe_step is not None else s.steps
            s.unsafe_proposals += bool(reasons)
            s.unsafe_executed += bool(reasons and not blocked)
            for r in reasons:
                s.unsafe_reasons[r] = s.unsafe_reasons.get(r, 0) + 1
            if blocked:
                s.blocked += 1
                s.block_reasons[blocked] = s.block_reasons.get(blocked, 0) + 1
            s.max_surprise = max(s.max_surprise, surprise)
            s.min_perception_conf = min(s.min_perception_conf, perception_confidence(feats))
            for k, v in (("feats", feats), ("pose", o.pose), ("proposed", proposed), ("executed", executed), ("expert", expert), ("surprise", surprise), ("unsafe", bool(reasons)), ("blocked", blocked is not None),
                         ("eef_pos", o.state["eef_pos"]), ("cube_pos", o.state["cube_pos"]), ("contact_force", o.state["contact_force"]), ("gripper_closed", o.state["gripper_closed"]), ("true_uv", tuv)):
                log[k].append(v)
            z_prev_pred = z_pred
            o = task.step(executed)
            frames.append(o.frame)
            s.steps += 1
        s.success = task.success
        s.failure = "unsafe" if s.first_unsafe_step is not None else (None if s.success else "no_lift")
        s.probe_err_px = round(float(np.median(errs)), 1) if errs else 0.0
        s.min_perception_conf = round(s.min_perception_conf, 3)
        d = self.root / "episodes"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{s.id}.npz", table_z=np.float32(o.state["table_z"]), success=np.bool_(s.success), **{k: np.asarray(v) for k, v in log.items()})
        import imageio

        imageio.imwrite(d / f"{s.id}.png", frames[-1])
        imageio.mimwrite(d / f"{s.id}.mp4", frames, fps=10, macro_block_size=8)
        return s

    def run(self, cand: Candidate, specs: list[EpisodeSpec]) -> list[dict]:
        head = load_head(cand) if any(sp.policy == "head" for sp in specs) else None
        readout = Readout(cand.probes)
        return [self.episode(cand, sp, head, readout).dict() for sp in specs]
