"""The model under test: frozen V-JEPA 2-AC (encoder + action-conditioned predictor)
plus a small controller that reads named probe features off the encoder.

Interface used by the rest of the system:
    wm = WorldModel.load()                 # on a Modal H100
    z  = wm.encode(frames)                 # (T, N, D) patch latents, layer-normed
    z1 = wm.predict(z0, actions, poses)    # next-frame latents
    r  = wm.residual(z_pred, z_real)       # surprise, per patch -> pooled
    a  = wm.plan(z0, pose, z_goal)         # CEM teacher (slow)
    a  = head.act(features)                # controller on probe features (embodied.probes)

Weights are always reachable as torch tensors (`wm.encoder`, `wm.predictor`,
`head.state_dict()`), which is what weight editing needs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VJEPA_REPO = os.environ.get("VJEPA_REPO", "/opt/vjepa2")
CKPT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class WorldModel:
    def __init__(self, encoder: nn.Module, predictor: nn.Module, device: str = "cuda"):
        self.encoder = encoder.to(device).eval()
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.tokens_per_frame = (256 // encoder.patch_size) ** 2
        self.dim = encoder.embed_dim

    @classmethod
    def load(cls, device: str = "cuda") -> "WorldModel":
        if VJEPA_REPO not in sys.path:
            sys.path.insert(0, VJEPA_REPO)
        import src.hub.backbones as bb

        bb.VJEPA_BASE_URL = CKPT_URL.rsplit("/", 1)[0]  # upstream points at localhost
        encoder, predictor = bb.vjepa2_ac_vit_giant(pretrained=True)
        return cls(encoder, predictor, device)

    # ---- latents ---------------------------------------------------------
    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """frames (T, 256, 256, 3) uint8 -> (T, N, D). Each frame encoded alone (tubelet = frame x2)."""
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - MEAN.to(self.device)) / STD.to(self.device)
        x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (T, C, 2, H, W)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.encoder(x)
        return F.layer_norm(h.float(), (h.size(-1),))

    @torch.no_grad()
    def encode_layers(self, frames: np.ndarray, layers: list[int]) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """One encoder pass -> (final latents (T,N,D), {layer: layer-normed block output (T,N,D)}). layer -1 = final."""
        acts: dict[int, torch.Tensor] = {}
        hooks = [self.encoder.blocks[i].register_forward_hook(lambda m, inp, out, i=i: acts.__setitem__(i, out))
                 for i in layers if i >= 0]
        try:
            z = self.encode(frames)
        finally:
            for h in hooks:
                h.remove()
        return z, {i: (z if i < 0 else F.layer_norm(acts[i].float(), (self.dim,))) for i in layers}

    @property
    def n_layers(self) -> int:
        return len(self.encoder.blocks)

    @torch.no_grad()
    def predict(self, z: torch.Tensor, actions: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """z (B, T, N, D), actions (B, T, 7), poses (B, T, 7) -> next latents (B, N, D)."""
        B, T, N, D = z.shape
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.predictor(z.flatten(1, 2), actions, poses)[:, -self.tokens_per_frame :]
        return F.layer_norm(out.float(), (D,))

    @staticmethod
    def residual(z_pred: torch.Tensor, z_real: torch.Tensor) -> torch.Tensor:
        """(..., N, D) -> per-patch L1 surprise (..., N)."""
        return (z_pred - z_real).abs().mean(-1)

    @torch.no_grad()
    def plan(self, z: torch.Tensor, pose: np.ndarray, z_goal: torch.Tensor, **cem_args) -> np.ndarray:
        """CEM over predicted latents toward z_goal (upstream mpc_utils.cem). Slow: the teacher, not the policy."""
        if VJEPA_REPO not in sys.path:
            sys.path.insert(0, VJEPA_REPO)
        sys.path.insert(0, os.path.join(VJEPA_REPO, "notebooks"))
        from utils.mpc_utils import cem, compute_new_pose

        def step(reps, acts, poses):
            nxt = self.predict(reps, acts.float(), poses.float())
            return nxt.unsqueeze(1), compute_new_pose(poses[:, -1:], acts[:, -1:])

        args = dict(rollout=1, samples=256, topk=16, cem_steps=6, maxnorm=0.05, momentum_mean=0.15, momentum_std=0.15)
        args.update(cem_args)
        p = torch.from_numpy(pose).to(self.device).view(1, 1, 7)
        a = cem(context_frame=z.view(1, 1, *z.shape[-2:]), context_pose=p, goal_frame=z_goal.view(1, 1, *z_goal.shape[-2:]), world_model=step, **args)
        return a[0, 0].float().cpu().numpy()


def grid_pool(z: torch.Tensor, g: int = 8) -> torch.Tensor:
    """(..., N=256, D) patch latents -> (..., g*g, D). Keeps coarse 'where', which a mean-pool loses (RoPE: no pos in tokens)."""
    *lead, N, D = z.shape
    s = int(N**0.5)
    x = z.reshape(-1, s, s, D).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x, g).permute(0, 2, 3, 1).reshape(*lead, g * g, D)
    return x


class ActionHead(nn.Module):
    """The controller: 21 named probe features -> [dx, dy, dz, dclosedness] (xyz in units of max step).

    ~1.9k params, deliberately. Inputs are named (embodied.probes.FEATURE_NAMES), so a
    weight is legible ("fc1[3, cube_conf_side]") and an LLM can read and edit the whole
    thing as text. `mu`/`sd` are fixed input normalisers fitted at first training.
    """

    def __init__(self, n_in: int = 21, hidden: int = 32):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 4)

    def hidden(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.fc1((f - self.mu) / self.sd))
        return h1, F.relu(self.fc2(h1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(f)[1])

    @torch.no_grad()
    def act(self, f: np.ndarray) -> np.ndarray:
        dev = self.mu.device
        return head_out_to_action(self(torch.as_tensor(f, device=dev).float().view(1, -1))[0].cpu().numpy())


def head_out_to_action(out: np.ndarray) -> np.ndarray:
    a = np.zeros(7, dtype=np.float32)
    a[:3] = out[:3] * 0.05
    a[6] = out[3]
    return a


def action_to_head_target(a: np.ndarray) -> np.ndarray:
    return np.array([a[0] / 0.05, a[1] / 0.05, a[2] / 0.05, a[6]], dtype=np.float32)
