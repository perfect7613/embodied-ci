"""Internal fixes on the controller: DAgger fine-tune, attribution, circuit-restricted fine-tune.

Every logged step carries the scripted expert's action for that exact state, so
episodes the controller drove itself are relabelled for free (DAgger).

Attribution ("locate before you fix", applied to
controller units): on failing steps, score each hidden unit by |activation x grad|
of the imitation loss, and each named input feature by |input x grad|. A
circuit-restricted fine-tune then updates only the top-k units' weights, leaving
the rest of the controller (and so the behaviour that already works) untouched.

The controller is ~1.9k params on 21 features: this runs on CPU in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from embodied.candidate import Candidate
from embodied.probes import FEATURE_NAMES
from embodied.world_model import ActionHead, action_to_head_target


def load_dataset(root: str | Path, episode_ids: list[str], failed: set[str] | None = None, fail_weight: float = 3.0):
    """-> F (S,21), Y (S,4), W (S,), fail_mask (S,) as tensors."""
    Fs, Ys, Ws, Ms = [], [], [], []
    for eid in episode_ids:
        p = Path(root) / "episodes" / f"{eid}.npz"
        if not p.exists():
            continue
        with np.load(p) as d:
            if "feats" not in d.files:
                continue
            T = len(d["feats"])
            bad = bool(failed and eid in failed)
            Fs.append(d["feats"].astype(np.float32))
            Ys.append(np.stack([action_to_head_target(a) for a in d["expert"]]))
            Ws.append(np.full(T, fail_weight if bad else 1.0, np.float32))
            Ms.append(np.full(T, bad))
    if not Fs:
        raise ValueError("no episodes with features")
    return tuple(torch.from_numpy(np.concatenate(a)) for a in (Fs, Ys, Ws, Ms))


def _head(cand: Candidate | None, F_: torch.Tensor, fresh: bool) -> ActionHead:
    head = ActionHead()
    if cand is not None and not fresh and "fc1.weight" in cand.head:
        head.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in cand.controller.items()})
    else:
        head.mu.copy_(F_.mean(0))
        head.sd.copy_(F_.std(0).clamp_min(1e-3))
    return head


def train_head(cand: Candidate | None, root: str | Path, episode_ids: list[str], failed: set[str] | None = None,
               epochs: int = 200, lr: float = 3e-3, seed: int = 0, fresh: bool = False,
               mask: dict[str, torch.Tensor] | None = None) -> tuple[dict, dict]:
    """-> (controller state as numpy, metrics). `mask`: param name -> 0/1 tensor; only 1-entries train."""
    torch.manual_seed(seed)
    X, Y, W, _ = load_dataset(root, episode_ids, failed)
    head = _head(cand, X, fresh)
    n = len(X)
    perm = torch.randperm(n)
    val, tr = perm[: max(n // 10, 1)], perm[max(n // 10, 1):]
    params = dict(head.named_parameters())
    opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=0.0 if mask else 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    before = {k: v.detach().clone() for k, v in params.items()}
    for _ in range(epochs):
        for idx in tr[torch.randperm(len(tr))].split(256):
            loss = (F.mse_loss(head(X[idx]), Y[idx], reduction="none").mean(-1) * W[idx]).mean()
            opt.zero_grad()
            loss.backward()
            if mask:
                for k, p in params.items():
                    p.grad *= mask.get(k, torch.zeros_like(p))
            opt.step()
            if mask:  # AdamW momentum can still move masked entries: pin them
                with torch.no_grad():
                    for k, p in params.items():
                        m = mask.get(k, torch.zeros_like(p))
                        p.copy_(m * p + (1 - m) * before[k])
        sched.step()
    with torch.no_grad():
        vloss = F.mse_loss(head(X[val]), Y[val]).item()
    state = {k: v.detach().cpu().numpy() for k, v in head.state_dict().items()}
    return state, {"steps": n, "val_mse": round(vloss, 5), "epochs": epochs, "restricted": bool(mask)}


def attribute(cand: Candidate, root: str | Path, failed_ids: list[str], top_k: int = 6) -> dict:
    """Which controller units and which named features carry the failing behaviour?"""
    X, Y, _, _ = load_dataset(root, failed_ids)
    head = _head(cand, X, fresh=False)
    x = X.clone().requires_grad_(True)
    h1, h2 = head.hidden(x)
    h1.retain_grad(), h2.retain_grad()
    F.mse_loss(head.out(h2), Y).backward()
    s1 = (h1 * h1.grad).abs().mean(0).detach()
    s2 = (h2 * h2.grad).abs().mean(0).detach()
    sf = (((x - head.mu) / head.sd) * x.grad * head.sd).abs().mean(0).detach()
    u1 = s1.argsort(descending=True)[:top_k].tolist()
    u2 = s2.argsort(descending=True)[:top_k].tolist()
    w1 = head.fc1.weight.detach().abs()
    return {
        "fc1_units": u1, "fc2_units": u2,
        "fc1_scores": [round(float(s1[i]), 4) for i in u1],
        "features": [(FEATURE_NAMES[i], round(float(sf[i]), 4)) for i in sf.argsort(descending=True)[:8].tolist()],
        "unit_reads": {f"fc1[{u}]": [FEATURE_NAMES[j] for j in w1[u].argsort(descending=True)[:3].tolist()] for u in u1},
        "concentration": round(float(s1[u1].sum() / s1.sum().clamp_min(1e-9)), 3),
        "steps": len(X),
    }


def circuit_mask(attr: dict, hidden: int = 32) -> dict[str, torch.Tensor]:
    """Trainable entries for a circuit-restricted fine-tune: the located units' rows, plus the readout of fc2 units."""
    n_in = len(FEATURE_NAMES)
    m = {"fc1.weight": torch.zeros(hidden, n_in), "fc1.bias": torch.zeros(hidden),
         "fc2.weight": torch.zeros(hidden, hidden), "fc2.bias": torch.zeros(hidden),
         "out.weight": torch.zeros(4, hidden), "out.bias": torch.ones(4)}
    for u in attr["fc1_units"]:
        m["fc1.weight"][u] = 1
        m["fc1.bias"][u] = 1
        m["fc2.weight"][:, u] = 1
    for u in attr["fc2_units"]:
        m["fc2.weight"][u] = 1
        m["fc2.bias"][u] = 1
        m["out.weight"][:, u] = 1
    return m
