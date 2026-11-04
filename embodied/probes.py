"""Mechanistic-interpretability probes on V-JEPA 2-AC latents.

1. Patch probes (linear, per object): "is the cube / gripper in this 16x16 patch?"
   Labels come free from sim segmentation. Swept across encoder depths to find
   where object identity becomes linearly readable (cf. Joseph et al. 2026).
2. Readout: soft-argmax over each probe's patch logits gives an object's image
   position per view, plus a confidence. These NAMED features are what the
   controller reads, so the controller is small, legible and editable.
3. Failure-ahead probe: logistic regression on per-step features that predicts
   "a failure starts within k steps", scored by AUROC and lead time.
4. Drift signal: probe confidence collapses under perceptual drift (dim light)
   before the controller acts on a wrong position.

Principle: locate before you fix
(attribution over controller units -> circuit-restricted fine-tune lives in train.py),
and judge a fix by faithfulness-style checks (the gate), not by loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from embodied.sim import CAMERAS, IMG, OBJECTS, PATCH

G = IMG // PATCH  # 16 patches per side
FEATURE_NAMES = (
    [f"{o}_{ax}_{c}" for c in ("side", "agent") for o in OBJECTS for ax in ("u", "v", "conf")]
    + [f"cube_minus_gripper_{ax}_{c}" for c in ("side", "agent") for ax in ("u", "v")]
    + ["eef_x", "eef_y", "eef_z", "gripper_closed", "grip_settled"]
)
N_FEATURES = len(FEATURE_NAMES)  # 21

_coords = None


def _grid(device) -> torch.Tensor:
    global _coords
    if _coords is None or _coords.device != torch.device(device):
        ys, xs = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij")
        _coords = torch.stack([(xs.flatten() + 0.5) / G, (ys.flatten() + 0.5) / G], -1).float()  # (256, 2) u=x, v=y
    return _coords


# --------------------------------------------------------------------------- patch probes


def fit_patch_probe(X: torch.Tensor, Y: torch.Tensor, steps: int = 400, lr: float = 3e-3, wd: float = 1e-4, seed: int = 0) -> tuple[dict, dict]:
    """X (M, 256, D) fp16 tokens, Y (M, 256) in [0,1] coverage. Logistic probe shared over patches.
    Returns ({w, b}, metrics on a held-out 15% of images: patch AUROC, centroid error in px)."""
    torch.manual_seed(seed)
    M, N, D = X.shape
    # Consecutive frames of one episode are near-duplicates: hold out a contiguous block, not random frames,
    # so the reported error is on frames the probe has not effectively seen.
    nv = max(M * 15 // 100, 1)
    va = torch.arange(M - nv, M, device=X.device)
    tr = torch.arange(0, M - nv, device=X.device)
    w = torch.zeros(D, device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    pos = Y[tr].float().mean().clamp(1e-4, 1)
    pw = ((1 - pos) / pos).clamp(max=50.0).sqrt()  # soft coverage targets keep sub-patch position information
    for _ in range(steps):
        idx = tr[torch.randint(len(tr), (min(256, len(tr)),), device=X.device)]
        logit = X[idx].float() @ w + b
        loss = F.binary_cross_entropy_with_logits(logit, Y[idx].float(), pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit = torch.nan_to_num(X[va].float() @ w + b)
        auroc = _auroc(logit.flatten(), (Y[va] > 0.2).flatten())
        has = Y[va].sum(-1) > 0.5
        pred_uv = local_soft_argmax(logit)[has]
        true_uv = (Y[va][has].unsqueeze(-1) * _grid(X.device)).sum(1) / Y[va][has].sum(-1, keepdim=True)
        err_px = float(((pred_uv - true_uv).norm(dim=-1) * IMG).median()) if has.any() else float("nan")
    return {"w": w.detach().cpu().numpy(), "b": b.detach().cpu().numpy()}, {"auroc": round(auroc, 4), "median_err_px": round(err_px, 1)}


def soft_argmax(logit: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """(..., 256) patch logits -> (..., 2) expected (u, v) in [0, 1]. exp(logit) ~ coverage odds, so
    partially covered patches pull the estimate below patch resolution."""
    p = torch.softmax(logit / temp, -1)
    return p @ _grid(logit.device)


def local_soft_argmax(logit: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Soft-argmax restricted to a (2r+1)^2 window around the strongest patch.

    The global version averages over all 256 patches, so when the object is faint (blue cube, dim light)
    the many weakly-positive background patches drag the estimate toward the image centre. Near an image
    edge (moved camera) the window is clipped, never wrapped."""
    shp = logit.shape
    lg = torch.nan_to_num(logit.reshape(-1, G * G).float(), nan=-30.0, posinf=30.0, neginf=-30.0)
    peak = lg.argmax(-1)
    py, px = peak // G, peak % G
    ys = torch.arange(G, device=lg.device).repeat_interleave(G)
    xs = torch.arange(G, device=lg.device).repeat(G)
    near = ((ys[None] - py[:, None]).abs() <= radius) & ((xs[None] - px[:, None]).abs() <= radius)
    p = torch.softmax(lg.masked_fill(~near, float("-inf")), -1)
    return (p @ _grid(lg.device)).reshape(*shp[:-1], 2)


def _auroc(score: torch.Tensor, label: torch.Tensor) -> float:
    s, y = score.float().cpu().numpy(), label.cpu().numpy().astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


# --------------------------------------------------------------------------- readout


class Readout:
    """Probe weights -> named features for the controller."""

    VISIBLE = 0.15  # below this peak probability the object is treated as not visible this frame

    def __init__(self, probes: dict, device: str = "cuda"):
        self.layer = int(probes["layer"])
        # readout v0 (original): global soft-argmax. v1: local soft-argmax + hold the last confident position.
        # Kept per candidate so an old candidate behaves exactly as when it was gated.
        self.version = int(np.asarray(probes.get("readout", 0)))
        self.reset()
        self.W = torch.from_numpy(np.stack([probes[f"{o}_w"] for o in OBJECTS])).float().to(device)  # (O, D)
        self.B = torch.from_numpy(np.concatenate([probes[f"{o}_b"] for o in OBJECTS])).float().to(device)  # (O,)

    @torch.no_grad()
    def heat(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (V, 256, D) -> logits (V, O, 256)."""
        return torch.einsum("vnd,od->von", tokens.float(), self.W) + self.B[None, :, None]

    def reset(self) -> None:
        """Call at the start of every episode: forget the tracked positions."""
        self._last = None

    def locate(self, logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """logits (V, O, 256) -> uv (V, O, 2), conf (V, O)."""
        logit = torch.nan_to_num(logit.float(), nan=-30.0, posinf=30.0, neginf=-30.0)
        conf = torch.sigmoid(logit).amax(-1)
        if self.version == 0:
            return soft_argmax(logit), conf
        uv = local_soft_argmax(logit)
        seen = conf >= self.VISIBLE
        if self._last is not None:  # occluded / out of frame: keep the last place it was seen, not a guess at the centre
            uv = torch.where(seen[..., None], uv, self._last)
        self._last = uv
        return uv, conf

    @torch.no_grad()
    def features(self, tokens: torch.Tensor, pose: np.ndarray, grip_steps: int) -> np.ndarray:
        logit = self.heat(tokens)  # (V, O, 256)
        uv, conf = self.locate(logit)  # (V, O, 2), (V, O)
        f = []
        for v in range(len(CAMERAS)):
            for o in range(len(OBJECTS)):
                f += [uv[v, o, 0].item(), uv[v, o, 1].item(), conf[v, o].item()]
        for v in range(len(CAMERAS)):
            d = uv[v, 0] - uv[v, 1]
            f += [d[0].item(), d[1].item()]
        f += [float(pose[0]), float(pose[1]), float(pose[2]), float(pose[6]), float(min(grip_steps, 4) / 4)]
        return np.asarray(f, np.float32)


def perception_confidence(feats: np.ndarray) -> float:
    """Min probe confidence across views/objects: low = the model can't see what it needs."""
    idx = [FEATURE_NAMES.index(n) for n in FEATURE_NAMES if n.endswith("_conf_side") or n.endswith("_conf_agent")]
    return float(np.min(feats[..., idx], axis=-1)) if feats.ndim == 1 else np.min(feats[..., idx], axis=-1)


# --------------------------------------------------------------------------- failure-ahead probe


def fit_failure_probe(feats: list[np.ndarray], fail_step: list[int | None], horizon: int = 5, steps: int = 600, names=FEATURE_NAMES) -> dict:
    """Per-step features of many episodes -> P(failure begins within `horizon` steps).
    fail_step: first unsafe step, or the last step for a no-lift episode, or None for success.
    Returns weights + AUROC (held-out episodes) + median lead time at a 10% false-alarm threshold."""
    X, Y, E = [], [], []
    for e, (f, fs) in enumerate(zip(feats, fail_step)):
        T = len(f)
        y = np.zeros(T, np.float32)
        if fs is not None:
            y[max(0, fs - horizon) : fs + 1] = 1
        X.append(f), Y.append(y), E.append(np.full(T, e))
    X, Y, E = np.concatenate(X), np.concatenate(Y), np.concatenate(E)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    rng = np.random.default_rng(0)
    eps = rng.permutation(len(feats))
    va_eps = set(eps[: max(1, len(eps) // 4)].tolist())
    va = np.isin(E, list(va_eps))
    w, b = np.zeros(X.shape[1]), 0.0
    pw = (1 - Y[~va].mean()) / max(Y[~va].mean(), 1e-3)
    for _ in range(steps):  # plain gradient descent, numpy only
        z = Xn[~va] @ w + b
        p = 1 / (1 + np.exp(-z))
        g = (p - Y[~va]) * np.where(Y[~va] > 0, pw, 1.0)
        w -= 0.05 * (Xn[~va].T @ g / len(g) + 1e-3 * w)
        b -= 0.05 * g.mean()
    s = Xn[va] @ w + b
    au = _auroc(torch.from_numpy(s), torch.from_numpy(Y[va] > 0)) if va.any() else float("nan")
    thr = float(np.quantile(s[Y[va] == 0], 0.9)) if (Y[va] == 0).any() else 0.0
    leads = []
    for e in va_eps:
        fs = fail_step[e]
        if fs is None:
            continue
        se = Xn[E == e] @ w + b
        hit = np.nonzero(se > thr)[0]
        if len(hit) and hit[0] <= fs:
            leads.append(fs - hit[0])
    top = np.argsort(-np.abs(w))[:5]
    return {
        "w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist(),
        "auroc": round(float(au), 3), "median_lead_steps": float(np.median(leads)) if leads else None,
        "caught": f"{len(leads)}/{sum(fail_step[e] is not None for e in va_eps)}",
        "top_features": [(names[i], round(float(w[i]), 3)) for i in top],
    }


# --------------------------------------------------------------------------- fitting on the GPU


@torch.no_grad()
def collect(wm, task, seeds: list[int], drifts: list[str], layers: list[int], every: int = 2, noise: float = 0.02, labeler=None):
    """Run the (noisy) scripted expert; keep both views every `every` steps.
    labeler: None = simulator segmentation; a SamLabeler = labels from Meta SAM 3 on the raw frames.
    -> {layer: (M,256,D) fp16 on GPU}, labels (M, O, 256) [, sim labels for the same frames when labeler is set]."""
    from embodied.sim import scripted_expert

    X = {l: [] for l in layers}
    Y, Ysim = [], []
    for seed in seeds:
        for drift in drifts:
            rng = np.random.default_rng(seed)
            o, t = task.reset(seed, drift), 0
            while not task.done:
                if t % every == 0:
                    _, acts = wm.encode_layers(o.views, layers)
                    for l in layers:
                        X[l].append(acts[l].half())
                    if labeler is None:
                        Y.append(torch.from_numpy(o.patch_labels).float())
                    else:
                        Y.append(torch.from_numpy(labeler.coverage(o.views)).float())
                        Ysim.append(o.patch_labels)
                o, t = task.step(scripted_expert(o, noise=noise, rng=rng)), t + 1
    Y = torch.cat(Y).to(wm.device)  # (M, O, 256): views stacked into M
    X = {l: torch.cat(v) for l, v in X.items()}
    return (X, Y, np.concatenate(Ysim)) if labeler is not None else (X, Y)


def fit_probes(wm, task, seeds: list[int], drifts: list[str], layers: list[int] | None = None,
               label_source: str = "sim", readout: int = 0) -> tuple[dict, dict]:
    """Layer sweep -> best layer's probes as Candidate head entries ("probe.*") + a per-layer report.
    label_source "sam": labels come from Meta SAM 3 on the raw frames (what a real robot would use);
    the report then includes SAM-vs-sim agreement. Probe metrics are always scored against the labels used."""
    L = wm.n_layers
    layers = layers or [L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1, -1]
    sam_report = None
    if label_source in ("sam", "sam3.1"):
        from embodied.sam_labels import SamLabeler, agreement

        if label_source == "sam3.1":  # Meta SAM 3.1 over the Meta Model API: billed per image, so label fewer frames
            from embodied.meta_sam import PROMPTS, USD_PER_IMAGE, BudgetExceeded, MetaSamLabeler

            lab, every = MetaSamLabeler(), 6
            worst = len(seeds) * len(drifts) * (task.horizon // every + 1) * 2 * sum(len(p) for p in PROMPTS.values()) * USD_PER_IMAGE
            left = lab.ledger.cap - lab.ledger.spent()
            if worst > left:
                raise BudgetExceeded(f"labelling could cost up to ${worst:.2f}; ${left:.2f} left under the Meta API cap")
        else:
            lab, every = SamLabeler(device=wm.device), 2
        X, Y, Ysim = collect(wm, task, seeds, drifts, layers, every=every, labeler=lab)
        sam_report = agreement(Y.cpu().numpy(), Ysim)
    else:
        X, Y = collect(wm, task, seeds, drifts, layers)
    report, fitted = {}, {}
    for l in layers:
        fitted[l], report[l] = {}, {}
        for oi, name in enumerate(OBJECTS):
            p, m = fit_patch_probe(X[l], Y[:, oi])
            fitted[l][name], report[l][name] = p, m
    def err(l):  # the controller reads both objects; a layer with no measurable error (nan) never wins
        e = report[l]["cube"]["median_err_px"] + report[l]["gripper"]["median_err_px"]
        return (e if np.isfinite(e) else float("inf"), -report[l]["cube"]["auroc"])

    best = min(layers, key=err)
    out = {"probe.layer": np.asarray(best), "probe.readout": np.asarray(readout)}
    for name in OBJECTS:
        out[f"probe.{name}_w"] = fitted[best][name]["w"].astype(np.float32)
        out[f"probe.{name}_b"] = fitted[best][name]["b"].astype(np.float32)
    del X
    torch.cuda.empty_cache()
    rep = {"layers": {str(l): r for l, r in report.items()}, "best_layer": best, "images": int(len(Y)), "drifts": drifts,
           "label_source": label_source, "readout": "local+track" if readout else "global"}
    if sam_report:
        rep["sam_vs_sim"] = sam_report
    return out, rep


@torch.no_grad()
def eval_probes(wm, task, probes: dict, seeds: list[int], drifts: list[str]) -> dict:
    """Probe quality per drift (localisation error px, mean confidence): where perception breaks."""
    r = Readout(probes, wm.device)
    out = {}
    for d in drifts:
        X, Y = collect(wm, task, seeds, [d], [r.layer], every=4)
        logit = torch.einsum("mnd,od->mon", X[r.layer].float(), r.W) + r.B[None, :, None]
        logit = torch.nan_to_num(logit, nan=-30.0, posinf=30.0, neginf=-30.0)
        uv = (local_soft_argmax if r.version else soft_argmax)(logit[:, 0])
        cy = Y[:, 0]
        has = cy.sum(-1) > 0.3
        tuv = (cy[has].unsqueeze(-1) * _grid(cy.device)).sum(1) / cy[has].sum(-1, keepdim=True)
        if not has.any():  # cube never visible under this drift: report it rather than crash on an empty median
            out[d] = {"cube_err_px": float("nan"), "cube_conf": round(float(torch.sigmoid(logit[:, 0]).amax(-1).mean()), 3), "auroc": float("nan")}
            continue
        out[d] = {"cube_err_px": round(float(((uv[has] - tuv).norm(dim=-1) * IMG).median()), 1),
                  "cube_conf": round(float(torch.sigmoid(logit[:, 0]).amax(-1).mean()), 3),
                  "auroc": round(_auroc(logit[:, 0].flatten(), (cy > 0.2).flatten()), 3)}
    return out
