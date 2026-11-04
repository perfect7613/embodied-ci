"""Probe labels from Meta SAM 3 instead of the simulator.

The patch probes need "how much of this 16x16 patch is the cube / the gripper"
for every frame. In sim that comes free from segmentation; on a real robot it
does not. SAM 3 segments from a text prompt ("cube", "robot gripper"), so the
same coverage labels can be made from raw camera frames. That makes the probe
fitting, recalibration and the perception check work on real footage.

    lab = SamLabeler()                       # facebook/sam3 (gated on HF: accept the licence, set HF_TOKEN)
    cov = lab.coverage(views)                # (V, O, 256), same format as Obs.patch_labels
    agreement(cov, sim_cov)                  # how close SAM labels are to sim ground truth
"""

from __future__ import annotations

import os

import numpy as np

from embodied.sim import OBJECTS, PATCH

PROMPTS = {"cube": "small cube on the table", "gripper": "robot gripper"}


class SamLabeler:
    def __init__(self, model_id: str | None = None, device: str = "cuda", threshold: float = 0.4):
        from transformers import Sam3Model, Sam3Processor

        self.model_id = model_id or os.environ.get("SAM_MODEL", "facebook/sam3")
        token = os.environ.get("HF_TOKEN")
        self.model = Sam3Model.from_pretrained(self.model_id, token=token).to(device).eval()
        self.proc = Sam3Processor.from_pretrained(self.model_id, token=token)
        self.device, self.threshold = device, threshold

    def masks(self, image: np.ndarray, prompt: str) -> np.ndarray:
        """(H, W, 3) uint8 -> (H, W) bool: the highest-scoring SAM 3 instance for the prompt."""
        import torch
        from PIL import Image

        inputs = self.proc(images=Image.fromarray(image), text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_instance_segmentation(
            out, threshold=self.threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
        m = res["masks"]
        if len(m) == 0:
            return np.zeros(image.shape[:2], bool)
        best = int(torch.as_tensor(res["scores"]).argmax())  # one cube, one gripper: keep the most confident instance
        return torch.as_tensor(m[best]).bool().cpu().numpy()

    def coverage(self, views: np.ndarray) -> np.ndarray:
        """(V, H, W, 3) -> (V, O, 256) patch coverage in [0, 1]."""
        V, H, W, _ = views.shape
        g = H // PATCH
        out = np.zeros((V, len(OBJECTS), g * g), np.float32)
        for v in range(V):
            for o, name in enumerate(OBJECTS):
                m = self.masks(views[v], PROMPTS[name]).astype(np.float32)
                out[v, o] = m.reshape(g, PATCH, g, PATCH).mean((1, 3)).reshape(-1)
        return out


def agreement(sam: np.ndarray, sim: np.ndarray, thr: float = 0.2) -> dict:
    """SAM vs sim labels on the same frames: patch IoU and centroid distance (px), per object."""
    g = int(sam.shape[-1] ** 0.5)
    ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    grid = np.stack([(xs.ravel() + 0.5) / g, (ys.ravel() + 0.5) / g], -1)
    out = {}
    for o, name in enumerate(OBJECTS):
        a, b = sam[:, o] > thr, sim[:, o] > thr
        inter, union = (a & b).sum(), (a | b).sum()
        ma, mb = sam[:, o].sum(-1), sim[:, o].sum(-1)
        ok = (ma > 0.3) & (mb > 0.3)
        ca = sam[ok, o] @ grid / ma[ok, None]
        cb = sim[ok, o] @ grid / mb[ok, None]
        out[name] = {
            "patch_iou": round(float(inter / max(union, 1)), 3),
            "centroid_err_px": round(float(np.median(np.linalg.norm(ca - cb, axis=-1)) * g * PATCH), 1) if ok.any() else None,
            "detected": f"{int((ma > 0.3).sum())}/{len(ma)}",
        }
    return out
