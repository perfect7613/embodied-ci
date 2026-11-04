"""How good are Meta SAM 3.1 labels on our frames? Compare SAM 3.1 boxes with simulator ground truth on
logged episodes. No GPU: frames come from the episode videos, ground truth from the episode logs.

    uv run python -m infra.sam31_check --episodes-dir path/with/npz --frames 15

Prints, per drift and object: how often SAM 3.1 found it, and the median distance between the box
centre and the true centroid (px, 256 px frame). Spend goes through the same capped ledger.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from embodied.meta_sam import PROMPTS, MetaSamLabeler
from embodied.sim import OBJECTS

API = os.environ.get("EMBODIED_API", "https://ameymuke252003--embodied-ci-api.us-east.modal.run")


def load_env(path: str = ".env") -> None:
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def frames(eid: str) -> np.ndarray:
    """All frames of an episode video as (T, 256, 256, 3) uint8 (raw RGB through ffmpeg, no extra deps)."""
    with tempfile.TemporaryDirectory() as d:
        mp4 = Path(d) / "e.mp4"
        urllib.request.urlretrieve(f"{API}/episodes/{eid}.mp4", mp4)
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 256, 256, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--candidate", default="0a1e7fbb")  # v0: the cube is still on the table in many frames
    ap.add_argument("--frames", type=int, default=15, help="frames per drift")
    a = ap.parse_args()
    load_env()
    lab = MetaSamLabeler()
    print(f"Meta API spend before: ${lab.ledger.spent():.4f} of ${lab.ledger.cap:.2f}")
    rows = {}
    for drift, suite in (("nominal", "nominal"), ("cube_color", "heldout"), ("camera_shift", "heldout"), ("dim_light", "heldout")):
        paths = sorted(glob.glob(f"{a.episodes_dir}/{a.candidate}-{suite}-{drift}-*.npz"))
        picked, rng = [], np.random.default_rng(0)
        for p in paths:  # spread frames over episodes and over time in each episode
            with np.load(p) as d:
                tuv = d["true_uv"]  # (T, V, O, 2), nan where the object is not visible
            eid = Path(p).stem
            for t in rng.choice(len(tuv), size=min(3, len(tuv)), replace=False):
                picked.append((eid, int(t), tuv[t, 0]))  # view 0 = sideview, the camera the video records
            if len(picked) >= a.frames:
                break
        stats = {o: {"found": 0, "visible": 0, "err": []} for o in OBJECTS}
        cache = {}
        for eid, t, true in picked[: a.frames]:
            if eid not in cache:
                cache[eid] = frames(eid)
            img = cache[eid][t]
            for o, name in enumerate(OBJECTS):
                if np.isnan(true[o]).any():
                    continue
                stats[name]["visible"] += 1
                for prompt in PROMPTS[name]:
                    boxes = lab.segment(img, prompt)
                    if boxes:
                        b = boxes[0]
                        c = np.array([(b["x1"] + b["x2"]) / 2 / b["w"], (b["y1"] + b["y2"]) / 2 / b["h"]])
                        stats[name]["found"] += 1
                        stats[name]["err"].append(float(np.linalg.norm(c - true[o]) * 256))
                        break
        rows[drift] = {n: {"found": f"{s['found']}/{s['visible']}",
                           "median_err_px": round(float(np.median(s["err"])), 1) if s["err"] else None} for n, s in stats.items()}
        print(drift, json.dumps(rows[drift]))
    print(f"Meta API spend after: ${lab.ledger.spent():.4f} of ${lab.ledger.cap:.2f}")


if __name__ == "__main__":
    main()
