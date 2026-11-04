"""Probe labels from Meta SAM 3.1 through the Meta Model API (api.meta.ai), with a hard spend cap.

Same job as embodied/sam_labels.py (SAM 3 weights on our own GPU), but SAM 3.1 runs on Meta's side:
no gated weights to download and no GPU memory used. One frame is one billed image.

    lab = MetaSamLabeler()             # META_API_KEY from .env; cap META_API_BUDGET_USD (default 50)
    cov = lab.coverage(views)          # (V, O, 256) patch coverage, same format as Obs.patch_labels

Coverage comes from SAM 3.1's boxes. The cube is a few patches wide, so its box and its mask cover
nearly the same patches. The gripper box is looser than its mask.

Spend is tracked in a ledger file. A call that would take the total past the cap is refused before
it is sent (BudgetExceeded). The ledger lives on the Modal volume when there is one, so the cap holds
across runs and containers.
"""

from __future__ import annotations

import base64
import fcntl
import struct
import json
import os
import re
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from embodied.sim import OBJECTS, PATCH

API = "https://api.meta.ai/v1/responses"
MODEL = "sam-3.1"
USD_PER_IMAGE = 2.50 / 1000  # Meta Model API pricing, SAM 3.1 image segmentation
# First prompt that finds something wins. The blue cube drift needs its own phrasing.
PROMPTS = {"cube": ("red cube", "blue cube", "cube"), "gripper": ("robot gripper", "gripper", "end effector")}
_BOX = re.compile(r"<\|box;([^|]*)\|>")


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Total USD spent on the Meta API, in a JSON file guarded by a file lock."""

    def __init__(self, path: str | Path | None = None, cap_usd: float | None = None):
        default = "/data/meta_api_spend.json" if Path("/data").is_dir() else str(Path.home() / ".embodiedci" / "meta_api_spend.json")
        self.path = Path(path or os.environ.get("META_API_LEDGER", default))
        self.cap = float(cap_usd if cap_usd is not None else os.environ.get("META_API_BUDGET_USD", "50"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rw(self, fn):
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read()
            d = json.loads(raw) if raw.strip() else {"usd": 0.0, "images": 0, "calls": 0}
            out = fn(d)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(d))
            return out

    def spent(self) -> float:
        return self._rw(lambda d: d["usd"])

    def reserve(self, usd: float) -> None:
        """Count the cost up front, so a crash between call and bookkeeping cannot under-count."""
        def take(d):
            if d["usd"] + usd > self.cap + 1e-9:
                raise BudgetExceeded(f"Meta API cap ${self.cap:.2f} reached: ${d['usd']:.4f} spent, this call needs ${usd:.4f}")
            d["usd"] += usd
            d["calls"] += 1
        self._rw(take)

    def settle(self, reserved: float, images: int) -> None:
        """Replace the reservation with what the API reported (images_processed)."""
        def fix(d):
            d["usd"] += images * USD_PER_IMAGE - reserved
            d["images"] += images
        self._rw(fix)


def parse_boxes(text: str) -> list[dict]:
    """SAM 3.1 special-token text -> [{x1, y1, x2, y2, w, h, c}] in source pixels."""
    out = []
    for body in _BOX.findall(text):
        kv = dict(p.split("=", 1) for p in body.split(";") if "=" in p)
        try:
            out.append({k: float(kv[k]) for k in ("x1", "y1", "x2", "y2", "w", "h")} | {"c": float(kv.get("c", 1.0))})
        except (KeyError, ValueError):
            continue  # a malformed record is skipped, never guessed
    return out


def box_coverage(box: dict, size: int, patch: int = PATCH) -> np.ndarray:
    """A box in source pixels -> (g*g,) fraction of each patch it covers, on a size x size frame."""
    g = size // patch
    sx, sy = size / box["w"], size / box["h"]
    x1, x2 = np.clip([box["x1"] * sx, box["x2"] * sx], 0, size)
    y1, y2 = np.clip([box["y1"] * sy, box["y2"] * sy], 0, size)
    edges = np.arange(g + 1) * patch
    ox = np.clip(np.minimum(edges[1:], x2) - np.maximum(edges[:-1], x1), 0, patch) / patch
    oy = np.clip(np.minimum(edges[1:], y2) - np.maximum(edges[:-1], y1), 0, patch) / patch
    return (oy[:, None] * ox[None, :]).reshape(-1).astype(np.float32)


class MetaSamLabeler:
    def __init__(self, api_key: str | None = None, ledger: Ledger | None = None, min_conf: float = 0.5, timeout: float = 60.0):
        self.key = api_key or os.environ.get("META_API_KEY")
        if not self.key:
            raise RuntimeError("META_API_KEY is not set (put it in .env)")
        self.ledger = ledger or Ledger()
        self.min_conf, self.timeout = min_conf, timeout

    @staticmethod
    def _png(image: np.ndarray) -> str:
        """(H, W, 3) uint8 -> PNG data URL. Plain zlib, no imaging library needed."""
        img = np.ascontiguousarray(np.asarray(image, np.uint8)[..., :3])
        h, w, _ = img.shape
        raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))  # filter type 0 per scanline

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
        return "data:image/png;base64," + base64.b64encode(png).decode()

    def segment(self, image: np.ndarray, prompt: str) -> list[dict]:
        """One billed call: boxes for `prompt` in `image`, most confident first."""
        self.ledger.reserve(USD_PER_IMAGE)
        body = {"model": MODEL, "stream": True, "metadata": {"include_confidence": "true"},
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                                       {"type": "input_image", "image_url": self._png(image)}]}]}
        req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "Accept": "text/event-stream"})
        text, images = "", 1  # if usage never arrives, assume the image was billed
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    for line in r:
                        line = line.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            e = json.loads(line[5:])
                        except json.JSONDecodeError:
                            continue
                        if e.get("type") == "response.output_text.delta":
                            text += e.get("delta", "")
                        elif e.get("type") == "response.completed":
                            images = int((e.get("response") or {}).get("usage", {}).get("images_processed", 1))
                        elif e.get("type") in ("error", "response.failed"):
                            raise RuntimeError(f"SAM 3.1 error: {json.dumps(e)[:300]}")
                break
            except urllib.error.HTTPError as err:
                if err.code == 429 and attempt < 2:  # rate limited: back off, same reservation
                    time.sleep(2 ** attempt)
                    continue
                self.ledger.settle(USD_PER_IMAGE, 0)  # rejected requests are not billed
                raise RuntimeError(f"SAM 3.1 HTTP {err.code}: {err.read()[:300]!r}") from None
        self.ledger.settle(USD_PER_IMAGE, images)
        return sorted((b for b in parse_boxes(text) if b["c"] >= self.min_conf), key=lambda b: -b["c"])

    def coverage(self, views: np.ndarray) -> np.ndarray:
        """(V, H, W, 3) -> (V, O, 256): one object per prompt list, first prompt with a hit wins."""
        V, H, W, _ = views.shape
        out = np.zeros((V, len(OBJECTS), (H // PATCH) * (W // PATCH)), np.float32)
        for v in range(V):
            for o, name in enumerate(OBJECTS):
                for prompt in PROMPTS[name]:
                    boxes = self.segment(views[v], prompt)
                    if boxes:  # one cube, one gripper: keep the most confident instance
                        out[v, o] = box_coverage(boxes[0], H)
                        break
        return out
