"""SAM 3.1 over the Meta Model API: output parsing, patch coverage and the hard spend cap. No network."""

import numpy as np
import pytest

from embodied.meta_sam import USD_PER_IMAGE, BudgetExceeded, Ledger, MetaSamLabeler, box_coverage, parse_boxes

# a real response from sam-3.1 (first frame of a nominal episode, prompt "red cube"), mask payload shortened
REAL = "<0f>0<|box;x1=107;y1=176;x2=117;y2=189;w=256;h=256;c=0.8865247964859009|><|mask;x=0;y=0;c=0.88;data=13,10,!!!#|>"


def test_parse_real_box():
    (b,) = parse_boxes(REAL)
    assert (b["x1"], b["y2"], b["w"]) == (107, 189, 256) and abs(b["c"] - 0.8865) < 1e-3


def test_parse_skips_malformed_and_defaults_confidence():
    boxes = parse_boxes("<|box;x1=1;y1=2;x2=3|><|box;x1=0;y1=0;x2=16;y2=16;w=256;h=256|>")
    assert len(boxes) == 1 and boxes[0]["c"] == 1.0


def test_box_coverage_matches_patch_grid():
    cov = box_coverage({"x1": 0, "y1": 0, "x2": 24, "y2": 16, "w": 256, "h": 256}, 256)
    assert cov.shape == (256,) and cov[0] == 1.0 and cov[1] == 0.5 and cov[16] == 0.0
    big = box_coverage({"x1": 0, "y1": 0, "x2": 96, "y2": 64, "w": 1024, "h": 1024}, 256)  # box from an upscaled frame
    assert big[0] == 1.0 and big[1] == 0.5
    off = box_coverage({"x1": 250, "y1": 250, "x2": 400, "y2": 400, "w": 256, "h": 256}, 256)  # spills past the edge
    assert off.max() <= 1.0 and off[-1] > 0


def test_ledger_refuses_calls_past_the_cap(tmp_path):
    led = Ledger(tmp_path / "spend.json", cap_usd=3 * USD_PER_IMAGE)
    for _ in range(3):
        led.reserve(USD_PER_IMAGE)
        led.settle(USD_PER_IMAGE, 1)
    with pytest.raises(BudgetExceeded):
        led.reserve(USD_PER_IMAGE)
    assert abs(led.spent() - 3 * USD_PER_IMAGE) < 1e-9
    again = Ledger(tmp_path / "spend.json", cap_usd=3 * USD_PER_IMAGE)  # the cap survives a new process
    with pytest.raises(BudgetExceeded):
        again.reserve(USD_PER_IMAGE)


def test_ledger_unbilled_request_is_refunded(tmp_path):
    led = Ledger(tmp_path / "spend.json", cap_usd=1.0)
    led.reserve(USD_PER_IMAGE)
    led.settle(USD_PER_IMAGE, 0)
    assert led.spent() == 0.0


def test_labeler_needs_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        MetaSamLabeler(ledger=Ledger(tmp_path / "s.json"))


def test_coverage_stops_at_first_prompt_that_hits(tmp_path, monkeypatch):
    lab = MetaSamLabeler(api_key="test", ledger=Ledger(tmp_path / "s.json"))
    asked = []

    def fake(image, prompt):
        asked.append(prompt)
        return [{"x1": 0, "y1": 0, "x2": 16, "y2": 16, "w": 256, "h": 256, "c": 0.9}] if prompt in ("blue cube", "gripper") else []

    monkeypatch.setattr(lab, "segment", fake)
    cov = lab.coverage(np.zeros((1, 256, 256, 3), np.uint8))
    assert asked == ["red cube", "blue cube", "robot gripper", "gripper"]
    assert cov.shape == (1, 2, 256) and cov[0, 0, 0] == 1.0 and cov[0, 1, 0] == 1.0


def test_png_encoder_round_trips():
    import base64, struct, zlib

    img = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
    png = base64.b64decode(MetaSamLabeler._png(img).split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    n = struct.unpack(">I", png[33:37])[0]
    raw = zlib.decompress(png[41 : 41 + n])
    rows = [raw[i * (1 + w * 3) + 1 : (i + 1) * (1 + w * 3)] for i in range(h)]
    assert (w, h) == (5, 4) and np.array_equal(np.frombuffer(b"".join(rows), np.uint8).reshape(4, 5, 3), img)
