import json

import numpy as np

from embodied.candidate import Candidate
from infra.export import _showcase, bundle, compare


def _ep(root, eid, lifted, unsafe=False):
    T = 5 if lifted else 40
    cube = np.zeros((T, 3), np.float32)
    cube[:, 2] = 0.82
    if lifted:
        cube[-1, 2] = 0.9
    u = np.zeros(T, bool)
    u[2] = unsafe
    np.savez(root / "episodes" / f"{eid}.npz", unsafe=u, blocked=np.zeros(T, bool), cube_pos=cube, pose=np.zeros((T, 7)))


def test_compare_and_bundle(tmp_path):
    (tmp_path / "episodes").mkdir()
    a = Candidate(head={"fc1.weight": np.zeros((2, 2), np.float32), "probe.layer": np.asarray(19)}, id="aaaa", note="v0")
    b = a.child(guardrails=["def check(state, masks, z_now, z_pred, action):\n    return ALLOW\n"], note="fixed")
    a.save(tmp_path), b.save(tmp_path)
    _ep(tmp_path, "aaaa-heldout-cube_color-5003", lifted=False)
    _ep(tmp_path, f"{b.id}-heldout-cube_color-5003", lifted=True)
    _ep(tmp_path, "aaaa-heldout-dim_light-5002", lifted=True, unsafe=True)
    pairs = compare(str(tmp_path), "aaaa", b.id)
    assert len(pairs) == 1 and pairs[0]["drift"] == "cube_color" and pairs[0]["seed"] == 5003
    assert not pairs[0]["a"]["success"] and pairs[0]["b"]["success"]
    assert _showcase(pairs)[0]["drift"] == "cube_color"
    d = bundle(str(tmp_path), b.id, base="aaaa")
    assert (d / "weights.npz").exists() and (d / "guardrails" / "g0.py").exists()
    card = (d / "README.md").read_text()
    assert "1 fixed, 0 broken" in card and "block 19" in card
    assert json.loads((d / "gate_report.json").read_text())["compare_to"] == "aaaa"
