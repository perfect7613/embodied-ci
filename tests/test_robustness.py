"""Edge cases found while hardening the probes, the guardrail sandbox and the gate's log cache."""

import numpy as np
import pytest

from embodied.candidate import compile_guardrail, safety_envelope
from embodied.gate import EpisodeSpec, summary_from_log

TABLE = 0.8


def state(eef=(0.0, 0.0, 0.9), force=0.0):
    return {"eef_pos": np.array(eef, np.float32), "table_z": TABLE, "contact_force": force}


def act(dx=0.0, dy=0.0, dz=0.0, grip=0.0):
    return np.array([dx, dy, dz, 0, 0, 0, grip], np.float32)


def test_ndarray_methods_work_inside_the_sandbox():
    # ndarray.max() imports numpy internals through the guardrail's builtins; this used to fail closed every step
    f = compile_guardrail("def check(state, masks, z_now, z_pred, action):\n    x = action.max() + action.mean() + action.min()\n    return ALLOW\n")
    assert f(None, None, None, None, act(0.01)) is None


def test_sandbox_import_hook_only_resolves_numpy():
    from embodied.candidate import _numpy_only_import

    with pytest.raises(ImportError):
        _numpy_only_import("os")


def test_envelope_allows_normal_actions_and_clips_unsafe_ones():
    g = compile_guardrail(safety_envelope({"speed", "table_penetration"}))
    assert g(state(), None, None, None, act(0.02, 0.0, -0.02)) is None  # normal: untouched
    reason, a = g(state(), None, None, None, act(0.06))  # too fast: scaled, same direction
    assert reason == "envelope" and np.linalg.norm(a[:3]) <= 0.045 + 1e-6 and a[0] > 0
    reason, a = g(state(eef=(0, 0, TABLE + 0.01)), None, None, None, act(dz=-0.03))  # would hit the table: stop above it
    assert reason == "envelope" and TABLE + 0.01 + a[2] >= TABLE + 0.005
    reason, a = g(state(eef=(0.34, 0, 0.9)), None, None, None, act(dx=0.03))  # would leave the workspace
    assert reason == "envelope" and 0.34 + a[0] <= 0.35
    reason, a = g(state(force=55.0), None, None, None, act())  # pressing on the table: back off
    assert reason == "contact_force" and a[2] > 0
    _, a = g(state(), None, None, None, act(0.06, grip=1.0))  # the gripper command passes through
    assert a[6] == 1.0


def _log(path, T=20, success=None, unsafe_at=(), blocked_at=(), z_drop=False):
    unsafe, blocked = np.zeros(T, bool), np.zeros(T, bool)
    unsafe[list(unsafe_at)] = True
    blocked[list(blocked_at)] = True
    proposed = np.zeros((T, 7), np.float32)
    eef = np.tile(np.array([0, 0, 0.9], np.float32), (T, 1))
    for t in unsafe_at:
        if z_drop:
            eef[t, 2], proposed[t, 2] = TABLE + 0.012, -0.01  # ends 2 mm above the table: inside the workspace, under the margin
        else:
            proposed[t, 0] = 0.06
    d = dict(unsafe=unsafe, blocked=blocked, pose=np.zeros((T, 7)), proposed=proposed, eef_pos=eef,
             contact_force=np.zeros(T), surprise=np.linspace(0, 1, T), table_z=np.float32(TABLE))
    if success is not None:
        d["success"] = np.bool_(success)
    np.savez(path, **d)
    return path


def test_log_cache_rebuilds_outcomes(tmp_path):
    sp = EpisodeSpec(seed=1, suite="nominal")
    ok = summary_from_log(_log(tmp_path / "a.npz", T=30), sp, "c")  # old log, ended early: lifted
    assert ok["success"] and ok["failure"] is None and ok["cached"]
    timeout = summary_from_log(_log(tmp_path / "b.npz", T=40), sp, "c")  # old log, ran to the horizon
    assert timeout["failure"] == "no_lift"
    flagged = summary_from_log(_log(tmp_path / "c.npz", T=40, success=True), sp, "c")  # new log: the flag wins
    assert flagged["success"]
    fast = summary_from_log(_log(tmp_path / "d.npz", unsafe_at=[3, 5]), sp, "c")
    assert fast["failure"] == "unsafe" and fast["first_unsafe_step"] == 3 and fast["unsafe_reasons"] == {"speed": 2}
    caught = summary_from_log(_log(tmp_path / "e.npz", unsafe_at=[3], blocked_at=[3], z_drop=True), sp, "c")
    assert caught["unsafe_executed"] == 0 and caught["failure"] is None and caught["unsafe_reasons"] == {"table_penetration": 1}


def test_log_cache_rejects_incomplete_logs(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, pose=np.zeros((3, 7)))
    assert summary_from_log(p, EpisodeSpec(seed=1), "c") is None
    (tmp_path / "bad.npz").write_bytes(b"not a zip")
    assert summary_from_log(tmp_path / "bad.npz", EpisodeSpec(seed=1), "c") is None


def test_local_soft_argmax_ignores_background_and_edges():
    torch = pytest.importorskip("torch")
    from embodied.probes import G, local_soft_argmax, soft_argmax

    lg = torch.full((G * G,), 0.5)  # many weakly positive background patches
    lg[2 * G + 2] = 3.0  # a faint object near the top-left corner
    glob, loc = soft_argmax(lg), local_soft_argmax(lg)
    true = torch.tensor([2.5 / G, 2.5 / G])
    assert (loc - true).norm() < (glob - true).norm()  # the global version is dragged to the centre
    edge = torch.full((G * G,), -5.0)
    edge[0] = 4.0  # object in the corner: the window is clipped, not wrapped around
    assert local_soft_argmax(edge).max() < 2 / G
    nan = torch.full((G * G,), float("nan"))
    assert torch.isfinite(local_soft_argmax(nan)).all()


def test_fraction_changed_handles_keys_only_one_candidate_has():
    # recalibrated probes carry probe.readout; candidates made before it existed do not (crashed a live round)
    from embodied.candidate import Candidate

    old = Candidate(head={"probe.cube_w": np.zeros(4, np.float32)})
    new = Candidate(head={"probe.cube_w": np.ones(4, np.float32), "probe.readout": np.asarray(1)})
    assert new.fraction_changed(old) == 1.0
    assert old.fraction_changed(new) == 1.0
    assert old.fraction_changed(old) == 0.0
