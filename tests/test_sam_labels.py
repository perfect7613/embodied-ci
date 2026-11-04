import numpy as np

from embodied.sam_labels import agreement


def test_agreement_identical_and_shifted():
    sim = np.zeros((4, 2, 256), np.float32)
    sim[:, 0, 5 * 16 + 5] = 1.0  # cube at patch (5,5)
    sim[:, 1, 2 * 16 + 8] = 1.0  # gripper at (8,2)
    same = agreement(sim.copy(), sim)
    assert same["cube"]["patch_iou"] == 1.0 and same["cube"]["centroid_err_px"] == 0.0
    shifted = np.roll(sim, 1, axis=-1)  # one patch right = 16 px
    a = agreement(shifted, sim)
    assert a["cube"]["patch_iou"] == 0.0 and a["cube"]["centroid_err_px"] == 16.0
    assert a["gripper"]["detected"] == "4/4"
