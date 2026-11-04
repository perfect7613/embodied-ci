import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "posttrain" / "baseten_grpo"))
from reward import score  # noqa: E402

STEPS = [{"state": {"eef_pos": [0, 0, 1.0], "cube_pos": [0, 0, 0.82], "gripper_closed": 0, "table_z": 0.8, "contact_force": 0.0},
          "action": [0.08, 0, 0, 0, 0, 0, 0], "unsafe": True}] + \
        [{"state": {"eef_pos": [0, 0, 1.0], "cube_pos": [0, 0, 0.82], "gripper_closed": 0, "table_z": 0.8, "contact_force": 0.0},
          "action": [0.02, 0, 0, 0, 0, 0, 0], "unsafe": False}] * 20

GOOD = "```python\ndef check(state, masks, z_now, z_pred, action):\n    n = float(np.linalg.norm(action[:3]))\n    if n > 0.05:\n        a = action.copy(); a[:3] = a[:3] / n * 0.049\n        return block('speed', a)\n    return ALLOW\n```"
BLOCK_ALL = "```python\ndef check(state, masks, z_now, z_pred, action):\n    return block('no', HOLD)\n```"


def test_reward_orders_guardrails():
    assert score(GOOD, STEPS) == 1.0
    assert score(BLOCK_ALL, STEPS) < 0
    assert score("not code", STEPS) == -1.0
    assert score("```python\nimport os\ndef check(*a): pass\n```", STEPS) == -1.0
