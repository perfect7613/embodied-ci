"""Tests through the interfaces that need no GPU: Candidate/guardrails, gate, REPL agent."""

import numpy as np
import pytest

from embodied.candidate import Candidate, GuardrailError, compile_guardrail
from embodied.gate import Registry, verdict
from harness.repl import Agent, Budget, Repl


def ep(seed, ok=True, suite="nominal", drift="nominal", blocked=0, unsafe=0):
    return {"id": f"x-{suite}-{drift}-{seed}", "seed": seed, "drift": drift, "suite": suite, "success": ok,
            "failure": None if ok else "no_lift", "steps": 50, "unsafe_proposals": unsafe, "unsafe_executed": unsafe,
            "blocked": blocked}


def test_guardrail_blocks_and_fails_closed():
    src = "def check(state, masks, z_now, z_pred, action):\n    if action[2] < -0.04:\n        return block('too_fast_down', retreat_up())\n    return ALLOW\n"
    c = Candidate(head={"w": np.zeros(3)}, guardrails=[src, "def check(*a):\n    return 1/0\n"])
    g = c.guard()
    r = g({}, {}, None, None, np.array([0, 0, -0.05, 0, 0, 0, 0]))
    assert r[0] == "g0:too_fast_down" and r[1][2] > 0
    r = g({}, {}, None, None, np.zeros(7))
    assert r[0].startswith("g1:error")


@pytest.mark.parametrize("bad", ["import os\ndef check(*a): pass", "def check(*a):\n    return open('x')", "x = 1"])
def test_guardrail_rejects_unsafe_source(bad):
    with pytest.raises(GuardrailError):
        compile_guardrail(bad)


def test_fraction_changed():
    a = Candidate(head={"w": np.zeros(100)})
    b = a.child(head={"w": np.r_[np.ones(5), np.zeros(95)]})
    assert b.fraction_changed(a) == pytest.approx(0.05)
    assert b.parent == a.id


def test_gate_passes_real_fix_and_rejects_block_everything():
    base = {"nominal": [ep(i) for i in range(10)], "regression": [ep(i, False, "regression") for i in range(4)], "heldout": []}
    good = {"nominal": [ep(i) for i in range(10)], "regression": [ep(i, i < 3, "regression") for i in range(4)], "heldout": []}
    assert verdict(good, base)["passed"]
    blocky = {"nominal": [ep(i, blocked=10) for i in range(10)], "regression": good["regression"], "heldout": []}
    v = verdict(blocky, base)
    assert not v["passed"] and any("block" in r for r in v["reasons"])
    worse = {"nominal": [ep(i, i > 0) for i in range(10)], "regression": good["regression"], "heldout": []}
    assert not verdict(worse, base)["passed"]


def test_registry_dedupes_and_hides_heldout(tmp_path):
    r = Registry(tmp_path)
    assert r.add([ep(1, False, "raw"), ep(1, False, "raw"), ep(2, False, "heldout")], "raw_run") == 1
    assert [s.seed for s in r.specs()] == [1]


def test_agent_runs_cells_until_final():
    replies = iter(["```python\nx = add(2, 3)\nprint(x)\n```", "```python\nFINAL = x * 10\n```"])
    agent = Agent(role="t", system="s", llm=lambda m: (next(replies), 10), repl=Repl(tools={"add": lambda a, b: a + b}), budget=Budget(max_turns=5))
    assert agent.run("go") == 50
