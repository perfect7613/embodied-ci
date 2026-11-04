"""A Candidate is the one thing every fix produces and the gate judges:
model-side weights (internal fixes) + guardrail sources (external fix).

`head` holds both editable weight sets of the model under test:
  controller  fc1/fc2/out/mu/sd   reads named probe features -> action
  probes      probe.{cube,gripper}_{w,b}, probe.layer   linear readouts on a V-JEPA encoder layer

Guardrail contract — generated code defines exactly:

    def check(state, masks, z_now, z_pred, action):
        return ALLOW                      # or
        return block("reason", safe_action)

state:  dict  eef_pos (3,), cube_pos (3,), gripper_closed float, table_z float, contact_force float
masks:  dict  name -> (256, 256) bool  ("cube", "robot")
z_now, z_pred: (64, 1408) grid-pooled JEPA latents now / predicted after `action`
action: (7,) [dx, dy, dz, droll, dpitch, dyaw, dclosedness]
Namespace: np, ALLOW, block, HOLD, retreat_up. No imports, no I/O.
"""

from __future__ import annotations

import builtins
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HOLD = np.zeros(7, dtype=np.float32)
ALLOW = None
SAFE_BUILTINS = {k: getattr(builtins, k) for k in ("abs", "min", "max", "len", "range", "float", "int", "bool", "any", "all", "sum", "round", "zip", "enumerate", "isinstance", "dict", "list", "tuple")}


def _numpy_only_import(name, globals=None, locals=None, fromlist=(), level=0):
    """numpy's C code imports its own helpers lazily (ndarray.max -> numpy._core._methods) through the
    *calling* frame's builtins. Without this, `x.max()` inside a guardrail raised KeyError('__import__') and the
    guardrail failed closed (held the arm) on every step. Only numpy's own modules resolve; guardrail source
    still cannot contain "import" or dunders, so it cannot call this directly."""
    if level == 0 and name.split(".")[0] == "numpy":
        return builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is not allowed in a guardrail")


SAFE_BUILTINS["__import__"] = _numpy_only_import


def retreat_up(dz: float = 0.02) -> np.ndarray:
    a = np.zeros(7, dtype=np.float32)
    a[2] = dz
    return a


def block(reason: str, safe_action: np.ndarray = HOLD) -> tuple[str, np.ndarray]:
    return (reason, np.asarray(safe_action, dtype=np.float32))


class GuardrailError(Exception):
    pass


def compile_guardrail(src: str):
    if "import" in src or "__" in src or "open(" in src:
        raise GuardrailError("imports, dunders and I/O are not allowed")
    ns = {"__builtins__": SAFE_BUILTINS, "np": np, "ALLOW": ALLOW, "block": block, "HOLD": HOLD, "retreat_up": retreat_up}
    exec(compile(src, "<guardrail>", "exec"), ns)
    if "check" not in ns:
        raise GuardrailError("guardrail must define check(state, masks, z_now, z_pred, action)")
    return ns["check"]


def safety_envelope(rules: set[str]) -> str:
    """Guardrail source for the safety rules a candidate broke. It clips the action back inside the envelope
    (speed, table, workspace) instead of stopping, so normal behaviour barely changes; it blocks only on contact force."""
    from embodied.sim import MAX_CONTACT_FORCE, MAX_STEP_M, TABLE_Z_MARGIN, WORKSPACE

    lo, hi = WORKSPACE[:, 0] + 0.01, WORKSPACE[:, 1] - 0.01
    return f"""def check(state, masks, z_now, z_pred, action):
    # safety envelope for: {", ".join(sorted(rules))}
    if state["contact_force"] > {0.85 * MAX_CONTACT_FORCE:.1f}:
        return block("contact_force", retreat_up(0.02))
    a = np.array(action, dtype=np.float32)
    n = float(np.linalg.norm(a[:3]))
    if n > {0.9 * MAX_STEP_M:.4f}:
        a[:3] = a[:3] * ({0.9 * MAX_STEP_M:.4f} / n)
    lo = np.array([{lo[0]:.3f}, {lo[1]:.3f}, max({lo[2]:.3f}, state["table_z"] + {TABLE_Z_MARGIN + 0.004:.3f})])
    hi = np.array([{hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}])
    nxt = np.clip(state["eef_pos"] + a[:3], lo, hi)
    a[:3] = nxt - state["eef_pos"]
    if float(np.abs(a - action).max()) > 1e-7:
        return block("envelope", a)
    return ALLOW
"""


@dataclass
class Candidate:
    head: dict[str, np.ndarray]  # controller state_dict + "probe.*" readout weights, as numpy
    guardrails: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent: str | None = None
    note: str = ""

    # ---- guardrails --------------------------------------------------------
    def guard(self):
        """-> fn(state, masks, z_now, z_pred, action) -> (reason, safe_action) | None. First block wins."""
        checks = [compile_guardrail(s) for s in self.guardrails]

        def run(state, masks, z_now, z_pred, action):
            for i, c in enumerate(checks):
                try:
                    r = c(state, masks, z_now, z_pred, action)
                except Exception as e:  # fail closed: a crashing guardrail holds, and the gate's nominal rule rejects it
                    return (f"g{i}:error:{type(e).__name__}", HOLD)
                if r is not None:
                    return (f"g{i}:{r[0]}", r[1])
            return None

        return run

    @property
    def needs_prediction(self) -> bool:
        return any("z_pred" in s for s in self.guardrails)

    # ---- weights -----------------------------------------------------------
    @property
    def controller(self) -> dict[str, np.ndarray]:
        return {k: v for k, v in self.head.items() if not k.startswith("probe.")}

    @property
    def probes(self) -> dict[str, np.ndarray]:
        return {k[6:]: v for k, v in self.head.items() if k.startswith("probe.")}

    def fraction_changed(self, other: "Candidate", tol: float = 1e-8) -> float:
        changed = total = 0
        for k, v in self.head.items():
            o = other.head.get(k)
            if o is None or np.shape(o) != np.shape(v):  # a tensor the other candidate lacks (e.g. probe.readout): all of it is new
                changed += int(np.size(v))
            else:
                changed += int((np.abs(np.asarray(v, np.float64) - np.asarray(o, np.float64)) > tol).sum())
            total += int(np.size(v))
        return changed / max(total, 1)

    def child(self, head: dict | None = None, guardrails: list[str] | None = None, note: str = "") -> "Candidate":
        return Candidate(
            head={k: v.copy() for k, v in (head or self.head).items()},
            guardrails=list(self.guardrails if guardrails is None else guardrails),
            parent=self.id,
            note=note,
        )

    # ---- persistence -------------------------------------------------------
    def save(self, root: str | Path) -> Path:
        d = Path(root) / "candidates" / self.id
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "head.npz", **self.head)
        (d / "meta.json").write_text(json.dumps({"id": self.id, "parent": self.parent, "note": self.note, "guardrails": self.guardrails}, indent=2))
        return d

    @classmethod
    def load(cls, root: str | Path, cid: str) -> "Candidate":
        d = Path(root) / "candidates" / cid
        meta = json.loads((d / "meta.json").read_text())
        with np.load(d / "head.npz") as z:
            head = {k: z[k] for k in z.files}
        return cls(head=head, guardrails=meta["guardrails"], id=meta["id"], parent=meta["parent"], note=meta["note"])
