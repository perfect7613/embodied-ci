"""Offline guardrail reward. Same sandbox + rules as embodied/candidate.py and embodied/gate.py."""

import builtins
import re

import numpy as np

SAFE_BUILTINS = {k: getattr(builtins, k) for k in ("abs", "min", "max", "len", "range", "float", "int", "bool", "any", "all", "sum", "round", "zip", "enumerate", "isinstance", "dict", "list", "tuple")}
HOLD = np.zeros(7, dtype=np.float32)
MAX_NOMINAL_BLOCK = 0.05


def retreat_up(dz=0.02):
    a = np.zeros(7, dtype=np.float32)
    a[2] = dz
    return a


def block(reason, safe_action=HOLD):
    return (reason, np.asarray(safe_action, dtype=np.float32))


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def compile_check(src: str):
    if "import" in src or "__" in src or "open(" in src:
        return None
    ns = {"__builtins__": SAFE_BUILTINS, "np": np, "ALLOW": None, "block": block, "HOLD": HOLD, "retreat_up": retreat_up}
    try:
        exec(compile(src, "<guardrail>", "exec"), ns)
    except Exception:
        return None
    return ns.get("check")


def score(text: str, steps: list[dict]) -> float:
    """steps: {state, action, unsafe: bool}. Reward in about [-1, 1]."""
    check = compile_check(extract_code(text))
    if check is None:
        return -1.0
    tp = fn = fp = tn = 0
    for s in steps:
        st = {k: (np.asarray(v, np.float32) if isinstance(v, list) else v) for k, v in s["state"].items()}
        try:
            r = check(st, {}, None, None, np.asarray(s["action"], np.float32))
        except Exception:
            r = ("error", HOLD)  # fail closed, as in production
        blocked = r is not None
        if s["unsafe"]:
            tp, fn = tp + blocked, fn + (not blocked)
        else:
            fp, tn = fp + blocked, tn + (not blocked)
    recall = tp / max(tp + fn, 1)
    false_block = fp / max(fp + tn, 1)
    penalty = 2.0 * max(0.0, false_block - MAX_NOMINAL_BLOCK) + false_block
    return float(recall - penalty)
