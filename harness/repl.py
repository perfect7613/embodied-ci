"""REPL-driven agent harness.

The orchestrator LLM's only tool is this persistent Python REPL. Large context
(episodes, tensors, metrics) lives in REPL variables, never in the prompt; the
model reads what it needs with code. Subagents are child REPLs started with
`spawn()`. The REPL is not a security sandbox; isolation comes from the Modal
container it runs in.
"""

from __future__ import annotations

import contextlib
import io
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

CODE_RE = re.compile(r"```python\n(.*?)```", re.S)


@dataclass
class Budget:
    max_turns: int = 30
    max_tokens: int = 400_000
    max_seconds: float = 1800

    def check(self, turns: int, tokens: int, started: float) -> str | None:
        if turns >= self.max_turns:
            return "turn budget spent"
        if tokens >= self.max_tokens:
            return "token budget spent"
        if time.time() - started >= self.max_seconds:
            return "time budget spent"
        return None


@dataclass
class Repl:
    """Persistent namespace + exec. `tools` are injected as plain functions."""

    tools: dict[str, Callable[..., Any]]
    ns: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ns.update(self.tools)
        self.ns["FINAL"] = None

    def run(self, code: str, max_out: int = 4000) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(code, "<cell>", "exec"), self.ns)
            except Exception:
                traceback.print_exc()
        out = buf.getvalue()
        return out if len(out) <= max_out else out[:max_out] + f"\n... [{len(out) - max_out} chars truncated; slice the variable instead]"


@dataclass
class Agent:
    role: str
    system: str
    llm: Callable[[list[dict]], tuple[str, int]]  # messages -> (text, tokens used)
    repl: Repl
    budget: Budget = field(default_factory=Budget)
    emit: Callable[[dict], None] = lambda e: None

    def run(self, task: str) -> Any:
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        turns = tokens = 0
        started = time.time()
        while (stop := self.budget.check(turns, tokens, started)) is None:
            text, used = self.llm(msgs)
            turns, tokens = turns + 1, tokens + used
            msgs.append({"role": "assistant", "content": text})
            cells = CODE_RE.findall(text)
            left = self.budget.max_turns - turns
            nudge = f"\n[{left} turns left]" + (" Set FINAL in your next cell, with your best result so far." if left <= 2 else "")
            if not cells:
                msgs.append({"role": "user", "content": "Reply with one ```python``` cell. Set FINAL to finish." + nudge})
                continue
            out = self.repl.run(cells[0])
            self.emit({"agent": self.role, "code": cells[0], "stdout": out, "tokens": used})
            if self.repl.ns.get("FINAL") is not None:
                return self.repl.ns["FINAL"]
            msgs.append({"role": "user", "content": f"[stdout]\n{out}{nudge}"})
            # TODO: compaction at ~80% of window: summarise old turns, keep vars.
        return {"error": stop}
