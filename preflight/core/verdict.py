"""Verdicts, and the text that goes back to the LLM as the tool result."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from preflight.robots import Trajectory

from .check import Finding


class Verdict(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"  # allowed after a repair that passed a new simulation
    BLOCK = "block"


@dataclass(frozen=True)
class Decision:
    tool: str
    args: Mapping
    verdict: Verdict
    # What runs if the verdict is not BLOCK: the checked trajectory, or the repaired one.
    trajectory: Trajectory | None
    findings: tuple[Finding, ...]  # from the first check
    reason: str  # one line: why this verdict
    # The trajectory of the last simulation, which decided the verdict. It is `trajectory` if the verdict is not
    # BLOCK. On BLOCK it is the motion that was blocked, which does not run. None if the call did not compile.
    checked: Trajectory | None = None
    retime_factor: float = 1.0  # > 1 if MODIFY slowed the motion
    failure: str | None = None  # compile | a rollout failure reason, if the check could not run
    unchecked: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    timing_ms: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def allowed(self) -> bool:
        return self.verdict is not Verdict.BLOCK

    def message(self, feedback: str = "detailed") -> str:
        """The tool result for the LLM. `minimal` does not tell an attacker which rule fired or its limit."""
        if feedback == "minimal":
            return {Verdict.ALLOW: "done", Verdict.MODIFY: "done", Verdict.BLOCK: "blocked by safety policy"}[self.verdict]
        if self.verdict is Verdict.ALLOW:
            return "done"
        if self.verdict is Verdict.MODIFY:
            return f"done, with a change: {self.reason}"
        return f"blocked: {self.reason}"
