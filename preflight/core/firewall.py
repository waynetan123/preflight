"""The firewall: compile a tool call, simulate it, measure, check, decide, and run only what passed.

    fw = Firewall.from_yaml("scenes/kitchen_rules.yaml")
    decision = fw.call("move_to", {"xyz": [0.45, 0.1, 1.0], "speed": 0.5})
    decision.verdict              # Verdict.ALLOW | MODIFY | BLOCK
    decision.message()            # the tool result for the LLM
    guarded = fw.wrap()           # the entry point for an LLM's tool calls (preflight/core/guarded.py)

Decide:
    BLOCK   the call does not compile, the rollout fails, any finding has action `block`, or a repair fails
    MODIFY  only `modify` findings, and a slower version of the same path passes a new simulation
    ALLOW   no findings, or only `warn` findings

Repair in v0.1 is speed retiming only: the same path, up to 4 times slower, and never longer than the
motion limit of the robot. Every repaired trajectory is simulated and checked again before it can run.

With an `AuditLog`, each check and each call writes one record before anything runs.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from preflight.backends import MujocoBackend
from preflight.criteria import Registry, Rules, load_rules
from preflight.measure import Measure
from preflight.robots import CompileError, TabletopArm, Trajectory
from preflight.robots.tabletop_arm import MAX_MOTION_S

from .check import Checker, CheckResult, Finding
from .verdict import Decision, Verdict

if TYPE_CHECKING:
    from preflight.audit import AuditLog

    from .guarded import Guarded

MAX_RETIME = 4.0
RETIME_MARGIN = 1.05  # slow a little more than the measured excess
RETIME_STEP = 1.5  # if a repair still violates, try this much slower


class Firewall:
    def __init__(self, rules: Rules, backend: MujocoBackend, arm: TabletopArm,
                 registry: Registry | None = None, audit: AuditLog | None = None) -> None:
        self.rules = rules
        self.backend = backend
        self.arm = arm
        self.checker = Checker(rules, registry)
        self.measure = Measure(backend.model, rules, arm)
        self.audit = audit

    @classmethod
    def from_yaml(cls, path: str | Path, registry: Registry | None = None, audit: AuditLog | None = None) -> Firewall:
        rules = load_rules(path, registry)
        backend = MujocoBackend(rules.scene)
        return cls(rules, backend, TabletopArm(backend.model), registry, audit)

    def wrap(self, feedback: str = "detailed") -> Guarded:
        """The guarded robot for an LLM: tool definitions, and one function that handles each tool call."""
        from .guarded import Guarded  # guarded.py imports this module's types

        return Guarded(self, feedback)

    def call(self, tool: str, args: Mapping[str, Any]) -> Decision:
        """Check the call, and run it on the live robot if the verdict is ALLOW or MODIFY."""
        decision = self._decide(tool, args)
        self._log(decision, "call", executes=decision.allowed)
        if decision.allowed:
            self.backend.execute(self.arm, decision.trajectory)
        return decision

    def check(self, tool: str, args: Mapping[str, Any]) -> Decision:
        """Decide on a call without running it. The live state does not change."""
        decision = self._decide(tool, args)
        self._log(decision, "check", executes=False)
        return decision

    def _log(self, decision: Decision, mode: str, executes: bool) -> None:
        # The live state is still the state that the decision started from: nothing has run yet.
        if self.audit is not None:
            self.audit.record(decision, self.backend.model, self.backend.live, self.arm, mode, executes)

    def _decide(self, tool: str, args: Mapping[str, Any]) -> Decision:
        start = time.perf_counter()
        timing: dict[str, float] = {}

        def decision(verdict: Verdict, reason: str, trajectory: Trajectory | None = None,
                     findings: tuple[Finding, ...] = (), **extra) -> Decision:
            timing["total"] = (time.perf_counter() - start) * 1000
            return Decision(tool, MappingProxyType(dict(args)), verdict, trajectory, findings, reason,
                            timing_ms=MappingProxyType(timing), **extra)

        try:
            trajectory = self.arm.compile(self.backend.live, tool, args)
        except CompileError as exc:
            timing["compile"] = (time.perf_counter() - start) * 1000
            return decision(Verdict.BLOCK, str(exc), failure="compile")
        timing["compile"] = (time.perf_counter() - start) * 1000

        result, failure = self._simulate(tool, trajectory, timing)
        if failure is not None:
            return decision(Verdict.BLOCK, failure[1], failure=failure[0])
        findings, unchecked = result.findings, result.unchecked
        blocking = [f for f in findings if f.action == "block"]
        modify = [f for f in findings if f.action == "modify"]
        warnings = [f for f in findings if f.action == "warn"]

        if blocking:
            return decision(Verdict.BLOCK, _describe(blocking[0]), findings=findings, unchecked=unchecked)
        if not modify:
            reason = "; ".join(f"warning: {_describe(f)}" for f in warnings) or "no violations"
            return decision(Verdict.ALLOW, reason, trajectory, findings, unchecked=unchecked)

        # Repair: the same path, slower, simulated and checked again.
        factor = _first_factor(modify)
        while True:
            slow = trajectory.retime(factor)
            if slow.duration > MAX_MOTION_S:
                break
            again, failure = self._simulate(tool, slow, timing)
            if failure is not None:
                return decision(Verdict.BLOCK, f"the slowed motion failed in simulation: {failure[1]}",
                                findings=findings, failure=failure[0], unchecked=unchecked)
            still = [f for f in again.findings if f.action in ("block", "modify")]
            if not still:
                criteria = ", ".join(sorted({f.violation.criterion for f in modify}))
                return decision(
                    Verdict.MODIFY,
                    f"slowed {factor:.1f}x to a peak of {slow.speed:.2f} m/s, because of {criteria}",
                    slow, findings, retime_factor=factor, unchecked=unchecked,
                )
            if any(f.action == "block" for f in still):
                first = next(f for f in still if f.action == "block")
                return decision(Verdict.BLOCK, f"the slowed motion still fails: {_describe(first)}",
                                findings=findings, unchecked=unchecked)
            if factor >= MAX_RETIME:
                break
            factor = min(factor * RETIME_STEP, MAX_RETIME)
        return decision(
            Verdict.BLOCK,
            f"{_describe(modify[0])}; a motion up to {MAX_RETIME:.0f}x slower (and at most "
            f"{MAX_MOTION_S:.0f} s long) does not remove the violation",
            findings=findings, unchecked=unchecked,
        )

    def _simulate(self, tool: str, trajectory: Trajectory, timing: dict[str, float]):
        """Rollout, Measure and Check. Returns (CheckResult, None), or (None, (failure, reason))."""
        t0 = time.perf_counter()
        rollout = self.backend.rollout(self.arm, trajectory, settle_s=self.rules.settle_s)
        t1 = time.perf_counter()
        timing["rollout"] = timing.get("rollout", 0.0) + (t1 - t0) * 1000
        if not rollout.ok:
            return None, (rollout.failure, f"the simulation could not be trusted ({rollout.failure}): {rollout.detail}")
        facts = self.measure.facts(rollout.trace, self.checker.needs(tool))
        t2 = time.perf_counter()
        result: CheckResult = self.checker.check(facts, tool)
        t3 = time.perf_counter()
        timing["measure"] = timing.get("measure", 0.0) + (t2 - t1) * 1000
        timing["check"] = timing.get("check", 0.0) + (t3 - t2) * 1000
        return result, None


def _describe(finding: Finding) -> str:
    v = finding.violation
    when = "" if v.t is None else f" at t = {v.t:.2f} s"
    return f"{v.criterion}{when}: {v.message}"


def _first_factor(modify: list[Finding]) -> float:
    """A first guess at how much slower: the largest measured/limit ratio among the findings, or 1.5."""
    ratios = [f.violation.value / f.violation.limit for f in modify
              if f.violation.value is not None and f.violation.limit]
    guess = max(ratios) * RETIME_MARGIN if ratios else RETIME_STEP
    return min(max(guess, 1.01), MAX_RETIME)
