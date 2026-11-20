"""Check: apply the tool's exemptions, then run every active criterion on the facts of one rollout.

    checker = Checker(rules)
    facts = measure.facts(rollout.trace, checker.needs(tool))
    result = checker.check(facts, tool)
    result.findings        # earliest violation of each criterion, in time order, with its action
    result.unchecked       # criteria that could not be checked, and why

Exemptions follow the design doc: a tool's exemption relaxes one waivable criterion for the bodies of one
tag, and its replacement criteria take over. If a replacement cannot be checked yet, the exemption is not
applied: the original criterion stays fully active. A check that raises an exception gives a blocking
finding, so an error never lets an action through.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from preflight.criteria import REGISTRY, Need, Params, Registry, Rules
from preflight.criteria.checks import Context, Violation
from preflight.measure import Facts


@dataclass(frozen=True)
class Finding:
    violation: Violation
    action: str  # block | modify | warn, from the rules file; replacements and failed checks always block


@dataclass(frozen=True)
class CheckResult:
    tool: str
    findings: tuple[Finding, ...]
    checked: tuple[str, ...]
    unchecked: Mapping[str, str]  # criterion (or exemption) -> reason


@dataclass(frozen=True)
class _Active:
    criterion: str
    check: Callable
    params: Params
    exempt: frozenset[str]
    action: str
    needs: tuple[Need, ...]


class Checker:
    def __init__(self, rules: Rules, registry: Registry | None = None) -> None:
        self.rules = rules
        self.registry = registry or REGISTRY

    def needs(self, tool: str) -> set[Need]:
        """The facts that Measure must compute for a call of `tool`."""
        active, _ = self._plan(tool)
        return {need for a in active for need in a.needs}

    def check(self, facts: Facts, tool: str) -> CheckResult:
        active, unchecked = self._plan(tool)
        findings = []
        for a in active:
            ctx = Context(regions=self.rules.regions, tags=self.rules.tags, exempt=a.exempt)
            try:
                violation = a.check(facts, a.params, ctx)
            except Exception as exc:  # noqa: BLE001  fail closed
                violation = Violation(a.criterion, None, None, None, (), f"the check failed: {type(exc).__name__}: {exc}")
                findings.append(Finding(violation, "block"))
                continue
            if violation is not None:
                findings.append(Finding(violation, a.action))
        findings.sort(key=lambda f: -math.inf if f.violation.t is None else f.violation.t)
        return CheckResult(tool, tuple(findings), tuple(a.criterion for a in active), MappingProxyType(unchecked))

    def _plan(self, tool: str) -> tuple[list[_Active], dict[str, str]]:
        rules, registry = self.rules, self.registry
        exemptions = {e.relax: e for e in rules.exemptions_for(tool)}
        active, unchecked = [], {}
        for spec in registry.criteria():
            exempt = frozenset()
            exemption = exemptions.get(spec.id)
            if exemption is not None:
                missing = [name for name in exemption.replace_with if registry.get(name).check is None]
                if missing:
                    unchecked[f"{tool}: exemption from {spec.id}"] = (
                        f"not applied: the replacement {', '.join(missing)} cannot be checked yet, "
                        f"so {spec.id} stays fully active"
                    )
                else:
                    exempt = exemption.objects
                    for name, params in exemption.replace_with.items():
                        replacement = registry.get(name)
                        active.append(_Active(name, replacement.check, params, frozenset(), "block", replacement.needs))
            if spec.check is None:
                unchecked[spec.id] = "no check yet: Measure does not supply the facts it needs"
                continue
            active.append(_Active(spec.id, spec.check, rules.params[spec.id], exempt, rules.on_violation[spec.id],
                                  spec.needs))
        return active, unchecked
