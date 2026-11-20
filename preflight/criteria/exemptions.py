"""Task exemptions: one tool relaxes one waivable criterion for one tag, and a narrower criterion takes over.

The three rules from the design doc:
1. Scoped: an exemption names a tool and a tag, and applies only inside that tool's calls.
2. Replaced, not removed: `replace_with` names at least one replacement criterion.
3. Hard floor: a non-waivable criterion cannot be relaxed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .registry import Params, Registry
from .schema import ExemptionEntry


@dataclass(frozen=True)
class Exemption:
    tool: str
    relax: str
    objects: frozenset[str]  # bodies the relaxation covers
    replace_with: Mapping[str, Params]


def exemption_problems(entry: ExemptionEntry, registry: Registry) -> list[str]:
    """Problems with the relaxed criterion and the replacement names; parameters are checked by the loader."""
    problems = []
    spec = registry.get(entry.relax)
    if spec is None:
        problems.append(f"relax: unknown criterion {entry.relax!r}")
    elif spec.replacement:
        problems.append(f"relax: {entry.relax!r} is a replacement criterion and is only active inside an exemption")
    elif not spec.waivable:
        problems.append(f"relax: {entry.relax!r} is not waivable, so no exemption can relax it")
    elif entry.objects.tag not in spec.tags:
        problems.append(
            f"objects.tag: {entry.relax!r} does not read tag {entry.objects.tag!r}; "
            f"it reads {', '.join(spec.tags) or 'no tags'}"
        )

    for name in entry.replace_with:
        replacement = registry.get(name)
        if replacement is None or not replacement.replacement:
            known = ", ".join(s.id for s in registry.replacements())
            problems.append(f"replace_with.{name}: not a replacement criterion; known: {known}")
    return problems
