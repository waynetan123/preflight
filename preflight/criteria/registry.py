"""The registry of safety criteria: what each one is called, what it reads, and whether it can be waived.

Built-in criteria are registered in catalogue.py. A plug-in registers its own with `register()`; it can add
criteria and the tags they read, but it cannot replace a criterion that is already registered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ValidationInfo


def _known_tag(value: str, info: ValidationInfo) -> str:
    tags = (info.context or {}).get("tags")
    if tags is not None and value not in tags:
        raise ValueError(f"unknown tag {value!r}; known tags: {', '.join(sorted(tags))}")
    return value


# A parameter that names a tag. The loader passes the known tags as validation context.
TagName = Annotated[str, AfterValidator(_known_tag)]


class Params(BaseModel):
    """Base for a criterion's parameters: unknown keys are errors, and loaded values cannot change."""

    model_config = {"extra": "forbid", "frozen": True}


@dataclass(frozen=True)
class Need:
    """One kind of fact that a criterion reads. Measure computes only the facts that criteria need.

    kind:
        contact     contacts between the robot or the held object and bodies of tags[0]
        proximity   smallest distance between two groups: tags[0] and tags[1]
        speed       speed of the TCP and of the held object
        tilt        angle from upright of each body of tags[0]
        fall        drops of each body of tags[0]: height and impact speed
        position    positions of the TCP, the robot bodies and the held object
    A group is a tag from the rules file, or "gripper": the hand, the fingers and the held object.
    """

    kind: Literal["contact", "proximity", "speed", "tilt", "fall", "position"]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriterionSpec:
    id: str
    kind: Literal["state", "trace"]
    # False: no exemption may relax it, and on_violation cannot be `warn`.
    waivable: bool
    params: type[Params]
    # Tags whose bodies the criterion reads. The tags a rules file may use are the union over all criteria.
    tags: tuple[str, ...] = ()
    # A replacement is active only inside an exemption's `replace_with`, never on its own.
    replacement: bool = False
    # Facts the criterion reads. Empty means Measure does not supply its facts yet.
    needs: tuple[Need, ...] = ()
    # check(facts, params, context) -> Violation | None. None here means the criterion cannot be checked yet.
    check: Callable | None = None


class Registry:
    def __init__(self) -> None:
        self._specs: dict[str, CriterionSpec] = {}

    def register(self, spec: CriterionSpec) -> CriterionSpec:
        if spec.id in self._specs:
            raise ValueError(f"criterion {spec.id!r} is already registered")
        if spec.replacement and spec.waivable:
            raise ValueError(f"replacement criterion {spec.id!r} cannot be waivable")
        self._specs[spec.id] = spec
        return spec

    def get(self, criterion_id: str) -> CriterionSpec | None:
        return self._specs.get(criterion_id)

    def criteria(self) -> list[CriterionSpec]:
        """Criteria that are always active, in registration order."""
        return [s for s in self._specs.values() if not s.replacement]

    def replacements(self) -> list[CriterionSpec]:
        return [s for s in self._specs.values() if s.replacement]

    def tags(self) -> set[str]:
        return {tag for spec in self._specs.values() for tag in spec.tags}

    def needs(self, criterion_ids) -> set[Need]:
        """The union of the facts that these criteria read."""
        return {need for cid in criterion_ids for need in self._specs[cid].needs}

    def copy(self) -> Registry:
        other = Registry()
        other._specs = dict(self._specs)
        return other


REGISTRY = Registry()


def register(spec: CriterionSpec) -> CriterionSpec:
    """Add a criterion to the default registry."""
    return REGISTRY.register(spec)
