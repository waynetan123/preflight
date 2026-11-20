"""The structure of a rules file. Checks that need the scene or the registry are in loader.py."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .registry import TagName

NonEmpty = Annotated[str, Field(min_length=1)]
OnViolation = Literal["block", "modify", "warn"]


class Strict(BaseModel):
    model_config = {"extra": "forbid"}


class Horizon(Strict):
    settle_s: float = Field(1.0, gt=0, le=10)


class HumanModel(Strict):
    """Which body region each `human` body is, for per-region limits."""

    profile: NonEmpty | None = None  # a key of catalogue.REGION_PROFILES
    prefix: str = ""  # put before every pattern of the profile, e.g. "human_"
    # Body-name pattern -> region. Overrides the profile; full body names, no prefix added.
    regions: dict[NonEmpty, NonEmpty] = {}


class CriteriaConfig(Strict):
    params: dict[str, dict[str, Any]] = {}
    on_violation: dict[str, OnViolation] = {}


class ExemptionObjects(Strict):
    tag: TagName


class ExemptionEntry(Strict):
    tool: NonEmpty
    relax: NonEmpty
    objects: ExemptionObjects
    replace_with: dict[str, dict[str, Any]] = Field(min_length=1)


class RulesFile(Strict):
    version: Literal[1]
    scene: NonEmpty  # relative to the rules file
    horizon: Horizon = Horizon()
    tags: dict[TagName, list[NonEmpty]]  # tag -> body-name patterns (fnmatch)
    human_model: HumanModel = HumanModel()
    criteria: CriteriaConfig = CriteriaConfig()
    exemptions: list[ExemptionEntry] = []
