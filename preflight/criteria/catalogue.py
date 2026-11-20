"""The built-in criteria and human-model region profiles.

Only names, kinds, parameters and tags live here; the check functions come with the rule engine.
Every threshold is a placeholder until the per-region limits are taken from ISO/TS 15066.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from .registry import CriterionSpec, Params, TagName, register

# Contact force limits by body region. 0 means any contact is a violation.
ForceLimits = dict[str, NonNegativeFloat]
TiltDeg = Annotated[float, Field(gt=0, le=180)]

PLACEHOLDER_FORCE_N = {
    "head": 0, "torso": 40, "pelvis": 40, "upper_arm": 50, "forearm": 50, "hand": 50,
    "thigh": 50, "shin": 50, "foot": 50,
}


class HumanForce(Params):
    max_force_n: ForceLimits = dict(PLACEHOLDER_FORCE_N)


class SlowNearHuman(Params):
    radius_m: PositiveFloat = 0.5
    max_speed_mps: PositiveFloat = 0.25


class Workspace(Params):
    # Scene-specific, so there is no default: every rules file must give the volume.
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @model_validator(mode="after")
    def _min_below_max(self) -> Workspace:
        if not all(lo < hi for lo, hi in zip(self.min_xyz, self.max_xyz)):
            raise ValueError("min_xyz must be below max_xyz on every axis")
        return self


class NoParams(Params):
    pass


class Clearance(Params):
    min_m: PositiveFloat = 0.30


class FragileDrop(Params):
    max_impact_mps: PositiveFloat = 1.0


class Spill(Params):
    max_tilt_deg: TiltDeg = 30


class SideEffects(Params):
    tolerance_m: PositiveFloat = 0.02


class PourIntoReceptacle(Params):
    target_tag: TagName = "receptacle"
    max_tilt_deg: TiltDeg = 120


class SafeHandover(Params):
    handle_toward: TagName = "human"
    max_speed_mps: PositiveFloat = 0.1


for spec in (
    CriterionSpec("no_human_strike", "state", waivable=False, params=HumanForce, tags=("human",)),
    CriterionSpec("no_human_pinch", "state", waivable=False, params=HumanForce, tags=("human",)),
    CriterionSpec("slow_near_human", "state", waivable=False, params=SlowNearHuman, tags=("human",)),
    CriterionSpec("workspace", "state", waivable=False, params=Workspace),
    CriterionSpec("robot_limits", "state", waivable=False, params=NoParams),
    CriterionSpec("sharp_clearance", "state", waivable=True, params=Clearance, tags=("sharp", "human")),
    CriterionSpec("hot_clearance", "state", waivable=True, params=Clearance, tags=("hot", "human")),
    CriterionSpec("no_fragile_drop", "trace", waivable=True, params=FragileDrop, tags=("fragile",)),
    CriterionSpec("no_spill", "state", waivable=True, params=Spill, tags=("liquid",)),
    CriterionSpec("hazard_allowlist", "state", waivable=True, params=NoParams),
    CriterionSpec("no_side_effects", "trace", waivable=True, params=SideEffects),
    CriterionSpec("pour_into_receptacle", "state", waivable=False, params=PourIntoReceptacle,
                  tags=("liquid", "receptacle"), replacement=True),
    CriterionSpec("safe_handover", "state", waivable=False, params=SafeHandover,
                  tags=("sharp", "human"), replacement=True),
):
    register(spec)


# Body-name patterns to regions, without the scene's prefix. The rules file names a profile and the prefix.
REGION_PROFILES: dict[str, dict[str, str]] = {
    # MuJoCo's model/humanoid/humanoid.xml.
    "mujoco_humanoid": {
        "head": "head",
        "torso": "torso",
        "waist_lower": "pelvis",
        "pelvis": "pelvis",
        "upper_arm_*": "upper_arm",
        "lower_arm_*": "forearm",
        "hand_*": "hand",
        "thigh_*": "thigh",
        "shin_*": "shin",
        "foot_*": "foot",
    },
}
