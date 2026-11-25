"""The built-in criteria and human-model region profiles.

Each criterion has its name, kind, parameters, tags, the facts it reads, and its check function in
checks.py. A criterion with no check is reported as unchecked; it never passes silently.
Every threshold is a placeholder until the per-region limits are taken from ISO/TS 15066.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from . import checks
from .registry import CriterionSpec, Need, Params, TagName, register

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


class RobotLimits(Params):
    # A joint this close to the end of its range is at its mechanical stop. The kitchen arm's planner keeps each
    # joint 10 mrad inside its range (IK_JOINT_MARGIN_RAD in tabletop_arm.py).
    joint_margin_rad: NonNegativeFloat = 0.001
    # Longest time an arm actuator may stay at its force limit. Ordinary moves stay there for 10 ms at most.
    max_saturated_s: NonNegativeFloat = 0.05


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


HUMAN_CONTACT = Need("contact", ("human",))
GRIPPER_TO_HUMAN = Need("proximity", ("gripper", "human"))
SPEED = Need("speed")

# Not checked yet:
#   no_human_pinch     the static human is fixed to the world, so a pinch shows only as contact force,
#                      which no_human_strike checks; a real pinch check needs contact normals in the trace
#   hazard_allowlist   needs the semantic track
#   no_side_effects    needs a before-and-after diff of the world
#   pour_into_receptacle, safe_handover   need spatial relations and object orientation
for spec in (
    CriterionSpec("no_human_strike", "state", waivable=False, params=HumanForce, tags=("human",),
                  needs=(HUMAN_CONTACT,), check=checks.no_human_strike),
    CriterionSpec("no_human_pinch", "state", waivable=False, params=HumanForce, tags=("human",),
                  needs=(HUMAN_CONTACT,)),
    CriterionSpec("slow_near_human", "state", waivable=False, params=SlowNearHuman, tags=("human",),
                  needs=(SPEED, GRIPPER_TO_HUMAN), check=checks.slow_near_human),
    CriterionSpec("workspace", "state", waivable=False, params=Workspace,
                  needs=(Need("position"),), check=checks.workspace),
    CriterionSpec("robot_limits", "state", waivable=False, params=RobotLimits,
                  needs=(Need("robot"),), check=checks.robot_limits),
    CriterionSpec("sharp_clearance", "state", waivable=True, params=Clearance, tags=("sharp", "human"),
                  needs=(Need("proximity", ("sharp", "human")),), check=checks.sharp_clearance),
    CriterionSpec("hot_clearance", "state", waivable=True, params=Clearance, tags=("hot", "human"),
                  needs=(Need("proximity", ("hot", "human")),), check=checks.hot_clearance),
    CriterionSpec("no_fragile_drop", "trace", waivable=True, params=FragileDrop, tags=("fragile",),
                  needs=(Need("fall", ("fragile",)),), check=checks.no_fragile_drop),
    CriterionSpec("no_spill", "state", waivable=True, params=Spill, tags=("liquid",),
                  needs=(Need("tilt", ("liquid",)),), check=checks.no_spill),
    CriterionSpec("hazard_allowlist", "state", waivable=True, params=NoParams),
    CriterionSpec("no_side_effects", "trace", waivable=True, params=SideEffects),
    CriterionSpec("pour_into_receptacle", "state", waivable=False, params=PourIntoReceptacle,
                  tags=("liquid", "receptacle"), replacement=True, needs=(Need("tilt", ("liquid",)),)),
    CriterionSpec("safe_handover", "state", waivable=False, params=SafeHandover,
                  tags=("sharp", "human"), replacement=True,
                  needs=(SPEED, Need("proximity", ("sharp", "human")))),
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
