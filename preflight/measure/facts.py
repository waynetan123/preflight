"""Facts: what the criteria read about a rollout. Measure computes them from a Trace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# Distances above this are reported as this value.
REPORT_RADIUS_M = 1.0


@dataclass(frozen=True)
class ContactFacts:
    """Contacts between the robot (or the object it holds) and the bodies of one tag.

    One row for each contact at each physics step, in time order.
    """

    tag: str
    t: np.ndarray  # (M,) seconds
    robot_body: np.ndarray  # (M,) body name on the robot side: a robot body or the held object
    other_body: np.ndarray  # (M,) body name with the tag
    force_n: np.ndarray  # (M,) normal force
    other_fixed: np.ndarray | None = None  # (M,) True if the tag body is fixed to the world, so it cannot move away
    dt: float | None = None  # seconds between physics steps


@dataclass(frozen=True)
class Proximity:
    """Smallest distance between two groups at each sample. Negative means overlap."""

    groups: tuple[str, str]
    dist_m: np.ndarray  # (S,) REPORT_RADIUS_M if farther, or if a group has no bodies at that sample
    a: np.ndarray  # (S,) nearest body of groups[0], or None
    b: np.ndarray  # (S,) nearest body of groups[1], or None


@dataclass(frozen=True)
class Speed:
    tcp_mps: np.ndarray  # (S,)
    held_mps: np.ndarray  # (S,) NaN when the gripper holds nothing
    held: np.ndarray  # (S,) held body name, or None


@dataclass(frozen=True)
class Positions:
    """World positions of the robot's points and of the held object, at each sample."""

    names: tuple[str, ...]  # "tcp", then the robot body names
    xyz: np.ndarray  # (S, len(names), 3)
    held_xyz: np.ndarray  # (S, 3) origin of the held object; NaN when the gripper holds nothing
    held: np.ndarray  # (S,) held body name, or None


@dataclass(frozen=True)
class Tilt:
    """Angle between each body's z axis and the world's z axis, at each sample."""

    tag: str
    deg: Mapping[str, np.ndarray]  # body name -> (S,)


@dataclass(frozen=True)
class Fall:
    """A body that lost all contact and dropped."""

    body: str
    t_release: float  # last time with contact or grasp
    t_impact: float | None  # first time with contact again; None if still in the air at the end
    drop_m: float  # height lost from release to impact (or to the end)
    impact_mps: float | None  # speed just before impact


@dataclass(frozen=True)
class RobotHealth:
    """The arm's joints and actuators at every physics step, and the robot's contacts with itself.

    The gripper's joints and actuator are left out: the fingers rest at the open end of their range, and the
    gripper actuator's gain and bias keep its force far below its force limit.
    """

    t: np.ndarray  # (T,) seconds, every physics step
    joints: tuple[str, ...]  # the arm joints, from the base
    qpos: np.ndarray  # (T, J) radians
    q_range: np.ndarray  # (J, 2) lower and upper end of each joint's range
    actuators: tuple[str, ...]  # the arm actuators, from the base
    force: np.ndarray  # (T, A) actuator force (N m for a hinge joint); NaN where the trace has no value
    force_range: np.ndarray  # (A, 2) MuJoCo clamps the force to this range; -inf, inf if it has no limit
    self_contacts: ContactFacts  # contacts between two robot bodies, except between the fingers


@dataclass(frozen=True)
class Diff:
    """How far each free object is from where it started, at each sample.

    The distance is the largest movement of any corner of the object's bounding box, so a turn in place
    counts as well as a shift. v0.1 has no discrete state (appliances, doors), so positions are the diff.
    """

    names: tuple[str, ...]  # bodies with a free joint that are not part of the robot
    dist_m: np.ndarray  # (S, len(names)); row 0 is the start, so it is 0
    held: frozenset[str]  # objects the gripper holds at some time in the rollout


@dataclass(frozen=True)
class Facts:
    t: np.ndarray  # (S,) sample times for Proximity, Speed and Tilt
    contacts: Mapping[str, ContactFacts]  # tag -> contacts
    proximity: Mapping[tuple[str, str], Proximity]
    speed: Speed | None
    tilt: Mapping[str, Tilt]  # tag -> tilt
    falls: Mapping[str, tuple[Fall, ...]]  # tag -> falls
    positions: Positions | None = None
    robot: RobotHealth | None = None
    diff: Diff | None = None
