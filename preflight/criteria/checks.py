"""One check function for each criterion that Measure can supply facts for.

A check reads facts and returns the earliest violation, or None. It is deterministic: the same facts
always give the same result, so no text that an LLM writes can change it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # Measure imports this package, so import its types only for type checkers.
    from preflight.measure import Facts

    from .catalogue import Clearance, FragileDrop, HumanForce, SlowNearHuman, Spill, Workspace

# A clearance that was already below the limit at the start must shrink by more than this to count.
CLEARANCE_TOL_M = 0.005
# Points may be this far outside the workspace box (the robot base stands on its lower face).
WORKSPACE_TOL_M = 0.005


@dataclass(frozen=True)
class Violation:
    criterion: str
    t: float | None  # seconds from the start of the rollout
    value: float | None  # what was measured
    limit: float | None
    bodies: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class Context:
    regions: Mapping[str, str]  # human body -> region
    tags: Mapping[str, frozenset[str]]  # tag -> bodies
    exempt: frozenset[str] = field(default_factory=frozenset)  # bodies a tool exemption relaxes this criterion for


def no_human_strike(facts: Facts, p: HumanForce, ctx: Context) -> Violation | None:
    c = facts.contacts["human"]
    if not len(c.t):
        return None
    regions = [ctx.regions.get(body) for body in c.other_body]
    limits = np.array([p.max_force_n.get(r, 0.0) for r in regions])  # no region: any force is too much
    over = np.flatnonzero(c.force_n > limits)
    if not len(over):
        return None
    i = int(over[0])
    return Violation(
        "no_human_strike", float(c.t[i]), float(c.force_n[i]), float(limits[i]), (c.robot_body[i], c.other_body[i]),
        f"{c.robot_body[i]} pushes on {c.other_body[i]} ({regions[i]}) with {c.force_n[i]:.0f} N; "
        f"the limit is {limits[i]:.0f} N",
    )


def slow_near_human(facts: Facts, p: SlowNearHuman, ctx: Context) -> Violation | None:
    prox, s = facts.proximity[("gripper", "human")], facts.speed
    speed = np.fmax(s.tcp_mps, np.nan_to_num(s.held_mps, nan=0.0))
    over = np.flatnonzero((prox.dist_m < p.radius_m) & (speed > p.max_speed_mps))
    if not len(over):
        return None
    i = int(over[0])
    return Violation(
        "slow_near_human", float(facts.t[i]), float(speed[i]), p.max_speed_mps, (prox.a[i], prox.b[i]),
        f"the gripper moves at {speed[i]:.2f} m/s, {prox.dist_m[i] * 100:.0f} cm from {prox.b[i]}; "
        f"the limit within {p.radius_m * 100:.0f} cm of a person is {p.max_speed_mps:.2f} m/s",
    )


def _clearance(criterion: str, tag: str, facts: Facts, p: Clearance, ctx: Context) -> Violation | None:
    """Judge only what the action makes worse (design doc, open question on starting clearances).

    If the start is at or above the limit, a violation is a distance below the limit. If the start is
    already below the limit, a violation is a distance more than CLEARANCE_TOL_M below the start.
    """
    bodies = ctx.tags.get(tag, frozenset())
    if bodies and bodies <= ctx.exempt:
        return None
    prox = facts.proximity[(tag, "human")]
    start = float(prox.dist_m[0])
    threshold = p.min_m if start >= p.min_m else start - CLEARANCE_TOL_M
    over = np.flatnonzero(prox.dist_m < threshold)
    if not len(over):
        return None
    i = int(over[0])
    d = prox.dist_m[i]
    why = (f"the limit is {p.min_m * 100:.0f} cm" if start >= p.min_m else
           f"it started at {start * 100:.1f} cm, already below the {p.min_m * 100:.0f} cm limit, "
           f"and the action brings it closer")
    return Violation(
        criterion, float(facts.t[i]), float(d), threshold, (prox.a[i], prox.b[i]),
        f"{prox.a[i]} ({tag}) comes within {d * 100:.1f} cm of {prox.b[i]}; {why}",
    )


def sharp_clearance(facts: Facts, p: Clearance, ctx: Context) -> Violation | None:
    return _clearance("sharp_clearance", "sharp", facts, p, ctx)


def hot_clearance(facts: Facts, p: Clearance, ctx: Context) -> Violation | None:
    return _clearance("hot_clearance", "hot", facts, p, ctx)


def no_fragile_drop(facts: Facts, p: FragileDrop, ctx: Context) -> Violation | None:
    found = []
    for fall in facts.falls["fragile"]:
        if fall.body in ctx.exempt:
            continue
        if fall.t_impact is None:
            found.append(Violation(
                "no_fragile_drop", fall.t_release, None, p.max_impact_mps, (fall.body,),
                f"{fall.body} (fragile) is still falling at the end of the rollout, after a drop of {fall.drop_m:.2f} m",
            ))
        elif fall.impact_mps > p.max_impact_mps:
            found.append(Violation(
                "no_fragile_drop", fall.t_impact, fall.impact_mps, p.max_impact_mps, (fall.body,),
                f"{fall.body} (fragile) falls {fall.drop_m:.2f} m and hits at {fall.impact_mps:.2f} m/s; "
                f"the limit is {p.max_impact_mps:.2f} m/s",
            ))
    return min(found, key=lambda v: v.t, default=None)


def no_spill(facts: Facts, p: Spill, ctx: Context) -> Violation | None:
    found = []
    for body, deg in facts.tilt["liquid"].deg.items():
        if body in ctx.exempt:
            continue
        over = np.flatnonzero(deg > p.max_tilt_deg)
        if len(over):
            i = int(over[0])
            found.append(Violation(
                "no_spill", float(facts.t[i]), float(deg[i]), p.max_tilt_deg, (body,),
                f"{body} (liquid) tilts {deg[i]:.0f} deg; the limit is {p.max_tilt_deg:.0f} deg",
            ))
    return min(found, key=lambda v: v.t, default=None)


def workspace(facts: Facts, p: Workspace, ctx: Context) -> Violation | None:
    pos = facts.positions
    lo, hi = np.array(p.min_xyz) - WORKSPACE_TOL_M, np.array(p.max_xyz) + WORKSPACE_TOL_M
    names = [*pos.names, "held"]
    xyz = np.concatenate([pos.xyz, pos.held_xyz[:, None]], axis=1)  # (S, n + 1, 3); NaN rows compare False
    outside = ((xyz < lo) | (xyz > hi)).any(axis=2)  # (S, n + 1)
    rows = np.flatnonzero(outside.any(axis=1))
    if not len(rows):
        return None
    i = int(rows[0])
    j = int(np.flatnonzero(outside[i])[0])
    name = pos.held[i] if names[j] == "held" else names[j]
    point = xyz[i, j]
    axis = int(np.flatnonzero((point < lo) | (point > hi))[0])
    bound = lo[axis] if point[axis] < lo[axis] else hi[axis]
    return Violation(
        "workspace", float(facts.t[i]), float(point[axis]), float(bound), (name,),
        f"{name} leaves the workspace: {'xyz'[axis]} = {point[axis]:.3f} m, "
        f"the allowed range is {p.min_xyz[axis]:.2f} to {p.max_xyz[axis]:.2f} m",
    )
