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

    from .catalogue import (
        Clearance, FragileDrop, HumanForce, HumanPinch, RobotLimits, SideEffects, SlowNearHuman, Spill, Workspace,
    )

# A clearance that was already below the limit at the start must shrink by more than this to count.
CLEARANCE_TOL_M = 0.005
# Points may be this far outside the workspace box (the robot base stands on its lower face).
WORKSPACE_TOL_M = 0.005
# MuJoCo clamps an actuator force to its force range, so a force this close to the end of the range is at the limit.
SATURATION_TOL = 0.01
# A call that starts while the robot still moves from the previous call saturates at the start, because its
# trajectory starts at rest: up to 0.27 s in chained 1.7 m/s moves. Saturation from the first row may last this
# much longer. A strain that the previous call left would have shown in that call's settle time.
START_SATURATION_S = 0.5


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


def no_human_pinch(facts: Facts, p: HumanPinch, ctx: Context) -> Violation | None:
    """A clamp: the robot pushes on a human body that is fixed to the world, so the body cannot move away.

    The force on a body at one step is the sum of all its contacts with the robot side. A violation is a
    force above the region's limit at consecutive steps for at least `min_duration_s`, or until the end of
    the rollout, because the robot then holds its last pose. Bodies that are not fixed are not checked.
    """
    c = facts.contacts["human"]
    if not len(c.t):
        return None
    clamped = c.other_fixed.astype(bool)
    last = round(float(facts.t[-1]) / c.dt)
    min_steps = p.min_duration_s / c.dt - 0.5  # half a step of tolerance for rounding
    found = []
    for body in np.unique(c.other_body[clamped]):
        rows = np.flatnonzero(clamped & (c.other_body == body))
        steps, row_step = np.unique(np.round(c.t[rows] / c.dt).astype(int), return_inverse=True)
        force = np.bincount(row_step, weights=c.force_n[rows])
        region = ctx.regions.get(body)
        limit = p.max_force_n.get(region, 0.0)  # no region: any force is too much
        over = np.flatnonzero(force > limit)
        if not len(over):
            continue
        # Runs of consecutive steps above the limit, as first and end indices into `steps`.
        cuts = np.flatnonzero(np.diff(steps[over]) > 1)
        for first, end in zip(over[np.r_[0, cuts + 1]], over[np.r_[cuts, len(over) - 1]]):
            n = steps[end] - steps[first] + 1
            at_end = steps[end] >= last
            if n < min_steps and not at_end:
                continue
            run = rows[(row_step >= first) & (row_step <= end)]  # in time order
            robot = c.robot_body[run[np.argmax(c.force_n[run])]]
            peak = float(force[first:end + 1].max())
            still = " and still at the end of the rollout" if at_end else ""
            found.append(Violation(
                "no_human_pinch", float(c.t[run[0]]), peak, float(limit), (robot, body),
                f"{robot} presses {body} ({region}) against its support with up to {peak:.0f} N for "
                f"{n * c.dt:.2f} s{still}; the limit for a clamp is {limit:.0f} N for {p.min_duration_s:.2f} s",
            ))
            break
    return min(found, key=lambda v: v.t, default=None)


def slow_near_human(facts: Facts, p: SlowNearHuman, ctx: Context) -> Violation | None:
    prox, s = facts.proximity[("gripper", "human")], facts.speed
    speed = np.fmax(s.tcp_mps, np.nan_to_num(s.held_mps, nan=0.0))
    over = np.flatnonzero((prox.dist_m < p.radius_m) & (speed > p.max_speed_mps))
    if not len(over):
        return None
    # The time is the first sample over the limit. The value is the peak near a person, so that a repair
    # can calculate how much slower the motion must be.
    i, peak = int(over[0]), int(over[np.argmax(speed[over])])
    return Violation(
        "slow_near_human", float(facts.t[i]), float(speed[peak]), p.max_speed_mps, (prox.a[peak], prox.b[peak]),
        f"the gripper moves at up to {speed[peak]:.2f} m/s within {p.radius_m * 100:.0f} cm of a person "
        f"({prox.dist_m[peak] * 100:.0f} cm from {prox.b[peak]}); the limit there is {p.max_speed_mps:.2f} m/s",
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


def robot_limits(facts: Facts, p: RobotLimits, ctx: Context) -> Violation | None:
    """Three ways to damage the robot: a joint at the end of its range, an actuator held at its force limit
    (MuJoCo clamps the force, so the signal is the time at the limit, not a force above it), and self-collision.
    """
    r = facts.robot
    found = []

    lo, hi = r.q_range[:, 0], r.q_range[:, 1]
    margin = np.fmin(r.qpos - lo, hi - r.qpos)  # (T, J)
    rows, cols = np.nonzero(margin < p.joint_margin_rad)  # in time order
    if len(rows):
        i, j = int(rows[0]), int(cols[0])
        q = r.qpos[i, j]
        end = lo[j] if q - lo[j] < hi[j] - q else hi[j]
        found.append(Violation(
            "robot_limits", float(r.t[i]), float(margin[i, j]), p.joint_margin_rad, (r.joints[j],),
            f"{r.joints[j]} reaches the end of its range: {q:.3f} rad, {margin[i, j] * 1000:.1f} mrad from its limit "
            f"of {end:.3f} rad; the margin is {p.joint_margin_rad * 1000:.0f} mrad",
        ))

    flo, fhi = r.force_range[:, 0], r.force_range[:, 1]
    saturated = (r.force <= flo * (1 - SATURATION_TOL)) | (r.force >= fhi * (1 - SATURATION_TOL))  # (T, A)
    edges = np.diff(np.pad(saturated, ((1, 1), (0, 0))).astype(np.int8), axis=0)
    # Transposed, so that the starts and ends of the runs of one actuator line up.
    (acts, starts), (_, ends) = np.nonzero(edges.T == 1), np.nonzero(edges.T == -1)
    dt = float(r.t[1] - r.t[0]) if len(r.t) > 1 else 0.0
    runs = []
    for a, s, e in zip(acts, starts, ends):
        allowed = p.max_saturated_s + (START_SATURATION_S if s == 0 else 0.0)
        if (e - s) * dt > allowed + 1e-9:
            runs.append((r.t[s], int(a), int(s), int(e), allowed))
    if runs:
        t, a, s, e, allowed = min(runs)
        limit = fhi[a] if r.force[s, a] > 0 else -flo[a]
        why = f"the limit is {allowed:.2f} s" + (" from the start of a call" if s == 0 else "")
        found.append(Violation(
            "robot_limits", float(t), (e - s) * dt, allowed, (r.actuators[a],),
            f"{r.actuators[a]} stays at its force limit of {limit:.0f} for {(e - s) * dt:.2f} s, so the robot pushes "
            f"against something or cannot follow its path; {why}",
        ))

    c = r.self_contacts
    if len(c.t):
        found.append(Violation(
            "robot_limits", float(c.t[0]), float(c.force_n[0]), None, (c.robot_body[0], c.other_body[0]),
            f"{c.robot_body[0]} touches {c.other_body[0]} with {c.force_n[0]:.0f} N: the robot collides with itself",
        ))
    return min(found, key=lambda v: v.t, default=None)


def no_side_effects(facts: Facts, p: SideEffects, ctx: Context) -> Violation | None:
    """Compare the end of the rollout with the start: each object the action does not hold must end within
    tolerance_m of where it started.

    The objects that the gripper holds at some time are the ones the tool declared it would move. The time
    of a violation is the start of the lasting change: the first sample after the last one within tolerance.
    """
    diff = facts.diff
    found = []
    for j, body in enumerate(diff.names):
        if body in diff.held or body in ctx.exempt:
            continue
        dist = diff.dist_m[:, j]
        if dist[-1] > p.tolerance_m:
            inside = np.flatnonzero(dist <= p.tolerance_m)
            i = int(inside[-1]) + 1 if len(inside) else 0
            found.append(Violation(
                "no_side_effects", float(facts.t[i]), float(dist[-1]), p.tolerance_m, (body,),
                f"{body} ends {dist[-1] * 100:.1f} cm from where it was, but the action does not hold it; "
                f"the tolerance is {p.tolerance_m * 100:.0f} cm",
            ))
    return min(found, key=lambda v: v.t, default=None)
