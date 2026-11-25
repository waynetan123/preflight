"""Compute facts from a Trace. Only the facts that the active criteria need are computed.

    measure = Measure(sim.model, rules, arm)
    facts = measure.facts(rollout.trace, REGISTRY.needs(rules.params))
    facts.proximity[("hot", "human")].dist_m.min()

Contacts, falls and robot health use every physics step. Proximity, speed and tilt use every `sample_every`-th
step (default 5: every 10 ms) and the last step, because each sample restores the state and runs MuJoCo.
The diff uses the same samples, but reads the poses of free objects directly from qpos.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

import mujoco
import numpy as np

from preflight.core.trace import Trace
from preflight.criteria import Need, Rules
from preflight.robots import TabletopArm

from .facts import (
    REPORT_RADIUS_M, ContactFacts, Diff, Fall, Facts, Positions, Proximity, RobotHealth, Speed, Tilt,
)

GRIPPER = "gripper"  # group: the hand, the fingers and the held object
MIN_FALL_S = 0.05  # shorter losses of contact are bounces or contact noise, not falls
MIN_DROP_M = 0.02


class Measure:
    def __init__(self, model: mujoco.MjModel, rules: Rules, arm: TabletopArm, sample_every: int = 5) -> None:
        self.model = model
        self._arm = arm
        self._sample_every = sample_every
        self._names = np.array([model.body(b).name for b in range(model.nbody)], dtype=object)
        name_to_id = {name: b for b, name in enumerate(self._names)}
        self._tags = {tag: np.array(sorted(name_to_id[n] for n in names)) for tag, names in rules.tags.items()}
        collides = (model.geom_contype | model.geom_conaffinity) != 0
        self._geoms = [np.flatnonzero((model.geom_bodyid == b) & collides) for b in range(model.nbody)]
        self._robot = np.array(sorted(arm.bodies))
        self._fixed = model.body_weldid == 0  # no joint and no mocap body between it and the world
        self._gripper = np.array(sorted(arm.gripper_bodies))
        self._held_by_eq = {eq: name_to_id[obj] for obj, eq in arm.grasp_welds.items()}
        # Robot health reads the arm only: the limited joints outside the gripper, and the actuators that drive them.
        arm_body = np.isin(model.jnt_bodyid, sorted(arm.bodies - arm.gripper_bodies))
        self._arm_joints = np.flatnonzero(arm_body & (model.jnt_limited != 0))
        self._arm_actuators = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                             & np.isin(model.actuator_trnid[:, 0], self._arm_joints))
        # Free objects for the diff: (body, qpos address, corners of its geoms' bounding boxes in the body frame).
        self._objects = [(b, model.jnt_qposadr[model.body_jntadr[b]], _corners(model, self._geoms[b]))
                         for b in range(model.nbody) if b not in arm.bodies and _is_free(model, b)]
        self._scratch = mujoco.MjData(model)
        self._fromto = np.zeros(6)
        self._vel = np.zeros(6)

    def facts(self, trace: Trace, needs: Iterable[Need]) -> Facts:
        needs = set(needs)
        for need in needs:
            if need.kind not in ("contact", "proximity", "speed", "tilt", "fall", "position", "robot", "diff"):
                raise ValueError(f"unknown fact kind {need.kind!r}")
        held = self._held(trace)
        rows = np.unique(np.r_[np.arange(0, len(trace), self._sample_every), len(trace) - 1])

        contacts = {n.tags[0]: self._contacts(trace, held, n.tags[0]) for n in needs if n.kind == "contact"}
        falls = {n.tags[0]: self._falls(trace, held, n.tags[0]) for n in needs if n.kind == "fall"}
        diff = self._diff(trace, held, rows) if any(n.kind == "diff" for n in needs) else None
        pairs = sorted({n.tags for n in needs if n.kind == "proximity"})
        tilt_tags = sorted({n.tags[0] for n in needs if n.kind == "tilt"})
        want_speed = any(n.kind == "speed" for n in needs)
        want_position = any(n.kind == "position" for n in needs)

        dist = {p: np.full(len(rows), REPORT_RADIUS_M) for p in pairs}
        near = {p: (np.full(len(rows), None, dtype=object), np.full(len(rows), None, dtype=object)) for p in pairs}
        tcp_mps, held_mps = np.zeros(len(rows)), np.full(len(rows), np.nan)
        tilt = {tag: {self._names[b]: np.zeros(len(rows)) for b in self._tags.get(tag, ())} for tag in tilt_tags}
        points = np.zeros((len(rows), 1 + len(self._robot), 3))
        held_xyz = np.full((len(rows), 3), np.nan)

        if pairs or want_speed or tilt_tags or want_position:
            data, model = self._scratch, self.model
            for i, k in enumerate(rows):
                trace.restore(model, data, k)
                for p in pairs:
                    d, a, b = self._closest(data, self._group(p[0], held[k]), self._group(p[1], held[k]))
                    dist[p][i] = d
                    near[p][0][i], near[p][1][i] = a, b
                if want_speed:
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self._arm.tcp_site, self._vel, 0)
                    tcp_mps[i] = np.linalg.norm(self._vel[3:])
                    if held[k] >= 0:
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, held[k], self._vel, 0)
                        held_mps[i] = np.linalg.norm(self._vel[3:])
                for tag in tilt_tags:
                    for b in self._tags.get(tag, ()):
                        tilt[tag][self._names[b]][i] = np.degrees(np.arccos(np.clip(data.xmat[b][8], -1, 1)))
                if want_position:
                    points[i, 0] = data.site_xpos[self._arm.tcp_site]
                    points[i, 1:] = data.xpos[self._robot]
                    if held[k] >= 0:
                        held_xyz[i] = data.xpos[held[k]]

        held_names = np.array([self._names[h] if h >= 0 else None for h in held[rows]], dtype=object)
        positions = None
        if want_position:
            positions = Positions(("tcp", *self._names[self._robot]), points, held_xyz, held_names)
        return Facts(
            t=trace.t[rows],
            contacts=MappingProxyType(contacts),
            proximity=MappingProxyType({p: Proximity(p, dist[p], *near[p]) for p in pairs}),
            speed=Speed(tcp_mps, held_mps, held_names) if want_speed else None,
            tilt=MappingProxyType({tag: Tilt(tag, MappingProxyType(v)) for tag, v in tilt.items()}),
            falls=MappingProxyType(falls),
            positions=positions,
            robot=self._robot_health(trace) if any(n.kind == "robot" for n in needs) else None,
            diff=diff,
        )

    # --- Helpers -------------------------------------------------------------

    def _held(self, trace: Trace) -> np.ndarray:
        """Held body id at each row, or -1."""
        held = np.full(len(trace), -1)
        for eq, body in self._held_by_eq.items():
            held[trace.eq_active[:, eq]] = body
        return held

    def _group(self, group: str, held: int) -> np.ndarray:
        if group == GRIPPER:
            return self._gripper if held < 0 else np.r_[self._gripper, held]
        return self._tags.get(group, np.zeros(0, dtype=int))

    def _closest(self, data: mujoco.MjData, bodies_a: np.ndarray, bodies_b: np.ndarray):
        """Smallest distance between two body sets, and the two nearest bodies."""
        bodies_b = np.setdiff1d(bodies_b, bodies_a)
        ga = np.concatenate([self._geoms[b] for b in bodies_a]) if len(bodies_a) else np.zeros(0, int)
        gb = np.concatenate([self._geoms[b] for b in bodies_b]) if len(bodies_b) else np.zeros(0, int)
        if not len(ga) or not len(gb):
            return REPORT_RADIUS_M, None, None
        # Bounding spheres give a lower bound; test the geom pairs in that order and stop early.
        rb = self.model.geom_rbound
        lower = (np.linalg.norm(data.geom_xpos[ga][:, None] - data.geom_xpos[gb][None], axis=2)
                 - rb[ga][:, None] - rb[gb][None])
        best, pair = REPORT_RADIUS_M, None
        for flat in np.argsort(lower, axis=None):
            i, j = divmod(int(flat), len(gb))
            if lower[i, j] >= best:
                break
            d = mujoco.mj_geomDistance(self.model, data, ga[i], gb[j], best, self._fromto)
            if d < best:
                best, pair = d, (ga[i], gb[j])
        if pair is None:
            return REPORT_RADIUS_M, None, None
        bodyid = self.model.geom_bodyid
        return best, self._names[bodyid[pair[0]]], self._names[bodyid[pair[1]]]

    def _contacts(self, trace: Trace, held: np.ndarray, tag: str) -> ContactFacts:
        c = trace.contacts
        bodies = self.model.geom_bodyid[c.geom]  # (M, 2)
        held_at = held[c.step]
        robot_side = np.isin(bodies, self._robot) | (bodies == held_at[:, None])
        tag_side = np.isin(bodies, self._tags.get(tag, np.zeros(0, int)))
        forward, backward = robot_side[:, 0] & tag_side[:, 1], robot_side[:, 1] & tag_side[:, 0]
        rows = np.flatnonzero(forward | backward)
        flip = backward[rows]
        robot = np.where(flip, bodies[rows, 1], bodies[rows, 0])
        other = np.where(flip, bodies[rows, 0], bodies[rows, 1])
        return ContactFacts(tag, trace.t[c.step[rows]], self._names[robot], self._names[other], c.force[rows, 0],
                            self._fixed[other], self.model.opt.timestep)

    def _robot_health(self, trace: Trace) -> RobotHealth:
        model, joints, actuators = self.model, self._arm_joints, self._arm_actuators
        c = trace.contacts
        bodies = model.geom_bodyid[c.geom]  # (M, 2)
        # The fingers are siblings, so MuJoCo does not filter their contacts; they touch when the gripper closes.
        own = np.isin(bodies, self._robot).all(axis=1) & ~np.isin(bodies, self._gripper).all(axis=1)
        rows = np.flatnonzero(own)
        limited = model.actuator_forcelimited[actuators][:, None] != 0
        return RobotHealth(
            t=trace.t,
            joints=tuple(model.joint(j).name for j in joints),
            qpos=trace.qpos[:, model.jnt_qposadr[joints]],
            q_range=model.jnt_range[joints].copy(),
            actuators=tuple(model.actuator(a).name for a in actuators),
            force=trace.actuator_force[:, actuators],
            force_range=np.where(limited, model.actuator_forcerange[actuators], [-np.inf, np.inf]),
            self_contacts=ContactFacts("robot", trace.t[c.step[rows]], self._names[bodies[rows, 0]],
                                       self._names[bodies[rows, 1]], c.force[rows, 0]),
        )

    def _falls(self, trace: Trace, held: np.ndarray, tag: str) -> tuple[Fall, ...]:
        falls = []
        bodyid = self.model.geom_bodyid
        in_contact = bodyid[trace.contacts.geom]
        dt = trace.t[1] - trace.t[0] if len(trace) > 1 else 0.0
        for b in self._tags.get(tag, ()):
            joint = self.model.body_jntadr[b]
            if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                continue  # fixed to the world or to a parent: cannot fall on its own
            qadr, dadr = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
            z = trace.qpos[:, qadr + 2]
            speed = np.linalg.norm(trace.qvel[:, dadr:dadr + 3], axis=1)
            supported = held == b
            supported[trace.contacts.step[(in_contact == b).any(axis=1)]] = True

            k = 0
            while k < len(trace):
                if supported[k]:
                    k += 1
                    continue
                start = k
                while k < len(trace) and not supported[k]:
                    k += 1
                release = max(start - 1, 0)
                landed = k < len(trace)
                end = k if landed else len(trace) - 1
                drop = float(z[release] - z[end])
                if (end - release) * dt >= MIN_FALL_S and drop >= MIN_DROP_M:
                    falls.append(Fall(
                        body=self._names[b],
                        t_release=float(trace.t[release]),
                        t_impact=float(trace.t[end]) if landed else None,
                        drop_m=drop,
                        impact_mps=float(speed[end - 1]) if landed else None,
                    ))
        return tuple(falls)

    def _diff(self, trace: Trace, held: np.ndarray, rows: np.ndarray) -> Diff:
        qpos = trace.qpos[rows]
        dist = np.zeros((len(rows), len(self._objects)))
        for j, (_, qadr, corners) in enumerate(self._objects):
            points = _rotate(qpos[:, qadr + 3:qadr + 7], corners) + qpos[:, None, qadr:qadr + 3]  # (S, P, 3)
            dist[:, j] = np.linalg.norm(points - points[0], axis=2).max(axis=1)
        names = tuple(self._names[b] for b, _, _ in self._objects)
        return Diff(names, dist, frozenset(self._names[b] for b in np.unique(held[held >= 0])))


def _is_free(model: mujoco.MjModel, body: int) -> bool:
    joint = model.body_jntadr[body]
    return joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE


def _corners(model: mujoco.MjModel, geoms: np.ndarray) -> np.ndarray:
    """The 8 corners of each geom's bounding box, in the body frame. The body origin if it has no geoms."""
    if not len(geoms):
        return np.zeros((1, 3))
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners, rot = [], np.zeros(9)
    for g in geoms:
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        box = model.geom_aabb[g, :3] + signs * model.geom_aabb[g, 3:]  # in the geom frame
        corners.append(model.geom_pos[g] + box @ rot.reshape(3, 3).T)
    return np.concatenate(corners)


def _rotate(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate body-frame points by each quaternion (w x y z): (S, 4) and (P, 3) give (S, P, 3)."""
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, u = quat[:, None, :1], quat[:, None, 1:]
    t = 2 * np.cross(u, points[None])
    return points[None] + w * t + np.cross(u, t)
