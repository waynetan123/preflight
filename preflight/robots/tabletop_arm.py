"""The simulated Franka Panda: compile a tool call into a Trajectory, and play a Trajectory in a MuJoCo sim.

    arm = TabletopArm(model)
    traj = arm.compile(data, "move_to", {"xyz": [0.45, 0.0, 1.0], "speed": 0.5})
    arm.play(data, traj, settle_s=1.0)

Tools:
    move_to(xyz, speed)    move the TCP (the point between the fingertips) on a straight line to xyz
    pick(object, speed)    move above the object, lower onto its top, switch on its grasp weld, lift
    place(xyz, speed)      move the held object above xyz, lower it to xyz, switch off the weld, lift

The gripper always points down. The TCP follows a trapezoidal speed profile at `speed`, capped at the
robot's maximum; the requested value stays on the Trajectory so the rules can see it. Any argument or
target the arm cannot honour raises CompileError, which the firewall turns into BLOCK.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import mink
import mujoco
import numpy as np

from .trajectory import Trajectory, WeldEvent, trapezoid

# Franka Panda datasheet values.
MAX_SPEED_MPS = 1.7  # Cartesian TCP speed
JOINT_SPEED_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s

ACCEL_MPS2 = 2.0  # placeholder controller setting
DEFAULT_SPEED_MPS = 0.25
APPROACH_SPEED_MPS = 0.1  # the last part of pick and place, and the lift after them
HOVER_M = 0.10  # height above the grasp or place point where the approach starts
CLEARANCE_M = 0.005  # gap between the TCP and the top of an object when the weld switches on
# Hold still before a weld switches on or off: the position servos lag the targets by several mm in motion.
DWELL_S = 0.3
SAMPLE_DT = 0.02  # seconds between IK samples

IK_DT = 0.01
IK_MAX_ITERS = 50
IK_POS_TOL_M = 1e-3
IK_ROT_TOL_RAD = 1e-2

TOOL_ARGS = {
    "move_to": ({"xyz"}, {"speed"}),
    "pick": ({"object"}, {"speed"}),
    "place": ({"xyz"}, {"speed"}),
}
GRASP_PREFIX = "grasp_"


class CompileError(Exception):
    pass


class TabletopArm:
    tools = tuple(TOOL_ARGS)

    def __init__(self, model: mujoco.MjModel, prefix: str = "panda/") -> None:
        self.model = model
        joints = [model.joint(f"{prefix}joint{i}") for i in range(1, 8)]
        self._qadr = np.array([j.qposadr[0] for j in joints])
        self._dofadr = np.array([j.dofadr[0] for j in joints])
        self._arm_act = np.array([model.actuator(f"{prefix}actuator{i}").id for i in range(1, 8)])
        self._gripper_act = model.actuator(f"{prefix}actuator8").id
        self._tcp = model.site(f"{prefix}tcp").id
        self._hand = model.body(f"{prefix}hand").id
        self._welds = {
            model.equality(i).name.removeprefix(GRASP_PREFIX): i
            for i in range(model.neq)
            if model.equality(i).name.startswith(GRASP_PREFIX)
        }
        # For Measure: which bodies are the robot and the gripper, and which weld holds which object.
        self.tcp_site = self._tcp
        self.bodies = frozenset(b for b in range(model.nbody) if model.body(b).name.startswith(prefix))
        self.gripper_bodies = frozenset(b for b in self.bodies if b == self._hand or model.body_parentid[b] == self._hand)
        self.grasp_welds = MappingProxyType(dict(self._welds))  # object name -> equality id

        # The gripper keeps the orientation it has at the home keyframe: pointing down.
        home = model.key("home").qpos.copy()
        data = mujoco.MjData(model)
        data.qpos[:] = home
        mujoco.mj_kinematics(model, data)
        self._down = data.site_xmat[self._tcp].reshape(3, 3).copy()

        # IK runs on the full scene model. Only the arm joints move: no task depends on the other joints.
        self._config = mink.Configuration(model, home)
        self._tcp_task = mink.FrameTask(f"{prefix}tcp", "site", position_cost=1.0, orientation_cost=1.0)
        cost = np.zeros(model.nv)
        cost[self._dofadr] = 1e-3
        self._posture = mink.PostureTask(model, cost=cost)
        self._posture.set_target(home)
        self._limits = [mink.ConfigurationLimit(model)]

    # --- Compile -------------------------------------------------------------

    def compile(self, data: mujoco.MjData, tool: str, args: Mapping[str, Any]) -> Trajectory:
        """The trajectory for one tool call, starting from the state in `data`. `data` is not changed."""
        if tool not in TOOL_ARGS:
            raise CompileError(f"unknown tool {tool!r}; tools: {', '.join(self.tools)}")
        required, optional = TOOL_ARGS[tool]
        if missing := required - set(args):
            raise CompileError(f"{tool}: missing argument(s): {', '.join(sorted(missing))}")
        if extra := set(args) - required - optional:
            raise CompileError(f"{tool}: unknown argument(s): {', '.join(sorted(extra))}")
        requested = _speed(tool, args.get("speed", DEFAULT_SPEED_MPS))
        speed = min(requested, MAX_SPEED_MPS)

        if tool == "move_to":
            legs = [(_xyz(tool, args["xyz"]), speed, None)]
        elif tool == "pick":
            legs = self._pick_legs(data, args["object"], speed)
        else:
            legs = self._place_legs(data, _xyz(tool, args["xyz"]), speed)

        t, q, events, peak = self._plan(tool, data, legs)
        trajectory = Trajectory(
            tool=tool, args=MappingProxyType(dict(args)), t=t, q=q,
            gripper=float(data.ctrl[self._gripper_act]), events=events,
            requested_speed=requested, speed=peak,
        )
        # The robot cannot exceed its joint speed limits, so a path that needs more runs slower.
        if len(t) > 1:
            joint_speed = np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]
            factor = float((joint_speed / JOINT_SPEED_LIMITS).max())
            if factor > 1.0:
                trajectory = trajectory.retime(factor)
        return trajectory

    def holding(self, data: mujoco.MjData) -> str | None:
        """The object whose grasp weld is on, if any."""
        return next((obj for obj, eq in self._welds.items() if data.eq_active[eq]), None)

    def _pick_legs(self, data, obj: Any, speed: float) -> list:
        if not isinstance(obj, str) or obj not in self._welds:
            raise CompileError(f"pick: unknown object {obj!r}; graspable: {', '.join(self._welds)}")
        if (held := self.holding(data)) is not None:
            raise CompileError(f"pick: the gripper already holds {held!r}")
        body = self.model.body(obj).id
        grasp = np.array([*data.xpos[body][:2], self._top(data, body) + CLEARANCE_M])
        hover = grasp + [0, 0, HOVER_M]
        obj_pos, obj_quat = data.xpos[body].copy(), data.xquat[body].copy()

        def weld_on(t: float) -> WeldEvent:
            # The hand pose that IK planned for the grasp point, so check and execution use the same weld.
            hand_pos, hand_quat = self._config.data.xpos[self._hand], self._config.data.xquat[self._hand]
            relpose = _relative_pose(hand_pos, hand_quat, obj_pos, obj_quat)
            return WeldEvent(t, GRASP_PREFIX + obj, active=True, relpose=relpose)

        approach = min(speed, APPROACH_SPEED_MPS)
        return [(hover, speed, None), (grasp, approach, weld_on), (hover, approach, None)]

    def _place_legs(self, data, xyz: np.ndarray, speed: float) -> list:
        held = self.holding(data)
        if held is None:
            raise CompileError("place: the gripper holds nothing")
        eq = self._welds[held]
        obj_in_hand = self.model.eq_data[eq, 3:6]
        tcp_in_hand = self.model.site_pos[self._tcp]
        target = xyz + self._down @ (tcp_in_hand - obj_in_hand)  # TCP position that puts the object at xyz
        hover = target + [0, 0, HOVER_M]

        def weld_off(t: float) -> WeldEvent:
            return WeldEvent(t, GRASP_PREFIX + held, active=False)

        approach = min(speed, APPROACH_SPEED_MPS)
        return [(hover, speed, None), (target, approach, weld_off), (hover, approach, None)]

    def _plan(self, tool: str, data, legs: list[tuple[np.ndarray, float, Callable[[float], WeldEvent] | None]]):
        """Straight TCP lines through the legs' targets, sampled every SAMPLE_DT and solved with IK."""
        self._config.update(data.qpos.copy())
        p = data.site_xpos[self._tcp].copy()
        times, qs, events = [0.0], [data.qpos[self._qadr].copy()], []
        t_end, peak = 0.0, 0.0
        for target, speed, event in legs:
            distance = float(np.linalg.norm(target - p))
            t, s, v_peak = trapezoid(distance, speed, ACCEL_MPS2, SAMPLE_DT)
            for ti, si in zip(t[1:], s[1:]):
                times.append(t_end + ti)
                point = p + (target - p) * (si / distance)
                q = self._ik(point)
                if q is None:
                    # Name the target first: the caller knows it, and an LLM must not aim for the edge point.
                    raise CompileError(
                        f"{tool}: the arm cannot reach {_round(target)} with the gripper down; "
                        f"the path leaves the reach at {_round(point)}"
                    )
                qs.append(q)
            t_end += t[-1]
            peak = max(peak, v_peak)
            p = target
            if event is not None:
                t_end += DWELL_S
                times.append(t_end)
                qs.append(qs[-1])
                events.append(event(t_end))
        return np.array(times), np.array(qs), tuple(events), peak

    def _ik(self, point: np.ndarray) -> np.ndarray | None:
        """Arm joints that put the TCP at `point` with the gripper down, starting from the last solution.

        None if no solution is within IK_POS_TOL_M and IK_ROT_TOL_RAD.
        """
        self._tcp_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(self._down), point))
        tasks = [self._tcp_task, self._posture]
        for _ in range(IK_MAX_ITERS):
            pos_err, rot_err = self._tcp_error(point)
            if pos_err < 1e-4 and rot_err < 1e-3:
                break
            velocity = mink.solve_ik(self._config, tasks, IK_DT, "daqp", damping=1e-3, limits=self._limits)
            self._config.integrate_inplace(velocity, IK_DT)
        pos_err, rot_err = self._tcp_error(point)
        if pos_err > IK_POS_TOL_M or rot_err > IK_ROT_TOL_RAD:
            return None
        return self._config.q[self._qadr].copy()

    def _tcp_error(self, point: np.ndarray) -> tuple[float, float]:
        d = self._config.data
        pos_err = float(np.linalg.norm(d.site_xpos[self._tcp] - point))
        r = self._down.T @ d.site_xmat[self._tcp].reshape(3, 3)
        rot_err = math.acos(min(1.0, max(-1.0, (np.trace(r) - 1) / 2)))
        return pos_err, rot_err

    def _top(self, data, body: int) -> float:
        """Height of the highest point of a body's geoms, from their bounding boxes."""
        top = -math.inf
        for g in np.flatnonzero(self.model.geom_bodyid == body):
            xmat = data.geom_xmat[g].reshape(3, 3)
            center, half = self.model.geom_aabb[g, :3], self.model.geom_aabb[g, 3:]
            z = data.geom_xpos[g][2] + xmat[2] @ center + np.abs(xmat[2]) @ half
            top = max(top, z)
        return top

    # --- Play ----------------------------------------------------------------

    def play(self, data: mujoco.MjData, trajectory: Trajectory, settle_s: float = 0.0,
             on_step: Callable[[float, mujoco.MjData], None] | None = None) -> None:
        """Run the trajectory in `data`, then hold the last targets for `settle_s` seconds.

        `on_step(t, data)` runs after each physics step, with t in seconds from the start.
        Simulate (on a forked copy) and Execute (on the live sim) both use this function.
        """
        dt = self.model.opt.timestep
        pending = sorted(trajectory.events, key=lambda e: e.t)
        data.ctrl[self._gripper_act] = trajectory.gripper
        for k in range(round((trajectory.duration + settle_s) / dt)):
            t = k * dt
            while pending and pending[0].t <= t + 1e-9:
                self._apply(data, pending.pop(0))
            data.ctrl[self._arm_act] = trajectory.q_at(t)
            mujoco.mj_step(self.model, data)
            if on_step is not None:
                on_step(t + dt, data)
        for event in pending:
            self._apply(data, event)

    def _apply(self, data: mujoco.MjData, event: WeldEvent) -> None:
        eq = self.model.equality(event.weld).id
        if event.active and event.relpose is not None:
            # Shared by every MjData of this model. Compile sets the same value for check and execution.
            self.model.eq_data[eq, 3:10] = event.relpose
        data.eq_active[eq] = event.active


def _xyz(tool: str, value: Any) -> np.ndarray:
    try:
        xyz = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise CompileError(f"{tool}: xyz must be three numbers, not {value!r}") from None
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise CompileError(f"{tool}: xyz must be three finite numbers, not {value!r}")
    return xyz


def _speed(tool: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise CompileError(f"{tool}: speed must be a positive number in m/s, not {value!r}")
    return float(value)


def _round(point: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in point]


def _relative_pose(p1, q1, p2, q2) -> tuple[float, ...]:
    """Pose of frame 2 in frame 1: (x y z qw qx qy qz)."""
    q1_inv = np.zeros(4)
    mujoco.mju_negQuat(q1_inv, q1)
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, np.asarray(p2) - p1, q1_inv)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, q1_inv, q2)
    return (*pos.tolist(), *(quat / np.linalg.norm(quat)).tolist())
