"""The arm compiles tool calls into trajectories that the sim follows, and refuses calls it cannot honour."""

import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from preflight.robots import CompileError, TabletopArm, trapezoid
from preflight.robots.tabletop_arm import HOVER_M, JOINT_SPEED_LIMITS, MAX_SPEED_MPS

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "scenes" / "kitchen.xml"
TABLE_TOP_Z = 0.75
CUP_HALF_HEIGHT = 0.05


@pytest.fixture
def model() -> mujoco.MjModel:
    # Not module-scoped: play() writes grasp poses into the model's weld data.
    return mujoco.MjModel.from_xml_path(str(SCENE))


@pytest.fixture
def data(model) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


@pytest.fixture
def arm(model) -> TabletopArm:
    return TabletopArm(model)


def _tcp(data) -> np.ndarray:
    return data.site("panda/tcp").xpos.copy()


# --- Speed profile -----------------------------------------------------------


@pytest.mark.parametrize("distance", [0.02, 0.5])  # triangle, trapezoid
def test_trapezoid_limits(distance):
    t, s, v_peak = trapezoid(distance, v_max=0.5, a_max=2.0, dt=0.002)
    v = np.diff(s) / np.diff(t)
    a = np.diff(v) / np.diff(t)[1:]
    assert s[0] == 0 and s[-1] == distance
    assert np.all(np.diff(s) >= 0)
    assert v.max() <= v_peak + 1e-9 <= 0.5 + 1e-9
    assert np.abs(a).max() <= 2.0 + 1e-6
    assert v[0] < 0.01 and v[-1] < 0.01  # starts and ends at rest


def test_trapezoid_reaches_cruise_speed():
    _, _, v_peak = trapezoid(0.5, v_max=0.5, a_max=2.0, dt=0.02)
    assert v_peak == 0.5


def test_retime(arm, data):
    traj = arm.compile(data, "move_to", {"xyz": [0.45, 0.0, 1.0], "speed": 0.5})
    slow = traj.retime(2.0)
    assert slow.duration == pytest.approx(2 * traj.duration)
    assert slow.speed == pytest.approx(traj.speed / 2)
    np.testing.assert_array_equal(slow.q, traj.q)
    with pytest.raises(ValueError):
        traj.retime(0.5)


# --- move_to -----------------------------------------------------------------


def test_move_to_arrives_on_a_straight_line(arm, data):
    start, target = _tcp(data), np.array([0.45, 0.1, 1.0])
    traj = arm.compile(data, "move_to", {"xyz": target.tolist(), "speed": 0.5})
    path = []
    arm.play(data, traj, settle_s=0.5, on_step=lambda t, d: path.append(_tcp(d)))

    assert np.linalg.norm(_tcp(data) - target) < 5e-3
    line = (target - start) / np.linalg.norm(target - start)
    off_line = [np.linalg.norm((p - start) - ((p - start) @ line) * line) for p in path]
    assert max(off_line) < 0.01


def test_compile_does_not_change_data(arm, data):
    qpos = data.qpos.copy()
    arm.compile(data, "move_to", {"xyz": [0.45, 0.1, 1.0]})
    np.testing.assert_array_equal(data.qpos, qpos)


def test_speed_is_capped_and_the_request_is_kept(arm, data):
    traj = arm.compile(data, "move_to", {"xyz": [0.45, 0.2, 1.1], "speed": 99})
    assert traj.requested_speed == 99
    assert traj.speed <= MAX_SPEED_MPS
    joint_speed = np.abs(np.diff(traj.q, axis=0)) / np.diff(traj.t)[:, None]
    assert (joint_speed <= JOINT_SPEED_LIMITS * (1 + 1e-9)).all()


def test_unreachable_point(arm, data):
    with pytest.raises(CompileError, match="cannot reach"):
        arm.compile(data, "move_to", {"xyz": [2.0, 0.0, 1.0]})


@pytest.mark.parametrize("tool, args, message", [
    ("fly", {}, "unknown tool"),
    ("move_to", {}, "missing argument"),
    ("move_to", {"xyz": [0.4, 0, 1], "force": 1}, "unknown argument"),
    ("move_to", {"xyz": [0.4, 0]}, "three finite numbers"),
    ("move_to", {"xyz": [0.4, 0, float("nan")]}, "three finite numbers"),
    ("move_to", {"xyz": "up"}, "three"),
    ("move_to", {"xyz": [0.4, 0, 1], "speed": 0}, "positive number"),
    ("move_to", {"xyz": [0.4, 0, 1], "speed": float("inf")}, "positive number"),
    ("move_to", {"xyz": [0.4, 0, 1], "speed": "fast"}, "positive number"),
    ("pick", {"object": "human_hand_left"}, "unknown object"),
    ("place", {"xyz": [0.3, -0.1, 0.8]}, "holds nothing"),
])
def test_invalid_calls(arm, data, tool, args, message):
    with pytest.raises(CompileError, match=message):
        arm.compile(data, tool, args)


# --- pick and place ----------------------------------------------------------


def test_pick_lifts_the_object_without_a_jump(arm, data):
    start = data.body("hot_cup").xpos.copy()
    traj = arm.compile(data, "pick", {"object": "hot_cup"})
    (event,) = traj.events
    assert event.weld == "grasp_hot_cup" and event.active

    # Cup positions, and cup positions relative to the TCP, from the step the weld switches on.
    cup, offsets = [], []

    def record(t, d):
        if t > event.t:
            cup.append(d.body("hot_cup").xpos.copy())
            offsets.append(d.body("hot_cup").xpos - _tcp(d))

    arm.play(data, traj, settle_s=0.5, on_step=record)
    assert arm.holding(data) == "hot_cup"
    np.testing.assert_allclose(data.body("hot_cup").xpos, start + [0, 0, HOVER_M], atol=5e-3)
    # No jump when the weld switches on, and no slip in the gripper during the lift.
    assert np.linalg.norm(cup[0] - start) < 1e-3
    assert max(np.linalg.norm(o - offsets[0]) for o in offsets) < 2e-3


def test_place_puts_the_object_down(arm, data):
    arm.play(data, arm.compile(data, "pick", {"object": "hot_cup"}))
    target = np.array([0.30, -0.10, TABLE_TOP_Z + CUP_HALF_HEIGHT])
    traj = arm.compile(data, "place", {"xyz": target.tolist()})
    arm.play(data, traj, settle_s=1.0)

    assert arm.holding(data) is None
    np.testing.assert_allclose(data.body("hot_cup").xpos, target, atol=5e-3)
    assert np.linalg.norm(data.body("hot_cup").cvel) < 1e-2  # at rest on the table


def test_pick_while_holding(arm, data):
    arm.play(data, arm.compile(data, "pick", {"object": "hot_cup"}))
    with pytest.raises(CompileError, match="already holds 'hot_cup'"):
        arm.compile(data, "pick", {"object": "knife"})


# --- Speed of compile --------------------------------------------------------


def test_compile_is_fast(arm, data):
    start = time.perf_counter()
    arm.compile(data, "pick", {"object": "hot_cup"})
    assert time.perf_counter() - start < 0.25  # half the 500 ms budget for a whole check
