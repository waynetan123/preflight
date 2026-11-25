"""Rollouts run on a copy of the live world, record what happens, and fail closed."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from preflight.backends import MujocoBackend
from preflight.robots import TabletopArm, Trajectory, WeldEvent

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "scenes" / "kitchen.xml"
TABLE_TOP_Z = 0.75
CUP_HALF_HEIGHT = 0.05


@pytest.fixture
def sim() -> MujocoBackend:
    return MujocoBackend(SCENE)


@pytest.fixture
def arm(sim) -> TabletopArm:
    return TabletopArm(sim.model)


def _live_state(sim):
    return sim.live.time, sim.live.qpos.copy(), sim.live.qvel.copy(), sim.live.eq_active.copy()


def _assert_same(a, b):
    assert a[0] == b[0]
    for x, y in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(x, y)


# --- Fork and rollout --------------------------------------------------------


def test_fork_copies_the_live_state(sim):
    sim.live.qpos[sim.model.joint("panda/joint1").qposadr[0]] = 0.3
    fork = sim.fork()
    assert fork is not sim.live
    np.testing.assert_array_equal(fork.qpos, sim.live.qpos)


def test_rollout_does_not_change_the_live_state(sim, arm):
    before = _live_state(sim)
    rollout = sim.rollout(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    assert rollout.ok, rollout.detail
    _assert_same(_live_state(sim), before)
    assert arm.holding(sim.live) is None


def test_trace_covers_motion_and_settle_time(sim, arm):
    traj = arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0], "speed": 0.5})
    trace = sim.rollout(arm, traj, settle_s=1.0).trace
    steps = round((traj.duration + 1.0) / sim.model.opt.timestep)
    assert len(trace) == steps + 1  # the starting state, then one row per step
    assert trace.t[0] == 0.0
    assert trace.t[-1] == pytest.approx(traj.duration + 1.0, abs=sim.model.opt.timestep)
    assert trace.qpos.shape == (steps + 1, sim.model.nq)
    assert trace.trajectory is traj


def test_trace_records_contact_forces(sim, arm):
    trace = sim.rollout(arm, arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0]})).trace
    cup = sim.model.geom("hot_cup").id
    last = len(trace) - 1
    rows = [i for i in trace.contacts.at(last) if cup in trace.contacts.geom[i]]
    # The cup rests on the table: the normal forces carry its weight.
    weight = sim.model.body("hot_cup").mass[0] * -sim.model.opt.gravity[2]
    assert trace.contacts.force[rows, 0].sum() == pytest.approx(weight, rel=0.05)


def test_restore_gives_back_a_recorded_state(sim, arm):
    traj = arm.compile(sim.live, "pick", {"object": "hot_cup"})
    trace = sim.rollout(arm, traj, settle_s=0.5).trace
    end = mujoco.MjData(sim.model)  # the same motion again, on a separate copy
    mujoco.mj_copyData(end, sim.model, sim.live)
    arm.play(end, traj, settle_s=0.5)
    mujoco.mj_kinematics(sim.model, end)

    scratch = mujoco.MjData(sim.model)
    trace.restore(sim.model, scratch, len(trace) - 1)
    np.testing.assert_allclose(scratch.body("hot_cup").xpos, end.body("hot_cup").xpos, atol=1e-9)
    assert scratch.eq_active[sim.model.equality("grasp_hot_cup").id]


@pytest.mark.parametrize("where", [0.25, 0.5, 1.0])
def test_contacts_belong_to_the_state_in_their_row(sim, arm, where):
    trace = sim.rollout(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"})).trace
    step = min(int(where * len(trace)), len(trace) - 1)
    scratch = mujoco.MjData(sim.model)
    trace.restore(sim.model, scratch, step)
    mujoco.mj_forward(sim.model, scratch)

    rows = trace.contacts.at(step)
    recorded = sorted(zip(map(tuple, trace.contacts.geom[rows]), trace.contacts.dist[rows]))
    computed = sorted(zip(map(tuple, scratch.contact.geom[:scratch.ncon]), scratch.contact.dist[:scratch.ncon]))
    assert [g for g, _ in recorded] == [g for g, _ in computed]
    np.testing.assert_allclose([d for _, d in recorded], [d for _, d in computed], atol=1e-12)


@pytest.mark.parametrize("where", [0.0, 0.5, 1.0])
def test_actuator_forces_belong_to_the_state_in_their_row(sim, arm, where):
    traj = arm.compile(sim.live, "pick", {"object": "hot_cup"})
    trace = sim.rollout(arm, traj).trace
    step = min(int(where * len(trace)), len(trace) - 1)
    scratch = mujoco.MjData(sim.model)
    trace.restore(sim.model, scratch, step)
    scratch.ctrl[arm._arm_act] = traj.q_at(trace.t[step])  # the targets that play() set before this step
    scratch.ctrl[arm._gripper_act] = traj.gripper
    mujoco.mj_forward(sim.model, scratch)
    np.testing.assert_allclose(trace.actuator_force[step], scratch.actuator_force, atol=1e-9)


def test_settle_time_shows_a_drop(sim, arm):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    # Let go 0.25 m above the table: the cup falls after the motion ends.
    traj = arm.compile(sim.live, "place", {"xyz": [0.30, -0.10, 1.05]})
    rollout = sim.rollout(arm, traj, settle_s=1.0)
    assert rollout.ok, rollout.detail  # a drop is a result to judge, not a broken simulation
    trace = rollout.trace
    scratch = mujoco.MjData(sim.model)
    trace.restore(sim.model, scratch, len(trace) - 1)
    assert scratch.body("hot_cup").xpos[2] < TABLE_TOP_Z + CUP_HALF_HEIGHT + 0.02


# --- State carries across calls ----------------------------------------------


def test_execute_then_check_from_the_new_state(sim, arm):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    assert arm.holding(sim.live) == "hot_cup"
    traj = arm.compile(sim.live, "move_to", {"xyz": [0.40, 0.0, 1.0]})
    rollout = sim.rollout(arm, traj)
    assert rollout.ok
    cup = sim.model.joint("hot_cup").qposadr[0]
    assert rollout.trace.qpos[-1, cup + 2] > TABLE_TOP_Z + 0.15  # the check carried the cup along


# --- Fail closed -------------------------------------------------------------


def _trajectory(arm, sim, q=None, events=()) -> Trajectory:
    start = sim.live.qpos[arm._qadr].copy()
    q = start if q is None else q
    return Trajectory(tool="test", args={}, t=np.array([0.0, 0.2]), q=np.array([start, q]),
                      gripper=255.0, events=tuple(events))


def test_exception_fails_closed(sim, arm):
    rollout = sim.rollout(arm, _trajectory(arm, sim, events=[WeldEvent(0.1, "grasp_no_such_object", True)]))
    assert rollout.failure == "exception"
    assert "grasp_no_such_object" in rollout.detail


def test_bad_control_fails_closed(sim, arm):
    rollout = sim.rollout(arm, _trajectory(arm, sim, q=np.full(7, np.nan)))
    assert rollout.failure == "unstable"
    assert "BADCTRL" in rollout.detail


def test_timeout_fails_closed(sim, arm):
    traj = arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0]})
    rollout = sim.rollout(arm, traj, budget_s=1e-6)
    assert rollout.failure == "timeout"
    assert len(rollout.trace) < 100  # stopped early


def test_failed_rollout_has_no_actuator_forces_in_its_last_row(sim, arm):
    traj = arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0]})
    force = sim.rollout(arm, traj, budget_s=1e-6).trace.actuator_force
    assert np.isnan(force[-1]).all() and np.isfinite(force[:-1]).all()


def test_bad_starting_state_fails_closed(sim, arm):
    cup = sim.model.joint("hot_cup").qposadr[0]
    sim.live.qpos[cup + 2] = TABLE_TOP_Z  # the cup's centre on the table top: half of it inside the table
    mujoco.mj_forward(sim.model, sim.live)
    traj = _trajectory(arm, sim)
    rollout = sim.rollout(arm, traj)
    assert rollout.failure == "invalid_start"
    assert "hot_cup" in rollout.detail and "table" in rollout.detail
    assert len(rollout.trace) == 1


def test_penetration_during_rollout_fails_closed(sim, arm, monkeypatch):
    # Real overlaps this deep need broken physics. With a 0.01 mm limit, props that rest on the table
    # (up to about 0.04 mm of overlap) trigger the check.
    monkeypatch.setattr("preflight.backends.mujoco_backend.MAX_PENETRATION_M", 1e-5)
    rollout = sim.rollout(arm, arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0]}))
    assert rollout.failure == "penetration"
    assert "table" in rollout.detail


# --- Speed ---------------------------------------------------------------------


def test_pick_rollout_fits_the_budget(sim, arm):
    traj = arm.compile(sim.live, "pick", {"object": "hot_cup"})
    rollouts = [sim.rollout(arm, traj, settle_s=1.0) for _ in range(3)]
    assert all(r.ok for r in rollouts)
    # The fastest of 3 runs: a busy computer only makes a run slower, and that is not the code's cost.
    assert min(r.wall_s for r in rollouts) < 0.4
