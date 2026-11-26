"""`preflight replay`: the summary text, and a playback that matches Simulate. The viewer itself does not open."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from typer.testing import CliRunner

from preflight.audit import AuditLog, load, read_record
from preflight.audit.replay import Playback, summary
from preflight.cli import app
from preflight.core.firewall import Firewall
from preflight.robots import TabletopArm

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"
AWAY = [0.30, -0.10, 1.00]

runner = CliRunner()


@pytest.fixture(scope="module")
def log(tmp_path_factory) -> Path:
    """Records 1 to 4: pick (ALLOW), a move over the hand (BLOCK), a fast move away (MODIFY), no compile."""
    path = tmp_path_factory.mktemp("replay") / "audit.jsonl"
    fw = Firewall.from_yaml(RULES, audit=AuditLog(path, session="test"))
    fw.call("pick", {"object": "hot_cup"})
    x, y, z = map(float, fw.backend.live.geom("human_hand_left").xpos)
    fw.call("move_to", {"xyz": [x, y, z + 0.10], "speed": 1.0})
    fw.check("move_to", {"xyz": AWAY, "speed": 1.0})
    fw.call("move_to", {"xyz": [2.0, 0.0, 1.0]})
    return path


# --- Playback ------------------------------------------------------------------


def _settle_steps(model, trajectory, settle_s=1.0):
    return round((trajectory.duration + settle_s) / model.opt.timestep)


def test_playback_matches_simulate(log):
    """Step by step, the playback of `pick` (with its weld) gives the same state as `TabletopArm.play()`."""
    model, start, traj = load(read_record(log, 1))
    arm = TabletopArm(model)
    simulate, replayed = mujoco.MjData(model), mujoco.MjData(model)
    for data in (simulate, replayed):
        mujoco.mj_copyData(data, model, start)
    arm.play(simulate, traj, settle_s=1.0)
    playback = Playback(arm, traj)
    for _ in range(_settle_steps(model, traj)):
        playback.step(replayed)
    np.testing.assert_array_equal(replayed.qpos, simulate.qpos)
    assert replayed.eq_active.tolist() == simulate.eq_active.tolist()


def test_the_viewer_callback_stays_close_to_simulate(log):
    """In the viewer, the callback switches a weld one step late. The held cup still ends within 1 mm."""
    model, start, traj = load(read_record(log, 1))
    arm = TabletopArm(model)
    simulate, viewer = mujoco.MjData(model), mujoco.MjData(model)
    for data in (simulate, viewer):
        mujoco.mj_copyData(data, model, start)
    arm.play(simulate, traj, settle_s=1.0)
    playback = Playback(arm, traj)
    mujoco.set_mjcb_control(lambda m, d: playback.apply(d, d.time))
    try:
        for _ in range(_settle_steps(model, traj)):
            mujoco.mj_step(model, viewer)
    finally:
        mujoco.set_mjcb_control(None)
    gap = np.linalg.norm(viewer.body("hot_cup").xpos - simulate.body("hot_cup").xpos)
    assert gap < 1e-3


def test_a_reset_starts_the_weld_changes_again(log):
    model, data, traj = load(read_record(log, 1))
    assert traj.events, "pick has weld events"
    playback = Playback(TabletopArm(model), traj)
    playback.apply(data, traj.duration)
    assert playback._pending == []
    playback.apply(data, 0.0)
    assert len(playback._pending) == len(traj.events)


# --- Summary -------------------------------------------------------------------


def test_summary_of_a_block(log):
    text = summary(read_record(log, 2))
    assert "Record 2 of session test" in text
    assert "Verdict: BLOCK: hot_clearance at t = " in text
    # The cup started inside the 30 cm limit, so the limit of the finding is the start distance.
    assert "- hot_clearance (block) at t = 0.14 s: value 0.222, limit 0.222" in text
    assert "- no_human_strike (block) at t = 0.69 s: value " in text
    assert "Executed on the robot: no" in text
    assert "the blocked motion, which did not run" in text


def test_summary_of_a_modify(log):
    text = summary(read_record(log, 3))
    assert "Verdict: MODIFY: slowed " in text
    assert "Slowed: " in text and "- slow_near_human (modify)" in text
    assert "(check)" in text and "Executed on the robot: no" in text


def test_summary_without_a_trajectory(log):
    text = summary(read_record(log, 4))
    assert "Failure: compile" in text
    assert "Trajectory: none, because the call did not compile" in text


# --- Command -------------------------------------------------------------------


def test_replay_opens_the_viewer_at_the_starting_state(log, monkeypatch):
    seen = {}

    def launch(model, data):
        seen["qpos"] = data.qpos.copy()
        mujoco.mj_step(model, data)  # the viewer steps, so the control callback runs
        seen["ctrl"] = data.ctrl.copy()

    monkeypatch.setattr(mujoco.viewer, "launch", launch)
    result = runner.invoke(app, ["replay", str(log), "--id", "2"])
    assert result.exit_code == 0, result.output
    assert "Verdict: BLOCK" in result.output
    np.testing.assert_array_equal(seen["qpos"], read_record(log, 2)["state"]["qpos"])
    _, _, traj = load(read_record(log, 2))
    np.testing.assert_allclose(seen["ctrl"][:7], traj.q_at(0.0))
    assert mujoco.get_mjcb_control() is None  # removed after the window closes


def test_replay_without_a_trajectory_shows_the_state(log, monkeypatch):
    calls = []
    monkeypatch.setattr(mujoco.viewer, "launch", lambda model, data: calls.append(mujoco.get_mjcb_control()))
    result = runner.invoke(app, ["replay", str(log), "--id", "4"])
    assert result.exit_code == 0, result.output
    assert calls == [None]


def test_replay_of_an_unknown_id_fails(log):
    result = runner.invoke(app, ["replay", str(log), "--id", "99"])
    assert result.exit_code == 1
    assert "no record with id 99" in result.output
