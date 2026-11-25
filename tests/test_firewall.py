"""The firewall decides ALLOW, MODIFY or BLOCK, repairs by slowing down, runs only what passed, and fails
closed."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from preflight.core.firewall import Firewall
from preflight.core.verdict import Verdict

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"
AWAY = [0.30, -0.10, 1.00]  # across the table from the person
TABLE_TOP_Z = 0.75


@pytest.fixture
def fw() -> Firewall:
    return Firewall.from_yaml(RULES)


@pytest.fixture
def holding(fw) -> Firewall:
    assert fw.call("pick", {"object": "hot_cup"}).allowed
    return fw


def _over_hand(fw, height):
    x, y, z = map(float, fw.backend.live.geom("human_hand_left").xpos)
    return [x, y, z + height]


def _live(fw):
    return fw.backend.live.qpos.copy(), fw.backend.live.time


def _with_rules(tmp_path, change) -> Firewall:
    rules = yaml.safe_load(RULES.read_text())
    rules["scene"] = str(ROOT / "scenes" / "kitchen.xml")
    change(rules)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules))
    return Firewall.from_yaml(path)


# --- Verdicts ------------------------------------------------------------------


def test_allow(holding):
    d = holding.check("move_to", {"xyz": AWAY, "speed": 0.2})
    assert d.verdict is Verdict.ALLOW
    assert d.findings == () and d.message() == "done"
    assert d.trajectory is not None


def test_block_unsafe_move(holding):
    d = holding.check("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0})
    assert d.verdict is Verdict.BLOCK
    assert d.message().startswith("blocked: hot_clearance at t = ")
    assert d.trajectory is None


def test_modify_slows_the_motion_and_checks_it_again(holding):
    d = holding.check("move_to", {"xyz": AWAY, "speed": 1.0})
    assert d.verdict is Verdict.MODIFY
    assert [f.violation.criterion for f in d.findings] == ["slow_near_human"]
    assert 1 < d.retime_factor <= 4
    assert d.trajectory.speed < 0.25
    assert d.trajectory.requested_speed == 1.0
    assert d.message().startswith("done, with a change: slowed ")


def test_block_a_move_that_pushes_into_the_table(fw):
    # The TCP stays inside the workspace, but the fingertips below it push into the table top.
    d = fw.check("move_to", {"xyz": [0.35, 0.0, TABLE_TOP_Z], "speed": 0.2})
    assert d.verdict is Verdict.BLOCK
    assert [f.violation.criterion for f in d.findings] == ["robot_limits"]
    assert d.message().startswith("blocked: robot_limits at t = ")
    assert "stays at its force limit" in d.reason


def test_the_planner_keeps_joints_off_their_stops(fw):
    # Without the planner's joint margin, this move put joint 4 on its stop, and robot_limits blocked it.
    d = fw.check("move_to", {"xyz": [0.15, -0.2, 0.95], "speed": 0.5})
    assert d.verdict is Verdict.ALLOW, d.reason
    assert d.findings == ()


def test_ordinary_calls_do_not_trip_robot_limits(fw):
    decisions = [fw.call("pick", {"object": "hot_cup"}),          # from home
                 fw.call("move_to", {"xyz": AWAY, "speed": 1.0}),  # the gate's safe move: slowed, then run
                 fw.call("place", {"xyz": [0.30, -0.10, 0.80]})]
    assert [d.verdict for d in decisions] == [Verdict.ALLOW, Verdict.MODIFY, Verdict.ALLOW]
    assert not [f for d in decisions for f in d.findings if f.violation.criterion == "robot_limits"]


def test_block_slow_press_on_the_hand(fw):
    # Beside the hand's centre, so a finger lands on the hand. At 0.1 m/s, slow_near_human does not fire,
    # and each contact is under the strike limit: only no_human_pinch sees the clamp against the table.
    x, y, _ = map(float, fw.backend.live.geom("human_hand_left").xpos)
    assert fw.call("move_to", {"xyz": [x - 0.02, y - 0.03, 0.95], "speed": 0.2}).verdict is Verdict.ALLOW
    d = fw.check("move_to", {"xyz": [x - 0.02, y - 0.03, 0.80], "speed": 0.1})
    assert d.verdict is Verdict.BLOCK
    assert [f.violation.criterion for f in d.findings] == ["no_human_pinch"]
    assert d.message().startswith("blocked: no_human_pinch at t = ")
    assert d.trajectory is None


def test_ordinary_calls_do_not_pinch(fw):
    for tool, args in [("pick", {"object": "hot_cup"}), ("move_to", {"xyz": AWAY, "speed": 1.0}),
                       ("place", {"xyz": [0.30, -0.10, 0.80]})]:
        d = fw.call(tool, args)
        assert d.allowed, d.reason
        assert "no_human_pinch" not in {f.violation.criterion for f in d.findings}


def test_repair_that_cannot_work_blocks(tmp_path):
    # Slowing down does not move the cup away from the hand, so a `modify` clearance cannot be repaired.
    fw = _with_rules(tmp_path, lambda r: r["criteria"]["on_violation"].update(hot_clearance="modify"))
    fw.call("pick", {"object": "hot_cup"})
    # Stops 15 cm above the hand: no contact, so the only finding is the clearance (action modify).
    d = fw.check("move_to", {"xyz": _over_hand(fw, 0.25), "speed": 0.2})
    assert d.verdict is Verdict.BLOCK
    assert d.reason.startswith("hot_clearance")
    assert "does not remove the violation" in d.reason


def test_warning_allows(tmp_path):
    fw = _with_rules(tmp_path, lambda r: r["criteria"]["on_violation"].update(hot_clearance="warn"))
    fw.call("pick", {"object": "hot_cup"})
    # Toward the hand, slowly, and stopping 15 cm above it: closer than at the start, but no contact.
    d = fw.check("move_to", {"xyz": _over_hand(fw, 0.25), "speed": 0.2})
    assert d.verdict is Verdict.ALLOW
    assert d.reason.startswith("warning: hot_clearance")


# --- Side effects ----------------------------------------------------------------


def test_block_a_move_that_pushes_the_plate(holding):
    assert holding.call("move_to", {"xyz": [0.25, -0.15, 0.86]}).verdict is Verdict.ALLOW  # beside the plate
    d = holding.check("move_to", {"xyz": [0.45, -0.15, 0.86]})  # the cup sweeps across the plate
    assert d.verdict is Verdict.BLOCK
    (finding,) = [f for f in d.findings if f.violation.criterion == "no_side_effects"]
    assert finding.violation.bodies == ("plate",) and finding.violation.value > 0.10
    assert "plate ends " in d.message() and "the action does not hold it" in d.message()


def test_ordinary_calls_have_no_side_effects(fw):
    # The gate's setup pick, a move away, a place 8 mm from the plate, and a move with an empty gripper.
    calls = [("pick", {"object": "hot_cup"}), ("move_to", {"xyz": AWAY, "speed": 0.2}),
             ("place", {"xyz": [0.30, -0.10, 0.80]}), ("move_to", {"xyz": [0.45, 0.1, 1.0]})]
    for tool, args in calls:
        d = fw.call(tool, args)
        assert d.verdict is Verdict.ALLOW and d.findings == (), (tool, d.message())


# --- Fail closed ---------------------------------------------------------------


def test_compile_error_blocks(fw):
    d = fw.check("move_to", {"xyz": [2.0, 0.0, 1.0]})
    assert d.verdict is Verdict.BLOCK and d.failure == "compile"
    assert "cannot reach [2.0, 0.0, 1.0]" in d.message()


def test_motion_over_the_time_limit_blocks(fw):
    d = fw.check("move_to", {"xyz": [0.30, -0.45, 0.90], "speed": 0.05})
    assert d.verdict is Verdict.BLOCK and d.failure == "compile"
    assert "split it into shorter moves" in d.reason


def test_rollout_failure_blocks(fw, monkeypatch):
    monkeypatch.setattr("preflight.backends.mujoco_backend.MAX_PENETRATION_M", 1e-5)
    d = fw.check("move_to", {"xyz": AWAY, "speed": 0.2})
    assert d.verdict is Verdict.BLOCK and d.failure == "penetration"
    assert "could not be trusted" in d.reason


def test_minimal_feedback_hides_the_rule(holding):
    d = holding.check("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0})
    assert d.message("minimal") == "blocked by safety policy"
    assert "hot_clearance" not in d.message("minimal")


# --- Only what passed runs -------------------------------------------------------


def test_check_does_not_change_the_live_state(holding):
    before = _live(holding)
    holding.check("move_to", {"xyz": AWAY, "speed": 1.0})
    after = _live(holding)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1]


def test_call_does_not_run_a_blocked_move(holding):
    before = _live(holding)
    assert holding.call("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0}).verdict is Verdict.BLOCK
    np.testing.assert_array_equal(_live(holding)[0], before[0])


def test_call_runs_the_repaired_trajectory(holding):
    original = holding.arm.compile(holding.backend.live, "move_to", {"xyz": AWAY, "speed": 1.0})
    start = holding.backend.live.time
    d = holding.call("move_to", {"xyz": AWAY, "speed": 1.0})
    assert d.verdict is Verdict.MODIFY
    # The live sim ran the slow trajectory: its time advanced by the repaired duration, not the original.
    assert d.trajectory.duration == pytest.approx(original.duration * d.retime_factor)
    assert holding.backend.live.time - start == pytest.approx(d.trajectory.duration, abs=0.002)
    assert holding.backend.live.site("panda/tcp").xpos == pytest.approx(AWAY, abs=5e-3)
    assert holding.arm.holding(holding.backend.live) == "hot_cup"


def test_state_carries_across_calls(holding):
    holding.call("move_to", {"xyz": AWAY, "speed": 0.2})
    d = holding.check("place", {"xyz": [0.30, -0.10, 0.80]})
    assert d.verdict is Verdict.ALLOW


# --- The gate --------------------------------------------------------------------


def test_gate_script_passes():
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "gate.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GATE PASSED" in result.stdout
