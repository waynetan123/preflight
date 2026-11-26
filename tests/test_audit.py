"""Each decision writes one JSON line before anything runs, with enough data to run the decision again."""

import json
from pathlib import Path

import numpy as np
import pytest

from preflight.audit import AuditLog, digest, load, read_record
from preflight.core.firewall import Firewall
from preflight.robots import TabletopArm

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"


@pytest.fixture
def log(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl", session="test")


@pytest.fixture
def fw(log) -> Firewall:
    return Firewall.from_yaml(RULES, audit=log)


def _records(log):
    return [json.loads(line) for line in log.path.read_text().splitlines()]


def _over_hand(fw, height):
    x, y, z = map(float, fw.backend.live.geom("human_hand_left").xpos)
    return [x, y, z + height]


def test_one_record_per_decision(fw, log):
    fw.call("pick", {"object": "hot_cup"})
    fw.check("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    fw.call("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    records = _records(log)
    assert [(r["id"], r["mode"], r["tool"], r["verdict"], r["executes"]) for r in records] == [
        (1, "call", "pick", "allow", True),
        (2, "check", "move_to", "block", False),
        (3, "call", "move_to", "block", False),
    ]
    assert {r["session"] for r in records} == {"test"}


def test_a_record_has_the_call_the_verdict_and_the_findings(fw, log):
    fw.call("pick", {"object": "hot_cup"})
    d = fw.check("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    r = _records(log)[-1]
    assert r["args"] == {"xyz": _over_hand(fw, 0.10), "speed": 1.0}
    assert r["reason"] == d.reason and r["failure"] is None
    assert r["trajectory"]["requested_speed"] == 1.0 and r["executes"] is False  # the blocked motion, not run
    first = r["findings"][0]
    assert first["criterion"] == "hot_clearance" and first["action"] == "block"
    assert set(first) == {"criterion", "t", "value", "limit", "bodies", "message", "action"}
    assert r["unchecked"] == dict(d.unchecked) and "total" in r["timing_ms"]
    assert r["time"].endswith("+00:00")


def test_the_state_is_the_state_before_the_call(fw, log):
    before = fw.backend.live.qpos.copy()
    fw.call("pick", {"object": "hot_cup"})
    r = _records(log)[0]
    assert r["state"]["qpos"] == before.tolist()
    assert r["state_sha256"] == digest(r["state"]) and r["trajectory_sha256"] == digest(r["trajectory"])
    assert r["state"]["welds"] == {}


def test_hashes_repeat_for_the_same_state_and_call(fw, log):
    fw.check("pick", {"object": "hot_cup"})
    fw.check("pick", {"object": "hot_cup"})
    a, b = _records(log)
    assert a["state_sha256"] == b["state_sha256"] and a["trajectory_sha256"] == b["trajectory_sha256"]


def test_a_record_runs_again_to_the_same_state(fw, log):
    """Load the state and the trajectory from the record into a new sim: the result is the live result."""
    fw.call("pick", {"object": "hot_cup"})
    decision = fw.call("move_to", {"xyz": [0.30, -0.10, 1.00], "speed": 0.2})
    record = read_record(log.path, 2)
    assert record["scene"] == str(ROOT / "scenes" / "kitchen.xml")
    assert record["state"]["welds"].keys() == {"grasp_hot_cup"}

    model, data, traj = load(record)
    assert model is not fw.backend.model
    np.testing.assert_array_equal(traj.q, decision.trajectory.q)
    assert traj.events == decision.trajectory.events
    TabletopArm(model).play(data, traj)
    np.testing.assert_allclose(data.qpos, fw.backend.live.qpos, atol=1e-9)


def test_a_blocked_record_holds_the_blocked_motion(fw, log):
    """A replay of a BLOCK gives the motion that the firewall simulated, from the same state."""
    fw.call("pick", {"object": "hot_cup"})
    decision = fw.check("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    assert decision.verdict == "block" and decision.trajectory is None
    record = read_record(log.path, 2)
    assert record["trajectory_sha256"] == digest(record["trajectory"])

    model, data, traj = load(record)
    np.testing.assert_array_equal(traj.q, decision.checked.q)
    TabletopArm(model).play(data, traj, settle_s=fw.rules.settle_s)
    rollout = fw.backend.rollout(fw.arm, decision.checked, settle_s=fw.rules.settle_s)
    np.testing.assert_allclose(data.qpos, rollout.trace.qpos[-1], atol=1e-9)


def test_a_record_without_a_trajectory_loads_its_state(fw, log):
    fw.call("pick", {"object": "no_such_object"})
    model, data, traj = load(read_record(log.path, 1))
    assert traj is None
    np.testing.assert_array_equal(data.qpos, fw.backend.live.qpos)


def test_an_unknown_record_id_raises(fw, log):
    fw.call("pick", {"object": "hot_cup"})
    with pytest.raises(KeyError, match="no record with id 7"):
        read_record(log.path, 7)


def test_a_record_without_a_scene_cannot_load(fw, log):
    fw.call("pick", {"object": "hot_cup"})
    record = {**read_record(log.path, 1), "scene": None}
    with pytest.raises(ValueError, match="no scene file"):
        load(record)


def test_ids_go_on_in_an_existing_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    fw = Firewall.from_yaml(RULES, audit=AuditLog(path, session="one"))
    fw.check("pick", {"object": "hot_cup"})
    fw = Firewall.from_yaml(RULES, audit=AuditLog(path, session="two"))
    fw.check("pick", {"object": "hot_cup"})
    assert [(r["id"], r["session"]) for r in map(json.loads, path.read_text().splitlines())] == [(1, "one"), (2, "two")]


def test_every_line_is_standard_json(fw, log):
    fw.call("pick", {"object": "hot_cup"})
    fw.call("move_to", {"xyz": [0.30, -0.10, 1.00], "speed": 1.0})  # MODIFY
    for line in log.path.read_text().splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


def test_no_motion_runs_without_a_record(tmp_path):
    """If the log cannot be written, the call fails before the robot moves."""
    fw = Firewall.from_yaml(RULES, audit=AuditLog(tmp_path / "missing_dir" / "audit.jsonl"))
    before = fw.backend.live.qpos.copy()
    with pytest.raises(FileNotFoundError):
        fw.call("pick", {"object": "hot_cup"})
    assert (fw.backend.live.qpos == before).all()


def test_without_a_log_nothing_is_written(tmp_path):
    fw = Firewall.from_yaml(RULES)
    assert fw.audit is None
    fw.call("pick", {"object": "hot_cup"})
    assert list(tmp_path.iterdir()) == []
