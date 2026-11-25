"""Each decision writes one JSON line before anything runs, with enough data to run the decision again."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from preflight.audit import AuditLog, digest
from preflight.core.firewall import Firewall
from preflight.robots import Trajectory, WeldEvent

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
    assert r["reason"] == d.reason and r["failure"] is None and r["trajectory"] is None
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
    fw.call("move_to", {"xyz": [0.30, -0.10, 1.00], "speed": 0.2})
    r = _records(log)[1]
    assert r["state"]["welds"].keys() == {"grasp_hot_cup"}

    model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes" / "kitchen.xml"))
    data = mujoco.MjData(model)
    for name, relpose in r["state"]["welds"].items():
        model.eq_data[model.equality(name).id, 3:10] = relpose
    data.qpos[:], data.qvel[:], data.ctrl[:] = r["state"]["qpos"], r["state"]["qvel"], r["state"]["ctrl"]
    data.eq_active[:] = r["state"]["eq_active"]
    mujoco.mj_forward(model, data)
    t = r["trajectory"]
    traj = Trajectory(t["tool"], t["args"], np.array(t["t"]), np.array(t["q"]), t["gripper"],
                      tuple(WeldEvent(e["t"], e["weld"], e["active"], e["relpose"] and tuple(e["relpose"]))
                            for e in t["events"]))
    from preflight.robots import TabletopArm
    TabletopArm(model).play(data, traj)
    np.testing.assert_allclose(data.qpos, fw.backend.live.qpos, atol=1e-9)


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
