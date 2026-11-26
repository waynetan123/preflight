"""The audit log: one JSON line for each decision.

    fw = Firewall.from_yaml("scenes/kitchen_rules.yaml", audit=AuditLog("audit.jsonl"))

Each record has the time, the session, the tool call, the verdict and its findings, the starting state and
the checked trajectory, with a SHA-256 hash of each. The checked trajectory is the last one that the firewall
simulated: on BLOCK it is the blocked motion, which did not run. The scene file, the state and the trajectory are enough
to run the decision again. v0.1 has no rollout seed: it runs one rollout with no random changes.

    model, data, trajectory = load(read_record("audit.jsonl", 42))   # the state before decision 42
    TabletopArm(model).play(data, trajectory)                       # the motion that the firewall checked

The firewall writes the record before the robot moves. If the write fails, the error stops the call, so no
motion runs without a record.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from preflight.robots import TabletopArm, Trajectory, WeldEvent

if TYPE_CHECKING:
    from preflight.core.verdict import Decision


class AuditLog:
    def __init__(self, path: str | Path, session: str | None = None) -> None:
        self.path = Path(path)
        self.session = session or uuid.uuid4().hex[:12]
        # Record ids go on from the records already in the file, so ids stay unique when sessions share a file.
        self._next_id = 1
        if self.path.exists():
            with self.path.open() as f:
                self._next_id += sum(1 for _ in f)

    @property
    def next_id(self) -> int:
        """The id of the next record."""
        return self._next_id

    def record(self, decision: Decision, model: mujoco.MjModel, data: mujoco.MjData, arm: TabletopArm,
               mode: str, executes: bool, scene: str | Path | None = None, video: str | Path | None = None) -> int:
        """Append one record. `data` must hold the state that the decision started from. Returns its id.

        `scene` is the scene file of `model`. Without it, `load()` cannot rebuild the model from the record.
        `video` is the MP4 of the checked motion, if the firewall wrote one.
        """
        state = snapshot(model, data, arm)
        trajectory = None if decision.checked is None else _trajectory(decision.checked)
        record = {
            "id": self._next_id,
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session": self.session,
            "scene": None if scene is None else str(Path(scene).resolve()),
            "mode": mode,  # check | call
            "executes": executes,  # True if the firewall runs the trajectory after this record
            "tool": decision.tool,
            "args": decision.args,
            "verdict": decision.verdict,
            "reason": decision.reason,
            "failure": decision.failure,
            "retime_factor": decision.retime_factor,
            "findings": [{**asdict(f.violation), "action": f.action} for f in decision.findings],
            "unchecked": decision.unchecked,
            "timing_ms": decision.timing_ms,
            "state_sha256": digest(state),
            "trajectory_sha256": None if trajectory is None else digest(trajectory),
            "video": None if video is None else str(Path(video).resolve()),
            "state": state,
            "trajectory": trajectory,
        }
        line = json.dumps(jsonable(record), separators=(",", ":"), allow_nan=False)
        with self.path.open("a") as f:
            f.write(line + "\n")
        self._next_id += 1
        return record["id"]


def snapshot(model: mujoco.MjModel, data: mujoco.MjData, arm: TabletopArm) -> dict[str, Any]:
    """What a replay needs to start from the same state, including the pose of each active grasp weld."""
    welds = {model.equality(eq).name: model.eq_data[eq, 3:10] for eq in arm.grasp_welds.values() if data.eq_active[eq]}
    return jsonable({"qpos": data.qpos, "qvel": data.qvel, "ctrl": data.ctrl,
                     "eq_active": data.eq_active.astype(bool), "welds": welds})


def restore(model: mujoco.MjModel, data: mujoco.MjData, state: Mapping[str, Any]) -> None:
    """Write a `snapshot()` back: the weld poses go into `model`, and the rest goes into `data`."""
    for name, relpose in state["welds"].items():
        model.eq_data[model.equality(name).id, 3:10] = relpose
    data.qpos[:], data.qvel[:], data.ctrl[:] = state["qpos"], state["qvel"], state["ctrl"]
    data.eq_active[:] = state["eq_active"]
    mujoco.mj_forward(model, data)


def read_record(path: str | Path, record_id: int) -> dict[str, Any]:
    """The record with this id from an audit log file."""
    with Path(path).open() as f:
        for line in f:
            record = json.loads(line)
            if record["id"] == record_id:
                return record
    raise KeyError(f"{path}: no record with id {record_id}")


def load(record: Mapping[str, Any]) -> tuple[mujoco.MjModel, mujoco.MjData, Trajectory | None]:
    """A new model and data at the starting state of the record, and its checked trajectory.

    The trajectory is the checked one, also on BLOCK. It is None if the call did not compile. Each call loads a new model, because the weld poses
    of the state go into the model.
    """
    if record.get("scene") is None:
        raise ValueError(f"record {record.get('id')} has no scene file, so its model cannot be loaded")
    model = mujoco.MjModel.from_xml_path(record["scene"])
    data = mujoco.MjData(model)
    restore(model, data, record["state"])
    trajectory = None if record["trajectory"] is None else trajectory_from(record["trajectory"])
    return model, data, trajectory


def trajectory_from(value: Mapping[str, Any]) -> Trajectory:
    """The trajectory of a record: the inverse of `_trajectory()`."""
    events = tuple(WeldEvent(e["t"], e["weld"], e["active"], None if e["relpose"] is None else tuple(e["relpose"]))
                   for e in value["events"])
    return Trajectory(value["tool"], value["args"], np.array(value["t"]), np.array(value["q"]), value["gripper"],
                      events, value["requested_speed"], value["speed"])


def digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of a value. Python writes each float so that it reads back exactly."""
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonable(value: Any) -> Any:
    """Plain JSON types. A NaN or an infinity becomes None, so every line is standard JSON."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, (set, frozenset)):
        return [jsonable(v) for v in sorted(value, key=str)]  # a fixed order, so the hash is stable
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def _trajectory(trajectory: Trajectory) -> dict[str, Any]:
    return jsonable({
        "tool": trajectory.tool, "args": trajectory.args, "t": trajectory.t, "q": trajectory.q,
        "gripper": trajectory.gripper, "events": trajectory.events,
        "requested_speed": trajectory.requested_speed, "speed": trajectory.speed,
    })
