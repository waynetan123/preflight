"""`preflight replay`: open one decision of an audit log in the MuJoCo viewer.

    preflight replay audit.jsonl --id 42

The command prints the decision, then opens the viewer at the starting state of the record and plays the
checked trajectory in real time. On BLOCK this is the blocked motion, which did not run on the robot. After
the end of the trajectory, the arm holds its last targets until you close the window. Thus the window also
shows the settle time that the firewall simulated (`horizon.settle_s` in the rules file), and more.

The viewer runs its own physics loop. A control callback (`mujoco.set_mjcb_control`) sets the targets of the
trajectory at each step. With this callback, plain `python` can animate the viewer on macOS: `launch_passive`
needs `mjpython`, and `mjpython` fails with MuJoCo 4.0.1 (see README.md).

The callback runs inside `mj_step`, after MuJoCo makes the constraints of the step. Thus in the viewer, a weld
switches on or off one physics step (2 ms) later than in Simulate. `Playback.step()` applies the weld first,
as `TabletopArm.play()` does, and gives the same result as Simulate.

The reset key of the viewer (Backspace) goes to the default pose of the model, not to the state of the
record. To see the decision again, close the window and run the command again.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer

from preflight.robots import TabletopArm, Trajectory, WeldEvent

from .log import load, read_record


class Playback:
    """Sets the targets of a trajectory at a time t, as `TabletopArm.play()` does at each physics step.

    After the end of the trajectory, the last targets hold. If t goes back (a reset of the viewer), the weld
    changes start again from the beginning. The viewer gives t from `data.time`. `step()` counts its steps, as
    `play()` does, because a sum of time steps differs from k * dt in the last bits.
    """

    def __init__(self, arm: TabletopArm, trajectory: Trajectory) -> None:
        self.arm = arm
        self.trajectory = trajectory
        self._last_t = 0.0
        self._pending = self._events()
        self._steps = 0

    def apply(self, data: mujoco.MjData, t: float) -> None:
        if t < self._last_t:
            self._pending = self._events()
        self._last_t = t
        while self._pending and self._pending[0].t <= t + 1e-9:
            self.arm._apply(data, self._pending.pop(0))
        data.ctrl[self.arm._gripper_act] = self.trajectory.gripper
        data.ctrl[self.arm._arm_act] = self.trajectory.q_at(t)

    def step(self, data: mujoco.MjData) -> None:
        """Apply the targets of the next step, then do one physics step."""
        self.apply(data, self._steps * self.arm.model.opt.timestep)
        mujoco.mj_step(self.arm.model, data)
        self._steps += 1

    def _events(self) -> list[WeldEvent]:
        return sorted(self.trajectory.events, key=lambda e: e.t)


def summary(record: Mapping[str, Any]) -> str:
    """The decision of a record as text for the terminal."""
    lines = [
        f"Record {record['id']} of session {record['session']}, {record['time']} ({record['mode']})",
        f"Scene: {record.get('scene')}",
        f"Call: {record['tool']} {json.dumps(record['args'])}",
        f"Verdict: {str(record['verdict']).upper()}: {record['reason']}",
    ]
    if record.get("failure"):
        lines.append(f"Failure: {record['failure']}")
    if record.get("retime_factor", 1.0) != 1.0:
        lines.append(f"Slowed: {record['retime_factor']:.2f}x")
    if record["findings"]:
        lines.append("Findings:")
        lines += [f"- {_finding(f)}" for f in record["findings"]]
    else:
        lines.append("Findings: none")
    if record.get("unchecked"):
        lines.append("Not checked: " + ", ".join(sorted(record["unchecked"])))
    lines.append(f"Executed on the robot: {'yes' if record['executes'] else 'no'}")
    trajectory = record.get("trajectory")
    if trajectory is None:
        lines.append("Trajectory: none, because the call did not compile. The viewer shows the starting state.")
    else:
        what = "the blocked motion, which did not run" if record["verdict"] == "block" else "the checked motion"
        lines.append(f"Trajectory: {trajectory['t'][-1]:.2f} s, {what}. Then the arm holds its last targets.")
    return "\n".join(lines)


def replay(log: Path, record_id: int) -> None:
    """Print the decision, then play it in the MuJoCo viewer until the window closes."""
    record = read_record(log, record_id)
    print(summary(record), flush=True)
    model, data, trajectory = load(record)
    data.time = 0.0
    if trajectory is None:
        mujoco.viewer.launch(model, data)
        return
    playback = Playback(TabletopArm(model), trajectory)
    mujoco.set_mjcb_control(lambda m, d: playback.apply(d, d.time))
    try:
        mujoco.viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)


def _finding(f: Mapping[str, Any]) -> str:
    when = "" if f.get("t") is None else f" at t = {f['t']:.2f} s"
    keys = [k for k in ("value", "limit") if f.get(k) is not None]
    digits = 3
    if len(keys) == 2 and f["value"] != f["limit"]:
        # A finding is the first sample over the limit, so the value is often near the limit. Add digits until
        # the two numbers differ, so that the line shows the violation.
        while f"{f['value']:.{digits}g}" == f"{f['limit']:.{digits}g}":
            digits += 1
    numbers = ", ".join(f"{k} {f[k]:.{digits}g}" for k in keys)
    return f"{f['criterion']} ({f['action']}){when}" + (f": {numbers}" if numbers else "")
