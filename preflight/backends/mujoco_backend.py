"""The MuJoCo backend: the live world, and scratch copies of it for each check.

    sim = MujocoBackend("scenes/kitchen.xml")
    arm = TabletopArm(sim.model)
    traj = arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0]})
    result = sim.rollout(arm, traj, settle_s=1.0)   # runs on a copy; sim.live does not change
    if result.ok:
        sim.execute(arm, traj)

A rollout that cannot be trusted has `ok == False` and a `failure` reason. The firewall turns every
failure into BLOCK: an exception, an unstable simulation, deep penetration, a bad starting state, or
a rollout that runs past its time budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from preflight.core.trace import Trace, TraceRecorder
from preflight.robots import TabletopArm, Trajectory

# Contacts deeper than this mean the physics is not believable (placeholder).
MAX_PENETRATION_M = 0.01
# MuJoCo resets the state when it finds a bad value, so a NaN check alone does not see instability.
INSTABILITY_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADCTRL,
    mujoco.mjtWarning.mjWARN_CONTACTFULL,
    mujoco.mjtWarning.mjWARN_CNSTRFULL,
)
BUDGET_CHECK_STEPS = 25  # how often the rollout reads the clock
# A backstop for broken physics, not a performance limit: compile() caps the motion at 6 s, which a normal
# rollout finishes in about 300 ms. The verdict must depend on the physics, not on the computer's speed.
ROLLOUT_BUDGET_S = 2.0


class RolloutFailure(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Rollout:
    trace: Trace  # partial if the rollout failed
    failure: str | None  # None, or invalid_start | exception | unstable | penetration | timeout
    detail: str | None
    wall_s: float

    @property
    def ok(self) -> bool:
        return self.failure is None


class MujocoBackend:
    def __init__(self, scene: str | Path | mujoco.MjModel, keyframe: str | None = "home") -> None:
        self.model = scene if isinstance(scene, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(scene))
        self.live = mujoco.MjData(self.model)
        if keyframe is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.live, self.model.key(keyframe).id)
        mujoco.mj_forward(self.model, self.live)
        self._scratch = mujoco.MjData(self.model)

    def fork(self) -> mujoco.MjData:
        """A copy of the live state. The same scratch MjData is used again on the next call."""
        mujoco.mj_copyData(self._scratch, self.model, self.live)
        self._scratch.warning.number[:] = 0
        return self._scratch

    def rollout(self, arm: TabletopArm, trajectory: Trajectory, settle_s: float = 1.0,
                budget_s: float = ROLLOUT_BUDGET_S) -> Rollout:
        """Run the trajectory and `settle_s` more seconds on a fork of the live state, and record it."""
        start = time.perf_counter()
        data = self.fork()
        recorder = TraceRecorder(self.model)
        recorder.start(0.0, data)
        steps = 0

        def on_step(t: float, d: mujoco.MjData) -> None:
            nonlocal steps
            steps += 1
            recorder.step(t, d)
            self._check_state(d, when=f"t={t:.3f} s")
            if steps % BUDGET_CHECK_STEPS == 0 and time.perf_counter() - start > budget_s:
                raise RolloutFailure("timeout", f"rollout passed its {budget_s * 1000:.0f} ms budget at t={t:.2f} s")

        failure = detail = None
        try:
            self._check_state(data, when="start")
        except RolloutFailure as exc:
            failure, detail = "invalid_start", exc.detail
        if failure is None:
            try:
                arm.play(data, trajectory, settle_s=settle_s, on_step=on_step)
            except RolloutFailure as exc:
                failure, detail = exc.reason, exc.detail
            except Exception as exc:  # noqa: BLE001  fail closed on anything unexpected
                failure, detail = "exception", f"{type(exc).__name__}: {exc}"
        if failure is None:
            recorder.finish(data)
        return Rollout(recorder.trace(trajectory), failure, detail, time.perf_counter() - start)

    def execute(self, arm: TabletopArm, trajectory: Trajectory) -> None:
        """Run a verified trajectory on the live state."""
        arm.play(self.live, trajectory)

    def _check_state(self, data: mujoco.MjData, when: str) -> None:
        for warning in INSTABILITY_WARNINGS:
            if data.warning[warning].number:
                raise RolloutFailure("unstable", f"{when}: MuJoCo warning {warning.name}")
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RolloutFailure("unstable", f"{when}: NaN or Inf in the state")
        if data.ncon:
            i = int(np.argmin(data.contact.dist[:data.ncon]))
            if data.contact.dist[i] < -MAX_PENETRATION_M:
                c = data.contact[i]
                names = self.model.geom(c.geom1).name, self.model.geom(c.geom2).name
                raise RolloutFailure(
                    "penetration", f"{when}: {names[0]} and {names[1]} overlap by {-c.dist * 1000:.0f} mm"
                )
