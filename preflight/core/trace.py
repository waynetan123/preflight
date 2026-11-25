"""What a rollout recorded: the state at every physics step, every contact with its force, and actuator forces.

Measure reads a Trace to compute facts. `restore()` puts one recorded step back into an MjData, so
Measure can call MuJoCo functions (distances, velocities) on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from preflight.robots import Trajectory


@dataclass(frozen=True)
class Contacts:
    """All contacts of a rollout in one table: row i is one contact at step `step[i]`."""

    step: np.ndarray  # (M,) index into Trace.t
    geom: np.ndarray  # (M, 2) geom ids
    dist: np.ndarray  # (M,) metres; negative means penetration
    pos: np.ndarray  # (M, 3) world position
    force: np.ndarray  # (M, 3) newtons in the contact frame: normal, tangent 1, tangent 2

    def at(self, step: int) -> np.ndarray:
        """Row indices of the contacts at one step."""
        return np.flatnonzero(self.step == step)


@dataclass(frozen=True)
class Trace:
    """Row k is the state at t[k], and the contacts and actuator forces in that state."""

    trajectory: Trajectory
    t: np.ndarray  # (T,) seconds from the start; t[0] is the starting state
    qpos: np.ndarray  # (T, nq)
    qvel: np.ndarray  # (T, nv)
    eq_active: np.ndarray  # (T, neq) bool
    contacts: Contacts
    actuator_force: np.ndarray  # (T, nu) after MuJoCo clamps it to forcerange; NaN in the last row of a failed rollout

    def __len__(self) -> int:
        return len(self.t)

    def restore(self, model: mujoco.MjModel, data: mujoco.MjData, step: int) -> None:
        """Put recorded step `step` into `data` and compute positions and velocities of all bodies."""
        data.qpos[:] = self.qpos[step]
        data.qvel[:] = self.qvel[step]
        data.eq_active[:] = self.eq_active[step]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)


class TraceRecorder:
    """Collects one row per state: its time, qpos, qvel, active welds, contacts and actuator forces.

    After mj_step, MjData holds the new qpos and qvel, but its contacts and forces belong to the state
    before the step. So `step()` gives the contacts and actuator forces to the previous row, then adds a row
    for the new state. `finish()` computes the contacts and actuator forces of the last state.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._t: list[float] = []
        self._qpos: list[np.ndarray] = []
        self._qvel: list[np.ndarray] = []
        self._eq: list[np.ndarray] = []
        self._contacts: list[tuple[np.ndarray, ...]] = []  # one tuple of arrays per row that has contacts
        self._force = np.zeros(6)
        self._actuator_force: list[np.ndarray] = []

    def start(self, t: float, data: mujoco.MjData) -> None:
        self._row(t, data)

    def step(self, t: float, data: mujoco.MjData) -> None:
        self._record_contacts(data)
        self._actuator_force.append(data.actuator_force.copy())
        self._row(t, data)

    def finish(self, data: mujoco.MjData) -> None:
        mujoco.mj_forward(self._model, data)
        self._record_contacts(data)
        self._actuator_force.append(data.actuator_force.copy())

    def _row(self, t: float, data: mujoco.MjData) -> None:
        self._t.append(t)
        self._qpos.append(data.qpos.copy())
        self._qvel.append(data.qvel.copy())
        self._eq.append(data.eq_active.astype(bool))

    def _record_contacts(self, data: mujoco.MjData) -> None:
        step = len(self._t) - 1
        n = data.ncon
        if n:
            force = np.empty((n, 3))
            for i in range(n):
                mujoco.mj_contactForce(self._model, data, i, self._force)
                force[i] = self._force[:3]
            c = data.contact
            self._contacts.append((np.full(n, step), c.geom[:n].copy(), c.dist[:n].copy(), c.pos[:n].copy(), force))

    def trace(self, trajectory: Trajectory) -> Trace:
        parts = list(zip(*self._contacts)) or [[np.zeros(0, int)], [np.zeros((0, 2), int)], [np.zeros(0)],
                                                [np.zeros((0, 3))], [np.zeros((0, 3))]]
        contacts = Contacts(*(np.concatenate(p) for p in parts))
        # A failed rollout does not call finish(), so its last row has no actuator forces.
        actuator_force = np.full((len(self._t), self._model.nu), np.nan)
        if self._actuator_force:
            actuator_force[:len(self._actuator_force)] = self._actuator_force
        return Trace(
            trajectory=trajectory,
            t=np.array(self._t),
            qpos=np.array(self._qpos),
            qvel=np.array(self._qvel),
            eq_active=np.array(self._eq),
            contacts=contacts,
            actuator_force=actuator_force,
        )
