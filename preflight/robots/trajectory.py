"""A compiled motion: time-stamped arm joint targets and grasp-weld events.

The firewall checks a Trajectory and the executor runs the same object, so what was checked is what runs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WeldEvent:
    t: float  # seconds from the start of the trajectory
    weld: str  # equality name, e.g. "grasp_hot_cup"
    active: bool
    # Pose of the object in the hand frame (x y z qw qx qy qz), written to the weld before it switches on.
    relpose: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Trajectory:
    tool: str
    args: Mapping[str, Any]
    t: np.ndarray  # (N,) seconds, starting at 0
    q: np.ndarray  # (N, 7) arm joint targets
    gripper: float  # gripper actuator control, constant for the whole motion
    events: tuple[WeldEvent, ...] = ()
    requested_speed: float | None = None  # m/s, as the caller asked
    speed: float | None = None  # m/s, peak TCP speed as planned, after the robot's limits

    def __post_init__(self) -> None:
        for array in (self.t, self.q):
            array.setflags(write=False)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def q_at(self, t: float) -> np.ndarray:
        """Joint targets at time t, linear between samples and held after the end."""
        i = int(np.clip(np.searchsorted(self.t, t, side="right"), 1, len(self.t) - 1))
        t0, t1 = self.t[i - 1], self.t[i]
        a = 0.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        return self.q[i - 1] + a * (self.q[i] - self.q[i - 1])

    def retime(self, factor: float) -> Trajectory:
        """The same path, `factor` times slower. MODIFY repair and the joint speed limits use this."""
        if not factor >= 1.0:
            raise ValueError(f"retime factor must be at least 1, not {factor}")
        return replace(
            self,
            t=self.t * factor,
            q=self.q.copy(),
            events=tuple(replace(e, t=e.t * factor) for e in self.events),
            speed=None if self.speed is None else self.speed / factor,
        )


def trapezoid(distance: float, v_max: float, a_max: float, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Constant-acceleration speed profile along a path: (times, distances along the path, peak speed).

    Accelerates at a_max, cruises at v_max, decelerates at a_max. A short path never reaches v_max and
    has a triangular profile instead. The last sample is exactly at the end of the path.
    """
    if distance <= 0:
        return np.zeros(1), np.zeros(1), 0.0
    t_acc = v_max / a_max
    if a_max * t_acc**2 >= distance:  # triangle
        t_acc = math.sqrt(distance / a_max)
        v_peak, t_cruise = a_max * t_acc, 0.0
    else:
        v_peak, t_cruise = v_max, (distance - a_max * t_acc**2) / v_max
    total = 2 * t_acc + t_cruise

    t = np.linspace(0.0, total, max(1, math.ceil(total / dt)) + 1)
    d_acc = 0.5 * a_max * t_acc**2
    s = np.where(
        t < t_acc,
        0.5 * a_max * t**2,
        np.where(
            t < t_acc + t_cruise,
            d_acc + v_peak * (t - t_acc),
            distance - 0.5 * a_max * (total - t) ** 2,
        ),
    )
    s[-1] = distance
    return t, s, v_peak
