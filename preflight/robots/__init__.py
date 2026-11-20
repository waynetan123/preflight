"""Robots: compile tool calls into trajectories, and play trajectories in a simulation."""

from .tabletop_arm import CompileError, TabletopArm
from .trajectory import Trajectory, WeldEvent, trapezoid

__all__ = ["CompileError", "TabletopArm", "Trajectory", "WeldEvent", "trapezoid"]
