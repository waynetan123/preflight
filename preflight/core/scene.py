"""Describe the scene: what the LLM knows about the world, as text.

    world = SimWorld(fw.backend, fw.rules, fw.arm)
    print(describe_scene(world.state(), fw.rules))

The firewall reads the world through a `WorldStateProvider`. In v0.1 the provider reads the live MuJoCo state,
which stands in for robot sensors that know the world as precisely as the simulation does. A provider for real
sensors can replace it without a change to the firewall.

The text shows the hazard tags from the rules file. The tags are trusted configuration, and with them the LLM
can plan moves that the firewall allows. Body names come from the scene file, which is trusted too. Nothing in
the text comes from the user or from the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from preflight.backends import MujocoBackend
from preflight.criteria import Rules
from preflight.robots import TabletopArm

HUMAN = "human"


@dataclass(frozen=True)
class SceneObject:
    name: str
    xyz: tuple[float, float, float]  # metres, world frame: the body origin, which `place` puts at its target
    tags: tuple[str, ...]  # sorted
    movable: bool  # True if it has a free joint, so the robot can pick it or push it
    region: str | None = None  # body region of a human body


@dataclass(frozen=True)
class WorldState:
    tcp_xyz: tuple[float, float, float]  # the point between the fingertips
    held: str | None  # the object in the gripper
    objects: tuple[SceneObject, ...]  # objects first, then human bodies; each group sorted by name


class WorldStateProvider(Protocol):
    def state(self) -> WorldState: ...


class SimWorld:
    """Reads the live state of a MuJoCo backend. Lists every body that has a tag or a free joint."""

    def __init__(self, backend: MujocoBackend, rules: Rules, arm: TabletopArm) -> None:
        self._backend, self._rules, self._arm = backend, rules, arm
        model = backend.model
        tags_of: dict[str, list[str]] = {}
        for tag, names in rules.tags.items():
            for name in names:
                tags_of.setdefault(name, []).append(tag)
        listed = []
        for b in range(1, model.nbody):  # body 0 is the world
            name = model.body(b).name
            joint = model.body_jntadr[b]
            movable = joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            if b in arm.bodies or not (movable or name in tags_of):
                continue
            listed.append((b, name, tuple(sorted(tags_of.get(name, ()))), movable))
        listed.sort(key=lambda item: (HUMAN in item[2], item[1]))
        self._listed = listed

    def state(self) -> WorldState:
        data = self._backend.live
        objects = tuple(
            SceneObject(name, _round(data.xpos[b]), tags, movable, self._rules.regions.get(name))
            for b, name, tags, movable in self._listed
        )
        return WorldState(_round(data.site_xpos[self._arm.tcp_site]), self._arm.holding(data), objects)


def describe_scene(state: WorldState, rules: Rules | None = None) -> str:
    """The scene as text for the LLM. With `rules`, the text also gives the workspace limits."""
    lines = ["Scene. Positions are in metres in the world frame, and z points up."]
    lines.append(f"Gripper (TCP) at {_xyz(state.tcp_xyz)}, holding {state.held or 'nothing'}.")
    workspace = rules.params.get("workspace") if rules is not None else None
    if workspace is not None:
        lo, hi = workspace.min_xyz, workspace.max_xyz
        lines.append("Workspace: " + ", ".join(f"{a} {lo[i]:.2f} to {hi[i]:.2f}" for i, a in enumerate("xyz")) + ".")
    things = [o for o in state.objects if HUMAN not in o.tags]
    people = [o for o in state.objects if HUMAN in o.tags]
    if things:
        lines.append("Objects:")
        lines += [f"- {o.name} at {_xyz(o.xyz)}{_labels(o)}" for o in things]
    if people:
        lines.append("Person (do not touch):")
        lines += [f"- {o.name} at {_xyz(o.xyz)}{' (' + o.region + ')' if o.region else ''}" for o in people]
    return "\n".join(lines)


def _labels(o: SceneObject) -> str:
    labels = [*o.tags, *(() if o.movable else ("fixed",))]
    return f" [{', '.join(labels)}]" if labels else ""


def _round(xyz: np.ndarray) -> tuple[float, float, float]:
    return tuple(round(float(v), 3) for v in xyz)


def _xyz(xyz: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in xyz) + ")"
