"""The tools that an LLM can call, in a neutral form. anthropic.py and openai.py turn them into each API's format.

The arguments match `TOOL_ARGS` in preflight/robots/tabletop_arm.py. The schemas only help the LLM: the
firewall checks every argument again when it compiles the call, so a call that ignores the schema is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from preflight.robots import TabletopArm
from preflight.robots.tabletop_arm import DEFAULT_SPEED_MPS, MAX_SPEED_MPS

DESCRIBE_SCENE = "describe_scene"

_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
        "description": "Target [x, y, z] in metres, in the world frame. z points up."}
_SPEED = {"type": "number", "exclusiveMinimum": 0,
          "description": f"Speed of the gripper in m/s. Default {DEFAULT_SPEED_MPS}. "
                         f"The robot limits it to {MAX_SPEED_MPS}."}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments


def tool_specs(arm: TabletopArm) -> tuple[ToolSpec, ...]:
    """describe_scene, then the robot's tools. `pick` lists the objects that the robot can grasp."""
    objects = sorted(arm.grasp_welds)
    return (
        ToolSpec(DESCRIBE_SCENE,
                 "Return the position of the gripper, of each object and of each part of the person, with the "
                 "hazard tags of each object. Call it before you plan a move.",
                 _schema({})),
        ToolSpec("move_to",
                 "Move the gripper on a straight line, so that the point between its fingertips ends at xyz. "
                 "The gripper always points down. A held object moves with it.",
                 _schema({"xyz": _XYZ, "speed": _SPEED}, required=["xyz"])),
        ToolSpec("pick",
                 "Move above the object, lower onto its top, grasp it, and lift it 10 cm.",
                 _schema({"object": {"type": "string", "enum": objects, "description": "The object to pick."},
                          "speed": _SPEED}, required=["object"])),
        ToolSpec("place",
                 "Move the held object so that its origin is at xyz, release it, and lift the gripper. "
                 "If xyz is in the air, the object falls.",
                 _schema({"xyz": _XYZ, "speed": _SPEED}, required=["xyz"])),
    )


def _schema(properties: dict[str, Any], required: list[str] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
