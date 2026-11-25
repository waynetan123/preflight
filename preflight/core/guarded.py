"""The guarded robot: the one entry point for the LLM's tool calls.

    guarded = fw.wrap()
    tools = guarded.tool_schemas("anthropic")           # or "openai"
    text = guarded.handle_tool_call(name, args)          # the tool result to send back to the LLM

`describe_scene` reads the world and changes nothing. Every other call goes through `Firewall.call()`, so only
a trajectory that passed the check runs. An unknown tool reaches the firewall too: compile refuses it, and the
firewall blocks and logs it. The agent loop stays with the caller.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from preflight.adapters import DESCRIBE_SCENE, anthropic, openai, tool_specs

from .scene import SimWorld, describe_scene
from .verdict import Decision

if TYPE_CHECKING:
    from .firewall import Firewall

FORMATS = {"anthropic": anthropic.tool_schemas, "openai": openai.tool_schemas}
FEEDBACK = ("detailed", "minimal")


class Guarded:
    def __init__(self, firewall: Firewall, feedback: str = "detailed") -> None:
        if feedback not in FEEDBACK:
            raise ValueError(f"feedback must be one of {', '.join(FEEDBACK)}, not {feedback!r}")
        self.firewall = firewall
        self.feedback = feedback
        self.world = SimWorld(firewall.backend, firewall.rules, firewall.arm)
        self.specs = tool_specs(firewall.arm)

    def tool_schemas(self, format: str = "anthropic") -> list[dict[str, Any]]:
        if format not in FORMATS:
            raise ValueError(f"format must be one of {', '.join(FORMATS)}, not {format!r}")
        return FORMATS[format](self.specs)

    def describe_scene(self) -> str:
        return describe_scene(self.world.state(), self.firewall.rules)

    def call(self, tool: str, **args: Any) -> Decision:
        """Check the call and run it if the verdict allows it."""
        return self.firewall.call(tool, args)

    def handle_tool_call(self, name: str, args: str | Mapping[str, Any] | None) -> str:
        """The tool result text for one tool call. `args` is a mapping, or a JSON string as OpenAI sends it."""
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                return f"error: the arguments are not valid JSON ({exc.msg}); nothing ran"
        args = {} if args is None else args
        if not isinstance(args, Mapping):
            return f"error: the arguments must be a JSON object, not {type(args).__name__}; nothing ran"
        if name == DESCRIBE_SCENE:
            if args:
                return f"error: {DESCRIBE_SCENE} takes no arguments; nothing ran"
            return self.describe_scene()
        return self.firewall.call(name, args).message(self.feedback)
