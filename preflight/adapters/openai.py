"""Tool definitions in the format of the OpenAI Chat Completions API (`tools=[...]`). No SDK is needed.

OpenAI sends the arguments of a tool call as a JSON string. `Guarded.handle_tool_call()` accepts that string.
"""

from __future__ import annotations

from typing import Any

from .tools import ToolSpec


def tool_schemas(specs: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.parameters}}
            for s in specs]
