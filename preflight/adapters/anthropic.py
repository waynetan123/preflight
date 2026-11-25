"""Tool definitions in the format of the Anthropic Messages API (`tools=[...]`). No SDK is needed."""

from __future__ import annotations

from typing import Any

from .tools import ToolSpec


def tool_schemas(specs: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [{"name": s.name, "description": s.description, "input_schema": s.parameters} for s in specs]
