"""`preflight serve --mcp`: the guarded robot as MCP tools. It needs the extra `preflight-robot[mcp]`.

`Firewall.from_yaml(rules, audit=...).wrap(feedback)` gives the tool definitions and the tool-call handler.
"""

from __future__ import annotations

from pathlib import Path


def serve(rules: Path, feedback: str = "detailed", audit: Path | None = None) -> None:
    raise NotImplementedError("preflight serve --mcp is not built yet")
