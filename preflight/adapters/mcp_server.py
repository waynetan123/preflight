"""`preflight serve --mcp`: the guarded robot as MCP tools. It needs the extra `preflight-robot[mcp]`.

    preflight serve --mcp --rules scenes/kitchen_rules.yaml --audit audit.jsonl

The server speaks MCP over stdio. Its tools are the tools of `Guarded.tool_schemas()`, and each call goes to
`Guarded.handle_tool_call()`, so an MCP client gets the same checks, results and log as any other caller.

- A BLOCK is a normal tool result with `isError` set. The text tells the LLM why, so it can plan again, and the
  flag tells the client that the action did not run. The flag says nothing that the text does not say, so it
  is also safe with `minimal` feedback.
- The physics runs in a worker thread, so the protocol stays responsive during a long call. A lock lets only
  one call at a time use the single live robot. A call that the client cancels still finishes, so the robot
  never stops in the middle of a motion that the firewall did not check.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from mcp import types
from mcp.server import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

import preflight

if TYPE_CHECKING:
    from preflight.core.guarded import Guarded

INSTRUCTIONS = (
    "This server controls a robot arm through a safety firewall. Before a call runs, the firewall simulates it "
    "and checks the result against safety rules. The firewall can block a call or run it more slowly. A blocked "
    "call does not move the robot. Call describe_scene before you plan a move."
)
# The start of each tool result that means nothing ran: a BLOCK (`Decision.message()` in preflight/core/verdict.py)
# or bad arguments (`Guarded.handle_tool_call()` in preflight/core/guarded.py).
FAILED = ("blocked", "error:")


def build_server(guarded: Guarded) -> Server:
    """An MCP server for one guarded robot."""
    tools = [types.Tool(name=s["name"], description=s["description"], input_schema=s["input_schema"])
             for s in guarded.tool_schemas("anthropic")]
    lock = anyio.Lock()

    async def list_tools(ctx: ServerRequestContext, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
        async with lock:
            text = await anyio.to_thread.run_sync(guarded.handle_tool_call, params.name, params.arguments)
        return types.CallToolResult(content=[types.TextContent(text=text)], is_error=text.startswith(FAILED))

    return Server("preflight", version=preflight.__version__, instructions=INSTRUCTIONS,
                  on_list_tools=list_tools, on_call_tool=call_tool)


def serve(rules: Path, feedback: str = "detailed", audit: Path | None = None) -> None:
    """Serve the guarded robot over stdio until the client closes the connection."""
    anyio.run(_serve_stdio, rules, feedback, audit)


async def _serve_stdio(rules: Path, feedback: str, audit: Path | None) -> None:
    from preflight.audit import AuditLog
    from preflight.core.firewall import Firewall

    async with stdio_server() as (read_stream, write_stream):
        # Load inside the stdio context: there, stray output of MuJoCo goes to stderr, not into the protocol.
        guarded = Firewall.from_yaml(rules, audit=AuditLog(audit) if audit else None).wrap(feedback)
        server = build_server(guarded)
        await server.run(read_stream, write_stream, server.create_initialization_options())
