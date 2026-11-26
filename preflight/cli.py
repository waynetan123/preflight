"""The `preflight` command.

    preflight replay audit.jsonl --id 42                      # open decision 42 in the MuJoCo viewer
    preflight serve --mcp --rules scenes/kitchen_rules.yaml   # the guarded robot as MCP tools

Each subcommand imports its module only when it runs, so `preflight --help` works without the optional extras.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

app = typer.Typer(help="Simulate each action of an LLM-driven robot, and block the unsafe ones.",
                  no_args_is_help=True, add_completion=False)


@app.command()
def replay(
    log: Annotated[Path, typer.Argument(help="An audit log file.", exists=True, dir_okay=False)],
    record_id: Annotated[int, typer.Option("--id", help="The id of the record to open.")],
) -> None:
    """Open one decision of an audit log in the MuJoCo viewer."""
    from preflight.audit.replay import replay as run
    _run(run, log, record_id)


@app.command()
def serve(
    rules: Annotated[Path, typer.Option(help="The rules file. It names the scene.", exists=True, dir_okay=False)],
    mcp: Annotated[bool, typer.Option("--mcp", help="Serve the guarded robot as MCP tools.")] = False,
    feedback: Annotated[Literal["detailed", "minimal"],
                        typer.Option(help="What a tool result tells the LLM about a block.")] = "detailed",
    audit: Annotated[Path | None, typer.Option(help="Add one record for each decision to this file.",
                                               dir_okay=False)] = None,
) -> None:
    """Serve the guarded robot. v0.1 has one server: --mcp."""
    if not mcp:
        typer.echo("error: give --mcp. It is the only server in v0.1.", err=True)
        raise typer.Exit(2)
    from preflight.adapters.mcp_server import serve as run
    _run(run, rules, feedback, audit)


def _run(command: Callable[..., Any], *args: Any) -> None:
    try:
        command(*args)
    except NotImplementedError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
