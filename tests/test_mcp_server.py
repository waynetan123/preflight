"""`preflight serve --mcp`: an MCP client gets the guarded robot's tools, and every call goes through the firewall."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from preflight.adapters.mcp_server import build_server  # noqa: E402
from preflight.core.firewall import Firewall  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"


@pytest.fixture
def fw() -> Firewall:
    return Firewall.from_yaml(RULES)


def _over_hand(fw, height):
    x, y, z = map(float, fw.backend.live.geom("human_hand_left").xpos)
    return [x, y, z + height]


def _session(server, *calls):
    """Connect a client to `server` in-process, run the calls in sequence, and return the results."""
    async def run():
        async with Client(server) as client:
            return [await client.call_tool(name, args) for name, args in calls]
    return anyio.run(run)


def _text(result):
    return result.content[0].text


def test_the_tools_are_the_guarded_robots_tools(fw):
    guarded = fw.wrap()

    async def run():
        async with Client(build_server(guarded)) as client:
            return (await client.list_tools()).tools
    tools = anyio.run(run)
    assert [t.name for t in tools] == ["describe_scene", "move_to", "pick", "place"]
    assert [t.input_schema for t in tools] == [s["input_schema"] for s in guarded.tool_schemas("anthropic")]


def test_describe_scene(fw):
    guarded = fw.wrap()
    [result] = _session(build_server(guarded), ("describe_scene", {}))
    assert not result.is_error
    assert _text(result) == guarded.describe_scene()


def test_a_safe_call_runs(fw):
    before = fw.backend.live.qpos.copy()
    [result] = _session(build_server(fw.wrap()), ("pick", {"object": "hot_cup"}))
    assert not result.is_error and _text(result) == "done"
    assert fw.arm.holding(fw.backend.live) == "hot_cup"
    assert (fw.backend.live.qpos != before).any()


def test_an_unsafe_call_is_blocked_with_the_reason(fw):
    unsafe = ("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    _, result = _session(build_server(fw.wrap()), ("pick", {"object": "hot_cup"}), unsafe)
    assert result.is_error
    assert _text(result).startswith("blocked: hot_clearance at t = ")


def test_minimal_feedback_hides_the_rule(fw):
    unsafe = ("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    _, result = _session(build_server(fw.wrap("minimal")), ("pick", {"object": "hot_cup"}), unsafe)
    assert result.is_error and _text(result) == "blocked by safety policy"


def test_bad_arguments_are_an_error_and_nothing_runs(fw):
    before = fw.backend.live.qpos.copy()
    scene, unknown = _session(build_server(fw.wrap()),
                              ("describe_scene", {"verbose": True}), ("move_to", {"xyz": [0.4, 0.0, 1.0], "fast": 1}))
    assert scene.is_error and _text(scene).startswith("error: describe_scene takes no arguments")
    assert unknown.is_error and _text(unknown).startswith("blocked: ")
    assert (fw.backend.live.qpos == before).all()


def test_calls_run_one_at_a_time(fw):
    guarded = fw.wrap()
    active, peak, lock = [0], [0], threading.Lock()

    def slow_call(name, args):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return "done"

    guarded.handle_tool_call = slow_call
    server = build_server(guarded)

    async def run():
        async with Client(server) as client, anyio.create_task_group() as tg:
            for _ in range(4):
                tg.start_soon(client.call_tool, "describe_scene", {})
    anyio.run(run)
    assert peak[0] == 1


def test_the_command_serves_over_stdio(fw, tmp_path):
    """The real command in a subprocess, as an MCP client such as Claude Code starts it."""
    audit = tmp_path / "audit.jsonl"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from preflight.cli import app; app()", "serve", "--mcp", "--rules", str(RULES),
              "--feedback", "minimal", "--audit", str(audit)],
        cwd=ROOT,
        # This checkout first on the path, not an editable install of another one (as in test_gate_script_passes).
        env={"PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))},
    )

    async def run():
        async with Client(params) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            pick = await client.call_tool("pick", {"object": "hot_cup"})
            # The server loads the same scene, and the person does not move: `fw` gives the same hand position.
            move = await client.call_tool("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
            return names, pick, move
    names, pick, move = anyio.run(run)
    assert names == ["describe_scene", "move_to", "pick", "place"]
    assert not pick.is_error and _text(pick) == "done"
    assert move.is_error and _text(move) == "blocked by safety policy"
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [(r["tool"], r["verdict"]) for r in records] == [("pick", "allow"), ("move_to", "block")]
