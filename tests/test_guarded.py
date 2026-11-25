"""The guarded robot gives tool definitions in each API format, and every tool call that can move the robot
goes through the firewall."""

import json
from pathlib import Path

import pytest

from preflight.core.firewall import Firewall
from preflight.core.guarded import Guarded
from preflight.core.verdict import Verdict

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"
TOOLS = ["describe_scene", "move_to", "pick", "place"]


@pytest.fixture
def fw() -> Firewall:
    return Firewall.from_yaml(RULES)


@pytest.fixture
def guarded(fw) -> Guarded:
    return fw.wrap()


def _over_hand(fw, height):
    x, y, z = map(float, fw.backend.live.geom("human_hand_left").xpos)
    return [x, y, z + height]


def _live(fw):
    return fw.backend.live.qpos.copy(), fw.backend.live.time


def _same(a, b):
    return (a[0] == b[0]).all() and a[1] == b[1]


# --- Tool definitions ----------------------------------------------------------


def test_anthropic_tool_schemas(guarded):
    tools = guarded.tool_schemas("anthropic")
    assert [t["name"] for t in tools] == TOOLS
    for t in tools:
        assert set(t) == {"name", "description", "input_schema"} and t["description"]
        assert t["input_schema"]["type"] == "object"
    by_name = {t["name"]: t["input_schema"] for t in tools}
    assert by_name["move_to"]["required"] == ["xyz"]
    assert by_name["pick"]["properties"]["object"]["enum"] == ["hot_cup", "knife", "plate"]
    assert by_name["describe_scene"]["properties"] == {}
    json.dumps(tools)  # plain JSON, ready to send


def test_openai_tool_schemas(guarded):
    tools = guarded.tool_schemas("openai")
    assert [t["function"]["name"] for t in tools] == TOOLS
    assert all(t["type"] == "function" and set(t["function"]) == {"name", "description", "parameters"}
               for t in tools)
    anthropic = guarded.tool_schemas("anthropic")
    assert [t["function"]["parameters"] for t in tools] == [t["input_schema"] for t in anthropic]


def test_tool_arguments_match_the_robot(fw, guarded):
    from preflight.robots.tabletop_arm import TOOL_ARGS
    for t in guarded.tool_schemas():
        if t["name"] in TOOL_ARGS:
            required, optional = TOOL_ARGS[t["name"]]
            assert set(t["input_schema"]["required"]) == required
            assert set(t["input_schema"]["properties"]) == required | optional


def test_unknown_format_and_feedback_are_refused(fw, guarded):
    with pytest.raises(ValueError, match="format must be one of"):
        guarded.tool_schemas("gemini")
    with pytest.raises(ValueError, match="feedback must be one of"):
        fw.wrap(feedback="verbose")


# --- Tool calls ----------------------------------------------------------------


def test_describe_scene_returns_text_and_runs_nothing(fw, guarded):
    before = _live(fw)
    text = guarded.handle_tool_call("describe_scene", {})
    assert text.startswith("Scene.") and "- hot_cup at " in text
    assert guarded.handle_tool_call("describe_scene", None) == text
    assert guarded.handle_tool_call("describe_scene", "") == text  # OpenAI may send an empty string
    assert _same(before, _live(fw))


def test_describe_scene_refuses_arguments(guarded):
    assert guarded.handle_tool_call("describe_scene", {"verbose": True}).startswith("error: ")


def test_an_allowed_call_runs(fw, guarded):
    assert guarded.handle_tool_call("pick", {"object": "hot_cup"}) == "done"
    assert fw.arm.holding(fw.backend.live) == "hot_cup"


def test_openai_json_arguments(fw, guarded):
    assert guarded.handle_tool_call("pick", '{"object": "hot_cup"}') == "done"
    assert fw.arm.holding(fw.backend.live) == "hot_cup"


def test_a_blocked_call_does_not_run(fw, guarded):
    assert guarded.handle_tool_call("pick", {"object": "hot_cup"}) == "done"
    before = _live(fw)
    text = guarded.handle_tool_call("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    assert text.startswith("blocked: hot_clearance at t = ")
    assert _same(before, _live(fw))


def test_minimal_feedback_hides_the_rule(fw):
    guarded = fw.wrap(feedback="minimal")
    assert guarded.handle_tool_call("pick", {"object": "hot_cup"}) == "done"
    assert guarded.handle_tool_call("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0}) == "blocked by safety policy"


@pytest.mark.parametrize("name, args, start", [
    ("fly", {}, "blocked: unknown tool 'fly'"),                      # the firewall refuses it
    ("move_to", {"xyz": [0.3, 0.0, 1.0], "force": 9}, "blocked: move_to: unknown argument(s): force"),
    ("move_to", "{not json", "error: the arguments are not valid JSON"),
    ("move_to", "[0.3, 0.0, 1.0]", "error: the arguments must be a JSON object"),
])
def test_bad_calls_run_nothing(fw, guarded, name, args, start):
    before = _live(fw)
    assert guarded.handle_tool_call(name, args).startswith(start)
    assert _same(before, _live(fw))


def test_python_call_returns_the_decision(fw, guarded):
    d = guarded.call("pick", object="hot_cup")
    assert d.verdict is Verdict.ALLOW and fw.arm.holding(fw.backend.live) == "hot_cup"
