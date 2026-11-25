"""describe_scene gives the LLM each object and each part of the person, with positions and hazard tags."""

from pathlib import Path

import pytest

from preflight.core.firewall import Firewall
from preflight.core.scene import SimWorld, describe_scene

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"


@pytest.fixture
def fw() -> Firewall:
    return Firewall.from_yaml(RULES)


@pytest.fixture
def world(fw) -> SimWorld:
    return SimWorld(fw.backend, fw.rules, fw.arm)


def _objects(state):
    return {o.name: o for o in state.objects}


def test_objects_have_their_tags_and_positions(fw, world):
    objects = _objects(world.state())
    assert objects["hot_cup"].tags == ("hot", "liquid") and objects["hot_cup"].movable
    assert objects["knife"].tags == ("sharp",)
    assert objects["plate"].tags == ("fragile",)
    assert objects["sink"].tags == ("receptacle",) and not objects["sink"].movable
    for name, o in objects.items():
        assert o.xyz == pytest.approx(fw.backend.live.body(name).xpos, abs=5e-4)


def test_the_robot_and_untagged_fixed_bodies_are_not_listed(world):
    names = set(_objects(world.state()))
    assert not any(name.startswith("panda/") for name in names)
    assert "table" not in names and "world" not in names


def test_every_human_body_is_listed_with_its_region(fw, world):
    humans = {o.name: o for o in world.state().objects if "human" in o.tags}
    assert set(humans) == set(fw.rules.bodies("human"))
    assert humans["human_hand_left"].region == "hand" and humans["human_head"].region == "head"


def test_objects_come_before_the_person(world):
    is_human = ["human" in o.tags for o in world.state().objects]
    assert is_human == sorted(is_human)


def test_text_gives_gripper_workspace_objects_and_person(fw, world):
    text = describe_scene(world.state(), fw.rules)
    lines = text.splitlines()
    assert lines[1].startswith("Gripper (TCP) at (") and lines[1].endswith("holding nothing.")
    assert lines[2] == "Workspace: x -0.30 to 0.80, y -0.60 to 0.40, z 0.75 to 1.60."
    assert "- hot_cup at (0.250, 0.150, 0.800) [hot, liquid]" in lines
    assert "- sink at (0.350, -0.420, 0.750) [receptacle, fixed]" in lines
    assert lines.index("Objects:") < lines.index("Person (do not touch):")
    assert "- human_hand_left at (0.500, 0.320, 0.790) (hand)" in lines


def test_text_without_rules_has_no_workspace(world):
    assert "Workspace" not in describe_scene(world.state())


def test_state_follows_the_live_robot(fw, world):
    assert world.state().held is None
    assert fw.call("pick", {"object": "hot_cup"}).allowed
    state = world.state()
    assert state.held == "hot_cup"
    assert state.tcp_xyz == pytest.approx(fw.backend.live.site("panda/tcp").xpos, abs=5e-4)
    assert _objects(state)["hot_cup"].xyz[2] > 0.85  # lifted with the gripper
    assert "holding hot_cup." in describe_scene(state, fw.rules)


def test_reading_the_scene_does_not_change_the_live_state(fw, world):
    before = fw.backend.live.qpos.copy(), fw.backend.live.time
    describe_scene(world.state(), fw.rules)
    assert (fw.backend.live.qpos == before[0]).all() and fw.backend.live.time == before[1]
