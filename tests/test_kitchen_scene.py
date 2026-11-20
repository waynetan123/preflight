"""The kitchen scene loads, starts clean, stays still, and suits later stages."""

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "scenes" / "kitchen.xml"
TABLE_TOP_Z = 0.75

PROPS = ("hot_cup", "knife", "plate")
HUMAN_BODIES = (
    "human_torso", "human_head", "human_waist_lower", "human_pelvis",
    "human_upper_arm_left", "human_lower_arm_left", "human_hand_left",
    "human_upper_arm_right", "human_lower_arm_right", "human_hand_right",
    "human_thigh_left", "human_shin_left", "human_foot_left",
    "human_thigh_right", "human_shin_right", "human_foot_right",
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_kitchen", ROOT / "scenes" / "build_kitchen.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE))


@pytest.fixture
def data(model) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def test_expected_bodies_exist(model):
    for name in ("table", "sink", "panda/link0", "panda/hand", *PROPS, *HUMAN_BODIES):
        assert model.body(name).id >= 0


def test_human_is_static(model):
    human_ids = [model.body(name).id for name in HUMAN_BODIES]
    assert all(model.body_weldid[i] == 0 for i in human_ids), "every human body is welded to the world"


def test_grasp_welds_start_inactive(model):
    for obj in PROPS:
        eq = model.equality(f"grasp_{obj}")
        assert eq.type == mujoco.mjtEq.mjEQ_WELD
        assert not model.eq_active0[eq.id]


def test_no_interpenetration_at_start(model, data):
    for contact in data.contact[: data.ncon]:
        g1, g2 = model.geom(contact.geom1).name, model.geom(contact.geom2).name
        assert contact.dist > -1e-3, f"{g1} and {g2} overlap by {-contact.dist * 1000:.1f} mm"


def test_human_clear_of_table(model, data):
    # Static human and static table never generate contacts, so measure directly.
    fromto = np.zeros(6)
    names = [model.geom(g).name for g in range(model.ngeom)]
    table = [g for g, n in enumerate(names) if n.startswith("table")]
    for g in (g for g, n in enumerate(names) if n.startswith("human_")):
        for t in table:
            dist = mujoco.mj_geomDistance(model, data, g, t, 0.1, fromto)
            assert dist > -1e-3, f"{names[g]} is inside {names[t]} by {-dist * 1000:.1f} mm"


def test_scene_settles(model, data):
    start = {name: data.body(name).xpos.copy() for name in ("panda/hand", *PROPS, *HUMAN_BODIES)}
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    for name, pos in start.items():
        moved = np.linalg.norm(data.body(name).xpos - pos)
        limit = 0.0 if name.startswith("human_") else 2e-3
        assert moved <= limit, f"{name} moved {moved * 1000:.2f} mm"
    for name in PROPS:
        assert data.body(name).xpos[2] > TABLE_TOP_Z, f"{name} fell off the table"
        assert np.linalg.norm(data.body(name).cvel) < 1e-2, f"{name} is still moving"


@pytest.mark.parametrize("target", ["human_hand_left", "sink", *PROPS])
def test_within_panda_reach(data, target):
    shoulder = data.body("panda/link2").xpos  # joint 2 axis; the Panda reaches 0.855 m from here
    distance = np.linalg.norm(data.body(target).xpos - shoulder)
    assert distance < 0.80, f"{target} is {distance:.2f} m from the shoulder"


def test_human_hand_rests_on_table(model, data):
    hand = model.geom("human_hand_left")
    bottom = data.geom("human_hand_left").xpos[2] - hand.size[0]
    assert bottom == pytest.approx(TABLE_TOP_Z, abs=1e-3)


def test_geom_distance_cup_to_hand(model, data):
    fromto = np.zeros(6)
    dist = mujoco.mj_geomDistance(
        model, data, model.geom("hot_cup").id, model.geom("human_hand_left").id, 1.0, fromto
    )
    # The hand sphere's centre lies within the cup's height, so the gap is horizontal.
    gap = np.linalg.norm(data.body("hot_cup").xpos[:2] - data.body("human_hand_left").xpos[:2])
    expected = gap - model.geom("hot_cup").size[0] - model.geom("human_hand_left").size[0]
    assert dist == pytest.approx(expected, abs=2e-3)


def test_committed_xml_matches_builder():
    builder = _load_builder()
    assert builder.to_kitchen_xml(builder.build_kitchen()) == SCENE.read_text(), (
        "scenes/kitchen.xml is stale: run python scenes/build_kitchen.py"
    )
