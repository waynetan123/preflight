"""Measure turns a trace into correct facts, and computes only the facts that are needed."""

import itertools
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from preflight.backends import MujocoBackend
from preflight.criteria import REGISTRY, Need, load_rules
from preflight.measure import REPORT_RADIUS_M, Measure
from preflight.robots import TabletopArm

ROOT = Path(__file__).resolve().parent.parent
RULES = load_rules(ROOT / "scenes" / "kitchen_rules.yaml")
TABLE_TOP_Z = 0.75
G = 9.81

HOT_TO_HUMAN = Need("proximity", ("hot", "human"))
GRIPPER_TO_HUMAN = Need("proximity", ("gripper", "human"))
HUMAN_CONTACT = Need("contact", ("human",))
SPEED = Need("speed")


@pytest.fixture
def sim() -> MujocoBackend:
    return MujocoBackend(RULES.scene)


@pytest.fixture
def arm(sim) -> TabletopArm:
    return TabletopArm(sim.model)


@pytest.fixture
def measure(sim, arm) -> Measure:
    return Measure(sim.model, RULES, arm)


def _facts(sim, arm, measure, tool, args, needs):
    rollout = sim.rollout(arm, arm.compile(sim.live, tool, args))
    assert rollout.ok, rollout.detail
    return measure.facts(rollout.trace, needs), rollout


def _over_hand(sim, height):
    x, y, z = sim.live.geom("human_hand_left").xpos
    return [x, y, z + height]


# --- Which facts -------------------------------------------------------------


def test_kitchen_needs():
    needs = REGISTRY.needs(RULES.params)
    assert {HUMAN_CONTACT, SPEED, GRIPPER_TO_HUMAN, HOT_TO_HUMAN, Need("proximity", ("sharp", "human")),
            Need("tilt", ("liquid",)), Need("fall", ("fragile",)), Need("position")} == needs


def test_only_needed_facts_are_computed(sim, arm, measure):
    facts, _ = _facts(sim, arm, measure, "move_to", {"xyz": [0.45, 0.1, 1.0]}, {HUMAN_CONTACT})
    assert set(facts.contacts) == {"human"}
    assert not facts.proximity and not facts.tilt and not facts.falls
    assert facts.speed is None


def test_unknown_fact_kind(sim, arm, measure):
    rollout = sim.rollout(arm, arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0]}))
    with pytest.raises(ValueError, match="unknown fact kind"):
        measure.facts(rollout.trace, {Need("smell")})


def test_tag_with_no_bodies_reports_no_proximity(sim, arm, measure):
    facts, _ = _facts(sim, arm, measure, "move_to", {"xyz": [0.45, 0.1, 1.0]}, {Need("proximity", ("heavy", "human"))})
    assert (facts.proximity[("heavy", "human")].dist_m == REPORT_RADIUS_M).all()


# --- Proximity ---------------------------------------------------------------


def test_proximity_matches_brute_force(sim, arm, measure):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    rollout = sim.rollout(arm, arm.compile(sim.live, "move_to", {"xyz": _over_hand(sim, 0.25), "speed": 0.5}))
    facts = measure.facts(rollout.trace, {GRIPPER_TO_HUMAN})
    prox = facts.proximity[("gripper", "human")]

    model, data, fromto = sim.model, mujoco.MjData(sim.model), np.zeros(6)
    collides = (model.geom_contype | model.geom_conaffinity) != 0
    gripper = [g for g in range(model.ngeom) if collides[g] and (
        model.geom_bodyid[g] in arm.gripper_bodies or model.body(model.geom_bodyid[g]).name == "hot_cup")]
    human = [g for g in range(model.ngeom) if collides[g] and model.body(model.geom_bodyid[g]).name in RULES.bodies("human")]
    for i in (0, len(facts.t) // 2, len(facts.t) - 1):
        rollout.trace.restore(model, data, int(round(facts.t[i] / model.opt.timestep)))
        brute = min(mujoco.mj_geomDistance(model, data, a, b, REPORT_RADIUS_M, fromto)
                    for a, b in itertools.product(gripper, human))
        assert prox.dist_m[i] == pytest.approx(brute, abs=1e-9)


def test_starting_distances(sim, arm, measure):
    facts, _ = _facts(sim, arm, measure, "move_to", {"xyz": [0.45, 0.1, 1.0]},
                      {HOT_TO_HUMAN, Need("proximity", ("sharp", "human"))})
    hot, sharp = facts.proximity[("hot", "human")], facts.proximity[("sharp", "human")]
    assert hot.dist_m[0] == pytest.approx(0.222, abs=1e-3)
    assert (hot.a[0], hot.b[0]) == ("hot_cup", "human_hand_left")
    assert sharp.dist_m[0] == pytest.approx(0.189, abs=1e-3)
    assert sharp.a[0] == "knife"


# --- The unsafe move and the safe move ------------------------------------------


def test_hot_cup_into_the_hand(sim, arm, measure):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    needs = {HOT_TO_HUMAN, HUMAN_CONTACT, Need("tilt", ("liquid",))}
    facts, _ = _facts(sim, arm, measure, "move_to", {"xyz": _over_hand(sim, 0.10), "speed": 1.0}, needs)

    hot = facts.proximity[("hot", "human")]
    i = int(np.argmin(hot.dist_m))
    assert hot.dist_m[i] < 0
    assert (hot.a[i], hot.b[i]) == ("hot_cup", "human_hand_left")

    contacts = facts.contacts["human"]
    j = int(np.argmax(contacts.force_n))
    assert (contacts.robot_body[j], contacts.other_body[j]) == ("hot_cup", "human_hand_left")
    assert contacts.force_n[j] > 50  # above the placeholder hand limit
    assert facts.tilt["liquid"].deg["hot_cup"].max() > 5  # the impact tips the cup


def test_safe_move_away_from_the_person(sim, arm, measure):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    needs = {HOT_TO_HUMAN, HUMAN_CONTACT, Need("tilt", ("liquid",))}
    facts, _ = _facts(sim, arm, measure, "move_to", {"xyz": [0.30, -0.20, 1.00], "speed": 0.5}, needs)
    assert facts.proximity[("hot", "human")].dist_m.min() > 0.2
    assert len(facts.contacts["human"].t) == 0
    assert facts.tilt["liquid"].deg["hot_cup"].max() < 2


# --- Speed -------------------------------------------------------------------


def test_speed(sim, arm, measure):
    traj = arm.compile(sim.live, "move_to", {"xyz": [0.45, 0.1, 1.0], "speed": 0.5})
    facts = measure.facts(sim.rollout(arm, traj).trace, {SPEED})
    assert facts.speed.tcp_mps.max() == pytest.approx(traj.speed, rel=0.15)  # the servos lag a little
    assert np.isnan(facts.speed.held_mps).all()
    assert (facts.speed.held == None).all()  # noqa: E711  elementwise


def test_held_object_speed(sim, arm, measure):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    facts, _ = _facts(sim, arm, measure, "move_to", {"xyz": [0.30, -0.20, 1.00], "speed": 0.5}, {SPEED})
    assert (facts.speed.held == "hot_cup").all()
    # The cup moves with the TCP, so its speed is close to the TCP speed.
    assert np.abs(facts.speed.held_mps - facts.speed.tcp_mps).max() < 0.1


# --- Falls ---------------------------------------------------------------------


def test_dropped_plate_is_a_fall(sim, arm, measure):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "plate"}))
    plate_half = sim.model.geom("plate").size[1]
    height = 0.20
    facts, _ = _facts(sim, arm, measure, "place", {"xyz": [0.30, -0.10, TABLE_TOP_Z + plate_half + height]},
                      {Need("fall", ("fragile",))})
    (fall,) = facts.falls["fragile"]
    assert fall.body == "plate"
    assert fall.t_impact is not None and fall.t_impact > fall.t_release
    assert fall.drop_m == pytest.approx(height, abs=0.01)
    assert fall.impact_mps == pytest.approx(math.sqrt(2 * G * height), rel=0.05)


def test_normal_place_is_not_a_fall(sim, arm, measure):
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "plate"}))
    plate_half = sim.model.geom("plate").size[1]
    facts, _ = _facts(sim, arm, measure, "place", {"xyz": [0.30, -0.10, TABLE_TOP_Z + plate_half]},
                      {Need("fall", ("fragile",))})
    assert facts.falls["fragile"] == ()


# --- Speed of Measure -------------------------------------------------------------


def test_measure_is_fast(sim, arm, measure):
    rollout = sim.rollout(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))
    start = time.perf_counter()
    measure.facts(rollout.trace, REGISTRY.needs(RULES.params))
    assert time.perf_counter() - start < 0.15
