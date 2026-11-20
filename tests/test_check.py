"""Each check gives the right result just inside and just outside its limit, and the engine applies
exemptions, reports what it cannot check, and fails closed."""

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from preflight.backends import MujocoBackend
from preflight.core.check import Checker
from preflight.criteria import REGISTRY, CriterionSpec, Need, Params, load_rules
from preflight.criteria import checks
from preflight.criteria.catalogue import Clearance, FragileDrop, HumanForce, SlowNearHuman, Spill, Workspace
from preflight.criteria.checks import Context
from preflight.measure import ContactFacts, Fall, Facts, Measure, Positions, Proximity, Speed, Tilt
from preflight.robots import TabletopArm

ROOT = Path(__file__).resolve().parent.parent
RULES = load_rules(ROOT / "scenes" / "kitchen_rules.yaml")
CTX = Context(regions=RULES.regions, tags=RULES.tags)
T = np.array([0.0, 0.1, 0.2])


def _facts(**kwargs) -> Facts:
    fields = dict(t=T, contacts={}, proximity={}, speed=None, tilt={}, falls={}, positions=None)
    fields.update(kwargs)
    return Facts(**fields)


def _proximity(pair, dist, a="hot_cup", b="human_hand_left"):
    n = len(dist)
    return {pair: Proximity(pair, np.array(dist, float), np.full(n, a, object), np.full(n, b, object))}


def _contacts(forces, other="human_hand_left"):
    n = len(forces)
    return {"human": ContactFacts("human", T[:n], np.full(n, "hot_cup", object), np.full(n, other, object),
                                  np.array(forces, float))}


# --- no_human_strike ---------------------------------------------------------


@pytest.mark.parametrize("force, other, fires", [
    (49.9, "human_hand_left", False),
    (50.1, "human_hand_left", True),   # hand limit 50 N
    (39.9, "human_torso", False),
    (40.1, "human_torso", True),       # torso limit 40 N
    (0.1, "human_head", True),         # head limit 0 N: any force
])
def test_strike_limits_by_region(force, other, fires):
    v = checks.no_human_strike(_facts(contacts=_contacts([force], other)), RULES.params["no_human_strike"], CTX)
    assert (v is not None) == fires
    if fires:
        assert v.bodies == ("hot_cup", other) and v.value == pytest.approx(force)


def test_strike_reports_the_earliest_contact():
    v = checks.no_human_strike(_facts(contacts=_contacts([10, 60, 90])), RULES.params["no_human_strike"], CTX)
    assert v.t == T[1] and v.value == 60


# --- slow_near_human ---------------------------------------------------------


def _slow(dist, tcp, held=np.nan):
    n = len(dist)
    return _facts(t=T[:n], proximity=_proximity(("gripper", "human"), dist),
                  speed=Speed(np.array(tcp, float), np.full(n, held, float), np.full(n, None, object)))


@pytest.mark.parametrize("dist, speed, fires", [
    (0.49, 0.24, False),   # near, slow
    (0.49, 0.26, True),    # near, fast
    (0.51, 1.00, False),   # fast, but outside the 0.5 m radius
])
def test_slow_near_human(dist, speed, fires):
    p = SlowNearHuman(radius_m=0.5, max_speed_mps=0.25)
    assert (checks.slow_near_human(_slow([dist], [speed]), p, CTX) is not None) == fires


def test_slow_near_human_reads_the_held_object_speed():
    p = SlowNearHuman(radius_m=0.5, max_speed_mps=0.25)
    assert checks.slow_near_human(_slow([0.3], [0.1], held=0.3), p, CTX).value == pytest.approx(0.3)


# --- Clearance (option A) ----------------------------------------------------


@pytest.mark.parametrize("dist, fires", [
    ([0.40, 0.31, 0.40], False),   # starts clear, stays above 30 cm
    ([0.40, 0.29, 0.40], True),    # starts clear, goes below the limit
    ([0.22, 0.22, 0.23], False),   # starts below the limit, does not get closer
    ([0.22, 0.216, 0.22], False),  # 4 mm closer: inside the 5 mm tolerance
    ([0.22, 0.21, 0.22], True),    # 1 cm closer than the start
])
def test_clearance_judges_only_what_the_action_makes_worse(dist, fires):
    facts = _facts(proximity=_proximity(("hot", "human"), dist))
    assert (checks.hot_clearance(facts, Clearance(min_m=0.30), CTX) is not None) == fires


def test_clearance_skips_a_tag_that_an_exemption_covers():
    facts = _facts(proximity=_proximity(("sharp", "human"), [0.4, 0.1, 0.1], a="knife"))
    ctx = Context(regions=RULES.regions, tags=RULES.tags, exempt=frozenset({"knife"}))
    assert checks.sharp_clearance(facts, Clearance(min_m=0.30), ctx) is None


# --- no_fragile_drop and no_spill --------------------------------------------


@pytest.mark.parametrize("fall, fires", [
    (Fall("plate", 0.1, 0.3, 0.04, 0.9), False),
    (Fall("plate", 0.1, 0.3, 0.06, 1.1), True),
    (Fall("plate", 0.1, None, 0.5, None), True),   # still falling at the end: unknown outcome
])
def test_fragile_drop(fall, fires):
    facts = _facts(falls={"fragile": (fall,)})
    assert (checks.no_fragile_drop(facts, FragileDrop(max_impact_mps=1.0), CTX) is not None) == fires


@pytest.mark.parametrize("deg, fires", [(29.9, False), (30.1, True)])
def test_spill(deg, fires):
    facts = _facts(tilt={"liquid": Tilt("liquid", MappingProxyType({"hot_cup": np.array([0.0, deg, 0.0])}))})
    assert (checks.no_spill(facts, Spill(max_tilt_deg=30), CTX) is not None) == fires


# --- workspace ---------------------------------------------------------------


def _positions(tcp, held=(np.nan,) * 3):
    return _facts(t=T[:1], positions=Positions(("tcp",), np.array([[tcp]], float), np.array([held], float),
                                               np.array(["hot_cup" if np.isfinite(held).all() else None], object)))


@pytest.mark.parametrize("tcp, held, fires", [
    ((0.4, 0.0, 0.76), (np.nan,) * 3, False),
    ((0.4, 0.0, 0.744), (np.nan,) * 3, True),   # 6 mm under the table top: outside the 5 mm tolerance
    ((0.4, 0.0, 1.0), (0.4, 0.0, 0.74), True),  # the held object goes under the table top
    ((0.4, 0.41, 1.0), (np.nan,) * 3, True),    # past y max
])
def test_workspace(tcp, held, fires):
    v = checks.workspace(_positions(tcp, held), RULES.params["workspace"], CTX)
    assert (v is not None) == fires
    if fires and np.isfinite(held).all():
        assert v.bodies == ("hot_cup",)


# --- Engine --------------------------------------------------------------------


def test_unchecked_criteria_are_reported():
    _, unchecked = Checker(RULES)._plan("move_to")
    assert set(unchecked) == {"no_human_pinch", "robot_limits", "hazard_allowlist", "no_side_effects"}


def test_exemption_with_an_unchecked_replacement_is_not_applied():
    checker = Checker(RULES)
    active, unchecked = checker._plan("pour")
    spill = next(a for a in active if a.criterion == "no_spill")
    assert spill.exempt == frozenset()  # pour_into_receptacle has no check, so no_spill stays fully active
    assert "pour: exemption from no_spill" in unchecked


def test_exemption_with_a_checked_replacement_is_applied():
    registry = REGISTRY.copy()
    registry._specs["pour_into_receptacle"] = CriterionSpec(
        "pour_into_receptacle", "state", waivable=False, params=registry.get("pour_into_receptacle").params,
        replacement=True, check=lambda facts, p, ctx: None)
    active, unchecked = Checker(RULES, registry)._plan("pour")
    assert next(a for a in active if a.criterion == "no_spill").exempt == {"hot_cup"}
    assert "pour_into_receptacle" in {a.criterion for a in active}
    assert "pour: exemption from no_spill" not in unchecked


def test_a_check_that_fails_blocks():
    class Nothing(Params):
        pass

    def broken(facts, p, ctx):
        raise RuntimeError("no facts")

    registry = REGISTRY.copy()
    registry.register(CriterionSpec("broken", "state", waivable=True, params=Nothing, check=broken))
    rules = load_rules(ROOT / "scenes" / "kitchen_rules.yaml", registry)
    facts = _facts(contacts=_contacts([]), proximity={**_proximity(("hot", "human"), [0.5]),
                                                      **_proximity(("sharp", "human"), [0.5]),
                                                      **_proximity(("gripper", "human"), [0.9])},
                   t=T[:1], speed=Speed(np.zeros(1), np.full(1, np.nan), np.full(1, None, object)),
                   tilt={"liquid": Tilt("liquid", {})}, falls={"fragile": ()},
                   positions=Positions(("tcp",), np.array([[[0.4, 0.0, 1.0]]]), np.full((1, 3), np.nan),
                                       np.full(1, None, object)))
    result = Checker(rules, registry).check(facts, "move_to")
    (finding,) = result.findings
    assert finding.violation.criterion == "broken" and finding.action == "block"
    assert "RuntimeError: no facts" in finding.violation.message


# --- The gate case, end to end -------------------------------------------------


@pytest.fixture(scope="module")
def gate():
    sim = MujocoBackend(RULES.scene)
    arm = TabletopArm(sim.model)
    measure, checker = Measure(sim.model, RULES, arm), Checker(RULES)
    sim.execute(arm, arm.compile(sim.live, "pick", {"object": "hot_cup"}))

    def run(xyz, speed):
        rollout = sim.rollout(arm, arm.compile(sim.live, "move_to", {"xyz": xyz, "speed": speed}))
        assert rollout.ok, rollout.detail
        return checker.check(measure.facts(rollout.trace, checker.needs("move_to")), "move_to")

    return sim, run


def test_gate_unsafe_move_is_blocked(gate):
    sim, run = gate
    x, y, z = sim.live.geom("human_hand_left").xpos
    result = run([x, y, z + 0.10], 1.0)
    blocking = {f.violation.criterion for f in result.findings if f.action == "block"}
    assert {"hot_clearance", "no_human_strike"} <= blocking


def test_gate_safe_move_has_no_violations(gate):
    _, run = gate
    assert run([0.30, -0.20, 1.00], 0.2).findings == ()


def test_gate_safe_move_too_fast_is_only_slowed(gate):
    _, run = gate
    result = run([0.30, -0.20, 1.00], 0.5)
    assert [(f.violation.criterion, f.action) for f in result.findings] == [("slow_near_human", "modify")]
