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
from preflight.criteria.catalogue import (
    Clearance, FragileDrop, HumanForce, RobotLimits, SideEffects, SlowNearHuman, Spill, Workspace,
)
from preflight.criteria.checks import Context
from preflight.measure import (
    ContactFacts, Diff, Fall, Facts, Measure, Positions, Proximity, RobotHealth, Speed, Tilt,
)
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


# --- no_human_pinch ----------------------------------------------------------

DT = 0.01
PINCH = RULES.params["no_human_pinch"]  # defaults: hand 25 N, torso 20 N, head 0 N, for 0.5 s


def _clamp(force, steps, other="human_hand_left", start=0.0, robot="panda/left_finger", fixed=True):
    """One contact at each of `steps` steps, DT apart, from `start`."""
    return dict(t=start + DT * np.arange(steps), robot=np.full(steps, robot, object),
                other=np.full(steps, other, object), force=np.full(steps, float(force)), fixed=np.full(steps, fixed))


def _pinch_facts(*parts, end=2.0) -> Facts:
    """Contact facts from `_clamp` parts, in time order. The rollout ends at `end`."""
    merged = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    order = np.argsort(merged["t"], kind="stable")
    m = {k: v[order] for k, v in merged.items()}
    contacts = ContactFacts("human", m["t"], m["robot"], m["other"], m["force"], m["fixed"], DT)
    return _facts(t=np.array([0.0, end]), contacts={"human": contacts})


@pytest.mark.parametrize("force, other, fires", [
    (24.9, "human_hand_left", False),
    (25.1, "human_hand_left", True),   # hand limit 25 N
    (19.9, "human_torso", False),
    (20.1, "human_torso", True),       # torso limit 20 N
    (0.1, "human_head", True),         # head limit 0 N: any force
])
def test_pinch_limits_by_region(force, other, fires):
    v = checks.no_human_pinch(_pinch_facts(_clamp(force, 60, other)), PINCH, CTX)
    assert (v is not None) == fires
    if fires:
        assert v.bodies == ("panda/left_finger", other) and v.value == pytest.approx(force)
        assert v.limit == PINCH.max_force_n[RULES.regions[other]]


def test_pinch_limits_are_below_strike_limits():
    strike = RULES.params["no_human_strike"].max_force_n
    assert all(PINCH.max_force_n[r] <= strike[r] for r in strike)
    assert PINCH.max_force_n["hand"] < strike["hand"]


@pytest.mark.parametrize("steps, fires", [
    (49, False),   # 0.49 s: a transient contact, which no_human_strike judges
    (50, True),    # 0.50 s: the minimum duration
    (80, True),
])
def test_pinch_minimum_duration(steps, fires):
    v = checks.no_human_pinch(_pinch_facts(_clamp(30, steps, start=0.5)), PINCH, CTX)
    assert (v is not None) == fires
    if fires:
        assert v.t == pytest.approx(0.5)
        assert f"for {steps * DT:.2f} s;" in v.message


def test_pinch_needs_consecutive_steps_above_the_limit():
    # 0.3 s above, one step below the limit, 0.3 s above: two transient contacts, not one clamp.
    v = checks.no_human_pinch(_pinch_facts(_clamp(30, 30), _clamp(20, 1, start=0.30), _clamp(30, 30, start=0.31)),
                              PINCH, CTX)
    assert v is None


def test_pinch_reports_the_earliest_clamp():
    short = _clamp(40, 30, start=0.0)                          # too short
    hand = _clamp(30, 60, start=0.5)                           # first clamp
    torso = _clamp(30, 60, other="human_torso", start=0.7)     # later clamp
    head = _clamp(5, 60, other="human_head", start=0.4, robot="hot_cup")  # earlier still
    v = checks.no_human_pinch(_pinch_facts(short, hand, torso), PINCH, CTX)
    assert (v.t, v.bodies) == (pytest.approx(0.5), ("panda/left_finger", "human_hand_left"))
    v = checks.no_human_pinch(_pinch_facts(short, hand, torso, head), PINCH, CTX)
    assert (v.t, v.bodies, v.limit) == (pytest.approx(0.4), ("hot_cup", "human_head"), 0)


def test_pinch_adds_the_contacts_on_one_body():
    # Two fingers at 15 N each: no single contact is a strike, but together they clamp the hand with 30 N.
    facts = _pinch_facts(_clamp(15, 60, robot="panda/left_finger"), _clamp(15, 60, robot="panda/right_finger"))
    assert checks.no_human_strike(facts, RULES.params["no_human_strike"], CTX) is None
    assert checks.no_human_pinch(facts, PINCH, CTX).value == pytest.approx(30)


def test_pinch_ignores_a_body_that_is_not_fixed():
    # A body that can move away (a free or mocap body) is not clamped against its support.
    assert checks.no_human_pinch(_pinch_facts(_clamp(100, 100, fixed=False)), PINCH, CTX) is None


def test_pinch_still_pressing_at_the_end_of_the_rollout():
    # The robot holds its last pose after the call, so a short press at the end continues.
    clamp = _clamp(30, 20, start=1.0)
    v = checks.no_human_pinch(_pinch_facts(clamp, end=1.19), PINCH, CTX)
    assert v is not None and "still at the end of the rollout" in v.message
    assert checks.no_human_pinch(_pinch_facts(clamp, end=2.0), PINCH, CTX) is None


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


# --- robot_limits --------------------------------------------------------------

ROBOT_DT = 0.002  # one physics step
N = 500


def _robot(q=None, force=None, contacts=()):
    """One joint with the range -1 to 1 rad, and actuators with the force range -10 to 10, for N steps."""
    q = np.zeros(N) if q is None else np.asarray(q, float)
    force = np.zeros((N, 1)) if force is None else np.asarray(force, float).reshape(N, -1)
    t, a, b, f = (list(c) for c in zip(*contacts)) if contacts else ([], [], [], [])
    n = force.shape[1]
    return _facts(robot=RobotHealth(
        np.arange(N) * ROBOT_DT, ("panda/joint4",), q[:, None], np.array([[-1.0, 1.0]]),
        tuple(f"panda/actuator{i + 1}" for i in range(n)), force, np.tile([-10.0, 10.0], (n, 1)),
        ContactFacts("robot", np.array(t, float), np.array(a, object), np.array(b, object), np.array(f, float)),
    ))


def _saturated(start, steps, value=10.0):
    force = np.zeros(N)
    force[start:start + steps] = value
    return force


@pytest.mark.parametrize("q, fires", [
    (0.9989, False),   # 1.1 mrad from the upper end
    (0.9991, True),    # 0.9 mrad: inside the 1 mrad margin
    (-0.9989, False),
    (-0.9991, True),   # the lower end
    (1.001, True),     # past the end: MuJoCo's joint limit is soft
])
def test_robot_joint_margin(q, fires):
    qs = np.zeros(N)
    qs[200] = q
    v = checks.robot_limits(_robot(q=qs), RobotLimits(joint_margin_rad=0.001), CTX)
    assert (v is not None) == fires
    if fires:
        assert v.t == pytest.approx(200 * ROBOT_DT) and v.bodies == ("panda/joint4",)


@pytest.mark.parametrize("start, steps, value, fires", [
    (100, 25, 10.0, False),   # 0.050 s at the limit: not more than 0.05 s
    (100, 26, 10.0, True),    # 0.052 s
    (100, 26, -10.0, True),   # the lower end of the force range
    (100, 200, 9.85, False),  # 98.5 % of the limit: MuJoCo clamps to the limit, so this is not saturation
    (0, 275, 10.0, False),    # from the first row, 0.5 s more is allowed: 0.55 s
    (0, 276, 10.0, True),
])
def test_robot_saturation(start, steps, value, fires):
    facts = _robot(force=_saturated(start, steps, value))
    v = checks.robot_limits(facts, RobotLimits(max_saturated_s=0.05), CTX)
    assert (v is not None) == fires
    if fires:
        assert v.t == pytest.approx(start * ROBOT_DT) and v.value == pytest.approx(steps * ROBOT_DT)
        assert v.bodies == ("panda/actuator1",)


def test_robot_self_collision():
    assert checks.robot_limits(_robot(), RobotLimits(), CTX) is None
    v = checks.robot_limits(_robot(contacts=[(0.3, "panda/link5", "panda/hand", 12.0)]), RobotLimits(), CTX)
    assert v.t == 0.3 and v.bodies == ("panda/link5", "panda/hand") and v.value == 12.0


def test_robot_limits_reports_the_earliest_violation():
    qs = np.zeros(N)
    qs[300:] = 0.9995  # at the joint limit from t = 0.6 s
    contact = [(0.8, "panda/link5", "panda/hand", 5.0)]
    force = np.stack([_saturated(150, 50), _saturated(100, 50)], axis=1)  # actuator2 saturates first, at 0.2 s
    v = checks.robot_limits(_robot(q=qs, force=force, contacts=contact), RobotLimits(), CTX)
    assert v.t == pytest.approx(0.2) and v.bodies == ("panda/actuator2",)
    v = checks.robot_limits(_robot(q=qs, contacts=contact), RobotLimits(), CTX)
    assert v.t == pytest.approx(0.6) and v.bodies == ("panda/joint4",)


# --- no_side_effects -----------------------------------------------------------


def _diff(dist, names=("knife",), held=()):
    dist = np.array(dist, float).reshape(len(T), -1)
    return _facts(diff=Diff(tuple(names), dist, frozenset(held)))


@pytest.mark.parametrize("end, fires", [(0.019, False), (0.021, True)])  # tolerance 2 cm
def test_side_effect_tolerance(end, fires):
    v = checks.no_side_effects(_diff([0.0, 0.01, end]), SideEffects(tolerance_m=0.02), CTX)
    assert (v is not None) == fires
    if fires:
        assert v.bodies == ("knife",) and v.value == pytest.approx(end) and v.limit == 0.02


def test_side_effect_judges_the_end_not_the_peak():
    # Pushed 5 cm, then back within the tolerance: the world at the end is as it was.
    assert checks.no_side_effects(_diff([0.0, 0.05, 0.01]), SideEffects(tolerance_m=0.02), CTX) is None


def test_side_effect_time_is_the_start_of_the_lasting_change():
    v = checks.no_side_effects(_diff([0.0, 0.03, 0.04]), SideEffects(tolerance_m=0.02), CTX)
    assert v.t == T[1]
    v = checks.no_side_effects(_diff([0.03, 0.01, 0.04]), SideEffects(tolerance_m=0.02), CTX)
    assert v.t == T[2]  # an earlier excursion that came back does not count


def test_side_effect_ignores_held_objects():
    facts = _diff([[0.0, 0.0], [0.1, 0.0], [0.3, 0.05]], names=("hot_cup", "plate"), held=("hot_cup",))
    v = checks.no_side_effects(facts, SideEffects(tolerance_m=0.02), CTX)
    assert v.bodies == ("plate",)


def test_side_effect_ignores_exempt_objects():
    ctx = Context(regions=RULES.regions, tags=RULES.tags, exempt=frozenset({"knife"}))
    assert checks.no_side_effects(_diff([0.0, 0.1, 0.2]), SideEffects(tolerance_m=0.02), ctx) is None


def test_side_effect_reports_the_earliest_object():
    facts = _diff([[0.0, 0.0], [0.0, 0.05], [0.05, 0.05]], names=("knife", "plate"))
    v = checks.no_side_effects(facts, SideEffects(tolerance_m=0.02), CTX)
    assert v.bodies == ("plate",) and v.t == T[1]


# --- Engine --------------------------------------------------------------------


def test_unchecked_criteria_are_reported():
    _, unchecked = Checker(RULES)._plan("move_to")
    assert set(unchecked) == {"hazard_allowlist"}


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
                                       np.full(1, None, object)),
                   robot=_robot().robot, diff=Diff(("knife",), np.zeros((1, 1)), frozenset()))
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


# --- Pinch, end to end: the open gripper lowered at 0.1 m/s onto the hand resting on the table -----


@pytest.fixture(scope="module")
def press():
    def run(dx, dy, z):
        """From 16 cm above the hand's centre, offset by (dx, dy), lower the TCP to height z."""
        sim = MujocoBackend(RULES.scene)
        arm = TabletopArm(sim.model)
        measure, checker = Measure(sim.model, RULES, arm), Checker(RULES)
        x, y, _ = sim.live.geom("human_hand_left").xpos
        sim.execute(arm, arm.compile(sim.live, "move_to", {"xyz": [x + dx, y + dy, 0.95], "speed": 0.2}))
        rollout = sim.rollout(arm, arm.compile(sim.live, "move_to", {"xyz": [x + dx, y + dy, z], "speed": 0.1}))
        assert rollout.ok, rollout.detail
        facts = measure.facts(rollout.trace, checker.needs("move_to"))
        return facts, checker.check(facts, "move_to")

    return run


def _criteria(result):
    return [f.violation.criterion for f in result.findings]


def test_pinch_finger_on_the_hand(press):
    # 2 cm toward -x and 3 cm toward the robot: the left finger lands on the hand and the palm misses the
    # forearm. Each contact stays under the 50 N strike limit, but the hand is held at more than 25 N.
    facts, result = press(-0.02, -0.03, 0.80)
    assert _criteria(result) == ["no_human_pinch"]
    c = facts.contacts["human"]
    assert set(c.other_body) == {"human_hand_left"} and c.other_fixed.all()
    assert 30 < c.force_n.max() < 45
    v = result.findings[0].violation
    assert v.bodies == ("panda/left_finger", "human_hand_left") and v.limit == 25
    assert "still at the end of the rollout" in v.message


def test_hard_press_is_a_strike_and_a_pinch(press):
    # The press also holds an actuator at its force limit, so robot_limits fires as well.
    _, result = press(-0.02, -0.03, 0.77)
    assert set(_criteria(result)) == {"no_human_pinch", "no_human_strike", "robot_limits"}


def test_short_bump_is_not_a_pinch(press):
    # 5 cm toward the robot, the palm bumps the hand: above 25 N for about 0.25 s, then it rests at about
    # 15 N. The bump is a transient contact under the strike limit, and the rest is under the pinch limit.
    facts, result = press(0.0, -0.05, 0.77)
    c = facts.contacts["human"]
    assert 25 < c.force_n.max() < 50
    assert "no_human_pinch" not in _criteria(result) and "no_human_strike" not in _criteria(result)
