"""The rules loader accepts the kitchen rules and refuses every kind of bad file, with a clear location."""

import copy
from pathlib import Path

import pytest
import yaml

from preflight.criteria import REGISTRY, CriterionSpec, Params, RulesError, load_rules
from preflight.criteria.catalogue import Spill

ROOT = Path(__file__).resolve().parent.parent
KITCHEN_RULES = ROOT / "scenes" / "kitchen_rules.yaml"
BASE = yaml.safe_load(KITCHEN_RULES.read_text())
BASE["scene"] = str(ROOT / "scenes" / "kitchen.xml")  # the test files live in tmp_path


def _write(tmp_path: Path, rules: dict | str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(rules if isinstance(rules, str) else yaml.safe_dump(rules, sort_keys=False))
    return path


def _problems(tmp_path: Path, change, registry=None) -> list[str]:
    rules = copy.deepcopy(BASE)
    change(rules)
    with pytest.raises(RulesError) as exc:
        load_rules(_write(tmp_path, rules), registry)
    return exc.value.problems


def _one_problem(tmp_path: Path, change, registry=None) -> str:
    problems = _problems(tmp_path, change, registry)
    assert len(problems) == 1, problems
    return problems[0]


# --- The kitchen file --------------------------------------------------------


def test_kitchen_rules_load():
    rules = load_rules(KITCHEN_RULES)
    assert rules.scene == ROOT / "scenes" / "kitchen.xml"
    assert {"human_hand_left", "human_hand_right", "human_head"} <= rules.bodies("human")
    assert rules.bodies("hot") == {"hot_cup"}
    assert rules.regions["human_hand_left"] == "hand"
    assert rules.regions["human_lower_arm_right"] == "forearm"


def test_every_criterion_is_active_with_defaults():
    rules = load_rules(KITCHEN_RULES)
    assert set(rules.params) == {s.id for s in REGISTRY.criteria()} == set(rules.on_violation)
    assert rules.params["hot_clearance"].min_m == 0.30  # default
    assert rules.params["no_human_strike"].max_force_n["forearm"] == 50  # default kept beside the file's values
    assert rules.params["no_human_pinch"].max_force_n["hand"] == 25  # default: half the strike limit
    assert rules.params["no_human_pinch"].min_duration_s == 0.5
    assert rules.on_violation["slow_near_human"] == "modify"
    assert rules.on_violation["no_human_strike"] == "block"


def test_exemptions_resolve_to_bodies():
    (pour,) = load_rules(KITCHEN_RULES).exemptions_for("pour")
    assert pour.relax == "no_spill"
    assert pour.objects == {"hot_cup"}
    assert pour.replace_with["pour_into_receptacle"].max_tilt_deg == 120


def test_loaded_rules_cannot_change():
    rules = load_rules(KITCHEN_RULES)
    with pytest.raises(TypeError):
        rules.on_violation["no_human_strike"] = "warn"
    with pytest.raises(Exception):
        rules.params["hot_clearance"].min_m = 0.0


# --- Structure ---------------------------------------------------------------


def test_unknown_key(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["horizon"].update(setle_s=2.0))
    assert problem.startswith("horizon.setle_s:")


def test_unknown_tag(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["tags"].update(pointy=["knife"]))
    assert problem.startswith("tags.pointy: unknown tag 'pointy'; known tags: fragile, hot, human")


def test_negative_threshold(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["criteria"]["params"].update(sharp_clearance={"min_m": -0.1}))
    assert problem.startswith("criteria.params.sharp_clearance.min_m:")


def test_workspace_is_required(tmp_path):
    problems = _problems(tmp_path, lambda r: r["criteria"]["params"].pop("workspace"))
    assert problems == [
        "criteria.params.workspace.min_xyz: Field required",
        "criteria.params.workspace.max_xyz: Field required",
    ]


def test_duplicate_key(tmp_path):
    text = KITCHEN_RULES.read_text().replace("criteria:", "criteria: {}\ncriteria:", 1)
    line = text.splitlines().index("criteria:") + 1
    with pytest.raises(RulesError) as exc:
        load_rules(_write(tmp_path, text))
    assert exc.value.problems == [f"invalid YAML at line {line}, column 1: found duplicate key 'criteria'"]


# --- Scene -------------------------------------------------------------------


def test_pattern_that_matches_no_body(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["tags"].update(sharp=["knife", "human_hand_l"]))
    assert problem == "tags.sharp[1]: 'human_hand_l' matches no body in the scene"


def test_human_tag_is_required(tmp_path):
    problems = _problems(tmp_path, lambda r: r["tags"].pop("human"))
    assert "tags.human: at least one body must be tagged human" in problems


def test_missing_scene(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r.update(scene="no_such_scene.xml"))
    assert problem.startswith("scene: cannot load")


def test_human_body_without_region(tmp_path):
    # Every body except human_head gets a region.
    regions = {"human_[!h]*": "torso", "human_hand_*": "hand"}
    problem = _one_problem(tmp_path, lambda r: r.update(human_model={"regions": regions}))
    assert problem == "human_model: human body 'human_head' has no region; add it to human_model.regions"


def test_explicit_region_overrides_profile(tmp_path):
    rules = copy.deepcopy(BASE)
    rules["human_model"]["regions"] = {"human_head": "torso"}
    assert load_rules(_write(tmp_path, rules)).regions["human_head"] == "torso"


def test_force_limit_for_unknown_region(tmp_path):
    problem = _one_problem(
        tmp_path, lambda r: r["criteria"]["params"]["no_human_strike"]["max_force_n"].update(tentacle=10)
    )
    assert problem == "criteria.params.no_human_strike.max_force_n.tentacle: no human body has this region"


def test_pinch_limit_for_unknown_region(tmp_path):
    problem = _one_problem(
        tmp_path, lambda r: r["criteria"]["params"].update(no_human_pinch={"max_force_n": {"tentacle": 10}})
    )
    assert problem == "criteria.params.no_human_pinch.max_force_n.tentacle: no human body has this region"


def test_pinch_limit_merges_with_defaults(tmp_path):
    rules = copy.deepcopy(BASE)
    rules["criteria"]["params"]["no_human_pinch"] = {"max_force_n": {"hand": 30}, "min_duration_s": 0.3}
    pinch = load_rules(_write(tmp_path, rules)).params["no_human_pinch"]
    assert (pinch.max_force_n["hand"], pinch.max_force_n["torso"], pinch.min_duration_s) == (30, 20, 0.3)


def test_pinch_duration_must_be_positive(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["criteria"]["params"].update(no_human_pinch={"min_duration_s": 0}))
    assert problem.startswith("criteria.params.no_human_pinch.min_duration_s:")


# --- Criteria and the hard floor --------------------------------------------


def test_unknown_criterion(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["criteria"]["on_violation"].update(no_human_stab="block"))
    assert "unknown criterion 'no_human_stab'" in problem


def test_warn_on_non_waivable_criterion(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["criteria"]["on_violation"].update(no_human_strike="warn"))
    assert "not waivable" in problem


def test_warn_on_waivable_criterion(tmp_path):
    rules = copy.deepcopy(BASE)
    rules["criteria"]["on_violation"]["no_side_effects"] = "warn"
    assert load_rules(_write(tmp_path, rules)).on_violation["no_side_effects"] == "warn"


def test_exemption_cannot_relax_non_waivable(tmp_path):
    def change(r):
        r["exemptions"][1]["relax"] = "no_human_strike"
        r["exemptions"][1]["objects"] = {"tag": "human"}
    assert "not waivable" in _one_problem(tmp_path, change)


def test_exemption_needs_replacement(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["exemptions"][0].pop("replace_with"))
    assert problem == "exemptions[0].replace_with: Field required"


def test_exemption_tag_must_belong_to_relaxed_criterion(tmp_path):
    problem = _one_problem(tmp_path, lambda r: r["exemptions"][0].update(objects={"tag": "sharp"}))
    assert "does not read tag 'sharp'" in problem


def test_replacement_parameter_names_unknown_tag(tmp_path):
    problem = _one_problem(
        tmp_path, lambda r: r["exemptions"][0]["replace_with"]["pour_into_receptacle"].update(target_tag="hand")
    )
    assert problem.startswith("exemptions[0].replace_with.pour_into_receptacle.target_tag:")


def test_all_problems_are_reported_together(tmp_path):
    def change(r):
        r["tags"]["sharp"] = ["no_such_body"]
        r["criteria"]["on_violation"]["no_human_strike"] = "warn"
        r["exemptions"][0]["relax"] = "workspace"
    assert len(_problems(tmp_path, change)) == 3


# --- Plug-ins ----------------------------------------------------------------


class Heavy(Params):
    max_mass_kg: float = 2.0


def test_plugin_criterion_adds_its_tag(tmp_path):
    registry = REGISTRY.copy()
    registry.register(CriterionSpec("heavy_over_human", "state", waivable=True, params=Heavy, tags=("heavy",)))
    rules = copy.deepcopy(BASE)
    rules["tags"]["heavy"] = ["plate"]
    loaded = load_rules(_write(tmp_path, rules), registry)
    assert loaded.bodies("heavy") == {"plate"}
    assert loaded.params["heavy_over_human"].max_mass_kg == 2.0

    # The default registry is unchanged, so there the tag is still unknown.
    with pytest.raises(RulesError, match="unknown tag 'heavy'"):
        load_rules(_write(tmp_path, rules))


def test_plugin_cannot_replace_a_builtin():
    registry = REGISTRY.copy()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(CriterionSpec("no_human_strike", "state", waivable=True, params=Spill))
