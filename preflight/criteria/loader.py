"""Load and check a rules file. Any problem stops the load, so the firewall never starts on a bad config.

    rules = load_rules("scenes/kitchen_rules.yaml")
    rules.bodies("human")        # frozenset of body names
    rules.params["hot_clearance"].min_m

The loader collects every problem it can find and reports them together in one RulesError.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import yaml
from pydantic import ValidationError

from .catalogue import REGION_PROFILES, HumanForce
from .exemptions import Exemption, exemption_problems
from .registry import REGISTRY, CriterionSpec, Params, Registry
from .schema import HumanModel, RulesFile


class RulesError(Exception):
    def __init__(self, source: Path, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"{source}: {len(problems)} problem(s) in the rules file\n{lines}")


@dataclass(frozen=True)
class Rules:
    source: Path
    scene: Path
    settle_s: float
    tags: Mapping[str, frozenset[str]]  # tag -> body names
    regions: Mapping[str, str]  # human body -> region
    params: Mapping[str, Params]  # every active criterion, defaults filled in
    on_violation: Mapping[str, str]  # every active criterion: block | modify | warn
    exemptions: Mapping[str, tuple[Exemption, ...]]  # tool -> its exemptions

    def bodies(self, tag: str) -> frozenset[str]:
        return self.tags.get(tag, frozenset())

    def exemptions_for(self, tool: str) -> tuple[Exemption, ...]:
        return self.exemptions.get(tool, ())


def load_rules(path: str | Path, registry: Registry | None = None) -> Rules:
    registry = registry or REGISTRY
    source = Path(path).resolve()
    known_tags = registry.tags()

    raw = _read_yaml(source)
    try:
        file = RulesFile.model_validate(raw, context={"tags": known_tags})
    except ValidationError as exc:
        raise RulesError(source, _format(exc)) from None

    scene = (source.parent / file.scene).resolve()
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as exc:  # noqa: BLE001  any load failure is a config error
        raise RulesError(source, [f"scene: cannot load {scene}: {exc}"]) from None
    body_names = [model.body(i).name for i in range(model.nbody)]

    problems: list[str] = []
    tags = _expand_tags(file.tags, body_names, problems)
    if not tags.get("human"):
        problems.append("tags.human: at least one body must be tagged human")
    regions = _resolve_regions(file.human_model, tags.get("human", frozenset()), problems)
    context = {"tags": known_tags}
    params = _criteria_params(file, registry, context, problems)
    _check_force_regions(params, file.criteria.params, regions, problems)
    on_violation = _on_violation(file, registry, problems)
    exemptions = _exemptions(file, registry, tags, context, problems)

    if problems:
        raise RulesError(source, problems)
    return Rules(
        source=source,
        scene=scene,
        settle_s=file.horizon.settle_s,
        tags=MappingProxyType(tags),
        regions=MappingProxyType(regions),
        params=MappingProxyType(params),
        on_violation=MappingProxyType(on_violation),
        exemptions=MappingProxyType(exemptions),
    )


# --- YAML --------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a repeated key; plain PyYAML keeps the last value without a warning."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:  # unhashable key: let SafeLoader report it
                break
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


def _read_yaml(source: Path) -> Any:
    try:
        text = source.read_text()
    except OSError as exc:
        raise RulesError(source, [f"cannot read the file: {exc.strerror}"]) from None
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506  a SafeLoader subclass
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else "somewhere"
        raise RulesError(source, [f"invalid YAML at {where}: {exc.problem or exc.context}"]) from None
    except yaml.YAMLError as exc:
        raise RulesError(source, [f"invalid YAML: {exc}"]) from None
    if not isinstance(raw, dict):
        raise RulesError(source, ["the file must be a YAML mapping"])
    return raw


def _format(exc: ValidationError, prefix: str = "") -> list[str]:
    problems = []
    for error in exc.errors():
        loc = prefix
        for part in error["loc"]:
            if part == "[key]":  # pydantic's marker for an error in a dict key, not in its value
                continue
            loc += f"[{part}]" if isinstance(part, int) else (f".{part}" if loc else str(part))
        message = error["msg"].removeprefix("Value error, ")
        problems.append(f"{loc or '(top level)'}: {message}")
    return problems


# --- Checks against the scene ------------------------------------------------


def _expand_tags(patterns: dict[str, list[str]], bodies: list[str], problems: list[str]) -> dict[str, frozenset[str]]:
    tags = {}
    for tag, tag_patterns in patterns.items():
        matched = set()
        for i, pattern in enumerate(tag_patterns):
            hits = [b for b in bodies if fnmatchcase(b, pattern)]
            if not hits:
                problems.append(f"tags.{tag}[{i}]: {pattern!r} matches no body in the scene")
            matched.update(hits)
        tags[tag] = frozenset(matched)
    return tags


def _resolve_regions(human_model: HumanModel, humans: frozenset[str], problems: list[str]) -> dict[str, str]:
    """Each human body's region: an explicit `regions` entry wins over the profile."""
    profile: dict[str, str] = {}
    if human_model.profile is not None:
        if human_model.profile not in REGION_PROFILES:
            known = ", ".join(sorted(REGION_PROFILES))
            problems.append(f"human_model.profile: unknown profile {human_model.profile!r}; known: {known}")
        else:
            profile = {human_model.prefix + p: r for p, r in REGION_PROFILES[human_model.profile].items()}

    regions = {}
    for body in sorted(humans):
        for where, mapping in (("human_model.regions", human_model.regions), ("human_model.profile", profile)):
            found = {r for p, r in mapping.items() if fnmatchcase(body, p)}
            if len(found) > 1:
                problems.append(f"{where}: {body!r} matches more than one region ({', '.join(sorted(found))})")
            if found:
                regions[body] = min(found)
                break
        else:
            problems.append(f"human_model: human body {body!r} has no region; add it to human_model.regions")
    return regions


# --- Criteria ----------------------------------------------------------------


def _with_defaults(model: type[Params], user: dict[str, Any]) -> dict[str, Any]:
    """The model's defaults with the user's values on top; a dict parameter is merged one key at a time."""
    merged = {
        name: field.get_default(call_default_factory=True)
        for name, field in model.model_fields.items()
        if not field.is_required()
    }
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _params(spec: CriterionSpec, user: dict[str, Any], context: dict, loc: str, problems: list[str]) -> Params | None:
    try:
        return spec.params.model_validate(_with_defaults(spec.params, user), context=context)
    except ValidationError as exc:
        problems.extend(_format(exc, loc))
        return None


def _unknown_or_replacement(criterion_id: str, registry: Registry, loc: str) -> str | None:
    spec = registry.get(criterion_id)
    if spec is None:
        known = ", ".join(s.id for s in registry.criteria())
        return f"{loc}: unknown criterion {criterion_id!r}; known: {known}"
    if spec.replacement:
        return f"{loc}: {criterion_id!r} is a replacement criterion; set it inside an exemption's replace_with"
    return None


def _criteria_params(file: RulesFile, registry: Registry, context: dict, problems: list[str]) -> dict[str, Params]:
    user = file.criteria.params
    for criterion_id in user:
        if problem := _unknown_or_replacement(criterion_id, registry, f"criteria.params.{criterion_id}"):
            problems.append(problem)

    params = {}
    for spec in registry.criteria():
        value = _params(spec, user.get(spec.id, {}), context, f"criteria.params.{spec.id}", problems)
        if value is not None:
            params[spec.id] = value
    return params


def _check_force_regions(params: dict[str, Params], user: dict[str, dict], regions: dict[str, str],
                         problems: list[str]) -> None:
    """Per-region force limits: every region in use has a limit, and every region the file names exists."""
    used = set(regions.values())
    for criterion_id, value in params.items():
        if not isinstance(value, HumanForce):
            continue
        defaults = set(type(value).model_fields["max_force_n"].get_default(call_default_factory=True))
        loc = f"criteria.params.{criterion_id}.max_force_n"
        for region in user.get(criterion_id, {}).get("max_force_n", {}):
            if region not in used | defaults:
                problems.append(f"{loc}.{region}: no human body has this region")
        for region in sorted(used - set(value.max_force_n)):
            problems.append(f"{loc}: no limit for region {region!r}")


def _on_violation(file: RulesFile, registry: Registry, problems: list[str]) -> dict[str, str]:
    user = file.criteria.on_violation
    for criterion_id, action in user.items():
        loc = f"criteria.on_violation.{criterion_id}"
        if problem := _unknown_or_replacement(criterion_id, registry, loc):
            problems.append(problem)
        elif action == "warn" and not registry.get(criterion_id).waivable:
            problems.append(f"{loc}: {criterion_id!r} is not waivable, so it can only block or modify")
    return {spec.id: user.get(spec.id, "block") for spec in registry.criteria()}


def _exemptions(file: RulesFile, registry: Registry, tags: dict[str, frozenset[str]], context: dict,
                problems: list[str]) -> dict[str, tuple[Exemption, ...]]:
    by_tool: dict[str, list[Exemption]] = {}
    seen = set()
    for i, entry in enumerate(file.exemptions):
        loc = f"exemptions[{i}]"
        entry_problems = exemption_problems(entry, registry)
        problems.extend(f"{loc}.{p}" for p in entry_problems)
        if (entry.tool, entry.relax) in seen:
            problems.append(f"{loc}: tool {entry.tool!r} already has an exemption for {entry.relax!r}")
        seen.add((entry.tool, entry.relax))
        if entry_problems:
            continue

        replace_with = {}
        for name, user in entry.replace_with.items():
            value = _params(registry.get(name), user, context, f"{loc}.replace_with.{name}", problems)
            if value is not None:
                replace_with[name] = value
        by_tool.setdefault(entry.tool, []).append(Exemption(
            tool=entry.tool,
            relax=entry.relax,
            objects=tags.get(entry.objects.tag, frozenset()),
            replace_with=MappingProxyType(replace_with),
        ))
    return {tool: tuple(items) for tool, items in by_tool.items()}
