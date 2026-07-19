from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cinelingus.util import read_json, utc_now, write_json

from .registry import FilterRegistry, default_filter_registry


@dataclass
class FilterRecipe:
    filter_id: str
    filter_version: str
    family_id: str
    input_media_roles: dict[str, Any]
    parameters: dict[str, Any]
    output_settings: dict[str, Any]
    random_seed: int
    target_duration: float | None
    progression_settings: dict[str, Any] = field(default_factory=dict)
    identity_mapping_settings: dict[str, Any] = field(default_factory=dict)
    selected_filter_stack: list[str] = field(default_factory=list)
    compatibility_decisions: list[dict[str, str]] = field(default_factory=list)
    requested_analysis_backends: dict[str, str] = field(default_factory=dict)
    actual_analysis_backends: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: str = "1.0"

    @classmethod
    def create(
        cls,
        filter_id: str,
        *,
        input_media_roles: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        output_settings: dict[str, Any] | None = None,
        random_seed: int = 1,
        target_duration: float | None = None,
        selected_filter_stack: list[str] | None = None,
        requested_analysis_backends: dict[str, str] | None = None,
        actual_analysis_backends: dict[str, str] | None = None,
        registry: FilterRegistry | None = None,
    ) -> "FilterRecipe":
        registry = registry or default_filter_registry()
        definition = registry.get(filter_id)
        stack = selected_filter_stack or [definition.id]
        compatibility = registry.validate_stack(stack)
        normalized = definition.normalize_parameters(parameters)
        missing_roles = [role for role in definition.required_inputs if role not in input_media_roles]
        if missing_roles:
            raise ValueError(f"{definition.name} requires media roles: {', '.join(missing_roles)}")
        progression = {key: value for key, value in normalized.items() if key in {"progression", "curve_shape", "starting_intensity", "ending_intensity"}}
        identity = {key: value for key, value in normalized.items() if "speaker" in key or "identity" in key or key == "initial_carrier"}
        return cls(
            filter_id=definition.id,
            filter_version=definition.version,
            family_id=definition.family_id,
            input_media_roles={
                key: [str(item) for item in value] if isinstance(value, (list, tuple)) else str(value)
                for key, value in input_media_roles.items()
            },
            parameters=normalized,
            output_settings={"form": "full_length", **(output_settings or {}), "form": "full_length"},
            random_seed=int(random_seed),
            target_duration=None,
            progression_settings=progression,
            identity_mapping_settings=identity,
            selected_filter_stack=[registry.get(item).id for item in stack],
            compatibility_decisions=compatibility,
            requested_analysis_backends=requested_analysis_backends or {},
            actual_analysis_backends=actual_analysis_backends or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def deterministic_signature(self) -> str:
        data = self.to_dict()
        data.pop("created_at", None)
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecipeLoadResult:
    recipe: FilterRecipe
    warnings: tuple[str, ...] = ()
    migrations: tuple[str, ...] = ()


def save_recipe(recipe: FilterRecipe, output_path: Path) -> Path:
    write_json(output_path, recipe.to_dict())
    return output_path


def load_recipe(path: Path, *, registry: FilterRegistry | None = None) -> RecipeLoadResult:
    registry = registry or default_filter_registry()
    data = read_json(path)
    if str(data.get("schema_version")) != "1.0":
        raise ValueError(f"Unsupported filter recipe schema: {data.get('schema_version')}")
    resolved, migration = registry.resolve_id(str(data.get("filter_id", "")))
    definition = registry.get(resolved)
    warnings: list[str] = []
    migrations: list[str] = []
    if migration:
        migrations.append(migration)
    stored_version = str(data.get("filter_version") or "unknown")
    if stored_version != definition.version:
        warnings.append(f"Recipe uses {definition.name} version {stored_version}; the installed implementation is {definition.version}.")
    stack = []
    for item in data.get("selected_filter_stack") or [resolved]:
        stack_id, stack_migration = registry.resolve_id(str(item))
        stack.append(stack_id)
        if stack_migration:
            migrations.append(stack_migration)
    compatibility = registry.validate_stack(stack)
    stored_output_settings = dict(data.get("output_settings") or {})
    if stored_output_settings.get("form") not in {None, "full_length"}:
        migrations.append(
            f"Migrated removed output form '{stored_output_settings.get('form')}' to full_length."
        )
    stored_output_settings["form"] = "full_length"
    recipe = FilterRecipe(
        filter_id=resolved,
        filter_version=stored_version,
        family_id=definition.family_id,
        created_at=str(data.get("created_at") or utc_now()),
        input_media_roles={
            str(key): [str(item) for item in value] if isinstance(value, list) else str(value)
            for key, value in (data.get("input_media_roles") or {}).items()
        },
        parameters=definition.normalize_parameters(data.get("parameters") or {}),
        output_settings=stored_output_settings,
        random_seed=int(data.get("random_seed", 1)),
        target_duration=None,
        progression_settings=dict(data.get("progression_settings") or {}),
        identity_mapping_settings=dict(data.get("identity_mapping_settings") or {}),
        selected_filter_stack=stack,
        compatibility_decisions=list(data.get("compatibility_decisions") or compatibility),
        requested_analysis_backends=dict(data.get("requested_analysis_backends") or {}),
        actual_analysis_backends=dict(data.get("actual_analysis_backends") or {}),
    )
    return RecipeLoadResult(recipe=recipe, warnings=tuple(warnings), migrations=tuple(migrations))
