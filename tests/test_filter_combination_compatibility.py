from pathlib import Path

import pytest

from cinelingus.filter_lab.combination import (
    CombinationStatus,
    compile_compatibility_matrix,
    compile_ordered_combination,
)
from cinelingus.filter_lab.recipe import FilterRecipe
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def test_primary_then_bloom_is_visible_but_not_executable_without_procedure_proof() -> None:
    decision = compile_ordered_combination("multiworld.translation", "experimental.bloom")
    assert decision.status == CombinationStatus.COMPATIBLE_UNPROVEN
    assert decision.executable is False
    assert decision.checks["one_primary_plus_bloom_shape"] is True
    assert decision.checks["successor_receives_transformed_specimen"] is False


def test_bloom_cannot_precede_a_primary_filter() -> None:
    decision = compile_ordered_combination("experimental.bloom", "multiworld.translation")
    assert decision.status == CombinationStatus.INCOMPATIBLE
    assert decision.executable is False


def test_translation_then_possession_requires_transformed_state_or_reanalysis() -> None:
    decision = compile_ordered_combination("multiworld.translation", "multiworld.possession")
    assert decision.status == CombinationStatus.REQUIRES_REANALYSIS
    assert set(decision.shared_relationship_domains) == {"dialogue", "identity"}
    assert decision.executable is False


def test_unimplemented_filter_pair_is_unavailable() -> None:
    decision = compile_ordered_combination("multiworld.translation", "multiworld.bleed")
    assert decision.status == CombinationStatus.UNAVAILABLE


def test_registry_actively_rejects_an_unproven_stack() -> None:
    registry = default_filter_registry()
    with pytest.raises(ValueError, match="COMPATIBLE_UNPROVEN"):
        registry.validate_stack(["multiworld.translation", "experimental.bloom"])
    with pytest.raises(ValueError, match="INCOMPATIBLE"):
        registry.validate_stack(["experimental.bloom", "multiworld.translation"])


def test_recipe_cannot_persist_an_unproven_combination() -> None:
    with pytest.raises(ValueError, match="cannot execute"):
        FilterRecipe.create(
            "multiworld.translation",
            input_media_roles={"films": ["a.mp4", "b.mp4"]},
            selected_filter_stack=["multiworld.translation", "experimental.bloom"],
        )


def test_complete_ordered_matrix_is_deterministic_and_schema_valid(tmp_path: Path) -> None:
    first = compile_compatibility_matrix()
    second = compile_compatibility_matrix()
    registry = default_filter_registry()
    expected_pairs = len(registry.definitions()) * (len(registry.definitions()) - 1)
    assert first["ordered_pair_count"] == expected_pairs
    assert first["matrix_signature"] == second["matrix_signature"]
    assert sum(first["status_counts"].values()) == expected_pairs
    assert first["executable_pair_ids"] == []
    path = tmp_path / "filter_combination_compatibility_matrix.json"
    write_json(path, first)
    validate_artifact("filter_combination_compatibility_matrix", path, Path.cwd() / "schemas")
