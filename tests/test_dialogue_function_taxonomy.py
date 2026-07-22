from pathlib import Path

from cinelingus.dialogue_function.taxonomy import AXES, TAXONOMY_PATH, TAXONOMY_VERSION, load_taxonomy, validate_taxonomy
from cinelingus.validation import validate_artifact


def test_phase3_taxonomy_is_complete_versioned_and_schema_valid() -> None:
    taxonomy = load_taxonomy()
    report = validate_taxonomy(taxonomy)

    assert report["status"] == "VALID"
    assert report["taxonomy_version"] == TAXONOMY_VERSION
    assert set(taxonomy["axes"]) == set(AXES)
    assert report["label_count"] == 39
    validate_artifact("dialogue_function_taxonomy", TAXONOMY_PATH, Path("schemas"))


def test_taxonomy_preserves_required_uncertainty_and_exclusions() -> None:
    taxonomy = load_taxonomy()
    interaction = {row["name"] for row in taxonomy["axes"]["interaction_function"]["labels"]}
    sequence = {row["name"] for row in taxonomy["axes"]["sequence_position"]["labels"]}

    assert {"unknown", "ambiguous", "not_applicable"} <= interaction
    assert "unavailable" in sequence
    assert "emotion" in taxonomy["scope"]
    assert taxonomy["migration_policy"]["mapping_requirement"]
