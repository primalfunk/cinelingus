from copy import deepcopy
from pathlib import Path

from cinelingus.semantic import DeterministicFakeProvider, SemanticConfig, build_semantic_bundle
from cinelingus.util import read_json, write_json
from cinelingus.validation import validate_artifact


def _model() -> dict:
    return {
        "film_id": "film_fixture", "created_from_signature": "model-signature",
        "speech_passages": [
            {"speech_passage_id": "speech_a", "start": 0.0, "end": 1.0, "duration": 1.0, "original_transcript": "yes", "language": "en", "provenance_id": "provenance_a"},
            {"speech_passage_id": "speech_b", "start": 1.0, "end": 3.0, "duration": 2.0, "original_transcript": "one two three four five", "language": None, "provenance_id": "provenance_b"},
        ],
    }


class CountingProvider(DeterministicFakeProvider):
    def __init__(self, config: SemanticConfig):
        super().__init__(config)
        self.encoded = 0

    def encode(self, texts, *, role):
        self.encoded += len(texts)
        return super().encode(texts, role=role)


def test_bundle_uses_separate_float32_vectors_and_accounts_for_every_passage(tmp_path: Path) -> None:
    config = SemanticConfig(dimensions=8, token_limit=4)
    result = build_semantic_bundle(_model(), tmp_path, DeterministicFakeProvider(config), config, batch_size=1)
    assert result.validation_report["status"] == "VALID"
    assert result.bundle["construction_state"] == "READY"
    assert [row["embedding_status"] for row in result.bundle["entities"]] == ["LOW_INFORMATION", "TRUNCATED"]
    assert all((tmp_path / row["vector_locator"]).stat().st_size == 32 for row in result.bundle["entities"])
    metadata = read_json(tmp_path / "semantic_bundle.json")
    assert all(not any(isinstance(value, list) for value in row.values()) for row in metadata["entities"])
    validate_artifact("semantic_bundle", tmp_path / "semantic_bundle.json", Path("schemas"))


def test_bundle_resumes_completed_entities_and_ignores_scheduling_mode(tmp_path: Path) -> None:
    base = SemanticConfig(dimensions=8)
    first_provider = CountingProvider(base)
    first = build_semantic_bundle(_model(), tmp_path, first_provider, base)
    assert first_provider.encoded == 2
    report_only = SemanticConfig(mode="SEMANTIC_REPORT_ONLY", dimensions=8)
    assert report_only.configuration_signature == base.configuration_signature
    second_provider = CountingProvider(report_only)
    second = build_semantic_bundle(_model(), tmp_path, second_provider, report_only)
    assert second_provider.encoded == 0
    assert second.cache_report["cache_hits"] == 2


def test_relevant_transcript_change_reencodes_only_that_entity(tmp_path: Path) -> None:
    config = SemanticConfig(dimensions=8)
    build_semantic_bundle(_model(), tmp_path, CountingProvider(config), config)
    changed = deepcopy(_model())
    changed["speech_passages"][1]["original_transcript"] = "changed transcript"
    provider = CountingProvider(config)
    result = build_semantic_bundle(changed, tmp_path, provider, config)
    assert provider.encoded == 1
    assert result.cache_report["cache_hits"] == 1


def test_fake_vectors_cannot_be_reused_by_a_different_provider(tmp_path: Path) -> None:
    from cinelingus.semantic import UnavailableProvider
    config = SemanticConfig(dimensions=8)
    build_semantic_bundle(_model(), tmp_path, DeterministicFakeProvider(config), config)
    result = build_semantic_bundle(_model(), tmp_path, UnavailableProvider(config), config)
    assert result.cache_report["cache_hits"] == 0
    assert {row["embedding_status"] for row in result.bundle["entities"]} == {"UNAVAILABLE"}


def test_building_checkpoint_never_validates_as_ready(tmp_path: Path) -> None:
    config = SemanticConfig(dimensions=8)
    result = build_semantic_bundle(_model(), tmp_path, DeterministicFakeProvider(config), config)
    partial = deepcopy(result.bundle)
    partial["construction_state"] = "BUILDING"
    partial["entities"].pop()
    write_json(tmp_path / "semantic_bundle.json", partial)
    from cinelingus.semantic import validate_semantic_bundle
    report = validate_semantic_bundle(read_json(tmp_path / "semantic_bundle.json"), tmp_path, _model())
    assert report["status"] == "INVALID"
    assert {row["category"] for row in report["errors"]} == {"COVERAGE", "STATE"}


def _model_with_ordered_turns() -> dict:
    model = _model()
    model["dialogue_turns"] = [{
        "dialogue_turn_id": "turn_a",
        "ordered_speech_passage_references": ["speech_a", "speech_b"],
        "start": 0.0, "end": 3.0, "duration": 3.0,
        "provenance_id": "turn_provenance",
    }]
    model["performances"] = [{
        "performance_id": "performance_a",
        "dialogue_turn_references": ["turn_a"],
        "start": 0.0, "end": 3.0, "duration": 3.0,
        "provenance_id": "performance_provenance",
    }]
    return model


def test_experimental_turn_and_sequence_bundles_use_only_ordered_structure(tmp_path: Path) -> None:
    model = _model_with_ordered_turns()
    config = SemanticConfig(dimensions=8)
    turns = build_semantic_bundle(
        model, tmp_path / "turns", DeterministicFakeProvider(config), config,
        entity_type="dialogue_turn",
    )
    sequences = build_semantic_bundle(
        model, tmp_path / "sequences", DeterministicFakeProvider(config), config,
        entity_type="turn_sequence",
    )

    assert turns.validation_report["status"] == sequences.validation_report["status"] == "VALID"
    assert turns.bundle["entities"][0]["source_entity_id"] == "turn_a"
    assert turns.bundle["entities"][0]["structural_references"] == {
        "ordered_speech_passage_ids": ["speech_a", "speech_b"]
    }
    assert sequences.bundle["entities"][0]["source_entity_id"] == "performance_a"
    assert sequences.bundle["entities"][0]["structural_references"] == {
        "ordered_dialogue_turn_ids": ["turn_a"]
    }
    validate_artifact("semantic_bundle", tmp_path / "turns" / "semantic_bundle.json", Path("schemas"))
    validate_artifact("semantic_bundle", tmp_path / "sequences" / "semantic_bundle.json", Path("schemas"))


def test_experimental_semantics_excludes_invalid_structure_without_fabricating_it(tmp_path: Path) -> None:
    model = _model_with_ordered_turns()
    model["dialogue_turns"][0]["ordered_speech_passage_references"] = ["missing_passage"]
    config = SemanticConfig(dimensions=8)

    result = build_semantic_bundle(
        model, tmp_path, DeterministicFakeProvider(config), config,
        entity_type="dialogue_turn",
    )

    assert result.validation_report["status"] == "VALID"
    assert result.bundle["entities"] == []
    assert result.bundle["structural_exclusions"] == [{
        "source_entity_type": "dialogue_turn",
        "source_entity_id": "turn_a",
        "reasons": ["missing_passage_reference"],
    }]
