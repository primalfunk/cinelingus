from pathlib import Path

from cinelingus.dialogue_function import FunctionClassifierConfig, RuleDialogueFunctionClassifier, build_function_bundle, validate_function_bundle
from cinelingus.validation import validate_artifact


def _model() -> dict:
    return {
        "film_id": "film_fixture", "created_from_signature": "model_signature",
        "speech_passages": [
            {"speech_passage_id": "p1", "start": 0.0, "end": 1.0, "duration": 1.0, "original_transcript": "Where are you?", "language": "en", "provenance_id": "prov1", "source_transcript_reference": "w1", "linked_dialogue_turn_id": "t1", "linked_performance_ids": ["perf1"]},
            {"speech_passage_id": "p2", "start": 1.0, "end": 2.0, "duration": 1.0, "original_transcript": "I am here.", "language": "en", "provenance_id": "prov2", "source_transcript_reference": "w2", "linked_dialogue_turn_id": "t2", "linked_performance_ids": ["perf1"]},
        ],
        "dialogue_turns": [
            {"dialogue_turn_id": "t1", "ordered_speech_passage_references": ["p1"], "preceding_turn_reference": None, "following_turn_reference": "t2"},
            {"dialogue_turn_id": "t2", "ordered_speech_passage_references": ["p2"], "preceding_turn_reference": "t1", "following_turn_reference": None},
        ],
        "performances": [{"performance_id": "perf1", "start": 0.0, "end": 2.0, "duration": 2.0, "provenance_id": "perf-prov", "dialogue_turn_references": ["t1", "t2"], "speaker_sequence": ["speaker-a", "speaker-b"]}],
    }


def test_function_bundle_classifies_every_passage_and_resumes(tmp_path: Path) -> None:
    model = _model()
    classifier = RuleDialogueFunctionClassifier(FunctionClassifierConfig(context_mode="dialogue_turn"))
    first = build_function_bundle(model, tmp_path, classifier)
    second = build_function_bundle(model, tmp_path, classifier)

    assert first["construction_state"] == "READY"
    assert first["coverage"]["accounted_entity_count"] == 2
    assert first["coverage"]["sequence_position_available_count"] == 2
    assert first["coverage"]["source_turn_count"] == 2
    assert first["coverage"]["turn_aggregate_available_count"] == 2
    assert first["turns"][0]["aggregation_state"] == "AVAILABLE"
    assert first["turns"][0]["axes"]["surface_form"]["primary_label"] == "interrogative"
    assert first["turns"][0]["axes"]["sequence_position"]["primary_label"] == "initiating"
    assert first["coverage"]["function_sequence_available_count"] == 1
    assert [row["dialogue_turn_id"] for row in first["sequences"][0]["function_sequence"]] == ["t1", "t2"]
    assert first["sequences"][0]["representation_policy"].endswith("no_flattening")
    assert second["cache_report"]["cache_hits"] == 2
    assert second["cache_report"]["resume_used"] is True
    assert validate_function_bundle(second, model, classifier=classifier)["status"] == "VALID"
    validate_artifact("dialogue_function_bundle", tmp_path / "dialogue_function_bundle.json", Path("schemas"))


def test_bundle_invalidates_entity_when_adjacent_context_changes(tmp_path: Path) -> None:
    model = _model()
    classifier = RuleDialogueFunctionClassifier(FunctionClassifierConfig(context_mode="adjacent_passages"))
    build_function_bundle(model, tmp_path, classifier)
    model["speech_passages"][0]["original_transcript"] = "Why are you here?"
    model["created_from_signature"] = "changed_model_signature"
    rebuilt = build_function_bundle(model, tmp_path, classifier)

    assert rebuilt["cache_report"]["cache_hits"] == 0


def test_bundle_does_not_classify_uncalibrated_language(tmp_path: Path) -> None:
    model = _model()
    model["speech_passages"][0]["language"] = "es"
    bundle = build_function_bundle(model, tmp_path, RuleDialogueFunctionClassifier())

    row = next(row for row in bundle["entities"] if row["source_entity_id"] == "p1")
    assert row["classification_state"] == "UNAVAILABLE"
    assert row["classification"]["abstention"]["reason"] == "LANGUAGE_OUTSIDE_CALIBRATED_SCOPE"
