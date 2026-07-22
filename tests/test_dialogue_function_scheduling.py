from pathlib import Path
import pytest

from cinelingus.dialogue_function import (
    FunctionClassifierConfig, FunctionMode, FunctionScheduleContext,
    RuleDialogueFunctionClassifier, apply_function_contribution, build_function_bundle,
    dialogue_function_compatibility,
)
from cinelingus.schedule import build_schedule


def _classification(label: str, surface: str = "declarative", confidence: float = 0.9) -> dict:
    return {
        "axes": {
            "surface_form": {"labels": [{"label": surface, "confidence": confidence}], "supported": True},
            "interaction_function": {"labels": [{"label": label, "confidence": confidence}], "supported": True, "multi_label": True},
            "sequence_position": {"labels": [{"label": "unavailable", "confidence": 1.0}], "supported": False},
        },
        "confidence": confidence, "ambiguity_state": "UNAMBIGUOUS", "abstention": {"abstained": False},
    }


def _annotated(classification: dict, mode: FunctionMode, weight: float) -> dict:
    return {
        "_function_classification": classification, "_function_entity_ids": ["p"],
        "_function_mode": mode.value, "_function_weight": weight,
        "_function_minimum_confidence": 0.62, "_function_identity": {"taxonomy_version": "v1"},
    }


def test_function_compatibility_is_separate_confidence_aware_and_sequence_neutral() -> None:
    source = _annotated(_classification("request_information", "interrogative"), FunctionMode.PRESERVING, 0.2)
    destination = _annotated(_classification("request_information", "interrogative"), FunctionMode.PRESERVING, 0.2)
    result = dialogue_function_compatibility(source, destination)

    assert result["normalized_function_contribution"] == 1.0
    assert result["per_axis_compatibility"]["sequence_position"]["supported"] is False
    assert result["scoring_policy"] == "dialogue_function_axis_compatibility_v1"
    assert apply_function_contribution(0.5, result) == pytest.approx(0.6)


def test_low_confidence_function_evidence_is_weakened_toward_neutral() -> None:
    source = _annotated(_classification("warning", confidence=0.4), FunctionMode.ASSISTED, 0.2)
    destination = _annotated(_classification("warning", confidence=0.4), FunctionMode.ASSISTED, 0.2)

    result = dialogue_function_compatibility(source, destination)

    assert 0.5 < result["normalized_function_contribution"] < 1.0
    assert result["warnings"]


def _model(film: str, reference: str, text: str) -> dict:
    return {
        "film_id": film, "created_from_signature": film + "_sig", "dialogue_turns": [],
        "speech_passages": [{"speech_passage_id": film + "_p", "start": 0.0, "end": 2.0, "duration": 2.0, "original_transcript": text, "language": "en", "provenance_id": film + "_prov", "source_transcript_reference": reference, "linked_performance_ids": []}],
    }


def test_report_only_and_zero_weight_preserve_schedule_selection(tmp_path: Path) -> None:
    source_model = _model("source", "c1", "Where are you?")
    destination_model = _model("destination", "w1", "What happened?")
    classifier = RuleDialogueFunctionClassifier(FunctionClassifierConfig())
    source_bundle = build_function_bundle(source_model, tmp_path / "source", classifier)
    destination_bundle = build_function_bundle(destination_model, tmp_path / "destination", classifier)
    clips = [
        {"id": "c1", "event_id": "c1", "event_ids": ["c1"], "path": "c1.wav", "movie_timestamp": 0.0, "duration": 2.0, "transcript": "Where are you?", "confidence": 0.9, "usable": True},
        {"id": "c2", "event_id": "c2", "event_ids": ["c2"], "path": "c2.wav", "movie_timestamp": 4.0, "duration": 2.0, "transcript": "Leave now.", "confidence": 0.9, "usable": True},
    ]
    windows = [{"id": "w1", "start": 0.0, "end": 2.0, "duration": 2.0, "transcript": "What happened?", "confidence": 0.9, "usable": True}]
    common = dict(clips=clips, windows=windows, source_hash="s", destination_hash="d", max_time_stretch=0.2, scheduling_mode="best_fit", best_fit_lookahead=2)
    control = build_schedule(**common, output_path=tmp_path / "control.json")
    report_context = FunctionScheduleContext.from_bundles(mode=FunctionMode.REPORT_ONLY, weight=0.0, source_model=source_model, source_bundle=source_bundle, destination_model=destination_model, destination_bundle=destination_bundle)
    report = build_schedule(**common, output_path=tmp_path / "report.json", function_context=report_context)
    zero_context = FunctionScheduleContext.from_bundles(mode=FunctionMode.ASSISTED, weight=0.0, source_model=source_model, source_bundle=source_bundle, destination_model=destination_model, destination_bundle=destination_bundle)
    zero = build_schedule(**common, output_path=tmp_path / "zero.json", function_context=zero_context)

    assert [row["clip_id"] for row in control["mappings"]] == [row["clip_id"] for row in report["mappings"]] == [row["clip_id"] for row in zero["mappings"]]
    assert [row["score"] for row in control["mappings"]] == [row["score"] for row in report["mappings"]] == [row["score"] for row in zero["mappings"]]
    assert report["dialogue_function_scoring"]["mode"] == FunctionMode.REPORT_ONLY.value
    assert "dialogue_function_scoring" not in zero
