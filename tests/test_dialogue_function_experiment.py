from pathlib import Path

from cinelingus.dialogue_function.experiment import run_function_schedule_screen
from cinelingus.dialogue_function.scheduling import FunctionMode, FunctionScheduleContext
from cinelingus.semantic import SemanticEntity, SemanticMode, SemanticScheduleContext
from cinelingus.validation import validate_artifact


def _classification(surface: str, function: str) -> dict:
    return {
        "axes": {
            "surface_form": {"labels": [{"label": surface, "confidence": 0.9}], "supported": True},
            "interaction_function": {"labels": [{"label": function, "confidence": 0.9}], "supported": True, "multi_label": True},
            "sequence_position": {"labels": [{"label": "unavailable", "confidence": 1.0}], "supported": False},
        }, "confidence": 0.9, "ambiguity_state": "UNAMBIGUOUS", "abstention": {"abstained": False},
    }


def _record(entity_id: str, surface: str, function: str) -> dict:
    return {"source_entity_id": entity_id, "classification": _classification(surface, function)}


def _semantic_context() -> SemanticScheduleContext:
    return SemanticScheduleContext(
        SemanticMode.REPORT_ONLY, 0.0,
        {
            "e1": SemanticEntity("s1", "source", "speech_passage", "en", (1.0, 0.0), {}),
            "e2": SemanticEntity("s2", "source", "speech_passage", "en", (0.8, 0.6), {}),
        },
        {"w1": SemanticEntity("d1", "destination", "speech_passage", "en", (1.0, 0.0), {})},
        {"model_id": "fixture"},
    )


def _function_context() -> FunctionScheduleContext:
    return FunctionScheduleContext(
        FunctionMode.REPORT_ONLY, 0.0,
        {"e1": _record("s1", "imperative", "command"), "e2": _record("s2", "interrogative", "request_information")},
        {"w1": _record("d1", "interrogative", "request_information")},
        {"taxonomy_version": "fixture", "classifier_version": "fixture"},
    )


def test_four_way_screen_preserves_zero_influence_and_blocks_uncalibrated_nominee(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": "c1.wav", "movie_timestamp": 0.0, "duration": 2.0, "transcript": "Leave now.", "confidence": 0.9, "usable": True},
        {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": "c2.wav", "movie_timestamp": 2.0, "duration": 2.0, "transcript": "Where are you?", "confidence": 0.9, "usable": True},
    ]
    windows = [{"id": "w1", "start": 0.0, "end": 2.0, "duration": 2.0, "transcript": "What happened?", "confidence": 0.9, "usable": True}]
    report = run_function_schedule_screen(
        clips=clips, windows=windows, semantic_evidence=_semantic_context(), function_evidence=_function_context(),
        output_dir=tmp_path, source_hash="source", destination_hash="destination", max_time_stretch=0.1,
        semantic_weight=0.05, function_weight=0.5,
    )

    assert all(report["invariants"][key] for key in (
        "report_only_selection_equivalent_to_semantic_only", "report_only_scores_equivalent_to_semantic_only",
        "zero_weight_selection_equivalent_to_semantic_only", "zero_weight_scores_equivalent_to_semantic_only",
    ))
    variants = {row["variant_id"]: row for row in report["variants"]}
    assert variants["function_preserving"]["placements_changed_from_report_only"] == 1
    assert variants["function_preserving"]["technical_regression_count"] == 0
    assert report["render_selection_state"] == "BLOCKED_PENDING_REVIEWED_CALIBRATION"
    assert report["counterfactuals"]["lower_cosine_right_function_candidates"]
    validate_artifact("function_schedule_screen", tmp_path / "function_schedule_screen.json", Path("schemas"))


def test_reviewed_calibration_unlocks_technically_safe_four_way_nomination(tmp_path: Path) -> None:
    report = run_function_schedule_screen(
        clips=[
            {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": "c1.wav", "movie_timestamp": 0.0, "duration": 2.0, "transcript": "Leave.", "confidence": 0.9, "usable": True},
            {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": "c2.wav", "movie_timestamp": 2.0, "duration": 2.0, "transcript": "Where?", "confidence": 0.9, "usable": True},
        ],
        windows=[{"id": "w1", "start": 0.0, "end": 2.0, "duration": 2.0, "transcript": "What?", "confidence": 0.9, "usable": True}],
        semantic_evidence=_semantic_context(), function_evidence=_function_context(), output_dir=tmp_path,
        source_hash="source", destination_hash="destination", max_time_stretch=0.1,
        semantic_weight=0.05, function_weight=0.5, calibration={"review_state": "COMPLETE"},
    )

    assert report["render_selection_state"] == "FOUR_WAY_CANDIDATE_SELECTED"
    assert report["render_selection"] == ["legacy_control", "semantic_only", "function_report_only", "function_preserving"]
