from pathlib import Path

from cinelingus.dialogue_function.render_verification import evaluate_rendered_function
from cinelingus.validation import validate_artifact
from cinelingus.util import write_json


def _classification(surface: str, function: str, confidence: float = 0.9) -> dict:
    return {
        "axes": {
            "surface_form": {"labels": [{"label": surface, "confidence": confidence}], "supported": True},
            "interaction_function": {"labels": [{"label": function, "confidence": confidence}], "supported": True},
            "sequence_position": {"labels": [{"label": "unavailable", "confidence": 1.0}], "supported": False},
        },
        "confidence": confidence, "ambiguity_state": "UNAMBIGUOUS", "abstention": {"abstained": False},
    }


def _mapping(clip_id: str, function: str = "request_information") -> dict:
    source = _classification("interrogative", function)
    destination = _classification("interrogative", "request_information")
    return {
        "enabled": True, "window_id": "w1", "clip_id": clip_id, "source_performance_id": clip_id,
        "source_transcript": "Where are you?", "destination_timestamp": 0.0, "planned_render_duration": 2.0,
        "score_components": {"duration_similarity": 1.0}, "performance_similarity_score": 0.8,
        "speaker_match_preserved": True, "cinematic_compatibility_categories": {"visual": 0.8},
        "dialogue_function_compatibility": {
            "available": True, "source_distribution": source, "destination_distribution": destination,
            "normalized_function_contribution": 1.0 if function == "request_information" else 0.0,
            "confidence": 0.9,
        },
    }


def test_rendered_function_reclassifies_actual_transcript_and_validates(tmp_path: Path) -> None:
    control = {"mappings": [_mapping("old", "command")]}
    schedule = {"mappings": [_mapping("new")]}
    report = evaluate_rendered_function(
        schedule=schedule, baseline_schedule=control,
        rendered_dialogue_verification={"mappings": [{
            "mapping_index": 0, "window_id": "w1", "rendered_transcript": "Where are you?",
            "confidence": 0.92, "word_coverage_percentage": 100.0, "status": "pass",
        }]}, calibration={"review_state": "COMPLETE"},
    )

    assert report["status"] == "PASS"
    assert report["claim_state"] == "ELIGIBLE"
    assert report["mappings"][0]["verification_state"] == "VERIFIED"
    assert report["mappings"][0]["rendered_transcript_function"]["axes"]["interaction_function"]["labels"][0]["label"] == "request_information"
    path = tmp_path / "function_render_verification.json"
    write_json(path, report)
    validate_artifact("function_render_verification", path, Path("schemas"))


def test_rendered_function_distinguishes_mismatch_from_low_confidence() -> None:
    schedule = {"mappings": [_mapping("new")]}
    mismatch = evaluate_rendered_function(
        schedule=schedule,
        rendered_dialogue_verification={"mappings": [{
            "mapping_index": 0, "window_id": "w1", "rendered_transcript": "Leave now.",
            "confidence": 0.9, "word_coverage_percentage": 100.0, "status": "pass",
        }]},
    )
    uncertain = evaluate_rendered_function(
        schedule=schedule,
        rendered_dialogue_verification={"mappings": [{
            "mapping_index": 0, "window_id": "w1", "rendered_transcript": "Where are you?",
            "confidence": 0.1, "word_coverage_percentage": 100.0, "status": "pass",
        }]},
    )

    assert mismatch["status"] == "FAIL"
    assert mismatch["mappings"][0]["verification_state"] == "FUNCTION_MISMATCH"
    assert uncertain["status"] == "WARN"
    assert uncertain["mappings"][0]["verification_state"] == "UNVERIFIABLE"
    assert uncertain["claim_state"] == "PROVISIONAL_PENDING_REVIEWED_CALIBRATION"
