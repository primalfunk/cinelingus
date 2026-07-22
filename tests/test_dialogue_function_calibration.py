from pathlib import Path

from cinelingus.dialogue_function import RuleDialogueFunctionClassifier, build_function_bundle
from cinelingus.dialogue_function.calibration import finalize_calibration_review, prepare_calibration_review
from cinelingus.util import read_json, write_json


def test_calibration_proposals_are_not_ground_truth_and_human_review_is_preserved(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model = {"film_id": "film", "created_from_signature": "sig", "dialogue_turns": [], "speech_passages": [
        {"speech_passage_id": "p1", "start": 0.0, "end": 1.0, "duration": 1.0, "original_transcript": "Where are you?", "language": "en", "provenance_id": "prov1", "source_transcript_reference": "w1", "linked_performance_ids": []},
        {"speech_passage_id": "p2", "start": 1.0, "end": 2.0, "duration": 1.0, "original_transcript": "I won't help you.", "language": "en", "provenance_id": "prov2", "source_transcript_reference": "w2", "linked_performance_ids": []},
    ]}
    write_json(model_path, model)
    bundle_dir = tmp_path / "bundle"
    build_function_bundle(model, bundle_dir, RuleDialogueFunctionClassifier())
    package = tmp_path / "calibration"
    manifest = prepare_calibration_review([{"case_id": "case", "media_class": "live_action", "model_path": model_path, "function_bundle_path": bundle_dir / "dialogue_function_bundle.json"}], package)

    assert manifest["review_state"] == "PENDING_HUMAN_ANNOTATION"
    assert all(row["proposal_is_ground_truth"] is False for row in manifest["samples"])
    assert (package / "calibration_review.md").is_file()
    annotations = read_json(package / "calibration_annotations.json")
    for row in annotations["samples"]:
        row["annotations"] = [{
            "annotator_id": "human", "axes": {"surface_form": ["interrogative"], "interaction_function": ["request_information"], "sequence_position": ["unavailable"]},
            "annotator_confidence": 0.8, "ambiguity_state": "AMBIGUOUS", "notes": "Context is limited.",
        }]
    write_json(package / "calibration_annotations.json", annotations)
    result = finalize_calibration_review(package)

    assert result["review_state"] == "COMPLETE"
    assert result["metrics"]["ambiguity_count"] == result["sample_count"]
    assert result["samples"][0]["human_annotations"][0]["annotator_id"] == "human"
    assert result["samples"][0]["disagreement_state"] == "NOT_MEASURED"
    metrics = result["metrics"]
    assert metrics["metrics_version"] == "dialogue_function_calibration_metrics_v2"
    assert metrics["axis_metrics"]["interaction_function"]["annotation_comparison_count"] == result["sample_count"]
    assert "request_information" in metrics["axis_metrics"]["interaction_function"]["per_label"]
    assert len(metrics["confidence_calibration"]) == 5
    assert metrics["abstention_analysis"]["annotation_comparison_count"] == result["sample_count"]
