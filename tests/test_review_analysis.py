from pathlib import Path

from cinelingus.review import REVIEW_LABEL_BAD_SHOT, REVIEW_LABEL_GOOD
from cinelingus.review_analysis import build_review_analysis, infer_mapping_causes
from cinelingus.validation import validate_artifact


def test_infer_mapping_causes_flags_measurable_risks() -> None:
    causes = infer_mapping_causes(
        {
            "score": 0.4,
            "visual_fit_score": 0.5,
            "mapping_crosses_shot_boundary": True,
            "boundary_overrun_seconds": 0.8,
            "timing_strategy": "trim_to_window",
            "enabled": False,
        }
    )

    assert "low_score" in causes
    assert "low_visual_fit" in causes
    assert "crosses_shot_boundary" in causes
    assert "large_boundary_overrun" in causes
    assert "long_trim" in causes
    assert "disabled" in causes


def test_build_review_analysis_writes_valid_artifact(tmp_path: Path) -> None:
    schedule = {
        "media_hash": "dest",
        "mappings": [
            {
                "window_id": "w1",
                "clip_id": "c1",
                "score": 0.4,
                "visual_fit_score": 0.5,
                "mapping_crosses_shot_boundary": True,
                "boundary_overrun_seconds": 0.8,
                "timing_strategy": "trim_to_window",
                "enabled": True,
            },
            {"window_id": "w2", "clip_id": "c2", "score": 0.9, "visual_fit_score": 1.0, "enabled": True},
        ],
    }
    review_notes = {
        "media_hash": "dest",
        "label_counts": {REVIEW_LABEL_BAD_SHOT: 1, REVIEW_LABEL_GOOD: 1},
        "notes": [
            {"mapping_index": 0, "review_label": REVIEW_LABEL_BAD_SHOT},
            {"mapping_index": 1, "review_label": REVIEW_LABEL_GOOD},
        ],
    }
    output = tmp_path / "review_analysis.json"

    analysis = build_review_analysis(review_notes=review_notes, schedule=schedule, output_path=output)

    assert analysis["reviewed_mappings"] == 2
    assert analysis["bad_mappings"] == 1
    assert analysis["good_mappings"] == 1
    assert analysis["cause_counts"]["crosses_shot_boundary"] == 1
    assert analysis["recommendations"]
    validate_artifact("review_analysis", output, Path.cwd() / "schemas")
