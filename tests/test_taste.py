from pathlib import Path

from movie_masher.taste import build_editorial_highlights, default_taste_profile
from movie_masher.validation import validate_artifact


def _schedule(review_label: str | None = None) -> dict:
    mapping = {
        "destination_performance_id": "p1",
        "source_performance_id": "sp1",
        "window_id": "w1",
        "clip_id": "c1",
        "enabled": True,
        "destination_timestamp": 4.0,
        "planned_render_duration": 3.0,
        "score": 0.72,
        "performance_similarity_score": 0.78,
        "performance_dialogue_density": 0.61,
        "stretch_factor": 1.02,
        "selection_reason": "performance_signature_match",
        "performance_similarity_components": {
            "contrast_bonus": 0.35,
            "pause": 0.82,
        },
    }
    if review_label:
        mapping["review_label"] = review_label
    return {
        "media_hash": "desthash",
        "mappings": [mapping],
        "destination_performance_fills": [
            {
                "destination_performance_id": "p1",
                "start": 4.0,
                "duration": 3.5,
                "coverage": 0.9,
            }
        ],
    }


def test_default_taste_profile_exports_valid_artifact(tmp_path: Path) -> None:
    output = tmp_path / "taste_profile.json"

    profile = default_taste_profile(output_path=output)

    assert profile["profile_name"] == "default_absurd_but_watchable"
    validate_artifact("taste_profile", output, Path.cwd() / "schemas")


def test_editorial_highlights_exports_valid_performance_artifact(tmp_path: Path) -> None:
    output = tmp_path / "editorial_highlights.json"
    diagnostics = {
        "diagnostics": [
            {
                "destination_performance_id": "p1",
                "average_similarity_score": 0.8,
                "highest_stretch_delta": 0.02,
                "reuse_count": 0,
                "warnings": [],
            }
        ]
    }

    artifact = build_editorial_highlights(
        schedule=_schedule(),
        performance_diagnostics=diagnostics,
        output_path=output,
    )

    assert artifact["summary"]["evaluated_performances"] == 1
    assert artifact["performances"][0]["performance_id"] == "p1"
    assert artifact["performances"][0]["editorial_score"] > 0
    assert artifact["highlights"]["most_convincing"]
    validate_artifact("editorial_highlights", output, Path.cwd() / "schemas")


def test_editorial_highlights_respect_positive_and_negative_review_labels(tmp_path: Path) -> None:
    positive = build_editorial_highlights(
        schedule=_schedule("very_funny"),
        output_path=tmp_path / "positive.json",
    )
    negative = build_editorial_highlights(
        schedule=_schedule("poor_match"),
        output_path=tmp_path / "negative.json",
    )

    assert positive["performances"][0]["review_status"] == "positively_reviewed"
    assert negative["performances"][0]["review_status"] == "needs_revision"
    assert positive["performances"][0]["components"]["comedic_potential"] > negative["performances"][0]["components"]["comedic_potential"]
