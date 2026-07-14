from pathlib import Path

from movie_masher.performance_diagnostics import build_performance_diagnostics
from movie_masher.validation import validate_artifact


def test_build_performance_diagnostics_explains_low_scoring_matches(tmp_path: Path) -> None:
    schedule = {
        "media_hash": "dest",
        "scheduling_mode": "whole_line_fill",
        "active_filter": "balanced",
        "destination_performance_fills": [
            {
                "destination_performance_id": "p1",
                "coverage": 0.4,
                "target_coverage": 0.9,
                "stop_reason": "remaining_gap_has_no_fitting_whole_line",
            }
        ],
        "mappings": [
            {
                "enabled": True,
                "destination_performance_id": "p1",
                "performance_type": "exchange",
                "source_performance_id": "sp1",
                "clip_id": "c1",
                "planned_render_duration": 1.0,
                "performance_similarity_score": 0.42,
                "performance_similarity_components": {
                    "speaker_pattern": 0.3,
                    "dialogue_density": 0.45,
                    "pause": 0.4,
                },
                "filter_weights": {"speaker_pattern": 0.08, "dialogue_density": 0.1},
                "stretch_factor": 1.12,
                "matching_rationale": "weak speaker_pattern",
            }
        ],
    }

    artifact = build_performance_diagnostics(schedule=schedule, output_path=tmp_path / "performance_diagnostics.json")

    assert artifact["summary"]["low_similarity_count"] == 1
    row = artifact["diagnostics"][0]
    assert row["destination_performance_id"] == "p1"
    assert row["average_similarity_score"] == 0.42
    assert "speaker_pattern_mismatch" in row["warnings"]
    assert "under_target_coverage" in row["warnings"]
    assert "high_stretch" in row["warnings"]
    validate_artifact("performance_diagnostics", tmp_path / "performance_diagnostics.json", Path.cwd() / "schemas")
