from pathlib import Path

from movie_masher.performance_report import build_performance_placement_report
from movie_masher.validation import validate_artifact


def test_build_performance_placement_report_scores_and_writes_files(tmp_path: Path) -> None:
    schedule = {
        "media_hash": "desthash",
        "scheduling_mode": "whole_line_fill",
        "destination_performance_fills": [
            {
                "destination_performance_id": "dp1",
                "destination_performance_type": "exchange",
                "duration": 2.5,
                "scheduled_duration": 2.0,
                "coverage": 0.8,
                "target_coverage": 0.75,
                "mapping_count": 2,
                "source_performance_ids": ["sp1"],
                "stop_reason": "target_coverage_met",
            }
        ],
        "performance_placements": [
            {
                "source_performance_id": "sp1",
                "source_performance_type": "exchange",
                "destination_performance_id": "dp1",
                "destination_performance_type": "exchange",
                "clip_ids": ["c1", "c2"],
                "mapping_count": 2,
                "scheduled_duration": 2.0,
            }
        ],
        "mappings": [
            {
                "clip_id": "c1",
                "source_performance_id": "sp1",
                "destination_performance_id": "dp1",
                "planned_render_duration": 1.0,
                "score": 0.8,
                "visual_fit_score": 0.9,
            },
            {
                "clip_id": "c2",
                "source_performance_id": "sp1",
                "destination_performance_id": "dp1",
                "planned_render_duration": 1.0,
                "score": 0.7,
                "visual_fit_score": 0.8,
            },
        ],
    }
    source_performances = {
        "performances": [
            {"id": "sp1", "duration": 2.0, "conversation_type": "exchange", "dialogue_density": 0.8}
        ]
    }
    destination_performances = {
        "performances": [
            {"id": "dp1", "duration": 2.5, "conversation_type": "exchange", "dialogue_density": 0.75}
        ]
    }

    report = build_performance_placement_report(
        schedule=schedule,
        source_performances=source_performances,
        destination_performances=destination_performances,
        output_json=tmp_path / "performance_placement_report.json",
        output_csv=tmp_path / "performance_placement_report.csv",
        output_txt=tmp_path / "performance_placement_report.txt",
    )

    assert report["placement_count"] == 1
    assert report["placements"][0]["quality_score"] > 0.75
    assert report["summary"]["warning_count"] == 0
    assert report["destination_fills"][0]["fill_quality"] == 1.0
    assert (tmp_path / "performance_placement_report.csv").exists()
    assert "Performance Placement Report" in (tmp_path / "performance_placement_report.txt").read_text()
    validate_artifact("performance_placement_report", tmp_path / "performance_placement_report.json", Path.cwd() / "schemas")
