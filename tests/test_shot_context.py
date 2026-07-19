from pathlib import Path

from cinelingus.shot_context import annotate_windows_with_shots, build_visual_schedule_report
from cinelingus.validation import validate_artifact


def test_annotate_windows_assigns_primary_shot_and_boundary_overlap() -> None:
    shots = [
        {"id": "shot_000001", "start": 0.0, "end": 2.0, "duration": 2.0},
        {"id": "shot_000002", "start": 2.0, "end": 5.0, "duration": 3.0},
    ]
    windows = [{"id": "w1", "start": 1.5, "end": 3.5, "duration": 2.0, "confidence": 0.9}]

    annotated = annotate_windows_with_shots(windows, shots)

    assert annotated[0]["shot_id"] == "shot_000002"
    assert annotated[0]["crosses_shot_boundary"] is True
    assert annotated[0]["boundary_overlap_seconds"] == 0.5


def test_build_visual_schedule_report_counts_crossings_and_empty_shots(tmp_path: Path) -> None:
    shots = {
        "media_hash": "dest",
        "shots": [
            {"id": "shot_000001", "start": 0.0, "end": 2.0, "duration": 2.0},
            {"id": "shot_000002", "start": 2.0, "end": 5.0, "duration": 3.0},
        ],
    }
    timeline = {"windows": [{"id": "w1", "shot_id": "shot_000001"}]}
    schedule = {
        "media_hash": "dest",
        "shot_boundary_mode": "soft",
        "mappings": [
            {
                "window_id": "w1",
                "clip_id": "c1",
                "shot_id": "shot_000001",
                "mapping_crosses_shot_boundary": True,
                "boundary_overrun_seconds": 0.25,
                "visual_fit_score": 0.5,
                "score": 0.7,
            }
        ],
    }
    output = tmp_path / "visual_schedule_report.json"

    report = build_visual_schedule_report(shots_artifact=shots, timeline=timeline, schedule=schedule, output_path=output)

    assert report["total_shots"] == 2
    assert report["empty_dialogue_shots"] == ["shot_000002"]
    assert len(report["crossing_mappings"]) == 1
    validate_artifact("visual_schedule_report", output, Path.cwd() / "schemas")
