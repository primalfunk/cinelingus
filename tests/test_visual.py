from pathlib import Path

from movie_masher.validation import validate_artifact
from movie_masher.visual import build_shots_from_boundaries, build_visual_report


def test_build_shots_from_boundaries_filters_short_boundaries() -> None:
    shots = build_shots_from_boundaries([0.2, 1.0, 1.2, 3.5, 9.8], duration=10.0, min_shot_duration=0.5, confidence=0.35)

    assert [shot["id"] for shot in shots] == ["shot_000001", "shot_000002", "shot_000003"]
    assert shots[0]["start"] == 0.0
    assert shots[0]["end"] == 1.0
    assert shots[-1]["end"] == 10.0


def test_build_visual_report_writes_valid_schema(tmp_path: Path) -> None:
    shots = {
        "schema_version": "1.0",
        "tool_version": "test",
        "media_hash": "abc",
        "creation_timestamp": "now",
        "config_signature": "sig",
        "detector": "ffmpeg_scene_change",
        "threshold": 0.35,
        "min_shot_duration": 0.5,
        "shots": [
            {"id": "shot_000001", "start": 0.0, "end": 2.0, "duration": 2.0, "confidence": 0.35},
            {"id": "shot_000002", "start": 2.0, "end": 5.0, "duration": 3.0, "confidence": 0.35},
        ],
    }
    report_path = tmp_path / "visual_report.json"

    report = build_visual_report(shots_artifact=shots, movie={"duration": 5.0}, output_path=report_path)

    assert report["total_shots"] == 2
    assert report["average_shot_duration"] == 2.5
    validate_artifact("visual_report", report_path, Path.cwd() / "schemas")
