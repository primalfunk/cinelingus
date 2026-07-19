from pathlib import Path

from cinelingus.validation import validate_artifact
from cinelingus.visual import (
    build_boundary_stability,
    build_gradual_transition_candidates,
    build_shots_from_boundaries,
    build_stillness_intervals,
    build_visual_report,
    parse_black_intervals,
    parse_frame_difference_samples,
)


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


def test_black_intervals_receive_conservative_fade_guards() -> None:
    rows = parse_black_intervals(
        "[blackdetect] black_start:10.0 black_end:11.5 black_duration:1.5",
        duration=20.0,
    )

    assert rows == [{
        "id": "transition_000001",
        "kind": "BLACK_OR_FADE_GUARD",
        "start": 9.75,
        "end": 11.75,
        "detected_black_start": 10.0,
        "detected_black_end": 11.5,
        "confidence": 0.7,
        "capability_tag": "CORE_HEURISTIC",
        "detector": "ffmpeg_blackdetect_guard_v1",
    }]


def test_frame_difference_samples_are_normalized_literal_evidence() -> None:
    text = """
frame:0 pts:0 pts_time:0.000
lavfi.signalstats.YAVG=12.75
frame:1 pts:1 pts_time:0.250
lavfi.signalstats.YAVG=25.5
"""

    assert parse_frame_difference_samples(text) == [
        {"time": 0.0, "frame_difference": 0.05},
        {"time": 0.25, "frame_difference": 0.1},
    ]


def test_boundary_stability_samples_both_sides_and_reports_unavailable_evidence() -> None:
    rows = build_boundary_stability(
        [3.0, 9.0],
        [
            {"time": 2.5, "frame_difference": 0.02},
            {"time": 3.5, "frame_difference": 0.04},
        ],
    )

    assert rows[0]["status"] == "AVAILABLE"
    assert rows[0]["low_boundary_motion"] is True
    assert rows[0]["frame_difference_magnitude"] == 0.03
    assert rows[1]["status"] == "UNAVAILABLE"
    assert rows[1]["capability_tag"] == "FALLBACK_INFERENCE"


def test_sustained_low_difference_becomes_literal_stillness_evidence() -> None:
    rows = build_stillness_intervals(
        [{"time": time, "frame_difference": 0.02} for time in (1.0, 1.25, 1.5, 1.75)],
        duration=5.0,
    )

    assert rows[0]["kind"] == "SUSTAINED_LOW_FRAME_DIFFERENCE"
    assert rows[0]["start"] == 0.875
    assert rows[0]["end"] == 1.875


def test_sustained_change_is_only_a_gradual_transition_candidate() -> None:
    rows = build_gradual_transition_candidates(
        [{"time": time, "frame_difference": 0.12} for time in (2.0, 2.25, 2.5)],
        duration=5.0,
    )

    assert rows[0]["kind"] == "GRADUAL_TRANSITION_CANDIDATE"
    assert "DISSOLVE" not in str(rows[0]).upper()
