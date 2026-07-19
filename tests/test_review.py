from pathlib import Path

from cinelingus.validation import validate_artifact
from cinelingus.review import (
    REVIEW_LABEL_BAD_SHOT,
    REVIEW_LABEL_DISABLE,
    REVIEW_LABEL_VERY_FUNNY,
    FILTER_CROSS_SHOT,
    FILTER_DISABLED,
    FILTER_LOW_SCORE,
    FILTER_LOW_VISUAL_FIT,
    FILTER_RISKY,
    apply_review_label,
    build_review_notes,
    filtered_mapping_indices,
    review_row_values,
    review_summary,
    write_review_notes,
    apply_performance_review_label,
    build_performance_review_rows,
    filtered_performance_rows,
    performance_mapping_indices,
    performance_review_row_values,
    PERFORMANCE_FILTER_REUSED,
)


def test_filtered_mapping_indices_finds_risky_rows() -> None:
    mappings = [
        {"enabled": True, "score": 0.9, "visual_fit_score": 1.0},
        {"enabled": True, "score": 0.4, "visual_fit_score": 1.0},
        {"enabled": True, "score": 0.9, "visual_fit_score": 0.5},
        {"enabled": True, "score": 0.9, "visual_fit_score": 1.0, "mapping_crosses_shot_boundary": True},
        {"enabled": False, "score": 0.9, "visual_fit_score": 1.0},
    ]

    assert filtered_mapping_indices(mappings, FILTER_LOW_SCORE) == [1]
    assert filtered_mapping_indices(mappings, FILTER_LOW_VISUAL_FIT) == [2]
    assert filtered_mapping_indices(mappings, FILTER_CROSS_SHOT) == [3]
    assert filtered_mapping_indices(mappings, FILTER_DISABLED) == [4]
    assert filtered_mapping_indices(mappings, FILTER_RISKY) == [1, 2, 3, 4]


def test_review_row_values_includes_shot_diagnostics() -> None:
    row = review_row_values(
        {
            "enabled": True,
            "window_id": "w1",
            "clip_id": "c1",
            "destination_timestamp": 1.23456,
            "planned_render_duration": 2.34567,
            "score": 0.81234,
            "shot_id": "shot_000001",
            "visual_fit_score": 0.73456,
            "mapping_crosses_shot_boundary": True,
            "boundary_overrun_seconds": 0.45678,
            "timing_strategy": "trim_to_window",
            "source_speaker_id": "speaker_001",
            "destination_speaker_id": "speaker_002",
            "speaker_match_preserved": False,
            "speaker_fallback_reason": "no_same_speaker_fit",
            "source_transcript": "hello",
        }
    )

    assert row[:3] == ("yes", "w1", "c1")
    assert row[6] == "shot_000001"
    assert row[7] == 0.735
    assert row[8] == "yes"
    assert row[9] == 0.457
    assert row[11:15] == ("speaker_001", "speaker_002", "no", "no_same_speaker_fit")


def test_review_summary_reports_visual_risk_counts() -> None:
    summary = review_summary(
        [
            {"enabled": True, "visual_fit_score": 1.0},
            {"enabled": False, "visual_fit_score": 0.5, "mapping_crosses_shot_boundary": True},
        ],
        visible_count=1,
    )

    assert "1 shown" in summary
    assert "1 enabled" in summary
    assert "1 cross-shot" in summary
    assert "1 low visual fit" in summary


def test_apply_review_label_marks_and_can_disable_mapping() -> None:
    mappings = [{"enabled": True}, {"enabled": True}]

    apply_review_label(mappings, [0], REVIEW_LABEL_BAD_SHOT)
    apply_review_label(mappings, [1], REVIEW_LABEL_DISABLE)

    assert mappings[0]["review_label"] == REVIEW_LABEL_BAD_SHOT
    assert mappings[0]["enabled"] is True
    assert mappings[1]["review_label"] == REVIEW_LABEL_DISABLE
    assert mappings[1]["enabled"] is False


def test_positive_review_labels_are_available_and_persisted() -> None:
    mappings = [{"enabled": True}]

    apply_review_label(mappings, [0], REVIEW_LABEL_VERY_FUNNY)

    assert mappings[0]["review_label"] == REVIEW_LABEL_VERY_FUNNY
    assert mappings[0]["enabled"] is True


def test_write_review_notes_exports_valid_artifact(tmp_path: Path) -> None:
    schedule = {
        "media_hash": "dest",
        "mappings": [
            {
                "window_id": "w1",
                "clip_id": "c1",
                "review_label": REVIEW_LABEL_BAD_SHOT,
                "enabled": True,
                "destination_timestamp": 1.0,
                "score": 0.5,
                "visual_fit_score": 0.4,
                "mapping_crosses_shot_boundary": True,
            },
            {"window_id": "w2", "clip_id": "c2", "enabled": True},
        ],
    }
    output = tmp_path / "review_notes.json"

    data = write_review_notes(schedule, output, schedule_path=tmp_path / "replacement_schedule.json")

    assert data["reviewed_mappings"] == 1
    assert data["label_counts"][REVIEW_LABEL_BAD_SHOT] == 1
    validate_artifact("review_notes", output, Path.cwd() / "schemas")



def test_build_performance_review_rows_aggregates_schedule_context() -> None:
    schedule = {
        "destination_performance_fills": [
            {
                "destination_performance_id": "p1",
                "destination_performance_type": "exchange",
                "start": 10.0,
                "duration": 8.0,
                "coverage": 0.875,
                "target_coverage": 0.9,
                "stop_reason": "source_dialogue_exhausted",
                "source_performance_ids": ["sp1"],
            }
        ],
        "mappings": [
            {
                "destination_performance_id": "p1",
                "clip_id": "c1",
                "planned_render_duration": 3.0,
                "score": 0.8,
                "source_transcript": "hello there",
                "review_label": REVIEW_LABEL_BAD_SHOT,
                "source_speaker_id": "speaker_001",
                "destination_speaker_id": "speaker_001",
                "speaker_match_preserved": True,
            },
            {
                "destination_performance_id": "p1",
                "clip_id": "c2",
                "planned_render_duration": 4.0,
                "score": 0.6,
                "source_transcript": "general kenobi",
                "rescue_reused_clip": True,
                "source_speaker_id": "speaker_002",
                "destination_speaker_id": "speaker_001",
                "speaker_match_preserved": False,
                "speaker_fallback_reason": "timing_fit_overrode_speaker",
            },
        ],
    }

    row = build_performance_review_rows(schedule)[0]

    assert row["performance_id"] == "p1"
    assert row["mapping_indices"] == [0, 1]
    assert row["reuse_count"] == 1
    assert row["reviewed_count"] == 1
    assert row["average_score"] == 0.7
    assert row["speaker_match_rate"] == 0.5
    assert row["speaker_fallbacks"] == ["timing_fit_overrode_speaker"]
    assert row["risky"] is True
    assert "hello there" in row["transcript_preview"]
    values = performance_review_row_values(row)
    assert values[:5] == ("p1", "exchange", 10.0, 8.0, 0.875)
    assert values[9:13] == (0.5, "speaker_001,speaker_002", "speaker_001", "timing_fit_overrode_speaker")


def test_performance_review_filter_and_label_updates_underlying_mappings() -> None:
    schedule = {
        "mappings": [
            {"destination_performance_id": "p1", "enabled": True, "rescue_reused_clip": True},
            {"destination_performance_id": "p2", "enabled": True},
        ]
    }

    reused = filtered_performance_rows(schedule, PERFORMANCE_FILTER_REUSED)
    assert [row["performance_id"] for row in reused] == ["p1"]
    assert performance_mapping_indices(schedule, ["p1"]) == [0]

    apply_performance_review_label(schedule, ["p1"], REVIEW_LABEL_DISABLE)

    assert schedule["mappings"][0]["review_label"] == REVIEW_LABEL_DISABLE
    assert schedule["mappings"][0]["enabled"] is False
    assert "review_label" not in schedule["mappings"][1]
