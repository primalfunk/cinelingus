from pathlib import Path

from cinelingus.schedule import _append_source_exhaustion_reuse_fill_speech_slots, _append_undercovered_speech_slot_fill, _reanchor_single_slot_mappings_to_speech_start, build_schedule


def test_build_schedule_keeps_order_and_stops(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "path": "c1.wav", "duration": 1.0, "confidence": 0.9},
        {"id": "c2", "path": "c2.wav", "duration": 2.0, "confidence": 0.9},
    ]
    windows = [
        {"id": "w1", "start": 10.0, "duration": 1.05},
        {"id": "w2", "start": 20.0, "duration": 2.0},
        {"id": "w3", "start": 30.0, "duration": 1.0},
    ]
    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
    )
    assert [m["clip_id"] for m in schedule["mappings"]] == ["c1", "c2"]
    assert schedule["mappings"][0]["destination_timestamp"] == 10.0
    assert schedule["mappings"][0]["score"] > 0
    assert schedule["mappings"][0]["selection_reason"] == "next_source_clip_in_order"
    assert schedule["mappings"][0]["enabled"] is True
    assert schedule["mappings"][0]["planned_render_duration"] == 1.05
    assert [op["operation"] for op in schedule["mappings"][0]["render_operations"]] == [
        "trim",
        "time_stretch",
        "normalize_loudness",
        "fade_in_out",
        "delay",
        "limit",
    ]
    assert schedule["scheduling_mode"] == "strict_order"
    assert schedule["transformation_name"] == "translation"
    assert [step["verb"] for step in schedule["transformation_history"]] == ["select", "select", "place", "replace", "render"]


def test_build_schedule_trims_long_clip_to_short_window(tmp_path: Path) -> None:
    clips = [{"id": "c1", "path": "c1.wav", "duration": 8.0}]
    windows = [{"id": "w1", "start": 5.0, "duration": 1.5}]
    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
    )
    assert len(schedule["mappings"]) == 1
    assert schedule["mappings"][0]["clip_trim_duration"] == 1.5
    assert schedule["mappings"][0]["trailing_silence"] == 0.0
    assert schedule["mappings"][0]["timing_strategy"] == "trim_to_window"
    assert schedule["mappings"][0]["planned_render_duration"] == 1.5



def test_window_fill_places_multiple_clips_inside_one_window(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "path": "c1.wav", "duration": 1.0, "confidence": 0.9},
        {"id": "c2", "path": "c2.wav", "duration": 1.0, "confidence": 0.9},
        {"id": "c3", "path": "c3.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [{"id": "w1", "start": 10.0, "duration": 3.0}]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="window_fill",
    )

    mappings = schedule["mappings"]
    assert [m["clip_id"] for m in mappings] == ["c1", "c2", "c3"]
    assert [m["destination_timestamp"] for m in mappings] == [10.0, 11.0, 12.0]
    assert all(m["window_id"] == "w1" for m in mappings)
    assert all(m["selection_reason"] == "whole_line_fill_destination_window" for m in mappings)
    assert schedule["scheduled_window_count"] == 1
    assert schedule["used_clip_count"] == 3



def test_whole_line_fill_does_not_trim_source_lines(tmp_path: Path) -> None:
    clips = [
        {"id": "too_long", "path": "long.wav", "duration": 5.0, "confidence": 0.9},
        {"id": "fits", "path": "fits.wav", "duration": 1.8, "confidence": 0.9},
        {"id": "also_fits", "path": "also.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [{"id": "w1", "start": 10.0, "duration": 3.0}]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    mappings = schedule["mappings"]
    assert [m["clip_id"] for m in mappings] == ["fits", "also_fits"]
    assert mappings[0]["skipped_source_clips"] == 1
    assert all(m["timing_strategy"] != "trim_to_window" for m in mappings)
    assert all(m["clip_trim_duration"] >= 1.0 for m in mappings)
    assert [m["destination_timestamp"] for m in mappings] == [10.0, 11.8]


def test_whole_line_fill_records_recovered_speech_source_kind(tmp_path: Path) -> None:
    clips = [
        {"id": "line", "path": "line.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "p1",
            "start": 0.0,
            "duration": 2.0,
            "performance_id": "p1",
            "speech_windows": [
                {"id": "w1", "start": 0.0, "end": 1.2, "duration": 1.2, "source_kind": "recovered_filtered_speech_window"},
            ],
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    assert schedule["mappings"][0]["alignment_mode"] == "speech_window_snap"
    assert schedule["mappings"][0]["alignment_source_kind"] == "recovered_filtered_speech_window"



def test_whole_line_fill_snaps_lines_to_child_speech_windows(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "path": "c1.wav", "duration": 1.0, "confidence": 0.9},
        {"id": "c2", "path": "c2.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "p1",
            "start": 10.0,
            "duration": 6.0,
            "performance_id": "p1",
            "speech_windows": [
                {"id": "w1", "start": 10.0, "end": 11.1, "duration": 1.1},
                {"id": "w2", "start": 14.0, "end": 15.1, "duration": 1.1},
            ],
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    mappings = schedule["mappings"]
    assert [m["clip_id"] for m in mappings] == ["c1", "c2"]
    assert [m["destination_timestamp"] for m in mappings] == [10.0, 14.0]
    assert [m["alignment_mode"] for m in mappings] == ["speech_window_snap", "speech_window_snap"]
    assert mappings[0]["alignment_source_window_ids"] == ["w1"]
    assert mappings[1]["alignment_source_window_ids"] == ["w2"]


def test_whole_line_fill_can_span_adjacent_child_speech_windows(tmp_path: Path) -> None:
    clips = [
        {"id": "line", "path": "line.wav", "duration": 1.6, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "p1",
            "start": 0.0,
            "duration": 3.0,
            "performance_id": "p1",
            "speech_windows": [
                {"id": "w1", "start": 0.0, "end": 0.9, "duration": 0.9},
                {"id": "w2", "start": 1.1, "end": 2.0, "duration": 0.9},
            ],
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    assert len(schedule["mappings"]) == 1
    mapping = schedule["mappings"][0]
    assert mapping["destination_timestamp"] == 0.0
    assert mapping["alignment_source_window_ids"] == ["w1", "w2"]
    assert mapping["alignment_spans_speech_windows"] is True
    assert mapping["timing_strategy"] != "trim_to_window"
    assert schedule["destination_performance_fills"][0]["covered_speech_window_count"] == 2


def test_whole_line_fill_skips_tiny_child_speech_window(tmp_path: Path) -> None:
    clips = [
        {"id": "line", "path": "line.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "p1",
            "start": 0.0,
            "duration": 5.0,
            "performance_id": "p1",
            "speech_windows": [
                {"id": "too_short", "start": 0.0, "end": 0.2, "duration": 0.2},
                {"id": "fits", "start": 2.0, "end": 3.1, "duration": 1.1},
            ],
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    assert len(schedule["mappings"]) == 1
    mapping = schedule["mappings"][0]
    assert mapping["destination_timestamp"] == 2.0
    assert mapping["alignment_source_window_ids"] == ["fits"]


def test_whole_line_fill_records_source_and_destination_performance_metadata(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "path": "c1.wav", "movie_timestamp": 10.0, "duration": 1.0, "confidence": 0.9},
        {"id": "c2", "path": "c2.wav", "movie_timestamp": 11.0, "duration": 1.0, "confidence": 0.9},
    ]
    source_performances = {
        "performances": [
            {
                "id": "sp1",
                "start": 10.0,
                "end": 12.0,
                "duration": 2.0,
                "conversation_type": "exchange",
                "estimated_turn_count": 2,
                "dialogue_density": 0.8,
            }
        ]
    }
    windows = [
        {
            "id": "dp1",
            "start": 100.0,
            "duration": 3.0,
            "performance_id": "dp1",
            "performance_type": "exchange",
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
        source_performances=source_performances,
    )

    mappings = schedule["mappings"]
    assert [m["clip_id"] for m in mappings] == ["c1", "c2"]
    assert [m["clip_movie_timestamp"] for m in mappings] == [10.0, 11.0]
    assert [m["source_movie_timestamp"] for m in mappings] == [10.0, 11.0]
    assert all(m["source_performance_id"] == "sp1" for m in mappings)
    assert all(m["source_performance_type"] == "exchange" for m in mappings)
    assert all(m["destination_performance_id"] == "dp1" for m in mappings)
    assert schedule["performance_placements"][0]["source_performance_id"] == "sp1"
    assert schedule["performance_placements"][0]["mapping_count"] == 2
    assert schedule["destination_performance_fills"][0]["destination_performance_id"] == "dp1"
    assert schedule["destination_performance_fills"][0]["coverage"] > 0.6


def test_whole_line_fill_rescues_empty_short_performance_with_unused_fitting_line(tmp_path: Path) -> None:
    clips = [
        {"id": "too_long", "path": "long.wav", "duration": 5.0, "confidence": 0.9},
        {"id": "fits_large", "path": "large.wav", "duration": 4.0, "confidence": 0.9},
        {"id": "short_rescue", "path": "short.wav", "duration": 1.2, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "short",
            "start": 0.0,
            "duration": 1.5,
            "performance_id": "short",
            "speech_windows": [{"id": "short_speech", "start": 0.2, "end": 1.5, "duration": 1.3}],
        },
        {"id": "large", "start": 10.0, "duration": 5.0, "performance_id": "large"},
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    rescue = [m for m in schedule["mappings"] if m["selection_reason"] == "short_performance_rescue"]
    assert len(rescue) == 1
    assert rescue[0]["window_id"] == "short"
    assert rescue[0]["clip_id"] == "short_rescue"
    assert rescue[0]["timing_strategy"] != "trim_to_window"
    assert rescue[0]["alignment_mode"] == "speech_window_snap"
    assert rescue[0]["alignment_source_window_ids"] == ["short_speech"]
    assert rescue[0]["rescue_allowed_reason"] == "otherwise_empty_short_destination_performance"


def test_whole_line_fill_rescue_snaps_to_child_speech_window(tmp_path: Path) -> None:
    clips = [
        {"id": "short", "path": "short.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "empty",
            "start": 5.0,
            "duration": 4.0,
            "performance_id": "empty",
            "speech_windows": [
                {"id": "speech", "start": 6.0, "end": 7.2, "duration": 1.2},
            ],
        },
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    mapping = schedule["mappings"][0]
    assert mapping["selection_reason"] == "whole_line_fill_destination_window"
    assert mapping["alignment_mode"] == "speech_window_snap"
    assert mapping["alignment_source_window_ids"] == ["speech"]


def test_whole_line_fill_rescue_reuses_fitting_line_when_no_unused_line_remains(tmp_path: Path) -> None:
    clips = [
        {"id": "short", "path": "short.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [
        {"id": "first", "start": 0.0, "duration": 1.0, "performance_id": "first"},
        {"id": "second", "start": 5.0, "duration": 1.0, "performance_id": "second"},
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
        allow_source_reuse=True,
    )

    assert [m["window_id"] for m in schedule["mappings"]] == ["first", "second"]
    rescue = schedule["mappings"][1]
    assert rescue["selection_reason"] == "short_performance_rescue_reuse"
    assert rescue["clip_id"] == "short"
    assert rescue["rescue_reused_clip"] is True


def test_whole_line_fill_forbids_implicit_source_reuse_by_default(tmp_path: Path) -> None:
    clips = [{"id": "only", "path": "only.wav", "duration": 1.0, "confidence": 0.9}]
    windows = [
        {"id": "first", "start": 0.0, "duration": 1.0},
        {"id": "second", "start": 5.0, "duration": 1.0},
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
    )

    assert [mapping["clip_id"] for mapping in schedule["mappings"]] == ["only"]
    assert schedule["source_reuse_policy"] == "forbidden"
    assert schedule["reused_clip_placement_count"] == 0
    assert schedule["source_clip_reuse_counts"] == {}


def test_source_exhaustion_reuse_fill_uses_free_tail_inside_speech_slot(tmp_path: Path) -> None:
    clips = [
        {"id": "first", "path": "first.wav", "duration": 2.0, "confidence": 0.9},
        {"id": "short", "path": "short.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "p1",
            "start": 0.0,
            "duration": 4.0,
            "performance_id": "p1",
            "speech_windows": [{"id": "speech", "start": 0.0, "end": 4.0, "duration": 4.0}],
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
        allow_source_reuse=True,
    )

    reuse = [m for m in schedule["mappings"] if m["selection_reason"] == "source_exhaustion_reuse_fill"]
    assert reuse
    assert reuse[0]["destination_timestamp"] == 3.0
    assert reuse[0]["alignment_source_window_ids"] == ["speech"]
    fill = schedule["destination_performance_fills"][0]
    assert fill["coverage"] >= 0.9


def test_source_exhaustion_reuse_fill_adds_whole_lines_to_underfilled_tail(tmp_path: Path) -> None:
    clips = [
        {"id": "a", "path": "a.wav", "duration": 2.0, "confidence": 0.9},
        {"id": "b", "path": "b.wav", "duration": 2.0, "confidence": 0.9},
    ]
    windows = [
        {"id": "early", "start": 0.0, "duration": 4.0, "performance_id": "early"},
        {"id": "tail", "start": 10.0, "duration": 40.0, "performance_id": "tail"},
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="whole_line_fill",
        allow_source_reuse=True,
    )

    reuse = [m for m in schedule["mappings"] if m["selection_reason"] == "source_exhaustion_reuse_fill"]
    assert reuse
    assert all(m["window_id"] == "tail" for m in reuse)
    assert all(m["timing_strategy"] != "trim_to_window" for m in reuse)
    assert all(m["reuse_allowed_reason"] == "source_dialogue_exhausted" for m in reuse)
    tail_fill = [row for row in schedule["destination_performance_fills"] if row["destination_performance_id"] == "tail"][0]
    assert tail_fill["coverage"] >= 0.75


def test_performance_fill_places_source_performance_group_without_trimming(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "path": "c1.wav", "movie_timestamp": 10.0, "duration": 1.0, "confidence": 0.9},
        {"id": "c2", "path": "c2.wav", "movie_timestamp": 11.0, "duration": 1.5, "confidence": 0.9},
        {"id": "c3", "path": "c3.wav", "movie_timestamp": 40.0, "duration": 5.0, "confidence": 0.9},
    ]
    source_performances = {
        "performances": [
            {"id": "sp1", "start": 10.0, "end": 12.5, "duration": 2.5, "conversation_type": "exchange"},
            {"id": "sp2", "start": 40.0, "end": 45.0, "duration": 5.0, "conversation_type": "monologue"},
        ]
    }
    windows = [
        {
            "id": "dp1",
            "start": 100.0,
            "duration": 3.0,
            "performance_id": "dp1",
            "performance_type": "exchange",
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="performance_fill",
        best_fit_lookahead=2,
        source_performances=source_performances,
    )

    mappings = schedule["mappings"]
    assert [m["clip_id"] for m in mappings] == ["c1", "c2"]
    assert [m["destination_timestamp"] for m in mappings] == [100.0, 101.0]
    assert all(m["source_performance_id"] == "sp1" for m in mappings)
    assert all(m["destination_performance_id"] == "dp1" for m in mappings)
    assert all(m["timing_strategy"] != "trim_to_window" for m in mappings)
    assert schedule["scheduling_mode"] == "performance_fill"



def test_performance_fill_prefers_signature_match_over_duration_tie(tmp_path: Path) -> None:
    clips = [
        {"id": "a1", "path": "a1.wav", "movie_timestamp": 0.0, "duration": 1.0, "confidence": 0.9},
        {"id": "a2", "path": "a2.wav", "movie_timestamp": 1.0, "duration": 1.0, "confidence": 0.9},
        {"id": "b1", "path": "b1.wav", "movie_timestamp": 10.0, "duration": 1.0, "confidence": 0.9},
        {"id": "b2", "path": "b2.wav", "movie_timestamp": 11.0, "duration": 1.0, "confidence": 0.9},
    ]
    source_performances = {
        "performances": [
            {
                "id": "sp_mono",
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "conversation_type": "exchange",
                "speaker_sequence": ["A"],
                "turn_pattern": "A",
                "signature": {
                    "duration": 2.0,
                    "speaker_count": 1,
                    "turn_count": 1,
                    "speaker_sequence": ["A"],
                    "turn_pattern": "A",
                    "average_turn_duration": 1.0,
                    "average_pause_duration": 0.0,
                    "dialogue_density": 1.0,
                    "estimated_energy": 0.8,
                    "shot_change_rate": 0.0,
                },
            },
            {
                "id": "sp_exchange",
                "start": 10.0,
                "end": 12.0,
                "duration": 2.0,
                "conversation_type": "exchange",
                "speaker_sequence": ["A", "B"],
                "turn_pattern": "A B",
                "signature": {
                    "duration": 2.0,
                    "speaker_count": 2,
                    "turn_count": 2,
                    "speaker_sequence": ["A", "B"],
                    "turn_pattern": "A B",
                    "average_turn_duration": 1.0,
                    "average_pause_duration": 0.0,
                    "dialogue_density": 1.0,
                    "estimated_energy": 0.8,
                    "shot_change_rate": 0.0,
                },
            },
        ]
    }
    windows = [
        {
            "id": "dp_exchange",
            "start": 100.0,
            "duration": 2.0,
            "performance_id": "dp_exchange",
            "performance_type": "exchange",
            "speaker_sequence": ["A", "B"],
            "turn_pattern": "A B",
            "signature": {
                "duration": 2.0,
                "speaker_count": 2,
                "turn_count": 2,
                "speaker_sequence": ["A", "B"],
                "turn_pattern": "A B",
                "average_turn_duration": 1.0,
                "average_pause_duration": 0.0,
                "dialogue_density": 1.0,
                "estimated_energy": 0.8,
                "shot_change_rate": 0.0,
            },
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="performance_fill",
        best_fit_lookahead=2,
        source_performances=source_performances,
    )

    assert [mapping["source_performance_id"] for mapping in schedule["mappings"]] == ["sp_exchange", "sp_exchange"]
    assert all(mapping["performance_similarity_score"] > 0.95 for mapping in schedule["mappings"])
    assert all(mapping["speaker_pattern_match"] == 1.0 for mapping in schedule["mappings"])
    assert "speaker_pattern" in schedule["mappings"][0]["performance_similarity_components"]



def test_performance_fill_uses_signature_v2_behavior_fields(tmp_path: Path) -> None:
    clips = [
        {"id": "flat", "path": "flat.wav", "movie_timestamp": 0.0, "duration": 2.0, "confidence": 0.9},
        {"id": "matched", "path": "matched.wav", "movie_timestamp": 10.0, "duration": 2.0, "confidence": 0.9},
    ]
    base_signature = {
        "duration": 2.0,
        "speaker_count": 2,
        "turn_count": 2,
        "speaker_sequence": ["A", "B"],
        "average_turn_duration": 1.0,
        "average_pause_duration": 0.3,
        "dialogue_density": 0.7,
        "estimated_energy": 0.6,
        "shot_change_rate": 0.0,
    }
    source_performances = {
        "performances": [
            {
                "id": "flat_perf",
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "conversation_type": "exchange",
                "performance_type": "monologue",
                "signature": {
                    **base_signature,
                    "performance_type": "monologue",
                    "speech_continuity": 1.0,
                    "response_delay": 0.0,
                    "silence_ratio": 0.0,
                    "words_per_second": 0.5,
                },
            },
            {
                "id": "matched_perf",
                "start": 10.0,
                "end": 12.0,
                "duration": 2.0,
                "conversation_type": "exchange",
                "performance_type": "dialogue_exchange",
                "signature": {
                    **base_signature,
                    "performance_type": "dialogue_exchange",
                    "speech_continuity": 0.55,
                    "response_delay": 0.35,
                    "silence_ratio": 0.18,
                    "words_per_second": 2.2,
                },
            },
        ]
    }
    windows = [
        {
            "id": "dest",
            "start": 100.0,
            "duration": 2.0,
            "performance_id": "dest",
            "performance_type": "exchange",
            "performance_type_v2": "dialogue_exchange",
            "signature": {
                **base_signature,
                "performance_type": "dialogue_exchange",
                "speech_continuity": 0.55,
                "response_delay": 0.35,
                "silence_ratio": 0.18,
                "words_per_second": 2.2,
            },
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="performance_fill",
        best_fit_lookahead=2,
        source_performances=source_performances,
        cinematic_filter="rhythm",
    )

    mapping = schedule["mappings"][0]
    assert mapping["source_performance_id"] == "matched_perf"
    assert mapping["performance_similarity_components"]["performance_type"] == 1.0
    assert mapping["performance_similarity_components"]["response_delay"] == 1.0
    assert mapping["performance_similarity_components"]["words_per_second"] == 1.0


def test_cinematic_filters_change_performance_choice(tmp_path: Path) -> None:
    clips = [
        {"id": "dense", "path": "dense.wav", "movie_timestamp": 0.0, "duration": 2.0, "confidence": 0.9},
        {"id": "dry", "path": "dry.wav", "movie_timestamp": 10.0, "duration": 2.0, "confidence": 0.9},
    ]
    source_performances = {
        "performances": [
            {
                "id": "dense_perf",
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "conversation_type": "exchange",
                "signature": {
                    "duration": 2.0,
                    "speaker_count": 2,
                    "turn_count": 4,
                    "speaker_sequence": ["A", "B", "A", "B"],
                    "average_turn_duration": 0.5,
                    "average_pause_duration": 0.0,
                    "dialogue_density": 1.0,
                    "estimated_energy": 1.0,
                    "shot_change_rate": 0.0,
                },
            },
            {
                "id": "deadpan_perf",
                "start": 10.0,
                "end": 12.0,
                "duration": 2.0,
                "conversation_type": "exchange",
                "signature": {
                    "duration": 2.0,
                    "speaker_count": 1,
                    "turn_count": 1,
                    "speaker_sequence": ["A"],
                    "average_turn_duration": 2.0,
                    "average_pause_duration": 1.5,
                    "dialogue_density": 0.25,
                    "estimated_energy": 0.1,
                    "shot_change_rate": 0.0,
                },
            },
        ]
    }
    windows = [
        {
            "id": "dest",
            "start": 100.0,
            "duration": 2.0,
            "performance_id": "dest",
            "performance_type": "exchange",
            "signature": {
                "duration": 2.0,
                "speaker_count": 1,
                "turn_count": 2,
                "speaker_sequence": ["A", "B"],
                "average_turn_duration": 1.0,
                "average_pause_duration": 0.4,
                "dialogue_density": 0.55,
                "estimated_energy": 0.45,
                "shot_change_rate": 0.0,
            },
        }
    ]

    dense = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "dense.json",
        scheduling_mode="performance_fill",
        best_fit_lookahead=2,
        source_performances=source_performances,
        cinematic_filter="dense_comedy",
    )
    deadpan = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "deadpan.json",
        scheduling_mode="performance_fill",
        best_fit_lookahead=2,
        source_performances=source_performances,
        cinematic_filter="deadpan",
    )

    assert dense["active_filter"] == "dense_comedy"
    assert deadpan["active_filter"] == "deadpan"
    assert dense["mappings"][0]["source_performance_id"] == "dense_perf"
    assert deadpan["mappings"][0]["source_performance_id"] == "deadpan_perf"
    assert dense["mappings"][0]["active_filter"] == "dense_comedy"
    assert deadpan["mappings"][0]["active_filter"] == "deadpan"
    assert dense["mappings"][0]["baseline_similarity_score"] != dense["mappings"][0]["performance_similarity_score"]


def test_best_fit_selects_duration_match_within_lookahead(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "path": "c1.wav", "duration": 7.0, "confidence": 0.9},
        {"id": "c2", "path": "c2.wav", "duration": 1.1, "confidence": 0.9},
        {"id": "c3", "path": "c3.wav", "duration": 4.0, "confidence": 0.9},
    ]
    windows = [{"id": "w1", "start": 10.0, "duration": 1.0}]
    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="best_fit",
        best_fit_lookahead=3,
    )
    mapping = schedule["mappings"][0]
    assert mapping["clip_id"] == "c2"
    assert mapping["skipped_source_clips"] == 1
    assert mapping["selection_reason"] == "best_duration_fit_within_lookahead"
    assert mapping["score_components"]["duration_similarity"] > 0.9


def test_build_schedule_accepts_custom_transformation_metadata(tmp_path: Path) -> None:
    schedule = build_schedule(
        clips=[{"id": "c1", "path": "c1.wav", "duration": 1.0}],
        windows=[{"id": "w1", "start": 0.0, "duration": 1.0}],
        source_hash="same",
        destination_hash="same",
        max_time_stretch=0.1,
        output_path=tmp_path / "self_shuffle_schedule.json",
        transformation_name="self_shuffle",
        transformation_history=[{"verb": "shuffle", "description": "shuffle test", "inputs": ["a"], "outputs": ["b"]}],
    )

    assert schedule["transformation_name"] == "self_shuffle"
    assert schedule["transformation_history"][0]["verb"] == "shuffle"


def test_best_fit_prefers_clip_that_stays_inside_shot(tmp_path: Path) -> None:
    clips = [
        {"id": "long", "path": "long.wav", "duration": 2.8, "confidence": 0.9},
        {"id": "short", "path": "short.wav", "duration": 1.0, "confidence": 0.9},
    ]
    windows = [
        {
            "id": "w1",
            "start": 1.0,
            "end": 4.0,
            "duration": 3.0,
            "shot_id": "shot_1",
            "shot_start": 0.0,
            "shot_end": 2.1,
            "crosses_shot_boundary": False,
            "boundary_overlap_seconds": 0.0,
        }
    ]

    schedule = build_schedule(
        clips=clips,
        windows=windows,
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        scheduling_mode="best_fit",
        best_fit_lookahead=2,
        shot_boundary_mode="soft",
    )

    mapping = schedule["mappings"][0]
    assert mapping["clip_id"] == "short"
    assert mapping["shot_id"] == "shot_1"
    assert mapping["visual_fit_score"] == 1.0


def test_strict_mode_limits_render_to_shot_boundary(tmp_path: Path) -> None:
    schedule = build_schedule(
        clips=[{"id": "c1", "path": "c1.wav", "duration": 4.0}],
        windows=[
            {
                "id": "w1",
                "start": 1.0,
                "end": 5.0,
                "duration": 4.0,
                "shot_id": "shot_1",
                "shot_start": 0.0,
                "shot_end": 2.5,
                "crosses_shot_boundary": False,
                "boundary_overlap_seconds": 0.0,
            }
        ],
        source_hash="source",
        destination_hash="dest",
        max_time_stretch=0.1,
        output_path=tmp_path / "replacement_schedule.json",
        shot_boundary_mode="strict",
    )

    mapping = schedule["mappings"][0]
    assert mapping["planned_render_duration"] == 1.5
    assert mapping["mapping_crosses_shot_boundary"] is False
    assert mapping["timing_strategy"].endswith("shot_limited")



def test_undercovered_speech_slot_fill_targets_empty_slot() -> None:
    mappings = [
        {
            "enabled": True,
            "window_id": "p1",
            "destination_performance_id": "p1",
            "clip_id": "existing",
            "destination_timestamp": 0.0,
            "planned_render_duration": 1.0,
            "alignment_source_window_ids": ["s1"],
        }
    ]
    clips = [{"id": "reuse", "path": "reuse.wav", "duration": 1.0, "confidence": 0.9}]
    windows = [
        {
            "id": "p1",
            "performance_id": "p1",
            "start": 0.0,
            "duration": 2.0,
            "speech_windows": [
                {"id": "s1", "start": 0.0, "end": 1.0, "duration": 1.0},
                {"id": "s2", "start": 1.0, "end": 2.0, "duration": 1.0},
            ],
        }
    ]

    _append_undercovered_speech_slot_fill(
        mappings=mappings,
        usable_clips=clips,
        windows=windows,
        max_time_stretch=0.1,
        shot_boundary_mode="off",
        cinematic_filter="balanced",
    )

    fill = mappings[-1]
    assert fill["selection_reason"] == "undercovered_speech_slot_reuse_fill"
    assert fill["destination_timestamp"] == 1.0
    assert fill["alignment_source_window_ids"] == ["s2"]
    assert fill["reuse_allowed_reason"] == "undercovered_speech_slot"



def test_reanchor_single_slot_mapping_moves_line_to_visible_speech_start() -> None:
    mappings = [
        {
            "enabled": True,
            "window_id": "p1",
            "destination_performance_id": "p1",
            "clip_id": "early",
            "destination_timestamp": 9.4,
            "planned_render_duration": 1.0,
            "alignment_source_window_ids": ["s1"],
            "alignment_slot_start": 9.4,
            "alignment_slot_end": 11.0,
            "render_operations": [{"operation": "delay", "seconds": 9.4}],
            "selection_reason": "whole_line_fill_destination_window",
        }
    ]
    windows = [
        {
            "id": "p1",
            "performance_id": "p1",
            "start": 9.0,
            "duration": 3.0,
            "speech_windows": [{"id": "s1", "start": 10.0, "end": 11.0, "duration": 1.0}],
        }
    ]

    _reanchor_single_slot_mappings_to_speech_start(mappings=mappings, fills=[{"speech_windows": windows[0]["speech_windows"]}])

    mapping = mappings[0]
    assert mapping["destination_timestamp"] == 10.0
    assert mapping["alignment_slot_start"] == 10.0
    assert mapping["alignment_slot_end"] == 11.0
    assert mapping["alignment_spillover_seconds"] == 0.0
    assert mapping["render_operations"][0]["seconds"] == 10.0
    assert mapping["selection_reason"].endswith("speech_start_reanchored")




def test_source_exhaustion_speech_slot_fill_advances_when_candidate_trims(monkeypatch) -> None:
    import cinelingus.schedule as schedule_module

    calls = {"count": 0}

    def trim_mapping(**_kwargs):
        calls["count"] += 1
        return {"timing_strategy": "trim_to_window"}

    monkeypatch.setattr(schedule_module, "_build_mapping", trim_mapping)
    mappings = []
    clips = [{"id": "long", "path": "long.wav", "duration": 1.0, "confidence": 0.9}]
    window = {
        "id": "p1",
        "performance_id": "p1",
        "start": 0.0,
        "duration": 3.0,
        "speech_windows": [
            {"id": "s1", "start": 0.0, "end": 3.0, "duration": 3.0},
        ],
    }

    _append_source_exhaustion_reuse_fill_speech_slots(
        mappings=mappings,
        usable_clips=clips,
        window=window,
        max_time_stretch=0.1,
        shot_boundary_mode="off",
        target_coverage=0.9,
        recent_performance_ids=[],
        used_clip_ids=set(),
        cinematic_filter="balanced",
    )

    assert mappings == []
    assert calls["count"] == 1


def test_reanchor_multi_slot_mapping_targets_trailing_speech_slot() -> None:
    mappings = [
        {
            "enabled": True,
            "window_id": "p1",
            "destination_performance_id": "p1",
            "clip_id": "prior",
            "destination_timestamp": 9.0,
            "planned_render_duration": 1.0,
            "alignment_source_window_ids": ["s0"],
        },
        {
            "enabled": True,
            "window_id": "p1",
            "destination_performance_id": "p1",
            "clip_id": "early",
            "destination_timestamp": 9.4,
            "planned_render_duration": 1.0,
            "alignment_source_window_ids": ["s0", "s1"],
            "alignment_slot_start": 9.0,
            "alignment_slot_end": 11.0,
            "alignment_spans_speech_windows": True,
            "render_operations": [{"operation": "delay", "seconds": 9.4}],
            "selection_reason": "whole_line_fill_destination_window",
        }
    ]
    fills = [
        {
            "speech_windows": [
                {"id": "s0", "start": 9.0, "end": 10.0, "duration": 1.0},
                {"id": "s1", "start": 10.0, "end": 11.0, "duration": 1.0},
            ]
        }
    ]

    _reanchor_single_slot_mappings_to_speech_start(mappings=mappings, fills=fills)

    mapping = mappings[1]
    assert mapping["destination_timestamp"] == 10.0
    assert mapping["alignment_source_window_ids"] == ["s1"]
    assert mapping["alignment_spans_speech_windows"] is False

