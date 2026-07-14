from pathlib import Path

from movie_masher.short_form import build_short_remix_candidates, build_short_remix_report, build_short_remix_schedule, expanded_short_window, select_best_short_candidate
from movie_masher.validation import validate_artifact


def _schedule():
    return {
        "mappings": [
            {
                "enabled": True,
                "destination_timestamp": 10.0,
                "planned_render_duration": 4.0,
                "clip_trim_duration": 4.0,
                "clip_trim_start": 0.0,
                "score": 0.9,
                "visual_fit_score": 0.95,
                "stretch_factor": 1.0,
                "destination_performance_id": "p1",
                "source_speaker_id": "speaker_001",
                "destination_speaker_id": "speaker_001",
                "speaker_match_preserved": True,
                "source_transcript": "hello",
                "clip_movie_timestamp": 40.0,
            },
            {
                "enabled": True,
                "destination_timestamp": 15.0,
                "planned_render_duration": 5.0,
                "clip_trim_duration": 5.0,
                "clip_trim_start": 0.0,
                "score": 0.85,
                "visual_fit_score": 0.9,
                "stretch_factor": 1.02,
                "destination_performance_id": "p1",
                "source_speaker_id": "speaker_001",
                "destination_speaker_id": "speaker_001",
                "speaker_match_preserved": True,
                "source_transcript": "there",
                "clip_movie_timestamp": 50.0,
            },
        ]
    }


def test_short_form_candidates_rank_and_report(tmp_path: Path):
    schedule = _schedule()
    for index, mapping in enumerate(schedule["mappings"]):
        mapping["_schedule_index"] = index
    candidates = build_short_remix_candidates(
        schedule=schedule,
        target_duration_seconds=12,
        minimum_duration_seconds=5,
        maximum_duration_seconds=30,
        output_path=tmp_path / "candidates.json",
    )

    selected = select_best_short_candidate(candidates)
    short_schedule = build_short_remix_schedule(schedule, selected, padding=1.0)

    assert selected["mapping_count"] == 2
    assert selected["suitability_status"] == "strong"
    assert selected["selection_strategy"] in {"highest_score", "best_sequence_fallback"}
    assert selected["target_window_speech_coverage"] > 0
    assert selected["source_segment_start"] == 40.0
    assert selected["source_segment_end"] == 55.0
    assert short_schedule["mappings"][0]["destination_timestamp"] == 1.0
    report = build_short_remix_report(
        selected_mode="best_short_remix",
        target_duration_seconds=12,
        actual_duration_seconds=12,
        candidate=selected,
        candidates=candidates,
        output_video=tmp_path / "out.mp4",
        output_audio=tmp_path / "out.wav",
        total_processing_time_seconds=1.5,
        output_path=tmp_path / "movie_masher_best_short_report.json",
    )
    assert report["selected_mode"] == "best_short_remix"
    assert report["selection_summary"]["candidate_id"] == selected["id"]
    assert report["selection_summary"]["selection_strategy"] == selected["selection_strategy"]
    assert "suitability_flags" in report["selection_summary"]
    validate_artifact("short_remix_report", tmp_path / "movie_masher_best_short_report.json", Path.cwd() / "schemas")



def test_short_remix_schedule_rebases_destination_relative_fields():
    schedule = {
        "mappings": [
            {
                "enabled": True,
                "destination_timestamp": 100.0,
                "planned_render_duration": 2.0,
                "clip_trim_duration": 2.0,
                "alignment_mode": "speech_window_snap",
                "alignment_slot_start": 100.0,
                "alignment_slot_end": 102.0,
                "shot_start": 99.0,
                "shot_end": 103.0,
            }
        ]
    }
    candidate = {"id": "candidate", "mapping_indices": [0], "destination_start": 100.0}

    short_schedule = build_short_remix_schedule(schedule, candidate, start_time=95.0)
    mapping = short_schedule["mappings"][0]

    assert mapping["destination_timestamp"] == 5.0
    assert mapping["alignment_slot_start"] == 5.0
    assert mapping["alignment_slot_end"] == 7.0
    assert mapping["shot_start"] == 4.0
    assert mapping["shot_end"] == 8.0


def test_short_remix_schedule_drops_mappings_outside_extracted_segment():
    schedule = {
        "mappings": [
            {"enabled": True, "destination_timestamp": 10.0, "planned_render_duration": 2.0, "clip_trim_duration": 2.0},
            {"enabled": True, "destination_timestamp": 20.0, "planned_render_duration": 2.0, "clip_trim_duration": 2.0},
            {"enabled": True, "destination_timestamp": 40.0, "planned_render_duration": 2.0, "clip_trim_duration": 2.0},
        ]
    }
    candidate = {"id": "wide", "mapping_indices": [0, 1, 2], "destination_start": 10.0}

    short_schedule = build_short_remix_schedule(schedule, candidate, start_time=15.0, duration=20.0)

    assert [row["destination_timestamp"] for row in short_schedule["mappings"]] == [5.0]


def test_short_remix_schedule_trims_partial_segment_overlap():
    schedule = {
        "mappings": [
            {
                "enabled": True,
                "destination_timestamp": 10.0,
                "planned_render_duration": 5.0,
                "clip_trim_start": 2.0,
                "clip_trim_duration": 5.0,
            }
        ]
    }
    candidate = {"id": "partial", "mapping_indices": [0], "destination_start": 10.0}

    short_schedule = build_short_remix_schedule(schedule, candidate, start_time=12.0, duration=2.0)
    mapping = short_schedule["mappings"][0]

    assert mapping["destination_timestamp"] == 0.0
    assert mapping["planned_render_duration"] == 2.0
    assert mapping["clip_trim_start"] == 4.0
    assert mapping["clip_trim_duration"] == 2.0



def test_expanded_short_window_targets_configured_duration():
    candidate = {"destination_start": 100.0, "destination_end": 110.0}

    start, end = expanded_short_window(
        candidate=candidate,
        destination_duration=600.0,
        target_duration_seconds=180.0,
        minimum_duration_seconds=120.0,
        maximum_duration_seconds=300.0,
    )

    assert end - start == 180.0
    assert start < 100.0
    assert end > 110.0


def test_expanded_short_window_caps_to_available_duration():
    candidate = {"destination_start": 20.0, "destination_end": 30.0}

    start, end = expanded_short_window(
        candidate=candidate,
        destination_duration=90.0,
        target_duration_seconds=180.0,
        minimum_duration_seconds=120.0,
        maximum_duration_seconds=300.0,
    )

    assert start == 0.0
    assert end == 90.0



def test_short_form_candidate_flags_sparse_weak_scene():
    schedule = {
        "mappings": [
            {
                "enabled": True,
                "destination_timestamp": 10.0,
                "planned_render_duration": 1.0,
                "clip_trim_duration": 1.0,
                "score": 0.2,
                "visual_fit_score": 0.5,
                "stretch_factor": 1.4,
                "mapping_crosses_shot_boundary": True,
            }
        ]
    }
    candidates = build_short_remix_candidates(
        schedule=schedule,
        target_duration_seconds=180,
        minimum_duration_seconds=120,
        maximum_duration_seconds=300,
    )
    selected = select_best_short_candidate(candidates)

    assert selected["suitability_status"] == "risky"
    assert "too_few_swaps" in selected["suitability_flags"]
    assert "weak_timing_fit" in selected["suitability_flags"]
    assert selected["reason_summary"].startswith("Risky candidate")



def test_short_form_selection_can_prefer_near_top_strong_candidate():
    candidates = {
        "candidates": [
            {"id": "risky", "final_combined_score": 0.8, "suitability_status": "risky", "suitability_flags": ["thin_dialogue_for_target_window"]},
            {"id": "strong", "final_combined_score": 0.72, "suitability_status": "strong", "suitability_flags": []},
        ]
    }

    selected = select_best_short_candidate(candidates)

    assert selected["id"] == "strong"
    assert selected["selection_strategy"] == "near_top_strong_candidate"


def test_short_form_candidates_include_best_sequence_when_useful():
    schedule = {
        "mappings": [
            {"enabled": True, "destination_timestamp": 10.0, "planned_render_duration": 4.0, "clip_trim_duration": 4.0, "score": 0.8, "visual_fit_score": 0.9, "stretch_factor": 1.0, "destination_performance_id": "p1", "speaker_match_preserved": True, "source_speaker_id": "speaker_001", "destination_speaker_id": "speaker_001"},
            {"enabled": True, "destination_timestamp": 20.0, "planned_render_duration": 5.0, "clip_trim_duration": 5.0, "score": 0.82, "visual_fit_score": 0.92, "stretch_factor": 1.0, "destination_performance_id": "p2", "speaker_match_preserved": True, "source_speaker_id": "speaker_002", "destination_speaker_id": "speaker_002"},
        ]
    }

    candidates = build_short_remix_candidates(
        schedule=schedule,
        target_duration_seconds=120,
        minimum_duration_seconds=60,
        maximum_duration_seconds=180,
    )

    assert any(row["candidate_type"] == "best_sequence" for row in candidates["candidates"])


def test_best_sequence_candidate_does_not_exceed_maximum_duration():
    schedule = {
        "mappings": [
            {"enabled": True, "destination_timestamp": 10.0, "planned_render_duration": 4.0, "clip_trim_duration": 4.0, "score": 0.8, "visual_fit_score": 0.9, "stretch_factor": 1.0, "destination_performance_id": "p1", "speaker_match_preserved": True},
            {"enabled": True, "destination_timestamp": 500.0, "planned_render_duration": 5.0, "clip_trim_duration": 5.0, "score": 0.82, "visual_fit_score": 0.92, "stretch_factor": 1.0, "destination_performance_id": "p2", "speaker_match_preserved": True},
        ]
    }

    candidates = build_short_remix_candidates(
        schedule=schedule,
        target_duration_seconds=120,
        minimum_duration_seconds=60,
        maximum_duration_seconds=180,
    )

    assert not any(row["candidate_type"] == "best_sequence" for row in candidates["candidates"])


def test_short_form_does_not_penalize_missing_speaker_evidence():
    schedule = {
        "mappings": [
            {
                "enabled": True,
                "destination_timestamp": 10.0,
                "planned_render_duration": 4.0,
                "clip_trim_duration": 4.0,
                "score": 0.85,
                "visual_fit_score": 0.9,
                "stretch_factor": 1.0,
                "destination_performance_id": "p1",
            },
            {
                "enabled": True,
                "destination_timestamp": 15.0,
                "planned_render_duration": 4.0,
                "clip_trim_duration": 4.0,
                "score": 0.84,
                "visual_fit_score": 0.9,
                "stretch_factor": 1.0,
                "destination_performance_id": "p1",
            },
        ]
    }

    candidates = build_short_remix_candidates(
        schedule=schedule,
        target_duration_seconds=30,
        minimum_duration_seconds=5,
        maximum_duration_seconds=60,
    )
    selected = select_best_short_candidate(candidates)

    assert selected["suitability_status"] == "strong"
    assert "weak_speaker_consistency" not in selected["suitability_flags"]


def test_best_sequence_candidate_rejects_large_internal_gap():
    schedule = {
        "mappings": [
            {"enabled": True, "destination_timestamp": 10.0, "planned_render_duration": 4.0, "clip_trim_duration": 4.0, "score": 0.8, "visual_fit_score": 0.9, "stretch_factor": 1.0, "destination_performance_id": "p1", "speaker_match_preserved": True, "source_speaker_id": "s1", "destination_speaker_id": "s1"},
            {"enabled": True, "destination_timestamp": 80.0, "planned_render_duration": 5.0, "clip_trim_duration": 5.0, "score": 0.82, "visual_fit_score": 0.92, "stretch_factor": 1.0, "destination_performance_id": "p2", "speaker_match_preserved": True, "source_speaker_id": "s2", "destination_speaker_id": "s2"},
        ]
    }

    candidates = build_short_remix_candidates(
        schedule=schedule,
        target_duration_seconds=120,
        minimum_duration_seconds=60,
        maximum_duration_seconds=180,
    )

    assert not any(row["candidate_type"] == "best_sequence" for row in candidates["candidates"])


def test_self_shuffle_short_candidates_require_substantial_replacement_coverage():
    schedule = {
        "mutation_id": "self_shuffle",
        "mappings": [
            {
                "enabled": True,
                "destination_timestamp": 0.0,
                "planned_render_duration": 10.0,
                "clip_trim_duration": 10.0,
                "score": 0.95,
                "visual_fit_score": 0.95,
                "stretch_factor": 1.0,
                "destination_performance_id": "p1",
                "speaker_match_preserved": True,
                "source_speaker_id": "speaker_001",
                "destination_speaker_id": "speaker_001",
                "clip_movie_timestamp": 100.0,
            }
        ],
    }

    candidates = build_short_remix_candidates(
        schedule=schedule,
        target_duration_seconds=180,
        minimum_duration_seconds=120,
        maximum_duration_seconds=300,
    )
    selected = select_best_short_candidate(candidates)

    assert selected["target_window_speech_coverage"] < 0.25
    assert selected["suitability_status"] == "risky"
    assert "insufficient_self_shuffle_coverage" in selected["suitability_flags"]
