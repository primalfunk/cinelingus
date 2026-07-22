from cinelingus.performance_curation import (
    analyze_dialogue_density,
    build_performance_curation_manifest,
    build_reviewed_seed_refinement_manifest,
)


def _schedule() -> dict:
    return {
        "mappings": [
            {
                "enabled": True, "clip_id": "a", "destination_performance_id": "p1",
                "destination_timestamp": 0.0, "planned_render_duration": 1.0, "scheduler_tier": 1,
            },
            {
                "enabled": True, "clip_id": "b", "destination_performance_id": "p1",
                "destination_timestamp": 0.0, "planned_render_duration": 1.0, "scheduler_tier": 1,
            },
            {
                "enabled": True, "clip_id": "c", "destination_performance_id": "p2",
                "destination_timestamp": 4.0, "planned_render_duration": 1.0, "scheduler_tier": 4,
                "alignment_spans_speech_windows": True,
            },
        ],
        "destination_performance_fills": [
            {"destination_performance_id": "p1", "start": 0.0, "duration": 2.0, "speech_duration": 0.5, "speech_window_count": 1, "coverage": 1.0},
            {"destination_performance_id": "p2", "start": 4.0, "duration": 2.0, "speech_duration": 1.0, "speech_window_count": 2, "coverage": 1.0},
            {"destination_performance_id": "p3", "start": 8.0, "duration": 1.0, "speech_duration": 1.0, "speech_window_count": 1, "coverage": 0.0},
        ],
        "performance_decisions": [
            {"destination_performance_id": "p1", "scheduler_tier": 1},
            {"destination_performance_id": "p2", "scheduler_tier": 2},
            {"destination_performance_id": "p3", "scheduler_tier": 5},
        ],
        "destination_speech_regions": [
            {"id": "micro", "start": 0.0, "end": 0.5, "duration": 0.5, "transcript": "Hey!"},
        ],
        "residue_correction_regions": [
            {"id": "correction", "start": 10.0, "end": 10.5, "duration": 0.5, "evidence_kind": "destination_transcript_contrast"},
        ],
    }


def test_performance_curation_selects_each_available_review_stratum(tmp_path) -> None:
    artifact = build_performance_curation_manifest(
        schedule=_schedule(),
        source_video=tmp_path / "output.mp4",
        output_path=tmp_path / "manifest.json",
        max_per_stratum=1,
    )

    assert {row["stratum"] for row in artifact["selected"]} == {
        "coupled_performance",
        "adapted_performance",
        "suppressed_unreplaced",
        "residue_corrected",
    }
    assert artifact["selected_count"] == 4
    assert artifact["review_rubric"][-1] == "editorial_intentionality"


def test_dialogue_density_uses_interval_union_and_flags_stacking() -> None:
    diagnostics = analyze_dialogue_density(_schedule())
    p1 = next(row for row in diagnostics["performances"] if row["destination_performance_id"] == "p1")

    assert p1["audible_replacement_duration"] == 1.0
    assert p1["replacement_to_speech_ratio"] == 2.0
    assert p1["overlap_stacking_ratio"] == 2.0
    assert "replacement_overdensity" in p1["warnings"]
    assert "overlapping_replacement_dialogue" in p1["warnings"]


def test_curation_deduplicates_same_mapping_selection_across_strata(tmp_path) -> None:
    schedule = _schedule()
    schedule["mappings"][2]["scheduler_tier"] = 2
    schedule["mappings"][2]["alignment_spans_speech_windows"] = True

    artifact = build_performance_curation_manifest(
        schedule=schedule,
        source_video=tmp_path / "output.mp4",
        output_path=tmp_path / "manifest.json",
        max_per_stratum=1,
    )

    repeated = [row for row in artifact["selected"] if row.get("mapping_indices") == [2]]
    assert len(repeated) == 1


def test_reviewed_seed_refinement_requires_speech_and_complete_review_bounds(tmp_path) -> None:
    schedule = {
        "mappings": [
            {"enabled": True, "clip_id": "spoken", "destination_timestamp": 10.0, "planned_render_duration": 2.0, "source_transcript": "That timing works!"},
            {"enabled": True, "clip_id": "sfx", "destination_timestamp": 13.0, "planned_render_duration": 1.0, "source_transcript": ""},
            {"enabled": True, "clip_id": "fragment", "destination_timestamp": 15.0, "planned_render_duration": 0.4, "source_transcript": "and then"},
        ]
    }
    reviewed = {"selected": [{"index": 4, "stratum": "coupled_performance", "preview_start": 9.0, "preview_end": 17.0}]}

    artifact = build_reviewed_seed_refinement_manifest(
        schedule=schedule,
        reviewed_manifest=reviewed,
        positive_indices=[4],
        source_video=tmp_path / "movie.mp4",
        output_path=tmp_path / "refined.json",
    )

    assert artifact["selected_count"] == 1
    assert artifact["selected"][0]["clip_id"] == "spoken"
    assert artifact["selected"][0]["duration"] >= 2.5
    assert {row["reason"] for row in artifact["rejected"]} == {"no_verified_spoken_transcript", "spoken_fragment_too_short"}


def test_reviewed_seed_refinement_rejects_inaudibly_dense_and_multi_speaker_lines(tmp_path) -> None:
    schedule = {"mappings": [
        {
            "enabled": True, "clip_id": "truncated", "destination_timestamp": 1.0,
            "planned_render_duration": 1.0, "source_transcript": "This sentence cannot possibly finish here.",
            "source_speaker_sequence": ["source_a"], "destination_speaker_sequence": ["dest_a"],
        },
        {
            "enabled": True, "clip_id": "turn_mismatch", "destination_timestamp": 3.0,
            "planned_render_duration": 2.0, "source_transcript": "One voice races across both characters speaking back and forth.",
            "source_speaker_sequence": ["source_a"], "destination_speaker_sequence": ["dest_a", "dest_b"],
        },
    ]}
    reviewed = {"selected": [{"index": 8, "preview_start": 0.0, "preview_end": 7.0}]}

    artifact = build_reviewed_seed_refinement_manifest(
        schedule=schedule, reviewed_manifest=reviewed, positive_indices=[8],
        source_video=tmp_path / "movie.mp4", output_path=tmp_path / "refined.json",
    )

    assert artifact["selected_count"] == 0
    assert {row["reason"] for row in artifact["rejected"]} == {
        "improbable_audible_line_completeness",
        "single_voice_over_multi_speaker_exchange",
    }
