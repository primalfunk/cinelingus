from cinelingus.semantic.render_proof import (
    _changed_mapping_count, _completed_variant, assess_render_acceptance,
    assess_semantic_intervention, _word_timeline_as_windows, _rollback_placements_to_control,
)


def _variant(*, coverage: float, failed: int, status: str = "PASS") -> dict:
    return {
        "audio_ready": True, "video_ready": True,
        "voice_residue_status": "NONE_DETECTED",
        "rendered_dialogue_status": status,
        "rendered_dialogue_average_word_coverage_percentage": coverage,
        "rendered_dialogue_failed_mapping_count": failed,
    }


def test_render_acceptance_rejects_semantic_quality_regression() -> None:
    result = assess_render_acceptance([
        _variant(coverage=72.22, failed=1, status="FAIL"),
        _variant(coverage=51.34, failed=2, status="FAIL"),
    ])

    assert result["state"] == "REJECTED"
    assert "semantic_word_coverage_regressed" in result["reasons"]
    assert "semantic_failed_mapping_count_regressed" in result["reasons"]


def test_word_timeline_conversion_preserves_precise_attribution_bounds() -> None:
    windows = _word_timeline_as_windows({"words": [
        {"id": "a", "start": 1.2, "end": 1.5, "text": "hello", "probability": 0.8},
        {"id": "bad", "start": 2.0, "end": 2.0, "text": "skip", "probability": 0.2},
    ]})
    assert windows == [{
        "id": "a", "start": 1.3490000000000002, "end": 1.351,
        "transcript": "hello", "confidence": 0.8,
        "word_start": 1.2, "word_end": 1.5,
    }]


def test_render_acceptance_allows_non_regressing_verified_pair() -> None:
    result = assess_render_acceptance([
        _variant(coverage=92.0, failed=0),
        _variant(coverage=94.0, failed=0),
    ])

    assert result["state"] == "ACCEPTED_FOR_HUMAN_REVIEW"
    assert result["reasons"] == []


def test_render_acceptance_allows_disclosed_shared_baseline_failure() -> None:
    result = assess_render_acceptance([
        _variant(coverage=70.0, failed=1, status="FAIL"),
        _variant(coverage=75.0, failed=1, status="FAIL"),
    ], intervention={"semantic_failed_mapping_count": 0})
    assert result["state"] == "ACCEPTED_FOR_HUMAN_REVIEW"
    assert result["shared_baseline_failed_mapping_count"] == 1
    assert result["shared_baseline_failures_disclosed"] is True


def test_changed_mapping_count_uses_stable_destination_placement() -> None:
    control = {"mappings": [
        {"editorial_placement_id": "one", "clip_id": "c1", "source_performance_id": "p1"},
        {"editorial_placement_id": "two", "clip_id": "c2", "source_performance_id": "p2"},
    ]}
    semantic = {"mappings": [
        {"editorial_placement_id": "one", "clip_id": "c1", "source_performance_id": "p1"},
        {"editorial_placement_id": "two", "clip_id": "c3", "source_performance_id": "p3"},
    ]}
    assert _changed_mapping_count(control, semantic) == 1


def test_failed_semantic_placement_can_roll_back_without_losing_passing_change() -> None:
    control = {"mappings": [
        {"editorial_placement_id": "one", "clip_id": "legacy-1"},
        {"editorial_placement_id": "two", "clip_id": "legacy-2"},
    ]}
    semantic = {"mappings": [
        {"editorial_placement_id": "one", "clip_id": "semantic-pass"},
        {"editorial_placement_id": "two", "clip_id": "semantic-fail"},
    ]}
    repaired = _rollback_placements_to_control(semantic, control, {"two"})
    assert [row["clip_id"] for row in repaired["mappings"]] == ["semantic-pass", "legacy-2"]


def test_completed_variant_requires_render_artifacts_and_matching_destination(tmp_path) -> None:
    variant = tmp_path / "control"
    variant.mkdir()
    (variant / "replacement_dialogue.wav").write_bytes(b"x" * 45)
    (variant / "translation_output.mp4").write_bytes(b"video")
    from cinelingus.util import write_json
    write_json(variant / "final_schedule.json", {
        "destination_media_hash": "destination",
        "voice_residue_verification": {"status": "NONE_DETECTED"},
        "rendered_dialogue_verification": {"status": "PASS"},
    })
    assert _completed_variant(variant, {"destination_media_hash": "destination"}) is not None
    assert _completed_variant(variant, {"destination_media_hash": "different"}) is None


def test_intervention_verification_identifies_failed_changed_donor() -> None:
    control = {"mappings": [{
        "editorial_placement_id": "one", "window_id": "d1", "clip_id": "c1", "source_performance_id": "p1",
    }]}
    semantic = {"mappings": [{
        "editorial_placement_id": "one", "window_id": "d1", "clip_id": "c2", "source_performance_id": "p2",
    }]}
    control_final = {**control, "rendered_dialogue_verification": {"mappings": [{
        "editorial_placement_id": "one", "status": "fail", "word_coverage_percentage": 0.0,
    }]}}
    semantic_final = {**semantic, "rendered_dialogue_verification": {"mappings": [{
        "editorial_placement_id": "one", "status": "pass", "word_coverage_percentage": 100.0,
    }]}}
    accepted = assess_semantic_intervention(
        control_requested=control, semantic_requested=semantic,
        control_final=control_final, semantic_final=semantic_final,
    )
    assert accepted["state"] == "PASS"
    semantic_final["rendered_dialogue_verification"]["mappings"][0]["status"] = "fail"
    rejected = assess_semantic_intervention(
        control_requested=control, semantic_requested=semantic,
        control_final=control_final, semantic_final=semantic_final,
    )
    assert rejected["rejected_source_performance_ids"] == ["p2"]
