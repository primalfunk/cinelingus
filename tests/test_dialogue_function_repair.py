from cinelingus.dialogue_function.repair import finalize_function_repairs, propose_function_repairs


def _compat(score: float, confidence: float = 0.9) -> dict:
    return {"available": True, "normalized_function_contribution": score, "confidence": confidence}


def _schedule() -> dict:
    return {"mappings": [{
        "window_id": "w1", "clip_id": "bad", "source_performance_id": "bad",
        "destination_timestamp": 0.0, "planned_render_duration": 2.0,
        "dialogue_function_compatibility": _compat(0.2),
    }]}


def _verification(state: str = "FUNCTION_MISMATCH", confidence: float = 0.9) -> dict:
    return {"verification_signature": "verify", "mappings": [{
        "mapping_index": 0, "placement_key": "w1", "verification_state": state,
        "planned_classification_confidence": confidence, "rendered_transcript_confidence": confidence,
    }]}


def test_function_repair_prefers_function_then_semantic_and_waits_for_render() -> None:
    donors = [{"id": "semantic", "usable": True}, {"id": "function", "usable": True}]
    scores = {
        "semantic": {"score": 0.9, "hard_gate_passed": True, "components": {"duration_similarity": 0.9}, "dialogue_function_compatibility": _compat(0.7), "semantic_compatibility": {"normalized_semantic_contribution": 0.95}},
        "function": {"score": 0.8, "hard_gate_passed": True, "components": {"duration_similarity": 0.9}, "dialogue_function_compatibility": _compat(0.95), "semantic_compatibility": {"normalized_semantic_contribution": 0.6}},
    }
    result = propose_function_repairs(
        schedule=_schedule(), function_verification=_verification(), windows=[{"id": "w1", "duration": 2.0}],
        legal_donors=donors, score_candidate=lambda _window, donor: scores[donor["id"]],
        build_mapping=lambda window, donor, score: {"window_id": window["id"], "clip_id": donor["id"], "dialogue_function_compatibility": score["dialogue_function_compatibility"]},
    )

    assert result["candidate_schedule"]["mappings"][0]["clip_id"] == "function"
    assert result["repair_report"]["repair_state"] == "PROPOSED_PENDING_RENDER_VERIFICATION"
    assert result["repair_report"]["proposals"][0]["semantic_secondary_score"] == 0.6


def test_uncertain_function_mismatch_does_not_force_repair() -> None:
    result = propose_function_repairs(
        schedule=_schedule(), function_verification=_verification(confidence=0.4),
        windows=[{"id": "w1", "duration": 2.0}], legal_donors=[{"id": "other"}],
        score_candidate=lambda *_: {}, build_mapping=lambda *_: {},
    )
    assert result["repair_report"]["proposal_count"] == 0
    assert result["candidate_schedule"] == _schedule()


def test_function_repair_commit_and_rollback_are_render_gated() -> None:
    original = _schedule()
    candidate = {"mappings": [{"window_id": "w1", "clip_id": "fixed"}]}
    report = {"proposals": [{"mapping_index": 0, "old_clip_id": "bad", "new_clip_id": "fixed"}]}
    accepted = finalize_function_repairs(
        original_schedule=original, candidate_schedule=candidate, repair_report=report,
        rendered_function_verification={"mappings": [{"mapping_index": 0, "verification_state": "VERIFIED"}]},
        quality_before={"0": 0.7}, quality_after={"0": 0.72},
    )
    rolled_back = finalize_function_repairs(
        original_schedule=original, candidate_schedule=candidate, repair_report=report,
        rendered_function_verification={"mappings": [{"mapping_index": 0, "verification_state": "VERIFIED"}]},
        quality_before={"0": 0.7}, quality_after={"0": 0.6},
    )
    assert accepted["schedule"]["mappings"][0]["clip_id"] == "fixed"
    assert accepted["repair_report"]["accepted_count"] == 1
    assert rolled_back["schedule"]["mappings"][0]["clip_id"] == "bad"
    assert rolled_back["repair_report"]["rollback_count"] == 1
