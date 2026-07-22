from pathlib import Path

from cinelingus.semantic import (
    SemanticEntity,
    SemanticMode,
    SemanticScheduleContext,
)
from cinelingus.semantic.experiment import (
    _aggregate_opportunity_audits,
    _eligible_render_variant,
    _select_direct_pareto_admissions,
    run_semantic_schedule_screen,
)
from cinelingus.validation import validate_artifact


def _context() -> SemanticScheduleContext:
    return SemanticScheduleContext(
        SemanticMode.REPORT_ONLY,
        0.0,
        {
            "e1": SemanticEntity("source_1", "source", "speech_passage", "en", (1.0, 0.0), {}),
            "e2": SemanticEntity("source_2", "source", "speech_passage", "en", (0.0, 1.0), {}),
        },
        {"w1": SemanticEntity("destination_1", "destination", "speech_passage", "en", (0.0, 1.0), {})},
        {"model_id": "deterministic-test"},
    )


def test_schedule_screen_proves_zero_influence_and_selects_changed_candidate(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": "1.wav", "movie_timestamp": 0.0, "duration": 2.0, "transcript": "weather", "confidence": 0.8, "usable": True},
        {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": "2.wav", "movie_timestamp": 2.0, "duration": 2.0, "transcript": "sunny", "confidence": 0.8, "usable": True},
    ]
    windows = [{"id": "w1", "start": 0.0, "end": 2.0, "duration": 2.0, "transcript": "sunny", "confidence": 0.8, "usable": True}]

    report = run_semantic_schedule_screen(
        clips=clips, windows=windows, semantic_evidence=_context(), output_dir=tmp_path,
        source_hash="source", destination_hash="destination", max_time_stretch=0.1,
    )

    assert all(report["invariants"][key] for key in (
        "report_only_selection_equivalent", "report_only_scores_equivalent",
        "zero_weight_selection_equivalent", "zero_weight_scores_equivalent",
    ))
    variants = {row["variant_id"]: row for row in report["variants"]}
    assert variants["assisted_000"]["mode"] == "SEMANTIC_ASSISTED"
    assert variants["assisted_020"]["placements_changed"] == 1
    assert variants["assisted_020"]["conflict_count"] == 0
    assert report["render_selection"] == ["control", "report_only", "assisted_005"]
    validate_artifact("semantic_schedule_screen", tmp_path / "semantic_schedule_screen.json", Path("schemas"))


def test_schedule_screen_rejects_invalid_weight_grid(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="between 0 and 1"):
        run_semantic_schedule_screen(
            clips=[], windows=[], semantic_evidence=_context(), output_dir=tmp_path,
            source_hash="source", destination_hash="destination", max_time_stretch=0.1,
            weights=(-0.1, 0.2),
        )


def test_repair_quarantine_applies_to_every_rescreen_variant_after_stable_grouping(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": "1.wav", "movie_timestamp": 0.0, "duration": 2.0, "transcript": "weather", "confidence": 0.8, "usable": True},
        {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": "2.wav", "movie_timestamp": 2.0, "duration": 2.0, "transcript": "sunny", "confidence": 0.8, "usable": True},
    ]
    report = run_semantic_schedule_screen(
        clips=clips,
        windows=[{"id": "w1", "start": 0.0, "end": 2.0, "duration": 2.0, "transcript": "sunny", "confidence": 0.8, "usable": True}],
        semantic_evidence=_context(), output_dir=tmp_path,
        source_hash="source", destination_hash="destination", max_time_stretch=0.1,
        prohibited_source_performance_ids={"source_group_000001"},
        repair_preflight={"evidence_type": "acoustic_preflight", "preflight_signature": "repair"},
    )
    assert all(report["invariants"][key] for key in (
        "report_only_selection_equivalent", "zero_weight_selection_equivalent",
    ))
    for path in tmp_path.glob("*_schedule.json"):
        artifact = __import__("json").loads(path.read_text(encoding="utf-8"))
        assert artifact["prohibited_source_performance_ids"] == ["source_group_000001"]
        assert all(row["clip_id"] != "c1" for row in artifact["mappings"])


def test_render_nomination_requires_full_selected_placement_semantic_coverage() -> None:
    row = {
        "mode": "SEMANTIC_ASSISTED", "weight": 0.05, "placements_changed": 1,
        "conflict_count": 0, "mapping_count": 4, "semantic_placement_count": 3,
    }
    assert not _eligible_render_variant(row)
    row["semantic_placement_count"] = 4
    assert _eligible_render_variant(row)


def test_render_nomination_allows_only_explicitly_bounded_legacy_tradeoff() -> None:
    row = {
        "mode": "SEMANTIC_ASSISTED", "weight": 0.1, "placements_changed": 1,
        "conflict_count": 0, "mapping_count": 1, "semantic_placement_count": 1,
        "legacy_score_tradeoff_count": 1,
        "legacy_score_tradeoffs": [{"legacy_score_delta": -0.0024}],
    }
    assert _eligible_render_variant(row)
    row["legacy_score_tradeoffs"][0]["legacy_score_delta"] = -0.0026
    assert not _eligible_render_variant(row)


def test_guarded_render_nomination_requires_direct_admitted_mapping_coverage() -> None:
    row = {
        "mode": "SEMANTIC_ASSISTED", "weight": 0.0,
        "admission_mode": "DIRECT_EVIDENCE_GLOBAL_PARETO_V1", "admission_count": 1,
        "placements_changed": 1, "conflict_count": 0, "legacy_score_tradeoff_count": 0,
        "mapping_count": 4, "semantic_placement_count": 2,
        "admitted_mapping_count": 1, "admitted_direct_semantic_mapping_count": 0,
    }
    assert not _eligible_render_variant(row)
    row["admitted_direct_semantic_mapping_count"] = 1
    assert _eligible_render_variant(row)


def test_opportunity_audit_aggregate_is_report_only_and_counts_destinations() -> None:
    schedule = {"mappings": [{"source_performance_id": "s2", "destination_performance_id": "p2"}], "performance_decisions": [{
        "destination_performance_id": "p1",
        "semantic_opportunity_audit": {
            "legal_candidate_count": 3, "higher_semantic_candidate_count": 2,
            "placement_valid_candidate_count": 1, "fully_covered_candidate_count": 1,
            "opportunities": [{"source_performance_id": "s2", "semantic_delta": 0.1}],
        },
    }]}
    report = _aggregate_opportunity_audits(schedule)
    assert report["audited_destination_performance_count"] == 1
    assert report["local_pareto_safe_opportunity_count"] == 1
    assert report["pareto_safe_opportunity_count"] == 0
    assert report["opportunities"][0]["destination_performance_id"] == "p1"
    assert report["opportunities"][0]["conflicting_destination_performance_ids"] == ["p2"]

    schedule["performance_decisions"][0]["semantic_opportunity_audit"]["opportunities"][0]["two_cycle_swap"] = {
        "state": "ADMISSIBLE_TWO_CYCLE"
    }
    admitted = _aggregate_opportunity_audits(schedule)
    assert admitted["pareto_safe_opportunity_count"] == 1
    assert admitted["opportunities"][0]["global_admission_mode"] == "TWO_CYCLE"


def test_direct_pareto_admission_is_deterministic_and_rejects_broad_or_cyclic_evidence() -> None:
    common = {
        "globally_admissible": True, "global_admission_mode": "DIRECT",
        "selected_source_evidence_scope": "direct_passage",
        "candidate_source_evidence_scope": "direct_passage",
        "destination_evidence_scope": "direct_passage",
        "selected_mapping_count": 1, "selected_direct_semantic_mapping_count": 1,
        "candidate_mapping_count": 1, "candidate_direct_semantic_mapping_count": 1,
        "compatibility_deltas": {"performance": 0.0},
    }
    admissions = _select_direct_pareto_admissions({"opportunities": [
        {**common, "destination_performance_id": "d1", "displaced_source_performance_id": "old", "source_performance_id": "lower", "semantic_delta": 0.1},
        {**common, "destination_performance_id": "d1", "displaced_source_performance_id": "old", "source_performance_id": "higher", "semantic_delta": 0.2},
        {**common, "destination_performance_id": "d2", "displaced_source_performance_id": "old2", "source_performance_id": "cyclic", "semantic_delta": 0.3, "global_admission_mode": "TWO_CYCLE"},
        {**common, "destination_performance_id": "d3", "displaced_source_performance_id": "old3", "source_performance_id": "broad", "semantic_delta": 0.4, "candidate_source_evidence_scope": "performance_passage_aggregate"},
        {**common, "destination_performance_id": "d4", "displaced_source_performance_id": "old4", "source_performance_id": "clip_fallback", "semantic_delta": 0.5, "candidate_direct_semantic_mapping_count": 0},
    ]})

    assert list(admissions) == ["d1"]
    assert admissions["d1"]["source_performance_id"] == "higher"
    assert admissions["d1"]["semantic_delta"] == 0.2

    repaired = _select_direct_pareto_admissions(
        {"opportunities": [
            {**common, "destination_performance_id": "d1", "displaced_source_performance_id": "old", "source_performance_id": "higher", "semantic_delta": 0.2},
            {**common, "destination_performance_id": "d1", "displaced_source_performance_id": "old", "source_performance_id": "lower", "semantic_delta": 0.1},
        ]},
        prohibited_source_performance_ids={"higher"},
    )
    assert repaired["d1"]["source_performance_id"] == "lower"


def test_direct_pareto_admission_selects_fully_direct_positive_two_cycle_atomically() -> None:
    direct = {
        "globally_admissible": True, "global_admission_mode": "TWO_CYCLE",
        "selected_source_evidence_scope": "direct_passage",
        "candidate_source_evidence_scope": "direct_passage",
        "destination_evidence_scope": "direct_passage",
        "selected_mapping_count": 1, "selected_direct_semantic_mapping_count": 1,
        "candidate_mapping_count": 1, "candidate_direct_semantic_mapping_count": 1,
    }
    admissions = _select_direct_pareto_admissions({"opportunities": [{
        **direct, "destination_performance_id": "d1", "displaced_source_performance_id": "s1",
        "source_performance_id": "s2", "semantic_delta": 0.2,
        "compatibility_deltas": {"performance": 0.0},
        "two_cycle_swap": {
            "state": "ADMISSIBLE_TWO_CYCLE", "target_destination_performance_id": "d2",
            "replacement_source_performance_id": "s1", "net_semantic_delta": 0.1,
            "selected_mapping_count": 1, "selected_direct_semantic_mapping_count": 1,
            "candidate_mapping_count": 1, "candidate_direct_semantic_mapping_count": 1,
            "compatibility_deltas": {"performance": 0.0},
        },
    }]})

    assert set(admissions) == {"d1", "d2"}
    assert admissions["d1"]["source_performance_id"] == "s2"
    assert admissions["d2"]["source_performance_id"] == "s1"
    assert admissions["d1"]["cycle_id"] == admissions["d2"]["cycle_id"]
