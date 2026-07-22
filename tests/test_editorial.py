from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from cinelingus.editorial import EditorialMemory, EditorialPassManager, build_repair_batch, evaluate_editorial_decisions
from cinelingus.editorial.editorial_reports import build_editorial_report, build_repair_effectiveness_report
from cinelingus.editorial.repair_strategies import STRATEGIES, repair_strategy_for
from cinelingus.editorial.repair_engine import _has_boundary_integrity_failure, _strategy_diverse_selection, build_repair_neighborhoods
from cinelingus.editorial.quality_model import placement_quality
from cinelingus.editorial.pass_manager import _overlapping_repair_groups
from cinelingus.render_verification import merge_rendered_dialogue_verification
from cinelingus.render import _localized_schedule, _mapping_suppression_regions, render_schedule_regions_over_original_audio
from cinelingus.pipeline import Pipeline
from cinelingus.util import write_json


def _mapping(clip_id: str = "clip-a", score: float = 0.8) -> dict:
    return {
        "id": f"map-{clip_id}", "window_id": "window-1", "clip_id": clip_id,
        "editorial_placement_id": "editorial_placement_000001",
        "destination_timestamp": 10.0, "planned_render_duration": 2.0,
        "alignment_slot_start": 10.0, "alignment_slot_end": 12.0,
        "source_transcript": "we have to leave now", "score": score,
        "stretch_factor": 1.0, "cinematic_compatibility_score": score,
        "cinematic_compatibility_categories": {"visual": 0.8, "editing": 0.9},
        "performance_similarity_score": 0.8, "speaker_match_preserved": True,
        "destination_performance_signature": {"speaker_confidence": 0.9},
        "render_operations": [{"operation": "delay", "seconds": 10.0}],
    }


def _verification(*, coverage: float, status: str = "fail", clip_id: str = "clip-a") -> dict:
    row = {
        "mapping_index": 0, "mapping_id": f"map-{clip_id}", "window_id": "window-1",
        "editorial_placement_id": "editorial_placement_000001",
        "clip_id": clip_id, "intended_transcript": "we have to leave now",
        "rendered_transcript": "we have", "word_coverage_percentage": coverage,
        "missing_sentence_beginning": False, "missing_sentence_ending": coverage < 100,
        "mid_word_cut": False, "audio_masking_possible": False,
        "fade_masking_possible": False, "confidence": 0.9, "status": status,
    }
    return {
        "status": status.upper(), "mapping_count": 1, "measurable_mapping_count": 1,
        "failed_mapping_count": int(status == "fail"), "warning_mapping_count": int(status == "warning"),
        "average_word_coverage_percentage": coverage, "mappings": [row],
    }


def test_quality_model_normalizes_only_available_evidence() -> None:
    quality = placement_quality(mapping=_mapping(), verification=_verification(coverage=80)["mappings"][0])
    assert 0.0 <= quality["score"] <= 1.0
    assert abs(sum(quality["weights"].values()) - 1.0) < 0.001
    assert "render_completeness" in quality["contributors"]
    assert "reuse_integrity" in quality["unavailable_contributors"]


def test_decision_engine_classifies_incomplete_low_coverage_placement() -> None:
    result = evaluate_editorial_decisions(
        schedule={"mappings": [_mapping()]}, rendered_verification=_verification(coverage=40),
    )
    decision = result["decisions"][0]
    assert decision["recommendation"] == "repair"
    assert {row["category"] for row in decision["failures"]} >= {"incomplete_sentence", "low_rendered_coverage"}


def test_decision_engine_consumes_matching_problem_report_evidence() -> None:
    result = evaluate_editorial_decisions(
        schedule={"mappings": [_mapping()]},
        rendered_verification=_verification(coverage=100, status="pass"),
        problem_report={"problems": [{
            "problem_type": "uncertain_speech_boundary", "severity": "medium",
            "start": 10.0, "end": 12.0, "reason": "uncertain consonant tail",
        }]},
    )
    decision = result["decisions"][0]
    assert any(row["category"] == "transition_artifact" for row in decision["failures"])
    assert decision["problem_evidence"][0]["problem_type"] == "uncertain_speech_boundary"


def test_strategy_benchmark_promotes_observed_secondary_failure_without_changing_normal_routing() -> None:
    mapping = {
        **_mapping(),
        "cinematic_compatibility_categories": {"visual": 0.1, "editing": 0.9},
    }
    verification = _verification(coverage=40)
    normal = EditorialPassManager(maximum_passes=1).decide(
        {"mappings": [mapping]}, verification, {"regions": []}
    )
    focused = EditorialPassManager(
        maximum_passes=1, benchmark_target_failure_category="visual_mismatch"
    ).decide({"mappings": [mapping]}, verification, {"regions": []})

    assert normal["decisions"][0]["repair_strategy"] == "repair_sentence_boundaries"
    assert focused["benchmark_mode"] == "strategy_isolation"
    assert focused["benchmark_target_placement_count"] == 1
    assert focused["decisions"][0]["repair_strategy"] == "repair_visual_intent"
    assert focused["decisions"][0]["recommendation"] == "repair"
    assert _has_boundary_integrity_failure(normal["decisions"][0]) is True
    assert _has_boundary_integrity_failure(focused["decisions"][0]) is False


def test_targeted_verification_replaces_stable_placement_after_donor_change() -> None:
    merged = merge_rendered_dialogue_verification(
        _verification(coverage=40), _verification(coverage=100, status="pass", clip_id="clip-b"),
    )
    assert merged["mapping_count"] == 1
    assert merged["mappings"][0]["clip_id"] == "clip-b"
    assert merged["status"] == "PASS"


def test_shared_performance_window_does_not_conflate_distinct_placements() -> None:
    first = _mapping("clip-a")
    second = {**_mapping("clip-b"), "editorial_placement_id": "editorial_placement_000002", "id": "map-clip-b"}
    first_verification = _verification(coverage=40)["mappings"][0]
    second_verification = {
        **_verification(coverage=80, status="warning", clip_id="clip-b")["mappings"][0],
        "editorial_placement_id": "editorial_placement_000002",
        "mapping_index": 1,
    }
    result = evaluate_editorial_decisions(
        schedule={"mappings": [first, second]},
        rendered_verification={"mappings": [first_verification, second_verification]},
    )
    assert [row["placement_key"] for row in result["decisions"]] == [
        "editorial_placement_000001", "editorial_placement_000002",
    ]


def test_repair_batch_uses_alternative_and_remembers_rejected_pair() -> None:
    schedule = {"mappings": [_mapping()]}
    decisions = evaluate_editorial_decisions(schedule=schedule, rendered_verification=_verification(coverage=40))
    memory = EditorialMemory()
    batch = build_repair_batch(
        schedule=schedule, decisions=decisions,
        donor_candidates=[{"id": "clip-a", "duration": 2.0}, {"id": "clip-b", "duration": 2.0}],
        memory=memory,
        score_candidate=lambda _window, clip: {"score": 0.95 if clip["id"] == "clip-b" else 0.1},
        build_mapping=lambda window, clip, score: {**_mapping(clip["id"], score["score"]), "window_id": window["id"]},
    )
    assert batch["repaired_count"] == 1
    assert batch["schedule"]["mappings"][0]["clip_id"] == "clip-b"
    assert memory.rejected("editorial_placement_000001", "clip-a")


def test_repair_batch_extends_truncated_source_before_reassigning_donor() -> None:
    mapping = {
        **_mapping(), "clip_trim_start": 0.0, "clip_trim_duration": 1.7,
        "planned_render_duration": 1.7, "alignment_slot_end": 12.4,
    }
    decisions = evaluate_editorial_decisions(
        schedule={"mappings": [mapping]}, rendered_verification=_verification(coverage=40),
    )
    batch = build_repair_batch(
        schedule={"mappings": [mapping]}, decisions=decisions,
        donor_candidates=[{"id": "clip-a", "duration": 2.2}, {"id": "clip-b", "duration": 2.0}],
        memory=EditorialMemory(),
        score_candidate=lambda _window, clip: {"score": 0.9 if clip["id"] == "clip-b" else 0.2},
        build_mapping=lambda window, clip, score: {**_mapping(clip["id"], score["score"]), "window_id": window["id"]},
    )

    repaired = batch["schedule"]["mappings"][0]
    assert repaired["clip_id"] == "clip-a"
    assert repaired["clip_trim_duration"] == 2.2
    assert repaired["planned_render_duration"] == 2.2
    assert repaired["editorial_candidate_family"] == "source_boundary_extension"
    assert batch["attempts"][0]["repair_strategy"] == "extend_source_boundary"
    assert batch["attempts"][0]["candidate_family"] == "source_boundary_extension"


def test_required_failure_strategies_expose_executable_contracts() -> None:
    assert set(STRATEGIES) == {
        "incomplete_sentence", "mid_word_cut", "low_rendered_coverage", "duration_failure",
        "speaker_mismatch", "visual_mismatch", "transition_artifact", "residual_dialogue",
        "masking", "performance_mismatch", "reuse_exhaustion", "confidence_collapse",
    }
    for spec in STRATEGIES.values():
        assert spec["families"] and spec["hard_constraints"] and spec["scoring_focus"]
        assert spec["maximum_attempts"] >= 1 and spec["fallback"] and spec["verification"]


def test_repair_budget_reserves_space_for_distinct_strategies() -> None:
    decisions = [
        {
            "placement_key": f"sentence-{index}", "mapping_index": index,
            "overall_quality": 0.1 + index / 1000, "recommendation": "repair",
            "failures": [{"category": "incomplete_sentence"}],
        }
        for index in range(10)
    ] + [{
        "placement_key": "speaker", "mapping_index": 20, "overall_quality": 0.6,
        "recommendation": "repair", "failures": [{"category": "speaker_mismatch"}],
    }]

    selected = _strategy_diverse_selection(decisions, 3)

    assert {repair_strategy_for(row)["strategy"] for row in selected} == {
        "repair_sentence_boundaries", "repair_speaker_role",
    }
    assert len(selected) == 3


def test_adjacent_performance_failures_form_atomic_repair_neighborhood() -> None:
    first = {**_mapping("clip-a"), "destination_performance_id": "performance-1"}
    second = {
        **_mapping("clip-c"), "id": "map-clip-c", "editorial_placement_id": "editorial_placement_000002",
        "destination_timestamp": 12.2, "alignment_slot_start": 12.2, "alignment_slot_end": 14.2,
        "destination_performance_id": "performance-1",
    }
    decisions = {"decisions": [
        {"placement_key": "editorial_placement_000001", "mapping_index": 0, "overall_quality": 0.3, "recommendation": "repair", "failures": [{"category": "performance_mismatch"}]},
        {"placement_key": "editorial_placement_000002", "mapping_index": 1, "overall_quality": 0.3, "recommendation": "repair", "failures": [{"category": "performance_mismatch"}]},
    ]}
    schedule = {"mappings": [first, second]}

    neighborhoods = build_repair_neighborhoods(schedule, decisions)
    assert len(neighborhoods) == 1
    assert neighborhoods[0]["placement_keys"] == ["editorial_placement_000001", "editorial_placement_000002"]

    def score(window, clip):
        current = "clip-a" if float(window["start"]) == 10.0 else "clip-c"
        return {"score": 0.5 if clip["id"] == current else 0.2 if clip["id"] in {"clip-a", "clip-c"} else 0.9,
                "editorial_components": {"performance_fit": 0.9}}

    batch = build_repair_batch(
        schedule=schedule, decisions=decisions,
        donor_candidates=[{"id": value, "duration": 2.0} for value in ("clip-a", "clip-b", "clip-c", "clip-d")],
        memory=EditorialMemory(), score_candidate=score,
        build_mapping=lambda window, clip, data: {
            **_mapping(clip["id"], data["score"]), "window_id": window["id"],
            "destination_timestamp": window["start"], "destination_performance_id": "performance-1",
        },
    )
    assert batch["coordinated_neighborhood_count"] == 1
    assert len({row["assignment_group_id"] for row in batch["repairs"]}) == 1
    assert all(row.get("coordination_mode") == "coordinated_neighborhood" for row in batch["repairs"])


def test_sentence_integrity_repairs_are_not_grouped_by_secondary_mismatch() -> None:
    first = {**_mapping("clip-a"), "destination_performance_id": "performance-1"}
    second = {
        **_mapping("clip-b"), "editorial_placement_id": "editorial_placement_000002",
        "destination_timestamp": 12.1, "destination_performance_id": "performance-1",
    }
    decisions = {"decisions": [
        {"placement_key": "editorial_placement_000001", "mapping_index": 0, "failures": [{"category": "incomplete_sentence"}, {"category": "performance_mismatch"}]},
        {"placement_key": "editorial_placement_000002", "mapping_index": 1, "failures": [{"category": "incomplete_sentence"}, {"category": "performance_mismatch"}]},
    ]}
    assert build_repair_neighborhoods({"mappings": [first, second]}, decisions) == []


@pytest.mark.parametrize(
    ("category", "family", "field"),
    [
        ("residual_dialogue", "suppression_expansion", "suppression_trailing_padding"),
        ("masking", "audio_treatment", "gain_db"),
        ("transition_artifact", "audio_edge_adjustment", "fade_duration"),
        ("duration_failure", "time_adaptation", "stretch_factor"),
    ],
)
def test_local_failure_strategies_change_renderer_visible_mapping(category: str, family: str, field: str) -> None:
    mapping = {**_mapping(), "clip_trim_duration": 1.5, "stretch_factor": 1.2, "planned_render_duration": 1.8}
    decision = {
        "placement_key": "editorial_placement_000001", "mapping_index": 0,
        "overall_quality": 0.3, "recommendation": "repair",
        "failures": [{"category": category, "severity": "critical"}],
    }
    batch = build_repair_batch(
        schedule={"mappings": [mapping]}, decisions={"decisions": [decision]},
        donor_candidates=[{"id": "clip-a", "duration": 1.5}], memory=EditorialMemory(),
        score_candidate=lambda _window, _clip: {"score": 0.5},
        build_mapping=lambda *_args: {},
    )

    assert batch["repaired_count"] == 1
    assert batch["attempts"][0]["candidate_family"] == family
    assert field in batch["schedule"]["mappings"][0]
    assert batch["attempts"][0]["strategy_plan"]["maximum_attempts"] >= 1


def test_confidence_collapse_conservatively_retains_without_speculative_render() -> None:
    decision = {
        "placement_key": "editorial_placement_000001", "mapping_index": 0,
        "overall_quality": 0.3, "recommendation": "repair",
        "failures": [{"category": "confidence_collapse", "severity": "high"}],
    }
    assert repair_strategy_for(decision)["strategy"] == "conservative_uncertainty_retention"
    batch = build_repair_batch(
        schedule={"mappings": [_mapping()]}, decisions={"decisions": [decision]},
        donor_candidates=[{"id": "clip-a"}, {"id": "clip-b"}], memory=EditorialMemory(),
        score_candidate=lambda *_args: {"score": 0.99}, build_mapping=lambda *_args: {},
    )
    assert batch["repairs"] == []
    assert batch["attempts"][0]["candidate_loss_stage"] == "conservative_retention"


def test_pass_manager_marks_conservative_uncertainty_as_skipped_unrepairable() -> None:
    verification = _verification(coverage=100, status="warning")
    verification["mappings"][0]["confidence"] = 0.1
    manager = EditorialPassManager(maximum_passes=1, acceptance_threshold=0.99)
    result = manager.run(
        schedule={"mappings": [_mapping()]}, verification=verification, residue={"regions": []},
        repair_callback=lambda _schedule, decisions, _memory, _pass: {
            "schedule": _schedule, "repairs": [], "regions": [],
            "attempts": [{
                "placement_key": decisions["decisions"][0]["placement_key"],
                "candidate_loss_stage": "conservative_retention", "proposed": False,
                "no_viable_alternative": True,
            }],
        },
        render_verify_callback=lambda *_args: (verification, {"regions": []}),
    )
    assert result["final_decisions"]["decisions"][0]["final_state"] == "SKIPPED_UNREPAIRABLE"


def test_local_suppression_padding_expands_only_touching_region() -> None:
    schedule = {
        "mappings": [{
            **_mapping(), "destination_timestamp": 10.0, "planned_render_duration": 2.0,
            "suppression_leading_padding": 0.1, "suppression_trailing_padding": 0.2,
        }],
        "destination_speech_regions": [
            {"id": "a", "start": 10.0, "end": 12.0},
            {"id": "b", "start": 20.0, "end": 22.0},
        ],
    }
    regions = _mapping_suppression_regions(schedule)
    assert regions[0]["start"] == 9.9 and regions[0]["end"] == 12.2
    assert regions[1]["start"] == 20.0 and regions[1]["end"] == 22.0


def test_repair_batch_can_atomically_swap_occupied_donors() -> None:
    first = _mapping("clip-a")
    second = {
        **_mapping("clip-b"), "editorial_placement_id": "editorial_placement_000002",
        "destination_timestamp": 20.0, "alignment_slot_start": 20.0, "alignment_slot_end": 22.0,
    }
    decisions = {
        "decisions": [
            {"placement_key": "editorial_placement_000001", "mapping_index": 0, "overall_quality": 0.2, "recommendation": "repair", "failures": []},
            {"placement_key": "editorial_placement_000002", "mapping_index": 1, "overall_quality": 0.2, "recommendation": "repair", "failures": []},
        ]
    }

    def score(window, clip):
        preferred = "clip-b" if float(window["start"]) == 10.0 else "clip-a"
        return {"score": 0.9 if clip["id"] == preferred else 0.2}

    def build(window, clip, score_data):
        return {
            "window_id": window["id"], "clip_id": clip["id"],
            "destination_timestamp": window["start"], "planned_render_duration": window["duration"],
            "score": score_data["score"], "enabled": True,
        }

    batch = build_repair_batch(
        schedule={"mappings": [first, second]}, decisions=decisions,
        donor_candidates=[{"id": "clip-a", "duration": 2.0}, {"id": "clip-b", "duration": 2.0}],
        memory=EditorialMemory(), score_candidate=score, build_mapping=build,
    )

    assert [row["clip_id"] for row in batch["schedule"]["mappings"]] == ["clip-b", "clip-a"]
    assert batch["repaired_count"] == 2
    assert len({row["assignment_group_id"] for row in batch["repairs"]}) == 1
    assert all(row["candidates_considered"] > 0 for row in batch["attempts"])
    assert all(row["candidate_family"] == "atomic_donor_swap" for row in batch["attempts"])


def test_atomic_assignment_repairs_group_even_when_regions_are_far_apart() -> None:
    repairs = [
        {"placement_key": "a", "assignment_group_id": "swap-1", "region": {"start": 1.0, "end": 2.0}},
        {"placement_key": "b", "assignment_group_id": "swap-1", "region": {"start": 50.0, "end": 51.0}},
        {"placement_key": "c", "region": {"start": 100.0, "end": 101.0}},
    ]

    groups = _overlapping_repair_groups(repairs)

    assert sorted(len(group) for group in groups) == [1, 2]


def test_render_failure_does_not_admit_unrelated_predicted_regression() -> None:
    decisions = {
        "decisions": [{
            "placement_key": "editorial_placement_000001", "mapping_index": 0,
            "overall_quality": 0.2, "recommendation": "repair",
            "failures": [{"category": "low_rendered_coverage"}],
        }]
    }

    def score(_window, clip):
        if clip["id"] == "clip-a":
            return {"score": 0.8, "editorial_components": {"sentence_fit": 0.8}}
        return {"score": 0.6, "editorial_components": {"sentence_fit": 0.8}}

    batch = build_repair_batch(
        schedule={"mappings": [_mapping("clip-a")]}, decisions=decisions,
        donor_candidates=[{"id": "clip-a", "duration": 2.0}, {"id": "clip-b", "duration": 2.0}],
        memory=EditorialMemory(), score_candidate=score,
        build_mapping=lambda window, clip, data: {
            "window_id": window["id"], "clip_id": clip["id"],
            "destination_timestamp": window["start"], "planned_render_duration": window["duration"],
            "score": data["score"],
        },
    )

    assert batch["repairs"] == []
    assert batch["attempts"][0]["candidate_loss_stage"] == "pre_render_quality_ceiling"


def test_rendered_boundary_failure_admits_one_credible_near_ceiling_challenger() -> None:
    decisions = {"decisions": [{
        "placement_key": "editorial_placement_000001", "mapping_index": 0,
        "overall_quality": 0.4, "recommendation": "repair",
        "failures": [{"category": "incomplete_sentence", "severity": "high"}],
    }]}

    def score(_window, clip):
        value = 0.8 if clip["id"] == "clip-a" else 0.78
        return {
            "score": value,
            "editorial_components": {
                "sentence_fit": 1.0, "timing_and_render_fit": 0.8,
            },
        }

    batch = build_repair_batch(
        schedule={"mappings": [_mapping("clip-a")]}, decisions=decisions,
        donor_candidates=[
            {"id": "clip-a", "duration": 2.0, "transcript": "we have to leave now."},
            {"id": "clip-b", "duration": 2.0, "transcript": "please come with me."},
        ],
        memory=EditorialMemory(), score_candidate=score,
        build_mapping=lambda window, clip, data: {
            **_mapping(clip["id"], data["score"]), "window_id": window["id"],
        },
    )

    assert batch["repaired_count"] == 1
    assert batch["attempts"][0]["bounded_rendered_exploration"] is True
    assert batch["attempts"][0]["current_pre_render_score"] == 0.8


def test_repair_effectiveness_report_exposes_attempt_funnel() -> None:
    result = {
        "passes": [{
            "repair_attempts": [
                {
                    "placement_key": "a", "failure_categories": ["masking"],
                    "repair_strategy": "adjust_audio", "candidate_family": "audio_correction",
                    "candidates_considered": 3, "rendered": True, "survived": True,
                    "quality_delta": 0.1, "original_restored": False,
                    "no_viable_alternative": False, "candidate_rejection_reasons": {"too_loud": 1},
                },
                {
                    "placement_key": "b", "failure_categories": ["duration_failure"],
                    "repair_strategy": "adjust_duration", "candidate_family": None,
                    "candidates_considered": 2, "rendered": False, "survived": False,
                    "quality_delta": None, "original_restored": False,
                    "no_viable_alternative": True, "candidate_loss_stage": "pre_render_quality_ceiling",
                    "candidate_rejection_reasons": {"predicted_ceiling_below_required_delta": 1},
                },
            ]
        }]
    }

    report = build_repair_effectiveness_report(result)

    assert report["attempted_placement_count"] == 2
    assert report["candidate_survival_rate"] == 1.0
    assert report["original_already_predicted_best_count"] == 1
    assert {row["name"] for row in report["by_failure_category"]} == {"masking", "duration_failure"}


def test_pass_manager_accepts_improvement_and_reports_it() -> None:
    manager = EditorialPassManager(maximum_passes=1, acceptance_threshold=0.7)
    initial_schedule = {"mappings": [_mapping()]}

    def repair(schedule, _decisions, _memory, _pass):
        candidate = {"mappings": [_mapping("clip-b", 0.95)]}
        return {"schedule": candidate, "repairs": [{"placement_key": "editorial_placement_000001", "new_clip_id": "clip-b"}], "regions": [{"start": 10.0, "end": 12.0}]}

    result = manager.run(
        schedule=initial_schedule, verification=_verification(coverage=40), residue={"regions": []},
        repair_callback=repair,
        render_verify_callback=lambda *_args: (_verification(coverage=100, status="pass", clip_id="clip-b"), {"regions": []}),
    )
    report = build_editorial_report(result)
    assert result["schedule"]["mappings"][0]["clip_id"] == "clip-b"
    assert result["quality_improvement"] > 0
    assert report["placements_repaired"] == 1
    assert report["resumed_from_candidate_checkpoint"] is False
    assert result["final_state_counts"] == {"IMPROVED_ACCEPTED": 1}


def test_pass_manager_resumes_prepared_candidate_without_reselecting_donor() -> None:
    checkpoint = {}
    render_calls = []

    def repair(_schedule, _decisions, memory, _pass):
        memory.remember({
            "placement_key": "prior-placement",
            "failures": [{"category": "speaker_mismatch"}],
        }, clip_id="rejected-donor")
        return {
            "schedule": {"mappings": [_mapping("clip-b", 0.95)]},
            "repairs": [{"placement_key": "editorial_placement_000001", "new_clip_id": "clip-b"}],
            "regions": [{"start": 10.0, "end": 12.0}],
        }

    def interrupt_after_checkpoint(stage, state):
        checkpoint.update(deepcopy(state))
        assert stage == "candidate_prepared"
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        EditorialPassManager(maximum_passes=1, acceptance_threshold=0.7).run(
            schedule={"mappings": [_mapping()]}, verification=_verification(coverage=40),
            residue={"regions": []}, repair_callback=repair,
            render_verify_callback=lambda *_args: render_calls.append(True),
            progress_callback=interrupt_after_checkpoint,
        )

    result = EditorialPassManager(maximum_passes=1, acceptance_threshold=0.7).run(
        schedule={"mappings": [_mapping()]}, verification=_verification(coverage=40),
        residue={"regions": []},
        repair_callback=lambda *_args: pytest.fail("donor selection must not rerun"),
        render_verify_callback=lambda *_args: (
            render_calls.append(True) or _verification(coverage=100, status="pass", clip_id="clip-b"),
            {"regions": []},
        ),
        resume_state=checkpoint,
    )

    assert render_calls == [True]
    assert result["schedule"]["mappings"][0]["clip_id"] == "clip-b"
    assert result["resumed_from_candidate_checkpoint"] is True
    assert result["memory"]["rejected_pairs"] == [
        {"placement_key": "prior-placement", "clip_id": "rejected-donor"}
    ]
    assert result["repair_capabilities"]["interruption_recovery"] == {
        "granularity": "prepared_candidate",
        "donor_selection_replayed": False,
        "render_replayed_atomically": True,
        "avoidance_memory_restored": True,
    }


def test_pass_manager_rolls_back_regression() -> None:
    manager = EditorialPassManager(maximum_passes=1, acceptance_threshold=0.99)
    rollbacks = []
    result = manager.run(
        schedule={"mappings": [_mapping()]}, verification=_verification(coverage=70, status="warning"), residue={"regions": []},
        repair_callback=lambda *_args: {
            "schedule": {"mappings": [_mapping("clip-b", 0.2)]},
            "repairs": [{"placement_key": "editorial_placement_000001", "new_clip_id": "clip-b"}],
            "regions": [{"start": 10.0, "end": 12.0}],
        },
        render_verify_callback=lambda *_args: (_verification(coverage=10, clip_id="clip-b"), {"regions": []}),
        rollback_callback=lambda *_args: rollbacks.append(True),
    )
    assert result["schedule"]["mappings"][0]["clip_id"] == "clip-a"
    assert rollbacks == [True]
    assert result["passes"][-1]["accepted"] is False
    assert result["schedule"]["mappings"][0]["enabled"] is True
    assert result["final_decisions"]["decisions"][0]["final_action"] == "retained_best_known_after_bounded_repair"
    assert result["final_decisions"]["decisions"][0]["final_state"] == "BEST_KNOWN_UNRESOLVED"
    assert result["final_state_counts"] == {"BEST_KNOWN_UNRESOLVED": 1}
    assert "source_boundary_extension" in result["repair_capabilities"]["implemented_families"]
    assert result["placements_rejected"] == 1


def test_pass_manager_accepts_disjoint_improvement_and_rolls_back_only_regression() -> None:
    manager = EditorialPassManager(maximum_passes=1, acceptance_threshold=0.7)
    first = _mapping("clip-a")
    second = {
        **_mapping("clip-c"), "id": "map-clip-c", "window_id": "window-2",
        "editorial_placement_id": "editorial_placement_000002", "destination_timestamp": 20.0,
        "alignment_slot_start": 20.0, "alignment_slot_end": 22.0,
    }
    first_initial = _verification(coverage=40)["mappings"][0]
    second_initial = {
        **_verification(coverage=40, clip_id="clip-c")["mappings"][0],
        "mapping_index": 1, "mapping_id": "map-clip-c", "window_id": "window-2",
        "editorial_placement_id": "editorial_placement_000002",
    }
    first_candidate = _verification(coverage=100, status="pass", clip_id="clip-b")["mappings"][0]
    second_candidate = {
        **_verification(coverage=10, clip_id="clip-d")["mappings"][0],
        "mapping_index": 1, "mapping_id": "map-clip-d", "window_id": "window-2",
        "editorial_placement_id": "editorial_placement_000002",
    }
    rollbacks = []
    result = manager.run(
        schedule={"mappings": [first, second]},
        verification={"mappings": [first_initial, second_initial]}, residue={"regions": []},
        repair_callback=lambda *_args: {
            "schedule": {"mappings": [
                _mapping("clip-b"),
                {**second, "clip_id": "clip-d", "id": "map-clip-d"},
            ]},
            "repairs": [
                {"placement_key": "editorial_placement_000001", "mapping_index": 0, "new_clip_id": "clip-b", "region": {"start": 10.0, "end": 12.0}},
                {"placement_key": "editorial_placement_000002", "mapping_index": 1, "new_clip_id": "clip-d", "region": {"start": 20.0, "end": 22.0}},
            ],
            "regions": [{"start": 10.0, "end": 12.0}, {"start": 20.0, "end": 22.0}],
        },
        render_verify_callback=lambda *_args: ({"mappings": [first_candidate, second_candidate]}, {"regions": []}),
        rollback_callback=lambda _schedule, regions, _pass: rollbacks.extend(regions),
    )
    assert result["schedule"]["mappings"][0]["clip_id"] == "clip-b"
    assert result["schedule"]["mappings"][1]["clip_id"] == "clip-c"
    assert result["placements_repaired"] == 1
    assert rollbacks == [{"start": 20.0, "end": 22.0}]


def test_pass_manager_commits_entire_improving_neighborhood_atomically() -> None:
    first = _mapping("clip-a")
    second = {
        **_mapping("clip-c"), "id": "map-clip-c", "window_id": "window-2",
        "editorial_placement_id": "editorial_placement_000002",
        "destination_timestamp": 12.2, "alignment_slot_start": 12.2, "alignment_slot_end": 14.2,
    }
    first_initial = _verification(coverage=40)["mappings"][0]
    second_initial = {
        **_verification(coverage=40, clip_id="clip-c")["mappings"][0],
        "mapping_index": 1, "mapping_id": "map-clip-c", "window_id": "window-2",
        "editorial_placement_id": "editorial_placement_000002",
    }
    first_final = _verification(coverage=100, status="pass", clip_id="clip-b")["mappings"][0]
    second_final = {
        **_verification(coverage=100, status="pass", clip_id="clip-d")["mappings"][0],
        "mapping_index": 1, "mapping_id": "map-clip-d", "window_id": "window-2",
        "editorial_placement_id": "editorial_placement_000002",
    }
    group = "repair_neighborhood_0001"
    repairs = [
        {"placement_key": "editorial_placement_000001", "mapping_index": 0, "new_clip_id": "clip-b", "assignment_group_id": group, "repair_neighborhood_id": group, "coordination_mode": "coordinated_neighborhood", "region": {"start": 10.0, "end": 12.0}},
        {"placement_key": "editorial_placement_000002", "mapping_index": 1, "new_clip_id": "clip-d", "assignment_group_id": group, "repair_neighborhood_id": group, "coordination_mode": "coordinated_neighborhood", "region": {"start": 12.2, "end": 14.2}},
    ]
    attempts = [
        {"placement_key": row["placement_key"], "repair_strategy": "repair_performance_structure", "candidate_family": "coordinated_neighborhood", "coordinated_candidate": True, "repair_neighborhood_id": group, "candidates_considered": 2, "proposed": True}
        for row in repairs
    ]

    result = EditorialPassManager(maximum_passes=1, acceptance_threshold=0.7).run(
        schedule={"mappings": [first, second]},
        verification={"mappings": [first_initial, second_initial]}, residue={"regions": []},
        repair_callback=lambda *_args: {
            "schedule": {"mappings": [
                {**first, "clip_id": "clip-b", "id": "map-clip-b"},
                {**second, "clip_id": "clip-d", "id": "map-clip-d"},
            ]},
            "repairs": repairs, "attempts": attempts,
            "regions": [{"start": 10.0, "end": 14.2}],
        },
        render_verify_callback=lambda *_args: (
            {"mappings": [first_final, second_final]}, {"regions": []}
        ),
    )
    effectiveness = build_repair_effectiveness_report(result)

    assert result["placements_repaired"] == 2
    assert [row["clip_id"] for row in result["schedule"]["mappings"]] == ["clip-b", "clip-d"]
    assert effectiveness["coordinated_candidate_count"] == 2
    assert effectiveness["coordinated_surviving_count"] == 2
    assert all(row["survived"] for row in effectiveness["attempts"])


def test_incremental_schedule_localizes_only_affected_mappings_without_mutation() -> None:
    first = _mapping()
    second = {**_mapping("clip-b"), "id": "map-b", "window_id": "window-2", "destination_timestamp": 30.0}
    second["render_operations"] = [{"operation": "delay", "seconds": 30.0}]
    schedule = {
        "mappings": [first, second],
        "destination_speech_regions": [{"start": 9.5, "end": 12.5}, {"start": 29.5, "end": 32.5}],
    }
    localized = _localized_schedule(schedule, start=9.0, end=13.0)
    assert [row["window_id"] for row in localized["mappings"]] == ["window-1"]
    assert localized["mappings"][0]["destination_timestamp"] == 1.0
    assert localized["mappings"][0]["render_operations"][0]["seconds"] == 1.0
    assert schedule["mappings"][0]["render_operations"][0]["seconds"] == 10.0

    clipped = _localized_schedule(schedule, start=11.0, end=12.0)
    assert clipped["mappings"][0]["destination_timestamp"] == 0.0
    assert clipped["mappings"][0]["clip_trim_start"] == 1.0
    assert clipped["mappings"][0]["clip_trim_duration"] == 1.0
    assert clipped["mappings"][0]["render_operations"][0]["seconds"] == 0.0


def test_incremental_renderer_merges_start_end_regions_without_duration_key(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "existing.wav"
    output.write_bytes(b"existing")

    def fake_run(args):
        Path(args[-1]).write_bytes(b"carrier")

    def fake_render(**kwargs):
        kwargs["output_path"].write_bytes(b"patch")

    def fake_replace(**kwargs):
        kwargs["output"].write_bytes(b"merged")

    monkeypatch.setattr("cinelingus.render.run", fake_run)
    monkeypatch.setattr("cinelingus.render.render_schedule_over_original_audio", fake_render)
    monkeypatch.setattr("cinelingus.render._replace_wav_region", fake_replace)
    report = render_schedule_regions_over_original_audio(
        original_media=tmp_path / "source.mp4", schedule={"mappings": []},
        regions=[{"start": 1.0, "end": 2.0}, {"start": 2.04, "end": 3.0}],
        duration=4.0, output_path=output, sample_rate=48000, channels=2,
        target_lufs=-18.0, fade_duration=0.015,
    )
    assert report["region_count"] == 1
    assert report["regions"][0] == {"start": 0.85, "end": 3.15}


def test_performance_render_pipeline_runs_editorial_repair_loop(monkeypatch, tmp_path: Path) -> None:
    config = type("Config", (), {
        "output_dir": tmp_path / "output", "render_sample_rate": 48000, "render_channels": 2,
        "target_lufs": -18.0, "audio_fade_duration": 0.015, "original_duck_db": -28.0,
        "dialogue_suppression": "hard_mute", "suppression_padding": 0.04,
        "background_reconstruction": "neighboring_non_speech_with_adaptive_crossfades",
        "verify_voice_residue": True, "residue_correction_passes": 0, "residue_correction_padding": 0.12,
        "speech_backend": "whisper", "whisper_model": "medium", "whisper_language": "en",
        "transcription_mode": "quality", "max_time_stretch": 0.1,
        "editorial_refinement_enabled": True, "editorial_max_passes": 1,
        "editorial_acceptance_threshold": 0.72, "editorial_min_word_coverage": 0.72,
        "editorial_max_repairs_per_pass": 2, "editorial_incremental_render": True,
    })()
    pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.schemas_dir = Path.cwd() / "schemas"
    pipeline.destination = type("Dest", (), {
        "cache_dir": tmp_path / "cache", "media_path": tmp_path / "dest.mp4", "media_hash": "dest-hash",
    })()
    pipeline.source = type("Source", (), {"media_hash": "source-hash"})()
    pipeline.logger = type("Logger", (), {"info": lambda self, message: None})()
    pipeline.destination.cache_dir.mkdir(parents=True)
    pipeline.build_clip_library = lambda force=False: {"clips": [{"id": "clip-a"}, {"id": "clip-b"}]}
    pipeline.build_source_performances = lambda force=False: {"performances": []}
    transcriptions = {"count": 0}

    def fake_render(**kwargs):
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"rendered-audio" * 10)
        kwargs["schedule"]["background_reconstruction_report"] = {"reconstructed_region_count": 1}
        kwargs["schedule"]["suppression_padding_report"] = {"regions": []}

    def fake_incremental(**kwargs):
        kwargs["schedule"]["editorial_incremental_render_report"] = {"region_count": len(kwargs["regions"])}
        return kwargs["schedule"]["editorial_incremental_render_report"]

    def fake_transcribe(**kwargs):
        transcriptions["count"] += 1
        transcript = "noise only" if transcriptions["count"] == 1 else "we leave now"
        timeline = {
            "media_hash": kwargs["media_hash"],
            "windows": [{"start": 10.0, "end": 12.0, "transcript": transcript, "confidence": 0.95}],
        }
        write_json(kwargs["output_path"], timeline)
        return timeline

    def fake_build_mapping(window, clip, score, **_kwargs):
        return {
            **_mapping(clip["id"], score["score"]), "id": f"map-{clip['id']}",
            "window_id": window["id"], "source_transcript": "we leave now",
            "performance_similarity_score": 0.95,
        }

    monkeypatch.setattr("cinelingus.pipeline.render_schedule_over_original_audio", fake_render)
    monkeypatch.setattr("cinelingus.pipeline.render_schedule_regions_over_original_audio", fake_incremental)
    monkeypatch.setattr("cinelingus.pipeline.transcribe_with_whisper", fake_transcribe)
    monkeypatch.setattr("cinelingus.pipeline.prepare_editorial_repair_candidates", lambda clips, _performances: clips)
    monkeypatch.setattr("cinelingus.pipeline.score_editorial_repair_candidate", lambda _window, _clip, **_kwargs: {"score": 0.95})
    monkeypatch.setattr("cinelingus.pipeline.build_editorial_repair_mapping", fake_build_mapping)
    schedule = {
        "scheduling_mode": "performance_fill", "shot_boundary_mode": "off", "active_filter": "balanced",
        "unmatched_policy": "suppress_original_dialogue",
        "destination_speech_regions": [{"id": "window-1", "start": 10.0, "end": 12.0, "transcript": "original dialogue"}],
        "mappings": [_mapping("clip-a")],
    }
    schedule["mappings"][0]["source_transcript"] = "a completely different sentence"
    original_schedule = deepcopy(schedule)
    output = pipeline.render_audio_from_schedule(
        schedule=schedule, dest_movie={"duration": 20.0}, force=True, persist_schedule=False,
    )
    assert output.exists()
    assert schedule["mappings"][0]["clip_id"] == "clip-b"
    assert schedule["editorial_refinement"]["placements_repaired"] == 1
    assert (config.output_dir / "editorial_decisions.json").exists()
    assert (config.output_dir / "editorial_report.json").exists()

    pipeline.render_audio_from_schedule(
        schedule=original_schedule, dest_movie={"duration": 20.0}, force=True, persist_schedule=False,
    )
    assert transcriptions["count"] == 2
