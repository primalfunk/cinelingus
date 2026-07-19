from __future__ import annotations

from collections import Counter
from typing import Any


EVALUATION_VERSION = "montage_evaluation_v1"
REQUIRED_CATEGORIES = (
    "positive_boundaries",
    "negative_boundaries",
    "speech_heavy",
    "silent_or_non_dialogue",
    "hard_cuts",
    "dissolves",
    "fades",
    "long_takes",
    "reaction_coverage",
    "low_motion",
    "high_motion",
    "ambiguous",
    "single_shot_moments",
    "multi_shot_moments",
)


def build_montage_evaluation(
    *,
    core_records: list[dict[str, Any]],
    naive_records: list[dict[str, Any]],
    configuration: dict[str, Any],
    source_manifest_version: str,
    corpus_split_version: str,
    planner_version: str,
    filter_contract_version: str,
    model_inventory: list[dict[str, Any]],
    capability_availability: dict[str, Any],
    random_seed: int,
    plan_reproducible: bool,
    montage_checks: dict[str, bool] | None = None,
    source_start_bias_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_start_bias = build_source_start_bias_check(source_start_bias_records or [])
    category_metrics = {
        category: _metrics([row for row in core_records if category in row.get("categories", [])])
        for category in REQUIRED_CATEGORIES
    }
    core = _metrics(core_records)
    naive = _metrics(naive_records)
    comparison = {
        "core": core,
        "naive": naive,
        "severe_speech_failure_reduction": _relative_reduction(core["severe_speech_failure_rate"], naive["severe_speech_failure_rate"]),
        "severe_motion_failure_reduction": _relative_reduction(core["severe_motion_failure_rate"], naive["severe_motion_failure_rate"]),
        "human_acceptability_improvement_percentage_points": round((core["human_acceptability"] - naive["human_acceptability"]) * 100.0, 3),
    }
    checks = _production_checks(
        records=core_records,
        metrics=core,
        comparison=comparison,
        plan_reproducible=plan_reproducible,
        montage_checks=montage_checks or {},
        source_start_bias_passed=source_start_bias["passed"],
    )
    verdict = "PRODUCTION_READY" if all(checks.values()) else ("PREVIEW" if len(core_records) >= 40 and plan_reproducible else "EXPERIMENTAL")
    return {
        "schema_version": "1.0",
        "evaluation_version": EVALUATION_VERSION,
        "configuration": configuration,
        "source_manifest_version": source_manifest_version,
        "corpus_split_version": corpus_split_version,
        "planner_version": planner_version,
        "filter_contract_version": filter_contract_version,
        "model_inventory": model_inventory,
        "capability_availability": capability_availability,
        "random_seed": int(random_seed),
        "boundary_results": core_records,
        "category_metrics": category_metrics,
        "naive_sampler_comparison": comparison,
        "safety_critical_failures": [row for row in core_records if row.get("failure_codes")],
        "fallback_frequency": round(sum(bool(row.get("fallback")) for row in core_records) / len(core_records), 4) if core_records else 0.0,
        "human_review": {
            "reviewed_count": len(core_records),
            "label_counts": dict(sorted(Counter(str(row.get("human_label", "UNREVIEWED")) for row in core_records).items())),
            "human_acceptability": core["human_acceptability"],
            "acceptable_boundary_recall": core["acceptable_boundary_recall"],
        },
        "plan_reproducible": bool(plan_reproducible),
        "source_start_bias_check": source_start_bias,
        "production_readiness_checks": checks,
        "verdict": verdict,
    }


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in records if row.get("human_label") == "POSITIVE"]
    negative = [row for row in records if row.get("human_label") == "NEGATIVE"]
    accepted = [row for row in records if row.get("accepted") is True]
    true_positive = sum(row.get("human_label") == "POSITIVE" for row in accepted)
    false_positive = sum(row.get("human_label") == "NEGATIVE" for row in accepted)
    false_negative = sum(row.get("accepted") is False for row in positive)
    severe_speech = sum(_has_any_failure(row, {"MID_WORD", "MID_SENTENCE_ONSET"}) for row in records)
    severe_motion = sum(_has_any_failure(row, {"SUBJECT_MOTION_INTERRUPTED", "CAMERA_MOTION_INTERRUPTED"}) for row in records)
    return {
        "sample_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(records) - len(accepted),
        "true_positive_rate": round(true_positive / len(positive), 4) if positive else 0.0,
        "false_positive_rate": round(false_positive / len(negative), 4) if negative else 0.0,
        "false_negative_rate": round(false_negative / len(positive), 4) if positive else 0.0,
        "severe_failure_count": sum(bool(row.get("failure_codes")) for row in records),
        "severe_speech_failure_rate": round(severe_speech / len(records), 4) if records else 0.0,
        "severe_motion_failure_rate": round(severe_motion / len(records), 4) if records else 0.0,
        "human_acceptability": round(true_positive / (true_positive + false_positive), 4) if true_positive + false_positive else 0.0,
        "acceptable_boundary_recall": round(true_positive / (true_positive + false_negative), 4) if true_positive + false_negative else 0.0,
        "fallback_frequency": round(sum(bool(row.get("fallback")) for row in records) / len(records), 4) if records else 0.0,
    }


def _production_checks(
    *,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    plan_reproducible: bool,
    montage_checks: dict[str, bool],
    source_start_bias_passed: bool,
) -> dict[str, bool]:
    accepted = [row for row in records if row.get("accepted") is True]
    word_sensitive = [row for row in accepted if "speech_heavy" in row.get("categories", [])]
    transition_sensitive = [row for row in accepted if {"dissolves", "fades"} & set(row.get("categories", []))]
    subject_motion = [row for row in accepted if "high_motion" in row.get("categories", [])]
    no_failure = lambda rows, codes: (1.0 - sum(_has_any_failure(row, codes) for row in rows) / len(rows)) if rows else 0.0
    return {
        "held_out_sample_size_sufficient": len(records) >= 40,
        "word_sensitive_sample_size_sufficient": len(word_sensitive) >= 50,
        "audible_word_safety": no_failure(word_sensitive, {"MID_WORD"}) >= 0.98,
        "sentence_onset_safety": no_failure(word_sensitive, {"MID_SENTENCE_ONSET"}) >= 0.95,
        "transition_safety": no_failure(transition_sensitive, {"ENTERS_DURING_TRANSITION", "EXITS_DURING_TRANSITION"}) >= 0.95,
        "subject_motion_safety": no_failure(subject_motion, {"SUBJECT_MOTION_INTERRUPTED"}) >= 0.90,
        "camera_motion_safety": no_failure(subject_motion, {"CAMERA_MOTION_INTERRUPTED"}) >= 0.90,
        "no_fabricated_long_take_boundary": not any("FABRICATED_LONG_TAKE_BOUNDARY" in row.get("failure_codes", []) for row in records),
        "human_acceptability": metrics["human_acceptability"] >= 0.85,
        "acceptable_boundary_recall": metrics["acceptable_boundary_recall"] >= 0.80,
        "speech_failure_reduction": comparison["severe_speech_failure_reduction"] >= 0.50,
        "motion_failure_reduction": comparison["severe_motion_failure_reduction"] >= 0.35,
        "acceptability_improvement": comparison["human_acceptability_improvement_percentage_points"] >= 30.0,
        "plan_reproducible": plan_reproducible,
        "minimum_moment_or_fallback": montage_checks.get("minimum_moment_or_fallback", False),
        "source_participation": montage_checks.get("source_participation", False),
        "complete_provenance": montage_checks.get("complete_provenance", False),
        "source_start_bias": source_start_bias_passed,
    }


def build_source_start_bias_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate opening distribution across seeds or materially different inputs."""
    applicable = [row for row in records if not row.get("chronology_required_by_filter", False)]
    positions = [float(row.get("normalized_timeline_position", 0.0)) for row in applicable]
    opening_ids = {str(row.get("moment_id")) for row in applicable if row.get("moment_id")}
    earliest_rate = round(sum(bool(row.get("earliest_eligible_selected")) for row in applicable) / len(applicable), 4) if applicable else 0.0
    quartiles = {min(3, int(max(0.0, min(0.9999, value)) * 4)) for value in positions}
    checks = {
        "sample_size_sufficient": len(applicable) >= 12,
        "multiple_openings_selected": len(opening_ids) >= 4,
        "timeline_span_sufficient": bool(positions) and max(positions) - min(positions) >= 0.5,
        "timeline_quartile_coverage": len(quartiles) >= 3,
        "earliest_selection_rate_not_excessive": earliest_rate <= 0.25,
        "no_implicit_timeline_tiebreaker_declared": bool(applicable) and all(row.get("timeline_position_primary_tiebreaker") is False for row in applicable),
    }
    return {
        "status": "PASS" if all(checks.values()) else "INSUFFICIENT_OR_BIASED",
        "passed": all(checks.values()),
        "sample_count": len(applicable),
        "distinct_opening_count": len(opening_ids),
        "earliest_selection_rate": earliest_rate,
        "normalized_timeline_span": round(max(positions) - min(positions), 4) if positions else 0.0,
        "covered_timeline_quartiles": sorted(quartiles),
        "checks": checks,
    }


def _relative_reduction(current: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return round((baseline - current) / baseline, 4)


def _has_any_failure(record: dict[str, Any], codes: set[str]) -> bool:
    return bool(set(record.get("failure_codes", [])) & codes)
