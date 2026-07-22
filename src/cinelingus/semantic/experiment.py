from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..schedule import build_schedule
from ..util import stable_hash, utc_now, write_json
from .config import SemanticMode
from .scheduling import SemanticScheduleContext

DEFAULT_WEIGHT_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)
MAXIMUM_RENDER_NOMINEE_LEGACY_SCORE_REGRESSION = 0.0025
COMPATIBILITY_FIELDS = {
    "performance": ("performance_similarity_score",),
    "duration": ("score_components", "duration_similarity"),
    "speaker": ("speaker_pattern_match",),
    "visual": ("visual_fit_score",),
    "completeness": ("cinematic_compatibility_components", "audio", "transcript_completeness"),
}


def run_semantic_schedule_screen(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]],
    semantic_evidence: SemanticScheduleContext, output_dir: Path,
    source_hash: str, destination_hash: str, max_time_stretch: float,
    weights: tuple[float, ...] = DEFAULT_WEIGHT_GRID,
    scheduling_mode: str = "best_fit", best_fit_lookahead: int = 8,
    shot_boundary_mode: str = "off", cinematic_filter: str = "balanced",
    source_performances: dict[str, Any] | None = None,
    speaker_mapping: dict[str, Any] | None = None,
    prohibited_source_performance_ids: set[str] | frozenset[str] | None = None,
    repair_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Screen semantic weights over the scheduler's existing legal candidate set."""
    grid = tuple(sorted(set(float(value) for value in weights)))
    if not grid or grid[0] < 0.0 or grid[-1] > 1.0:
        raise ValueError("Semantic screen weights must be between 0 and 1")
    if 0.0 not in grid:
        grid = (0.0, *grid)
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantined_sources = frozenset(str(value) for value in (prohibited_source_performance_ids or set()))

    common = {
        "clips": clips, "windows": windows, "source_hash": source_hash,
        "destination_hash": destination_hash, "max_time_stretch": max_time_stretch,
        "scheduling_mode": scheduling_mode, "best_fit_lookahead": best_fit_lookahead,
        "shot_boundary_mode": shot_boundary_mode, "cinematic_filter": cinematic_filter,
        "source_performances": source_performances, "speaker_mapping": speaker_mapping,
        "prohibited_source_performance_ids": quarantined_sources,
    }
    schedules: dict[str, dict[str, Any]] = {}
    schedules["control"] = build_schedule(
        **common, output_path=output_dir / "control_schedule.json", semantic_context=None,
    )
    schedules["report_only"] = build_schedule(
        **common, output_path=output_dir / "report_only_schedule.json",
        semantic_context=_with_mode(semantic_evidence, SemanticMode.REPORT_ONLY, 0.0),
    )
    opportunity_audit = _aggregate_opportunity_audits(schedules["report_only"])
    pareto_admissions = _select_direct_pareto_admissions(
        opportunity_audit, prohibited_source_performance_ids=quarantined_sources,
    )
    if scheduling_mode == "performance_fill" and pareto_admissions:
        schedules["pareto_guarded"] = build_schedule(
            **common, output_path=output_dir / "pareto_guarded_schedule.json",
            semantic_context=_with_mode(semantic_evidence, SemanticMode.REPORT_ONLY, 0.0),
            performance_admissions=pareto_admissions,
        )
    for weight in grid:
        variant = _weight_variant(weight)
        schedules[variant] = build_schedule(
            **common, output_path=output_dir / f"{variant}_schedule.json",
            semantic_context=_with_mode(semantic_evidence, SemanticMode.ASSISTED, weight),
        )

    control = schedules["control"]
    report_only = schedules["report_only"]
    invariants = {
        "report_only_selection_equivalent": _selection_signature(control) == _selection_signature(report_only),
        "report_only_scores_equivalent": _score_signature(control) == _score_signature(report_only),
        "zero_weight_selection_equivalent": _selection_signature(control) == _selection_signature(schedules["assisted_000"]),
        "zero_weight_scores_equivalent": _score_signature(control) == _score_signature(schedules["assisted_000"]),
        "candidate_generation_policy": "EXISTING_LEGAL_CANDIDATES_ONLY",
        "hard_constraints_remain_authoritative": True,
    }
    if not all(value is True for key, value in invariants.items() if key.endswith("_equivalent")):
        raise RuntimeError("Semantic zero-influence schedule invariant failed")

    variants = []
    for name, schedule in schedules.items():
        baseline_for_semantics = report_only if name.startswith("assisted_") or name == "pareto_guarded" else control
        variants.append(_variant_report(name, schedule, control, baseline_for_semantics))
    promising = sorted(
        [row for row in variants if _eligible_render_variant(row)],
        key=lambda row: (-float(row["mean_selected_semantic_similarity"] or -1.0), row["weight"]),
    )
    render_selection = ["control", "report_only"]
    if promising:
        render_selection.append(promising[0]["variant_id"])
    report = {
        "schema_version": "1.0",
        "experiment_version": "semantic_schedule_screen_v1",
        "creation_timestamp": utc_now(),
        "experiment_signature": stable_hash({
            "source_hash": source_hash, "destination_hash": destination_hash,
            "weights": grid, "scheduling_mode": scheduling_mode,
            "semantic_model_identity": semantic_evidence.model_identity,
            "clips": clips, "windows": windows,
            "quarantined_source_performance_ids": sorted(quarantined_sources),
            "repair_evidence_signature": (
                (repair_preflight or {}).get("render_proof_signature")
                or (repair_preflight or {}).get("opportunity_audit_signature")
                or (repair_preflight or {}).get("preflight_signature")
            ),
        }),
        "source_hash": source_hash, "destination_hash": destination_hash,
        "scheduling_mode": scheduling_mode, "weight_grid": list(grid),
        "semantic_model_identity": semantic_evidence.model_identity,
        "invariants": invariants, "variants": variants,
        "render_selection": render_selection,
        "render_selection_state": "ASSISTED_CANDIDATE_SELECTED" if promising else "NO_CONFLICT_FREE_CHANGED_ASSISTED_CANDIDATE",
        "render_selection_diagnostics": {
            "changed_assisted_variant_count": sum(1 for row in variants if row["mode"] == SemanticMode.ASSISTED.value and row["weight"] > 0.0 and row["placements_changed"] > 0),
            "conflict_free_changed_variant_count": sum(1 for row in variants if row["mode"] == SemanticMode.ASSISTED.value and row["weight"] > 0.0 and row["placements_changed"] > 0 and row["conflict_count"] == 0),
            "fully_covered_conflict_free_changed_variant_count": len(promising),
            "pareto_guarded_variant_present": any(row["variant_id"] == "pareto_guarded" for row in variants),
            "pareto_guarded_admission_count": sum(int(row.get("admission_count", 0) or 0) for row in variants if row["variant_id"] == "pareto_guarded"),
            "render_nomination_requires_full_selected_placement_semantic_coverage": True,
            "maximum_render_nominee_legacy_score_regression": MAXIMUM_RENDER_NOMINEE_LEGACY_SCORE_REGRESSION,
        },
        "semantic_opportunity_audit": opportunity_audit,
        "pareto_guarded_admission": {
            "policy": "DIRECT_EVIDENCE_GLOBAL_PARETO_V1",
            "admission_count": len(pareto_admissions),
            "admissions": [pareto_admissions[key] for key in sorted(pareto_admissions)],
            "schedule_variant": "pareto_guarded" if pareto_admissions and scheduling_mode == "performance_fill" else None,
        },
        "acoustic_repair": {
            "state": (
                "RENDER_INFORMED_RESCREEN"
                if (repair_preflight or {}).get("evidence_type") == "semantic_render_proof"
                else "OPPORTUNITY_AUDIO_INFORMED_RESCREEN"
                if (repair_preflight or {}).get("evidence_type") == "opportunity_acoustic_audit"
                else "PREFLIGHT_INFORMED_RESCREEN" if repair_preflight else "NOT_APPLIED"
            ),
            "evidence_type": (repair_preflight or {}).get("evidence_type"),
            "predecessor_screen_signature": (repair_preflight or {}).get("screen_signature"),
            "preflight_signature": (repair_preflight or {}).get("preflight_signature"),
            "render_proof_signature": (repair_preflight or {}).get("render_proof_signature"),
            "opportunity_audit_signature": (repair_preflight or {}).get("opportunity_audit_signature"),
            "quarantined_source_performance_ids": sorted(quarantined_sources),
            "quarantined_source_count": len(quarantined_sources),
        },
        "claim_scope": "Schedule-only transcript-vector screening. Render quality and human preference are not established.",
    }
    write_json(output_dir / "semantic_schedule_screen.json", report)
    return report


def _with_mode(context: SemanticScheduleContext, mode: SemanticMode, weight: float) -> SemanticScheduleContext:
    return SemanticScheduleContext(
        mode=mode, weight=weight,
        source_by_reference=context.source_by_reference,
        destination_by_reference=context.destination_by_reference,
        model_identity=context.model_identity,
        source_by_start=context.source_by_start,
        destination_by_start=context.destination_by_start,
        source_by_text=context.source_by_text,
        destination_by_text=context.destination_by_text,
        source_by_performance=context.source_by_performance,
        destination_by_performance=context.destination_by_performance,
    )


def _weight_variant(weight: float) -> str:
    return f"assisted_{int(round(weight * 100)):03d}"


def _selection_signature(schedule: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            row.get("window_id"), row.get("destination_performance_id"),
            round(float(row.get("destination_start", 0.0) or 0.0), 3),
            row.get("clip_id"), row.get("source_performance_id"),
        )
        for row in schedule.get("mappings") or []
    ]


def _score_signature(schedule: dict[str, Any]) -> list[float]:
    return [float(row.get("score", 0.0) or 0.0) for row in schedule.get("mappings") or []]


def _variant_report(
    name: str, schedule: dict[str, Any], control: dict[str, Any], semantic_baseline: dict[str, Any],
) -> dict[str, Any]:
    mappings = schedule.get("mappings") or []
    control_mappings = control.get("mappings") or []
    semantic_rows = [row.get("semantic_compatibility") for row in mappings if row.get("semantic_compatibility", {}).get("available")]
    boundary_bridge_rows = [row for row in semantic_rows if "direct_passage_boundary_bridge" in {row.get("source_evidence_scope"), row.get("destination_evidence_scope")}]
    text_bridge_rows = [row for row in semantic_rows if "direct_passage_text_bridge" in {row.get("source_evidence_scope"), row.get("destination_evidence_scope")}]
    performance_fallback_rows = [row for row in semantic_rows if "performance_passage_aggregate" in {row.get("source_evidence_scope"), row.get("destination_evidence_scope")}]
    exact_direct_rows = [
        row for row in semantic_rows
        if {row.get("source_evidence_scope"), row.get("destination_evidence_scope")} <= {"direct_passage"}
    ]
    donors = [str(row.get("clip_id")) for row in mappings if row.get("clip_id")]
    counts = Counter(donors)
    conflicts = _semantic_conflicts(semantic_baseline.get("mappings") or [], mappings)
    tradeoffs = _legacy_score_tradeoffs(semantic_baseline.get("mappings") or [], mappings)
    if name.startswith("assisted_"):
        mode = SemanticMode.ASSISTED.value
        weight = int(name.rsplit("_", 1)[1]) / 100.0
    elif name == "pareto_guarded":
        mode, weight = SemanticMode.ASSISTED.value, 0.0
    elif name == "report_only":
        mode, weight = SemanticMode.REPORT_ONLY.value, 0.0
    else:
        mode, weight = SemanticMode.DISABLED.value, 0.0
    report = {
        "variant_id": name, "mode": mode, "weight": weight,
        "mapping_count": len(mappings),
        "placements_changed": _changed_placement_count(control_mappings, mappings),
        "placements_unchanged": max(len(control_mappings), len(mappings)) - _changed_placement_count(control_mappings, mappings),
        "semantic_placement_count": len(semantic_rows),
        "semantic_placement_coverage": round(len(semantic_rows) / len(mappings), 6) if mappings else 0.0,
        "performance_aggregate_fallback_count": len(performance_fallback_rows),
        "boundary_bridge_semantic_placement_count": len(boundary_bridge_rows),
        "text_bridge_semantic_placement_count": len(text_bridge_rows),
        "exact_direct_semantic_placement_count": len(exact_direct_rows),
        "mean_selected_semantic_similarity": round(sum(float(row["raw_cosine_similarity"]) for row in semantic_rows) / len(semantic_rows), 6) if semantic_rows else None,
        "mean_legacy_candidate_score": _mean_field(mappings, ("legacy_candidate_score",)),
        "mean_compatibility": {key: _mean_field(mappings, path) for key, path in COMPATIBILITY_FIELDS.items()},
        "unique_donor_count": len(counts), "maximum_donor_reuse": max(counts.values(), default=0),
        "conflict_count": len(conflicts), "conflicts": conflicts,
        "legacy_score_tradeoff_count": len(tradeoffs), "legacy_score_tradeoffs": tradeoffs,
        "schedule_file": f"{name}_schedule.json",
    }
    if name == "pareto_guarded":
        admission = schedule.get("semantic_pareto_admission") or {}
        admitted_destinations = {
            str(row.get("destination_performance_id") or "")
            for row in admission.get("admissions") or []
        }
        admitted_rows = [
            row for row in mappings
            if str(row.get("destination_performance_id") or "") in admitted_destinations
        ]
        direct_rows = [
            row for row in admitted_rows
            if (semantic := row.get("semantic_compatibility") or {}).get("available")
            and str(semantic.get("source_evidence_scope") or "").startswith("direct_passage")
            and str(semantic.get("destination_evidence_scope") or "").startswith("direct_passage")
        ]
        report["admission_mode"] = admission.get("policy")
        report["admission_count"] = int(admission.get("admission_count", 0) or 0)
        report["admitted_mapping_count"] = len(admitted_rows)
        report["admitted_direct_semantic_mapping_count"] = len(direct_rows)
    return report


def _eligible_render_variant(row: dict[str, Any]) -> bool:
    guarded = row.get("admission_mode") == "DIRECT_EVIDENCE_GLOBAL_PARETO_V1"
    tradeoffs = row.get("legacy_score_tradeoffs") or []
    bounded_tradeoffs = all(
        float(item.get("legacy_score_delta", 0.0) or 0.0)
        >= -MAXIMUM_RENDER_NOMINEE_LEGACY_SCORE_REGRESSION
        for item in tradeoffs
    )
    return (
        row.get("mode") == SemanticMode.ASSISTED.value
        and (float(row.get("weight", 0.0) or 0.0) > 0.0 or guarded)
        and int(row.get("placements_changed", 0) or 0) > 0
        and int(row.get("conflict_count", 0) or 0) == 0
        and (int(row.get("legacy_score_tradeoff_count", 0) or 0) == 0 or (not guarded and bounded_tradeoffs))
        and int(row.get("mapping_count", 0) or 0) > 0
        and (
            int(row.get("semantic_placement_count", 0) or 0) == int(row.get("mapping_count", 0) or 0)
            or (
                guarded
                and int(row.get("admission_count", 0) or 0) > 0
                and int(row.get("admitted_mapping_count", 0) or 0) > 0
                and int(row.get("admitted_direct_semantic_mapping_count", 0) or 0)
                == int(row.get("admitted_mapping_count", 0) or 0)
            )
        )
    )


def _aggregate_opportunity_audits(schedule: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {"destination_performance_id": decision.get("destination_performance_id"), **audit}
        for decision in schedule.get("performance_decisions") or []
        if (audit := decision.get("semantic_opportunity_audit")) is not None
    ]
    local_opportunities = [
        {"destination_performance_id": row["destination_performance_id"], **opportunity}
        for row in rows for opportunity in row.get("opportunities", [])
    ]
    source_usage: dict[str, set[str]] = {}
    for mapping in schedule.get("mappings") or []:
        source_id = str(mapping.get("source_performance_id") or "")
        destination_id = str(mapping.get("destination_performance_id") or mapping.get("window_id") or "")
        if source_id:
            source_usage.setdefault(source_id, set()).add(destination_id)
    opportunities = []
    for opportunity in local_opportunities:
        source_id = str(opportunity.get("source_performance_id") or "")
        destination_id = str(opportunity.get("destination_performance_id") or "")
        conflicts = sorted(value for value in source_usage.get(source_id, set()) if value != destination_id)
        swap = opportunity.get("two_cycle_swap") or {}
        swap_admitted = swap.get("state") == "ADMISSIBLE_TWO_CYCLE"
        opportunities.append({
            **opportunity,
            "global_source_reuse_conflict": bool(conflicts),
            "conflicting_destination_performance_ids": conflicts,
            "global_admission_mode": "DIRECT" if not conflicts else "TWO_CYCLE" if swap_admitted else "NONE",
            "globally_admissible": not conflicts or swap_admitted,
        })
    opportunities.sort(key=lambda row: (-float(row.get("semantic_delta", 0.0)), str(row.get("destination_performance_id")), str(row.get("source_performance_id"))))
    return {
        "audit_version": "semantic_pareto_opportunity_v1",
        "audited_destination_performance_count": len(rows),
        "legal_candidate_count": sum(int(row.get("legal_candidate_count", 0) or 0) for row in rows),
        "higher_semantic_candidate_count": sum(int(row.get("higher_semantic_candidate_count", 0) or 0) for row in rows),
        "placement_valid_candidate_count": sum(int(row.get("placement_valid_candidate_count", 0) or 0) for row in rows),
        "fully_covered_candidate_count": sum(int(row.get("fully_covered_candidate_count", 0) or 0) for row in rows),
        "local_pareto_safe_opportunity_count": len(opportunities),
        "globally_admissible_opportunity_count": sum(1 for row in opportunities if row["globally_admissible"]),
        "pareto_safe_opportunity_count": sum(1 for row in opportunities if row["globally_admissible"]),
        "opportunities": opportunities,
        "claim_scope": "Report-only counterfactual evidence with full-schedule source-reuse admission; the selected schedule is unchanged.",
    }


def _select_direct_pareto_admissions(
    audit: dict[str, Any], *,
    prohibited_source_performance_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select a deterministic, non-cascading subset of direct-evidence opportunities."""
    prohibited_sources = frozenset(str(value) for value in (prohibited_source_performance_ids or set()))
    selected: dict[str, dict[str, Any]] = {}
    used_sources: set[str] = set()
    for row in sorted(
        audit.get("opportunities") or [],
        key=lambda value: (
            -float(value.get("semantic_delta", 0.0) or 0.0),
            str(value.get("destination_performance_id") or ""),
            str(value.get("source_performance_id") or ""),
        ),
    ):
        destination_id = str(row.get("destination_performance_id") or "")
        source_id = str(row.get("source_performance_id") or "")
        placement_evidence_direct = (
            int(row.get("selected_mapping_count", 0) or 0) > 0
            and int(row.get("selected_direct_semantic_mapping_count", 0) or 0)
            == int(row.get("selected_mapping_count", 0) or 0)
            and int(row.get("candidate_mapping_count", 0) or 0) > 0
            and int(row.get("candidate_direct_semantic_mapping_count", 0) or 0)
            == int(row.get("candidate_mapping_count", 0) or 0)
        )
        common_eligible = (
            not destination_id or not source_id
            or source_id in prohibited_sources
            or destination_id in selected or source_id in used_sources
            or not row.get("globally_admissible")
            or float(row.get("semantic_delta", 0.0) or 0.0) <= 0.0
            or not str(row.get("selected_source_evidence_scope") or "").startswith("direct_passage")
            or not str(row.get("candidate_source_evidence_scope") or "").startswith("direct_passage")
            or not str(row.get("destination_evidence_scope") or "").startswith("direct_passage")
            or not placement_evidence_direct
        )
        if common_eligible:
            continue
        admission_mode = row.get("global_admission_mode")
        if admission_mode == "TWO_CYCLE":
            swap = row.get("two_cycle_swap") or {}
            target_id = str(swap.get("target_destination_performance_id") or "")
            replacement_id = str(swap.get("replacement_source_performance_id") or "")
            swap_direct = (
                int(swap.get("selected_mapping_count", 0) or 0) > 0
                and int(swap.get("selected_direct_semantic_mapping_count", 0) or 0)
                == int(swap.get("selected_mapping_count", 0) or 0)
                and int(swap.get("candidate_mapping_count", 0) or 0) > 0
                and int(swap.get("candidate_direct_semantic_mapping_count", 0) or 0)
                == int(swap.get("candidate_mapping_count", 0) or 0)
            )
            if (
                swap.get("state") != "ADMISSIBLE_TWO_CYCLE"
                or float(swap.get("net_semantic_delta", 0.0) or 0.0) <= 0.0
                or not target_id or not replacement_id or target_id in selected
                or replacement_id in prohibited_sources
                or replacement_id in used_sources or not swap_direct
            ):
                continue
            cycle_id = f"{destination_id}:{target_id}:{source_id}:{replacement_id}"
            primary = {
                "destination_performance_id": destination_id,
                "displaced_source_performance_id": replacement_id,
                "source_performance_id": source_id,
                "semantic_delta": round(float(row.get("semantic_delta", 0.0)), 6),
                "net_cycle_semantic_delta": round(float(swap.get("net_semantic_delta", 0.0)), 6),
                "global_admission_mode": "TWO_CYCLE", "evidence_scope": "direct_passage",
                "compatibility_deltas": dict(row.get("compatibility_deltas") or {}),
                "cycle_id": cycle_id,
            }
            secondary = {
                "destination_performance_id": target_id,
                "displaced_source_performance_id": source_id,
                "source_performance_id": replacement_id,
                "semantic_delta": round(float(swap.get("net_semantic_delta", 0.0)), 6),
                "net_cycle_semantic_delta": round(float(swap.get("net_semantic_delta", 0.0)), 6),
                "global_admission_mode": "TWO_CYCLE", "evidence_scope": "direct_passage",
                "compatibility_deltas": dict(swap.get("compatibility_deltas") or {}),
                "cycle_id": cycle_id,
            }
            selected[destination_id], selected[target_id] = primary, secondary
            used_sources.update({source_id, replacement_id})
            continue
        if admission_mode != "DIRECT":
            continue
        admission = {
            "destination_performance_id": destination_id,
            "displaced_source_performance_id": str(row.get("displaced_source_performance_id") or ""),
            "source_performance_id": source_id,
            "semantic_delta": round(float(row.get("semantic_delta", 0.0)), 6),
            "global_admission_mode": "DIRECT",
            "evidence_scope": "direct_passage",
            "compatibility_deltas": dict(row.get("compatibility_deltas") or {}),
        }
        selected[destination_id] = admission
        used_sources.add(source_id)
    return selected


def _changed_placement_count(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    left_signature = _selection_signature({"mappings": left})
    right_signature = _selection_signature({"mappings": right})
    maximum = max(len(left_signature), len(right_signature))
    return sum(index >= len(left_signature) or index >= len(right_signature) or left_signature[index] != right_signature[index] for index in range(maximum))


def _semantic_conflicts(baseline: list[dict[str, Any]], assisted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    for index, (before, after) in enumerate(zip(baseline, assisted)):
        if _selection_signature({"mappings": [before]}) == _selection_signature({"mappings": [after]}):
            continue
        before_semantic = (before.get("semantic_compatibility") or {}).get("normalized_semantic_contribution")
        after_semantic = (after.get("semantic_compatibility") or {}).get("normalized_semantic_contribution")
        if before_semantic is None or after_semantic is None or float(after_semantic) <= float(before_semantic):
            continue
        regressions = {}
        for name, path in COMPATIBILITY_FIELDS.items():
            old, new = _nested_number(before, path), _nested_number(after, path)
            if old is not None and new is not None and new < old:
                regressions[name] = round(new - old, 6)
        if regressions:
            conflicts.append({
                "placement_index": index, "window_id": after.get("window_id"),
                "control_clip_id": before.get("clip_id"), "assisted_clip_id": after.get("clip_id"),
                "semantic_delta": round(float(after_semantic) - float(before_semantic), 6),
                "compatibility_regressions": regressions,
            })
    return conflicts


def _legacy_score_tradeoffs(baseline: list[dict[str, Any]], assisted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, (before, after) in enumerate(zip(baseline, assisted)):
        if _selection_signature({"mappings": [before]}) == _selection_signature({"mappings": [after]}):
            continue
        old = float(before.get("legacy_candidate_score", before.get("score", 0.0)) or 0.0)
        new = float(after.get("legacy_candidate_score", after.get("score", 0.0)) or 0.0)
        if new < old:
            rows.append({
                "placement_index": index, "window_id": after.get("window_id"),
                "control_clip_id": before.get("clip_id"), "assisted_clip_id": after.get("clip_id"),
                "legacy_score_delta": round(new - old, 6),
                "interpretation": "Bounded soft-score tradeoff; requires render and human review and is not a technical regression by itself.",
            })
    return rows


def _mean_field(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = [value for row in rows if (value := _nested_number(row, path)) is not None]
    return round(sum(values) / len(values), 6) if values else None


def _nested_number(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
