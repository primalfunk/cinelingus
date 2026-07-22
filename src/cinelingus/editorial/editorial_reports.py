from __future__ import annotations

from collections import Counter
from typing import Any


def build_editorial_report(result: dict[str, Any]) -> dict[str, Any]:
    passes = list(result.get("passes", []))
    recurring = Counter()
    for row in passes:
        recurring.update(row.get("failure_counts", {}))
    final_accepted = {
        str(row.get("placement_key"))
        for row in result.get("final_decisions", {}).get("decisions", [])
        if row.get("recommendation") == "accept"
    }
    accepted_repairs = [
        repair
        for row in passes if row.get("accepted")
        for repair in row.get("repairs", [])
        if str(repair.get("placement_key")) in final_accepted
    ]
    accepted_repairs.sort(
        key=lambda row: float(row.get("new_predicted_score", 0.0)) - float(row.get("old_predicted_score", 0.0)),
        reverse=True,
    )
    final = result.get("final_decisions", {})
    unrecoverable = sorted(
        (row for row in final.get("decisions", []) if row.get("recommendation") != "accept"),
        key=lambda row: float(row.get("overall_quality", 0.0)),
    )
    effectiveness = build_repair_effectiveness_report(result)
    return {
        "schema_version": "1.0",
        "editorial_system_version": result.get("editorial_system_version"),
        "status": result.get("status"),
        "termination_reason": result.get("termination_reason"),
        "resumed_from_candidate_checkpoint": bool(result.get("resumed_from_candidate_checkpoint")),
        "placements_accepted_immediately": result.get("placements_accepted_immediately", 0),
        "placements_repaired": result.get("placements_repaired", 0),
        "placements_rejected": result.get("placements_rejected", 0),
        "final_state_counts": result.get("final_state_counts", {}),
        "repair_capabilities": result.get("repair_capabilities", {}),
        "repair_passes": result.get("completed_repair_passes", 0),
        "initial_quality": result.get("initial_quality", 0.0),
        "final_quality": result.get("final_quality", 0.0),
        "quality_improvement": result.get("quality_improvement", 0.0),
        "quality_by_pass": [
            {"pass": row.get("pass"), "quality": row.get("average_quality"), "accepted": row.get("accepted")}
            for row in passes
        ],
        "top_recurring_failure_types": [
            {"category": category, "count": count} for category, count in recurring.most_common()
        ],
        "best_repaired_moments": accepted_repairs[:10],
        "worst_unrecoverable_moments": unrecoverable[:10],
        "passes": passes,
        "memory": result.get("memory", {}),
        "decisions": final.get("decisions", []),
        "repair_effectiveness": effectiveness,
    }


def build_repair_effectiveness_report(result: dict[str, Any]) -> dict[str, Any]:
    attempts = [
        dict(attempt)
        for pass_row in result.get("passes", [])
        for attempt in pass_row.get("repair_attempts", [])
    ]
    by_failure: dict[str, list[dict[str, Any]]] = {}
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        for category in row.get("failure_categories", []) or ["uncategorized"]:
            by_failure.setdefault(str(category), []).append(row)
        by_strategy.setdefault(str(row.get("repair_strategy") or "unspecified"), []).append(row)
        by_family.setdefault(str(row.get("candidate_family") or "no_candidate"), []).append(row)
    loss_stages = Counter(str(row.get("final_stage") or "unknown") for row in attempts)
    rejection_reasons = Counter()
    for row in attempts:
        rejection_reasons.update(row.get("candidate_rejection_reasons", {}))
    survived = sum(1 for row in attempts if row.get("survived"))
    rendered = sum(1 for row in attempts if row.get("rendered"))
    positive = sum(1 for row in attempts if float(row.get("quality_delta") or 0.0) > 0.0)
    return {
        "schema_version": "1.0",
        "report_version": "repair_effectiveness_baseline_v1",
        "attempted_placement_count": len(attempts),
        "candidates_considered": sum(int(row.get("candidates_considered", 0)) for row in attempts),
        "rendered_candidate_count": rendered,
        "surviving_repair_count": survived,
        "candidate_survival_rate": round(survived / rendered, 4) if rendered else 0.0,
        "positive_rendered_delta_count": positive,
        "coordinated_candidate_count": sum(1 for row in attempts if row.get("coordinated_candidate")),
        "coordinated_surviving_count": sum(1 for row in attempts if row.get("coordinated_candidate") and row.get("survived")),
        "original_restored_count": sum(1 for row in attempts if row.get("original_restored")),
        "no_viable_alternative_count": sum(1 for row in attempts if row.get("no_viable_alternative")),
        "original_already_predicted_best_count": sum(
            1 for row in attempts if row.get("candidate_loss_stage") == "pre_render_quality_ceiling"
        ),
        "donor_pool_exhaustion_count": sum(
            1 for row in attempts
            if row.get("candidate_loss_stage") == "candidate_generation"
            and int((row.get("candidate_rejection_reasons") or {}).get("donor_already_occupied", 0))
                >= max(0, int(row.get("donor_pool_size", 0)) - 2)
        ),
        "candidate_loss_stages": [
            {"stage": key, "count": value} for key, value in loss_stages.most_common()
        ],
        "candidate_rejection_reasons": [
            {"reason": key, "count": value} for key, value in rejection_reasons.most_common()
        ],
        "by_failure_category": _aggregate_attempt_groups(by_failure),
        "by_repair_strategy": _aggregate_attempt_groups(by_strategy),
        "by_candidate_family": _aggregate_attempt_groups(by_family),
        "attempts": attempts,
    }


def _aggregate_attempt_groups(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for key, attempts in groups.items():
        rendered = [row for row in attempts if row.get("rendered")]
        survived = [row for row in attempts if row.get("survived")]
        deltas = [float(row.get("quality_delta")) for row in rendered if row.get("quality_delta") is not None]
        rows.append({
            "name": key,
            "attempt_count": len(attempts),
            "candidates_considered": sum(int(row.get("candidates_considered", 0)) for row in attempts),
            "rendered_count": len(rendered),
            "survived_count": len(survived),
            "survival_rate": round(len(survived) / len(rendered), 4) if rendered else 0.0,
            "average_quality_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "no_viable_alternative_count": sum(1 for row in attempts if row.get("no_viable_alternative")),
        })
    return sorted(rows, key=lambda row: (-int(row["attempt_count"]), str(row["name"])))
