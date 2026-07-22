from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from ..util import stable_hash, utc_now

ScoreCandidate = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
BuildMapping = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]

FUNCTION_REPAIR_VERSION = "function_mismatch_repair_v1"


def propose_function_repairs(
    *,
    schedule: dict[str, Any],
    function_verification: dict[str, Any],
    windows: list[dict[str, Any]],
    legal_donors: list[dict[str, Any]],
    score_candidate: ScoreCandidate,
    build_mapping: BuildMapping,
    maximum_repairs: int = 8,
    minimum_confidence: float = 0.62,
    minimum_function_gain: float = 0.1,
) -> dict[str, Any]:
    """Propose only high-confidence, already-legal donor changes; rendering decides acceptance."""
    candidate_schedule = deepcopy(schedule)
    mappings = candidate_schedule.get("mappings") or []
    window_by_id = {str(row.get("id") or row.get("window_id")): row for row in windows}
    used = {str(row.get("clip_id")) for row in mappings if row.get("enabled", True)}
    attempts: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    mismatch_rows = [
        row for row in function_verification.get("mappings") or []
        if row.get("verification_state") == "FUNCTION_MISMATCH"
        and float(row.get("planned_classification_confidence", 0.0) or 0.0) >= minimum_confidence
        and float(row.get("rendered_transcript_confidence", 0.0) or 0.0) >= 0.45
    ][:max(0, int(maximum_repairs))]
    for failure in mismatch_rows:
        index = int(failure.get("mapping_index", -1))
        if not 0 <= index < len(mappings):
            continue
        current = mappings[index]
        window = window_by_id.get(str(current.get("window_id"))) or _window_from_mapping(current)
        current_function = float((current.get("dialogue_function_compatibility") or {}).get("normalized_function_contribution", 0.0) or 0.0)
        rejection_counts: dict[str, int] = {}
        ranked: list[tuple[float, float, float, str, dict[str, Any], dict[str, Any]]] = []
        for donor in legal_donors:
            donor_id = str(donor.get("id") or "")
            if not donor_id or not donor.get("usable", True) or donor_id == str(current.get("clip_id")) or donor_id in used:
                _reject(rejection_counts, "unavailable_or_already_used")
                continue
            score = score_candidate(window, donor)
            if score.get("eligible") is False or score.get("hard_gate_passed") is False:
                _reject(rejection_counts, "hard_delivery_gate")
                continue
            compatibility = score.get("dialogue_function_compatibility") or {}
            function_score = float(compatibility.get("normalized_function_contribution", 0.5) or 0.5)
            confidence = float(compatibility.get("confidence", 0.0) or 0.0)
            if not compatibility.get("available") or confidence < minimum_confidence:
                _reject(rejection_counts, "function_evidence_uncertain")
                continue
            if function_score < current_function + minimum_function_gain:
                _reject(rejection_counts, "insufficient_function_gain")
                continue
            if not _technical_constraints_preserved(current, score):
                _reject(rejection_counts, "technical_constraint_regression")
                continue
            semantic_score = float((score.get("semantic_compatibility") or {}).get("normalized_semantic_contribution", 0.5) or 0.5)
            ranked.append((function_score, semantic_score, float(score.get("score", 0.0) or 0.0), donor_id, donor, score))
        attempt = {
            "placement_key": failure.get("placement_key"), "mapping_index": index,
            "old_clip_id": current.get("clip_id"), "candidates_considered": len(legal_donors),
            "candidate_rejection_reasons": rejection_counts, "proposed": False,
            "repair_state": "RETAIN_BEST_KNOWN_UNCERTAIN" if not ranked else "PROPOSED_PENDING_RENDER_VERIFICATION",
        }
        if ranked:
            function_score, semantic_score, technical_score, donor_id, donor, score = max(ranked, key=lambda row: (row[0], row[1], row[2], row[3]))
            replacement = build_mapping(window, donor, score)
            replacement["function_repair"] = {
                "repair_version": FUNCTION_REPAIR_VERSION,
                "prior_clip_id": current.get("clip_id"),
                "function_score_before": round(current_function, 6),
                "function_score_after": round(function_score, 6),
                "semantic_secondary_score": round(semantic_score, 6),
                "acceptance_state": "PENDING_RENDER_VERIFICATION",
            }
            mappings[index] = replacement
            used.add(donor_id)
            proposal = {
                "placement_key": failure.get("placement_key"), "mapping_index": index,
                "old_clip_id": current.get("clip_id"), "new_clip_id": donor_id,
                "function_score_before": round(current_function, 6), "function_score_after": round(function_score, 6),
                "semantic_secondary_score": round(semantic_score, 6), "technical_candidate_score": round(technical_score, 6),
                "acceptance_state": "PENDING_RENDER_VERIFICATION",
            }
            proposals.append(proposal)
            attempt.update({"proposed": True, "new_clip_id": donor_id, "repair_state": "PROPOSED_PENDING_RENDER_VERIFICATION"})
        attempts.append(attempt)
    report = {
        "schema_version": "1.0", "repair_version": FUNCTION_REPAIR_VERSION,
        "creation_timestamp": utc_now(), "input_schedule_signature": stable_hash(schedule),
        "input_function_verification_signature": function_verification.get("verification_signature"),
        "attempt_count": len(attempts), "proposal_count": len(proposals),
        "repair_state": "PROPOSED_PENDING_RENDER_VERIFICATION" if proposals else "NO_CONFIDENT_REPAIR_PROPOSED",
        "attempts": attempts, "proposals": proposals,
        "constraints": {
            "legal_donors_only": True, "function_primary_semantic_secondary": True,
            "hard_delivery_gates_authoritative": True, "render_verification_required": True,
            "uncertain_evidence_retains_best_known": True,
        },
    }
    report["repair_signature"] = stable_hash({key: value for key, value in report.items() if key not in {"creation_timestamp", "repair_signature"}})
    return {"candidate_schedule": candidate_schedule, "repair_report": report}


def finalize_function_repairs(
    *,
    original_schedule: dict[str, Any],
    candidate_schedule: dict[str, Any],
    repair_report: dict[str, Any],
    rendered_function_verification: dict[str, Any],
    quality_before: dict[str, float],
    quality_after: dict[str, float],
    non_regression_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Commit each repair only after rendered function and overall measured quality pass."""
    final_schedule = deepcopy(candidate_schedule)
    original = original_schedule.get("mappings") or []
    mappings = final_schedule.get("mappings") or []
    verification = {int(row.get("mapping_index", -1)): row for row in rendered_function_verification.get("mappings") or []}
    outcomes: list[dict[str, Any]] = []
    for proposal in repair_report.get("proposals") or []:
        index = int(proposal.get("mapping_index", -1))
        observed = verification.get(index) or {}
        before = float(quality_before.get(str(index), 0.0) or 0.0)
        after = float(quality_after.get(str(index), 0.0) or 0.0)
        function_passed = observed.get("verification_state") == "VERIFIED"
        quality_passed = after + float(non_regression_tolerance) >= before
        accepted = bool(0 <= index < len(mappings) and index < len(original) and function_passed and quality_passed)
        if not accepted and 0 <= index < len(mappings) and index < len(original):
            mappings[index] = deepcopy(original[index])
        outcomes.append({
            **proposal,
            "rendered_function_state": observed.get("verification_state", "UNAVAILABLE"),
            "quality_before": round(before, 4), "quality_after": round(after, 4),
            "acceptance_state": "ACCEPTED" if accepted else "ROLLED_BACK",
            "rollback_reason": None if accepted else (
                "RENDERED_FUNCTION_NOT_VERIFIED" if not function_passed else "MEASURED_QUALITY_REGRESSION"
            ),
        })
    accepted_count = sum(row["acceptance_state"] == "ACCEPTED" for row in outcomes)
    result = {
        **{key: value for key, value in repair_report.items() if key not in {"repair_state", "proposals", "repair_signature"}},
        "repair_state": "FINALIZED", "proposals": outcomes,
        "accepted_count": accepted_count, "rollback_count": len(outcomes) - accepted_count,
        "output_schedule_signature": stable_hash(final_schedule),
    }
    result["repair_signature"] = stable_hash({key: value for key, value in result.items() if key not in {"creation_timestamp", "repair_signature"}})
    return {"schedule": final_schedule, "repair_report": result}


def _technical_constraints_preserved(current: dict[str, Any], score: dict[str, Any]) -> bool:
    components = score.get("components") or score.get("editorial_components") or {}
    checks = {
        "duration": components.get("duration_similarity", components.get("timing_and_render_fit")),
        "performance": components.get("performance_fit"),
        "speaker": components.get("speaker_role_fit"),
        "visual": components.get("visual_fit"),
        "completeness": components.get("sentence_fit", components.get("transcript_completeness")),
    }
    explicit = [float(value) for value in checks.values() if value is not None]
    return all(value >= 0.4 for value in explicit) and float(score.get("score", 0.0) or 0.0) >= 0.4


def _window_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
    duration = float(mapping.get("planned_render_duration", 0.0) or 0.0)
    return {**mapping, "id": mapping.get("window_id"), "start": start, "end": start + duration, "duration": duration}


def _reject(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1
