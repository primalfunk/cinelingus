from __future__ import annotations

from typing import Any

from ..util import stable_hash, utc_now
from .classifier import FunctionClassifierConfig, RuleDialogueFunctionClassifier
from .scheduling import FunctionMode, dialogue_function_compatibility

FUNCTION_RENDER_VERIFICATION_VERSION = "function_render_verification_v1"


def evaluate_rendered_function(
    *,
    schedule: dict[str, Any],
    rendered_dialogue_verification: dict[str, Any],
    baseline_schedule: dict[str, Any] | None = None,
    classifier: RuleDialogueFunctionClassifier | None = None,
    calibration: dict[str, Any] | None = None,
    minimum_transcript_confidence: float = 0.45,
    minimum_function_compatibility: float = 0.5,
) -> dict[str, Any]:
    """Reclassify observed render transcripts; schedule scores alone are not proof."""
    classifier = classifier or RuleDialogueFunctionClassifier(FunctionClassifierConfig(context_mode="passage_alone"))
    baseline = _mapping_index(baseline_schedule or {})
    rendered = _verification_index(rendered_dialogue_verification)
    rows: list[dict[str, Any]] = []
    for mapping_index, mapping in enumerate(schedule.get("mappings") or []):
        if not mapping.get("enabled", True):
            continue
        key = _mapping_key(mapping, mapping_index)
        control = baseline.get(key)
        changed = control is None or _donor_key(control) != _donor_key(mapping)
        if baseline_schedule is not None and not changed:
            continue
        planned = mapping.get("dialogue_function_compatibility") or {}
        donor_function = planned.get("source_distribution")
        destination_function = planned.get("destination_distribution")
        observed_row = rendered.get(key) or rendered.get(f"index:{mapping_index}")
        observed_text = str((observed_row or {}).get("rendered_transcript") or "").strip()
        transcript_confidence = float((observed_row or {}).get("confidence", 0.0) or 0.0)
        word_coverage = float((observed_row or {}).get("word_coverage_percentage", 0.0) or 0.0) / 100.0
        transcript_reliable = transcript_confidence >= minimum_transcript_confidence or (
            transcript_confidence >= 0.2 and word_coverage >= 0.72 and str((observed_row or {}).get("status") or "").lower() != "fail"
        )
        observed_function = classifier.classify(observed_text) if observed_text else None
        donor_comparison = _compare(observed_function, donor_function, classifier.config.confidence_threshold)
        destination_comparison = _compare(observed_function, destination_function, classifier.config.confidence_threshold)
        technical_status = str((observed_row or {}).get("status") or "unavailable").upper()
        if not planned.get("available") or not donor_function or not destination_function:
            state = "UNVERIFIABLE"
            reason = "PLANNED_FUNCTION_EVIDENCE_UNAVAILABLE"
        elif not observed_text or not transcript_reliable:
            state = "UNVERIFIABLE"
            reason = "RENDERED_TRANSCRIPT_UNAVAILABLE_OR_LOW_CONFIDENCE"
        elif technical_status == "FAIL":
            state = "TECHNICAL_FAILURE"
            reason = "RENDERED_LINE_FAILED_TECHNICAL_VERIFICATION"
        elif (observed_function or {}).get("abstention", {}).get("abstained"):
            state = "UNVERIFIABLE"
            reason = "RENDERED_FUNCTION_CLASSIFIER_ABSTAINED"
        elif float((destination_comparison or {}).get("normalized_function_contribution", 0.5)) < minimum_function_compatibility:
            state = "FUNCTION_MISMATCH"
            reason = "HIGH_CONFIDENCE_RENDERED_FUNCTION_DIVERGES_FROM_DESTINATION"
        else:
            state = "VERIFIED"
            reason = "RENDERED_TRANSCRIPT_SUPPORTS_INTENDED_FUNCTION"
        rows.append({
            "mapping_index": mapping_index,
            "placement_key": key,
            "window_id": mapping.get("window_id"),
            "clip_id": mapping.get("clip_id"),
            "changed_from_baseline": changed,
            "destination_function_distribution": destination_function,
            "donor_function_distribution": donor_function,
            "intended_function_preservation_score": planned.get("normalized_function_contribution"),
            "planned_classification_confidence": planned.get("confidence"),
            "rendered_transcript": observed_text,
            "rendered_transcript_confidence": round(transcript_confidence, 4),
            "rendered_transcript_reliability_basis": (
                "DIRECT_CONFIDENCE" if transcript_confidence >= minimum_transcript_confidence
                else "LEXICAL_COVERAGE_CORROBORATION" if transcript_reliable else "INSUFFICIENT"
            ),
            "rendered_transcript_function": observed_function,
            "rendered_to_donor_function": donor_comparison,
            "rendered_to_destination_function": destination_comparison,
            "technical_evidence": {
                "duration_fit": (mapping.get("score_components") or {}).get("duration_similarity"),
                "performance_fit": mapping.get("performance_similarity_score"),
                "speaker_compatibility": mapping.get("speaker_match_preserved"),
                "visual_compatibility": (mapping.get("cinematic_compatibility_categories") or {}).get("visual"),
                "rendered_word_coverage_percentage": (observed_row or {}).get("word_coverage_percentage"),
                "rendered_dialogue_verification": technical_status,
                "editorial_state": mapping.get("editorial_state", "UNCHANGED"),
            },
            "verification_state": state,
            "verification_reason": reason,
        })
    counts = {state: sum(row["verification_state"] == state for row in rows) for state in (
        "VERIFIED", "FUNCTION_MISMATCH", "TECHNICAL_FAILURE", "UNVERIFIABLE",
    )}
    calibration_state = str((calibration or {}).get("review_state") or "NOT_PROVIDED")
    status = (
        "FAIL" if counts["FUNCTION_MISMATCH"]
        else "WARN" if counts["TECHNICAL_FAILURE"] or counts["UNVERIFIABLE"]
        else "PASS" if rows else "INCONCLUSIVE"
    )
    result = {
        "schema_version": "1.0",
        "verification_version": FUNCTION_RENDER_VERIFICATION_VERSION,
        "creation_timestamp": utc_now(),
        "schedule_signature": stable_hash(schedule),
        "baseline_schedule_signature": stable_hash(baseline_schedule) if baseline_schedule is not None else None,
        "classifier": classifier.describe(),
        "calibration_state": calibration_state,
        "claim_state": "ELIGIBLE" if calibration_state == "COMPLETE" else "PROVISIONAL_PENDING_REVIEWED_CALIBRATION",
        "status": status,
        "changed_mapping_count": len(rows),
        "counts": counts,
        "minimum_transcript_confidence": minimum_transcript_confidence,
        "minimum_function_compatibility": minimum_function_compatibility,
        "mappings": rows,
        "claim_scope": "Observed rendered transcript function only; no emotion, character, relationship, scene, genre, irony, comedy, or narrative claim.",
    }
    result["verification_signature"] = stable_hash({key: value for key, value in result.items() if key not in {"creation_timestamp", "verification_signature"}})
    return result


def _compare(observed: dict[str, Any] | None, target: dict[str, Any] | None, threshold: float) -> dict[str, Any] | None:
    if not observed or not target:
        return None
    wrapper = {
        "_function_mode": FunctionMode.REPORT_ONLY.value,
        "_function_weight": 0.0,
        "_function_minimum_confidence": threshold,
        "_function_identity": {},
        "_function_entity_ids": [],
    }
    return dialogue_function_compatibility(
        {**wrapper, "_function_classification": observed, "_function_evidence_scope": "rendered_transcript"},
        {**wrapper, "_function_classification": target, "_function_evidence_scope": "planned_distribution"},
    )


def _mapping_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("editorial_placement_id") or row.get("id") or row.get("window_id") or f"index:{index}")


def _donor_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("clip_id"), row.get("source_performance_id")


def _mapping_index(schedule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {_mapping_key(row, index): row for index, row in enumerate(schedule.get("mappings") or []) if row.get("enabled", True)}


def _verification_index(verification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(verification.get("mappings") or []):
        for key in (
            row.get("editorial_placement_id"), row.get("mapping_id"), row.get("window_id"),
            f"index:{row.get('mapping_index', index)}",
        ):
            if key is not None:
                result[str(key)] = row
    return result
