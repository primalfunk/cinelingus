from __future__ import annotations

from collections import Counter
from typing import Any

from .failure_taxonomy import failure
from .quality_model import placement_quality
from .repair_strategies import repair_strategy_for


def evaluate_editorial_decisions(
    *,
    schedule: dict[str, Any],
    rendered_verification: dict[str, Any],
    residue_verification: dict[str, Any] | None = None,
    acceptance_threshold: float = 0.72,
    minimum_word_coverage: float = 0.72,
    max_time_stretch: float = 0.1,
    weights: dict[str, float] | None = None,
    problem_report: dict[str, Any] | None = None,
    ignore_problem_placement_keys: set[str] | None = None,
) -> dict[str, Any]:
    verification_by_key: dict[str, dict[str, Any]] = {}
    for row in rendered_verification.get("mappings", []):
        for candidate in (
            row.get("editorial_placement_id"), row.get("mapping_id"), row.get("window_id"),
            f"index:{row.get('mapping_index', -1)}",
        ):
            if candidate is not None:
                verification_by_key[str(candidate)] = row
    residue_regions = [
        row for row in (residue_verification or {}).get("regions", [])
        if row.get("possible_residue")
    ]
    decisions = []
    for index, mapping in enumerate(schedule.get("mappings", [])):
        if not mapping.get("enabled", True):
            continue
        key = _mapping_key(mapping, index)
        verification = (
            verification_by_key.get(key)
            or verification_by_key.get(str(mapping.get("id") or ""))
            or verification_by_key.get(str(mapping.get("window_id") or ""))
            or verification_by_key.get(f"index:{index}")
        )
        placement_problems = (
            [] if key in (ignore_problem_placement_keys or set())
            else _placement_problems(problem_report or {}, mapping, index)
        )
        residue_failed = any(_overlaps_mapping(mapping, row) for row in residue_regions)
        quality = placement_quality(
            mapping=mapping, verification=verification, residue_failed=residue_failed,
            weights=weights, max_time_stretch=max_time_stretch,
        )
        failures = _failures(
            mapping=mapping, verification=verification or {}, residue_failed=residue_failed,
            minimum_word_coverage=minimum_word_coverage, max_time_stretch=max_time_stretch,
            problems=placement_problems,
        )
        hard_gate_failures = _hard_gate_failures(failures)
        high_failure = any(row["severity"] in {"high", "critical"} for row in failures)
        if quality["score"] >= acceptance_threshold and not high_failure:
            recommendation = "accept"
        elif any(row["recommended_repair"] != "reject_unverifiable_placement" for row in failures):
            recommendation = "repair"
        else:
            recommendation = "reject"
        repair_plan = repair_strategy_for({"failures": failures})
        decisions.append({
            "placement_key": key,
            "mapping_index": index,
            "window_id": mapping.get("window_id"),
            "clip_id": mapping.get("clip_id"),
            "destination_start": mapping.get("destination_timestamp"),
            "destination_end": _mapping_end(mapping),
            "overall_quality": quality["score"],
            "quality": quality,
            "problem_evidence": placement_problems,
            "failures": failures,
            "hard_gate_passed": not hard_gate_failures,
            "hard_gate_failures": hard_gate_failures,
            "repairability": _repairability(mapping, failures),
            "recommendation": recommendation,
            "repair_strategy": repair_plan["strategy"] if failures else None,
            "repair_plan": repair_plan if failures else None,
        })
    counts = Counter(row["recommendation"] for row in decisions)
    failure_counts = Counter(item["category"] for row in decisions for item in row["failures"])
    average = sum(row["overall_quality"] for row in decisions) / max(1, len(decisions))
    minimum = min((row["overall_quality"] for row in decisions), default=0.0)
    return {
        "schema_version": "1.0",
        "decision_engine_version": "editorial_decision_v1",
        "acceptance_threshold": round(float(acceptance_threshold), 4),
        "minimum_word_coverage": round(float(minimum_word_coverage), 4),
        "placement_count": len(decisions),
        "average_quality": round(average, 4),
        "minimum_quality": round(minimum, 4),
        "accepted_count": counts["accept"],
        "repair_count": counts["repair"],
        "rejected_count": counts["reject"],
        "quality_gate_passed": bool(decisions) and counts["repair"] == 0 and counts["reject"] == 0,
        "failure_counts": dict(sorted(failure_counts.items())),
        "decisions": decisions,
    }


def _failures(
    *, mapping: dict[str, Any], verification: dict[str, Any], residue_failed: bool,
    minimum_word_coverage: float, max_time_stretch: float,
    problems: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    confidence = float(verification.get("confidence", 0.0) or 0.0)
    coverage = float(verification.get("word_coverage_percentage", 0.0) or 0.0) / 100.0
    if verification and (verification.get("missing_sentence_beginning") or verification.get("missing_sentence_ending")):
        rows.append(failure("incomplete_sentence", severity="high", confidence=confidence, evidence={"coverage": coverage}))
    if verification.get("mid_word_cut"):
        rows.append(failure("mid_word_cut", severity="critical", confidence=confidence))
    if verification and coverage < minimum_word_coverage:
        severity = "critical" if coverage < 0.25 else "high" if coverage < 0.6 else "medium"
        rows.append(failure("low_rendered_coverage", severity=severity, confidence=confidence, evidence={"coverage": round(coverage, 4)}))
    if mapping.get("speaker_match_preserved") is False:
        rows.append(failure("speaker_mismatch", severity="medium", confidence=_signature_confidence(mapping)))
    visual = _nested(mapping, "cinematic_compatibility_categories", "visual", 1.0)
    if visual < 0.4:
        rows.append(failure("visual_mismatch", severity="medium", confidence=_compatibility_confidence(mapping), evidence={"score": visual}))
    performance = float(mapping.get("performance_similarity_score", 1.0) or 0.0)
    if performance < 0.45:
        rows.append(failure("performance_mismatch", severity="medium", confidence=_signature_confidence(mapping), evidence={"score": performance}))
    stretch = abs(float(mapping.get("stretch_factor", 1.0) or 1.0) - 1.0)
    if stretch > max(0.001, max_time_stretch * 0.8) or "trim" in str(mapping.get("timing_strategy", "")):
        rows.append(failure("duration_failure", severity="medium", confidence=0.9, evidence={"stretch_delta": round(stretch, 4), "strategy": mapping.get("timing_strategy")}))
    reuse = _nested(mapping, "performance_similarity_components", "reuse_penalty", 1.0)
    if reuse < 0.5:
        rows.append(failure("reuse_exhaustion", severity="medium", confidence=1.0, evidence={"score": reuse}))
    editing = _nested(mapping, "cinematic_compatibility_categories", "editing", 1.0)
    if editing < 0.4 or mapping.get("mapping_crosses_shot_boundary"):
        rows.append(failure("transition_artifact", severity="medium", confidence=_compatibility_confidence(mapping), evidence={"score": editing}))
    if residue_failed:
        rows.append(failure("residual_dialogue", severity="critical", confidence=0.9))
    if verification.get("audio_masking_possible") or verification.get("fade_masking_possible"):
        rows.append(failure("masking", severity="medium", confidence=confidence))
    if verification and confidence < 0.35:
        rows.append(failure("confidence_collapse", severity="high", confidence=1.0 - confidence))
    problem_categories = {
        "fallback_mapping": "performance_mismatch",
        "underfilled_performance": "performance_mismatch",
        "undercovered_speech_window": "low_rendered_coverage",
        "low_fit_mapping": "visual_mismatch",
        "possible_destination_speech_residue": "residual_dialogue",
        "ambience_silence_fallback": "masking",
        "uncertain_speech_boundary": "transition_artifact",
    }
    existing = {row["category"] for row in rows}
    for problem in problems:
        category = problem_categories.get(str(problem.get("problem_type")))
        if not category or category in existing:
            continue
        severity = str(problem.get("severity", "medium"))
        rows.append(failure(category, severity=severity, confidence=0.8, evidence={
            "problem_type": problem.get("problem_type"), "reason": problem.get("reason"),
        }))
        existing.add(category)
    return rows


def _placement_problems(
    report: dict[str, Any], mapping: dict[str, Any], mapping_index: int,
) -> list[dict[str, Any]]:
    window_id = str(mapping.get("window_id") or "")
    start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
    end = _mapping_end(mapping)
    matched = []
    for row in report.get("problems", []):
        indices = {int(value) for value in row.get("mapping_indices", [])}
        same_window = bool(window_id) and str(row.get("window_id") or "") == window_id
        row_start = float(row.get("start", 0.0) or 0.0)
        row_end = float(row.get("end", row_start) or row_start)
        overlaps = min(end, row_end) > max(start, row_start)
        if mapping_index in indices or same_window or overlaps:
            matched.append(dict(row))
    return matched


def _mapping_key(mapping: dict[str, Any], index: int) -> str:
    return str(mapping.get("editorial_placement_id") or mapping.get("id") or mapping.get("window_id") or f"index:{index}")


def _verification_key(row: dict[str, Any]) -> str:
    return str(row.get("editorial_placement_id") or row.get("mapping_id") or row.get("window_id") or f"index:{row.get('mapping_index', -1)}")


def _mapping_end(mapping: dict[str, Any]) -> float:
    return round(float(mapping.get("destination_timestamp", 0.0) or 0.0) + float(mapping.get("planned_render_duration", 0.0) or 0.0), 3)


def _overlaps_mapping(mapping: dict[str, Any], region: dict[str, Any]) -> bool:
    start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
    end = _mapping_end(mapping)
    return min(end, float(region.get("end", 0.0) or 0.0)) > max(start, float(region.get("start", 0.0) or 0.0))


def _nested(row: dict[str, Any], parent: str, child: str, default: float) -> float:
    try:
        return float((row.get(parent) or {}).get(child, default))
    except (TypeError, ValueError):
        return default


def _signature_confidence(mapping: dict[str, Any]) -> float:
    return _nested(mapping, "destination_performance_signature", "speaker_confidence", 0.5)


def _compatibility_confidence(mapping: dict[str, Any]) -> float:
    return _nested(mapping, "cinematic_compatibility_axes", "confidence", 0.5)


def _hard_gate_failures(failures: list[dict[str, Any]]) -> list[str]:
    hard_categories = {"mid_word_cut", "residual_dialogue"}
    rows = []
    for item in failures:
        category = str(item.get("category"))
        coverage = float((item.get("evidence") or {}).get("coverage", 1.0) or 0.0)
        if category in hard_categories or (category == "low_rendered_coverage" and coverage < 0.25):
            rows.append(category)
    return sorted(set(rows))


def _repairability(mapping: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    categories = {str(row.get("category")) for row in failures}
    score = 0.55
    evidence = []
    if categories.intersection({"incomplete_sentence", "mid_word_cut", "low_rendered_coverage"}):
        trim = float(mapping.get("clip_trim_duration", 0.0) or 0.0)
        slot_start = float(mapping.get("alignment_slot_start", mapping.get("destination_timestamp", 0.0)) or 0.0)
        slot_end = float(mapping.get("alignment_slot_end", slot_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)) or slot_start)
        headroom = max(0.0, slot_end - slot_start - float(mapping.get("planned_render_duration", 0.0) or 0.0))
        score += 0.2 if trim > 0.0 and headroom >= 0.08 else -0.1
        evidence.append("source_or_destination_boundary_candidate" if headroom >= 0.08 else "boundary_headroom_limited")
    if categories.intersection({"speaker_mismatch", "performance_mismatch", "visual_mismatch"}):
        score += 0.05
        evidence.append("donor_reassignment_candidate")
    if "transition_artifact" in categories:
        score += 0.05
        evidence.append("local_timing_or_fade_candidate")
    if "residual_dialogue" in categories:
        score += 0.1
        evidence.append("local_suppression_candidate")
    if "confidence_collapse" in categories:
        score -= 0.3
        evidence.append("verification_confidence_insufficient")
    if "reuse_exhaustion" in categories:
        score -= 0.2
        evidence.append("donor_pool_pressure")
    bounded = round(max(0.0, min(1.0, score)), 4)
    return {
        "score": bounded,
        "class": "high" if bounded >= 0.7 else "medium" if bounded >= 0.4 else "low",
        "evidence": evidence,
        "estimate_version": "placement_repairability_v1",
    }
