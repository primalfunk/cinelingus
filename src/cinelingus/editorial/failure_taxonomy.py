from __future__ import annotations

from typing import Any


FAILURE_REPAIRS = {
    "incomplete_sentence": "search_alternative_donor",
    "mid_word_cut": "replace_with_complete_line",
    "low_rendered_coverage": "search_alternative_donor",
    "speaker_mismatch": "prefer_speaker_compatible_donor",
    "visual_mismatch": "prefer_visual_compatible_donor",
    "performance_mismatch": "prefer_performance_compatible_donor",
    "duration_failure": "prefer_duration_compatible_donor",
    "reuse_exhaustion": "search_unused_donor",
    "transition_artifact": "move_or_shorten_placement",
    "residual_dialogue": "expand_suppression_region",
    "masking": "replace_or_adjust_audio_edges",
    "confidence_collapse": "reject_unverifiable_placement",
}


def failure(category: str, *, severity: str, confidence: float, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    if category not in FAILURE_REPAIRS:
        raise ValueError(f"Unknown editorial failure category: {category}")
    if severity not in {"low", "medium", "high", "critical"}:
        raise ValueError(f"Unknown editorial severity: {severity}")
    return {
        "category": category,
        "severity": severity,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
        "recommended_repair": FAILURE_REPAIRS[category],
        "evidence": dict(evidence or {}),
    }
