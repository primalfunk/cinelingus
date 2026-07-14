from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .review import (
    REVIEW_LABEL_BAD_DURATION,
    REVIEW_LABEL_BAD_SHOT,
    REVIEW_LABEL_DISABLE,
    REVIEW_LABEL_GOOD,
    REVIEW_LABEL_TOO_EARLY,
    REVIEW_LABEL_TOO_LATE,
    REVIEW_LABEL_UNREVIEWED,
    REVIEW_LABEL_WRONG_ENERGY,
)
from .util import utc_now, write_json

BAD_LABELS = {
    REVIEW_LABEL_BAD_DURATION,
    REVIEW_LABEL_BAD_SHOT,
    REVIEW_LABEL_DISABLE,
    REVIEW_LABEL_TOO_EARLY,
    REVIEW_LABEL_TOO_LATE,
    REVIEW_LABEL_WRONG_ENERGY,
}

CAUSES = (
    "low_score",
    "low_visual_fit",
    "crosses_shot_boundary",
    "large_boundary_overrun",
    "long_trim",
    "high_stretch",
    "disabled",
)


def build_review_analysis(
    *,
    review_notes: dict[str, Any],
    schedule: dict[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    mappings = schedule.get("mappings", [])
    notes = review_notes.get("notes", [])
    reviewed = []
    bad = []
    good = []
    cause_counts = {cause: 0 for cause in CAUSES}
    label_counts = dict(review_notes.get("label_counts", {}))

    for note in notes:
        index = int(note.get("mapping_index", -1))
        if index < 0 or index >= len(mappings):
            continue
        mapping = mappings[index]
        label = str(note.get("review_label", REVIEW_LABEL_UNREVIEWED))
        causes = infer_mapping_causes(mapping)
        for cause in causes:
            cause_counts[cause] += 1
        item = {
            "mapping_index": index,
            "window_id": mapping.get("window_id"),
            "clip_id": mapping.get("clip_id"),
            "review_label": label,
            "causes": causes,
            "score": mapping.get("score"),
            "visual_fit_score": mapping.get("visual_fit_score"),
            "mapping_crosses_shot_boundary": bool(mapping.get("mapping_crosses_shot_boundary")),
            "boundary_overrun_seconds": mapping.get("boundary_overrun_seconds", 0.0),
            "stretch_factor": mapping.get("stretch_factor"),
            "clip_trim_duration": mapping.get("clip_trim_duration"),
            "planned_render_duration": mapping.get("planned_render_duration"),
            "enabled": bool(mapping.get("enabled", True)),
        }
        reviewed.append(item)
        if label == REVIEW_LABEL_GOOD:
            good.append(item)
        elif label in BAD_LABELS:
            bad.append(item)

    recommendations = build_recommendations(bad, good, cause_counts)
    analysis = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": schedule.get("media_hash", review_notes.get("media_hash", "")),
        "creation_timestamp": utc_now(),
        "total_mappings": len(mappings),
        "reviewed_mappings": len(reviewed),
        "good_mappings": len(good),
        "bad_mappings": len(bad),
        "label_counts": label_counts,
        "cause_counts": cause_counts,
        "worst_mappings": sorted(bad, key=_risk_sort_key)[:20],
        "good_examples": sorted(good, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:20],
        "recommendations": recommendations,
    }
    if output_path is not None:
        write_json(output_path, analysis)
    return analysis


def infer_mapping_causes(mapping: dict[str, Any]) -> list[str]:
    causes = []
    if _float(mapping.get("score"), 1.0) < 0.55:
        causes.append("low_score")
    if _float(mapping.get("visual_fit_score"), 1.0) < 0.75:
        causes.append("low_visual_fit")
    if mapping.get("mapping_crosses_shot_boundary"):
        causes.append("crosses_shot_boundary")
    if _float(mapping.get("boundary_overrun_seconds"), 0.0) > 0.5:
        causes.append("large_boundary_overrun")
    if _float(mapping.get("clip_trim_duration"), 0.0) + 0.25 < _float(mapping.get("planned_render_duration"), 0.0):
        causes.append("high_stretch")
    if str(mapping.get("timing_strategy", "")).startswith("trim"):
        causes.append("long_trim")
    if not mapping.get("enabled", True):
        causes.append("disabled")
    return causes


def build_recommendations(bad: list[dict[str, Any]], good: list[dict[str, Any]], cause_counts: dict[str, int]) -> list[str]:
    if not bad and not good:
        return ["No reviewed mappings yet. Mark examples in Review Schedule to guide scheduler tuning."]
    recommendations = []
    if cause_counts.get("crosses_shot_boundary", 0):
        recommendations.append("Increase shot-boundary penalty or use strict shot-boundary mode for similar regions.")
    if cause_counts.get("low_visual_fit", 0):
        recommendations.append("Prefer clips that fit inside the primary shot when visual fit is low.")
    if cause_counts.get("low_score", 0):
        recommendations.append("Reduce acceptance of low aggregate score mappings or increase best-fit lookahead.")
    if cause_counts.get("long_trim", 0):
        recommendations.append("Penalize clips requiring heavy trim for reviewed-bad mappings.")
    if cause_counts.get("high_stretch", 0):
        recommendations.append("Lower tolerance for high stretch when reviewed examples sound unnatural.")
    if good:
        recommendations.append("Use reviewed-good mappings as positive examples when tuning duration and visual-fit weights.")
    return recommendations or ["Reviewed examples do not show a recurring measurable cause yet."]


def _risk_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _float(item.get("visual_fit_score"), 1.0),
        _float(item.get("score"), 1.0),
        -_float(item.get("boundary_overrun_seconds"), 0.0),
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
