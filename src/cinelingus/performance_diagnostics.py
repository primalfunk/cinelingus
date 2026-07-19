from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json


def build_performance_diagnostics(
    *,
    schedule: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    mappings = [mapping for mapping in schedule.get("mappings", []) if mapping.get("enabled", True)]
    by_destination = _group_by_destination(mappings)
    fills_by_id = {str(row.get("destination_performance_id")): row for row in schedule.get("destination_performance_fills", [])}
    diagnostics = []
    for destination_id in sorted(by_destination, key=_natural_key):
        rows = by_destination[destination_id]
        fill = fills_by_id.get(destination_id, {})
        source_ids = [str(row.get("source_performance_id") or "unknown_source_performance") for row in rows]
        unique_source_ids = sorted(set(source_ids), key=_natural_key)
        similarity_values = [_float(row.get("performance_similarity_score"), None) for row in rows]
        similarity_values = [value for value in similarity_values if value is not None]
        components = _average_components([row.get("performance_similarity_components") for row in rows])
        weights = _average_components([row.get("filter_weights") for row in rows])
        stretch_values = [abs(_float(row.get("stretch_factor"), 1.0) - 1.0) for row in rows]
        reuse_count = sum(1 for row in rows if row.get("rescue_reused_clip") or row.get("reuse_allowed_reason"))
        warnings = _warnings(rows, fill, components, similarity_values, reuse_count)
        diagnostics.append(
            {
                "destination_performance_id": destination_id,
                "destination_performance_type": rows[0].get("performance_type"),
                "chosen_source_performance_ids": unique_source_ids,
                "mapping_count": len(rows),
                "clip_ids": [row.get("clip_id") for row in rows],
                "coverage": _round(_float(fill.get("coverage"), 0.0)),
                "target_coverage": _round(_float(fill.get("target_coverage"), 0.0)),
                "scheduled_duration": _round(sum(_float(row.get("planned_render_duration"), 0.0) for row in rows)),
                "average_similarity_score": _round(sum(similarity_values) / len(similarity_values)) if similarity_values else 0.0,
                "lowest_similarity_score": _round(min(similarity_values)) if similarity_values else 0.0,
                "highest_stretch_delta": _round(max(stretch_values)) if stretch_values else 0.0,
                "reuse_count": reuse_count,
                "similarity_breakdown": components,
                "weight_contributions": weights,
                "selection_rationale": _rationales(rows),
                "top_rejected_candidates": [],
                "split_required": len(unique_source_ids) > 1,
                "split_reason": _split_reason(unique_source_ids, rows, fill),
                "stop_reason": fill.get("stop_reason"),
                "warnings": warnings,
            }
        )
    summary = {
        "performance_count": len(diagnostics),
        "average_similarity_score": _average([row.get("average_similarity_score") for row in diagnostics]),
        "low_similarity_count": sum(1 for row in diagnostics if _float(row.get("average_similarity_score"), 1.0) < 0.55),
        "high_stretch_count": sum(1 for row in diagnostics if _float(row.get("highest_stretch_delta"), 0.0) > 0.1),
        "reuse_count": sum(int(row.get("reuse_count", 0) or 0) for row in diagnostics),
        "warning_count": sum(len(row.get("warnings", [])) for row in diagnostics),
    }
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": schedule.get("media_hash", ""),
        "creation_timestamp": utc_now(),
        "scheduling_mode": schedule.get("scheduling_mode"),
        "active_filter": schedule.get("active_filter"),
        "summary": summary,
        "diagnostics": sorted(diagnostics, key=lambda row: (_float(row.get("average_similarity_score"), 1.0), _natural_key(str(row.get("destination_performance_id", ""))))),
    }
    write_json(output_path, artifact)
    return artifact


def _group_by_destination(mappings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for mapping in mappings:
        destination_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id"))
        grouped.setdefault(destination_id, []).append(mapping)
    return grouped


def _average_components(rows: list[Any]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            numeric = _float(value, None)
            if numeric is not None:
                values.setdefault(str(key), []).append(numeric)
    return {key: _round(sum(items) / len(items)) for key, items in sorted(values.items()) if items}


def _warnings(rows: list[dict[str, Any]], fill: dict[str, Any], components: dict[str, float], similarity_values: list[float], reuse_count: int) -> list[str]:
    warnings = []
    average_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 0.0
    if average_similarity < 0.55:
        warnings.append("low_similarity")
    if _float(fill.get("coverage"), 1.0) < _float(fill.get("target_coverage"), 0.75):
        warnings.append("under_target_coverage")
    if components.get("speaker_pattern", 1.0) < 0.5:
        warnings.append("speaker_pattern_mismatch")
    if components.get("dialogue_density", 1.0) < 0.55:
        warnings.append("dialogue_density_mismatch")
    if components.get("pause", 1.0) < 0.55 or components.get("response_delay", 1.0) < 0.55:
        warnings.append("pause_rhythm_mismatch")
    if components.get("performance_type", 1.0) < 0.6:
        warnings.append("performance_type_mismatch")
    if components.get("speech_continuity", 1.0) < 0.55:
        warnings.append("speech_continuity_mismatch")
    if components.get("silence_ratio", 1.0) < 0.55:
        warnings.append("silence_ratio_mismatch")
    if any(abs(_float(row.get("stretch_factor"), 1.0) - 1.0) > 0.1 for row in rows):
        warnings.append("high_stretch")
    if reuse_count:
        warnings.append("source_reuse")
    return warnings


def _rationales(rows: list[dict[str, Any]]) -> list[str]:
    rationales = []
    for row in rows:
        rationale = str(row.get("matching_rationale") or row.get("selection_reason") or "")
        if rationale and rationale not in rationales:
            rationales.append(rationale)
        if len(rationales) >= 3:
            break
    return rationales


def _split_reason(unique_source_ids: list[str], rows: list[dict[str, Any]], fill: dict[str, Any]) -> str | None:
    if len(unique_source_ids) <= 1:
        return None
    if fill.get("stop_reason") == "remaining_gap_has_no_fitting_whole_line":
        return "whole_performance_too_long_for_remaining_gap"
    if any(row.get("reuse_allowed_reason") for row in rows):
        return "coverage_conflict_required_reuse"
    return "multiple_source_performances_needed_for_coverage"


def _average(values: list[Any]) -> float | None:
    numeric = [_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return _round(sum(numeric) / len(numeric))


def _round(value: float | int | None) -> float:
    return round(float(value or 0.0), 4)


def _float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _natural_key(value: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return (prefix, int(digits or 0))
