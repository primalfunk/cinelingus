from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json


def build_performance_placement_report(
    *,
    schedule: dict[str, Any],
    source_performances: dict[str, Any] | None = None,
    destination_performances: dict[str, Any] | None = None,
    output_json: Path,
    output_csv: Path,
    output_txt: Path,
) -> dict[str, Any]:
    source_by_id = _by_id((source_performances or {}).get("performances", []))
    destination_by_id = _by_id((destination_performances or {}).get("performances", []))
    mappings_by_pair = _mappings_by_pair(schedule.get("mappings", []))
    destination_fills = schedule.get("destination_performance_fills", [])
    destination_fill_by_id = {str(row.get("destination_performance_id")): row for row in destination_fills}
    placements = []

    placement_rows = schedule.get("performance_placements") or []
    if not placement_rows:
        placement_rows = [
            {
                "source_performance_id": source_id,
                "destination_performance_id": destination_id,
                "clip_ids": [mapping.get("clip_id") for mapping in mappings],
                "mapping_count": len(mappings),
                "scheduled_duration": round(sum(float(mapping.get("planned_render_duration", 0.0) or 0.0) for mapping in mappings), 3),
            }
            for (source_id, destination_id), mappings in mappings_by_pair.items()
        ]

    for row in placement_rows:
        source_id = str(row.get("source_performance_id") or "unknown_source_performance")
        destination_id = str(row.get("destination_performance_id") or "unknown_destination_performance")
        source = source_by_id.get(source_id, {})
        destination = destination_by_id.get(destination_id, {})
        mappings = mappings_by_pair.get((source_id, destination_id), [])
        destination_fill = destination_fill_by_id.get(destination_id, {})
        placements.append(_score_placement(row, source, destination, mappings, destination_fill))

    placements = sorted(placements, key=lambda item: item["quality_score"])
    summary = _summary(placements)
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": schedule.get("media_hash", ""),
        "creation_timestamp": utc_now(),
        "scheduling_mode": schedule.get("scheduling_mode"),
        "placement_count": len(placements),
        "summary": summary,
        "destination_fills": _score_destination_fills(destination_fills),
        "placements": placements,
    }
    write_json(output_json, report)
    _write_csv(placements, output_csv)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(_format_text(report), encoding="utf-8")
    return report


def _score_placement(
    row: dict[str, Any],
    source: dict[str, Any],
    destination: dict[str, Any],
    mappings: list[dict[str, Any]],
    destination_fill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_duration = _float(source.get("duration"), _float(row.get("source_performance_duration"), 0.0))
    destination_duration = _float(destination.get("duration"), 0.0)
    scheduled_duration = _float(row.get("scheduled_duration"), sum(_float(m.get("planned_render_duration"), 0.0) for m in mappings))
    duration_fit = _ratio_fit(source_duration, destination_duration or scheduled_duration)
    coverage = min(1.0, scheduled_duration / destination_duration) if destination_duration > 0 else 1.0
    source_density = _float(source.get("dialogue_density"), _float(row.get("source_performance_dialogue_density"), 0.0))
    destination_density = _float(destination.get("dialogue_density"), 0.0)
    density_fit = 1.0 - min(1.0, abs(source_density - destination_density)) if source_density or destination_density else 0.75
    source_type = row.get("source_performance_type") or source.get("conversation_type")
    destination_type = row.get("destination_performance_type") or destination.get("conversation_type")
    type_fit = 1.0 if source_type and source_type == destination_type else 0.7 if source_type and destination_type else 0.8
    average_mapping_score = _average([mapping.get("score") for mapping in mappings])
    average_visual_fit = _average([mapping.get("visual_fit_score") for mapping in mappings])
    destination_fill = destination_fill or {}
    aggregate_coverage = _float(destination_fill.get("coverage"), coverage)
    overcrowding_risk = max(0.0, (scheduled_duration / destination_duration) - 1.0) if destination_duration > 0 else 0.0
    sparsity_risk = max(0.0, 0.35 - aggregate_coverage)
    crossing_count = sum(1 for mapping in mappings if mapping.get("mapping_crosses_shot_boundary"))
    warnings = []
    if duration_fit < 0.55:
        warnings.append("duration_mismatch")
    if density_fit < 0.55:
        warnings.append("density_mismatch")
    if type_fit < 0.8:
        warnings.append("conversation_type_mismatch")
    if overcrowding_risk > 0.05:
        warnings.append("overcrowded_destination_performance")
    if sparsity_risk > 0.05:
        warnings.append("sparse_destination_performance")
    if crossing_count:
        warnings.append("contains_shot_boundary_crossings")

    quality_score = (
        duration_fit * 0.28
        + density_fit * 0.18
        + type_fit * 0.16
        + coverage * 0.14
        + (average_mapping_score or 0.0) * 0.14
        + (average_visual_fit or 0.0) * 0.10
    )
    quality_score = max(0.0, min(1.0, quality_score - min(0.25, overcrowding_risk * 0.2) - min(0.15, sparsity_risk * 0.3)))
    return {
        "source_performance_id": row.get("source_performance_id"),
        "source_performance_type": source_type,
        "destination_performance_id": row.get("destination_performance_id"),
        "destination_performance_type": destination_type,
        "mapping_count": int(row.get("mapping_count", len(mappings)) or 0),
        "clip_ids": row.get("clip_ids", []),
        "source_duration": round(source_duration, 3),
        "destination_duration": round(destination_duration, 3),
        "scheduled_duration": round(scheduled_duration, 3),
        "duration_fit": round(duration_fit, 4),
        "dialogue_density_fit": round(density_fit, 4),
        "conversation_type_fit": round(type_fit, 4),
        "destination_coverage": round(coverage, 4),
        "aggregate_destination_coverage": round(aggregate_coverage, 4),
        "destination_stop_reason": destination_fill.get("stop_reason"),
        "overcrowding_risk": round(overcrowding_risk, 4),
        "sparsity_risk": round(sparsity_risk, 4),
        "average_mapping_score": average_mapping_score,
        "average_visual_fit_score": average_visual_fit,
        "shot_boundary_crossing_count": crossing_count,
        "quality_score": round(quality_score, 4),
        "warnings": warnings,
    }


def _summary(placements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "average_quality_score": _average([item.get("quality_score") for item in placements]),
        "average_duration_fit": _average([item.get("duration_fit") for item in placements]),
        "average_density_fit": _average([item.get("dialogue_density_fit") for item in placements]),
        "warning_count": sum(len(item.get("warnings", [])) for item in placements),
        "lowest_quality": placements[:10],
    }


def _score_destination_fills(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        coverage = _float(row.get("coverage"), 0.0) or 0.0
        target = _float(row.get("target_coverage"), 0.75) or 0.75
        if coverage >= target:
            fill_quality = 1.0
        else:
            fill_quality = max(0.0, coverage / max(target, 0.001))
        warnings = []
        if coverage < 0.35:
            warnings.append("sparse_destination_performance")
        if row.get("stop_reason") == "remaining_gap_has_no_fitting_whole_line":
            warnings.append("whole_line_fit_blocked")
        if row.get("stop_reason") == "source_dialogue_exhausted":
            warnings.append("source_dialogue_exhausted")
        item = dict(row)
        item["fill_quality"] = round(fill_quality, 4)
        item["warnings"] = warnings
        scored.append(item)
    return sorted(scored, key=lambda item: item.get("fill_quality", 0.0))


def _write_csv(placements: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_performance_id",
        "source_performance_type",
        "destination_performance_id",
        "destination_performance_type",
        "mapping_count",
        "source_duration",
        "destination_duration",
        "scheduled_duration",
        "duration_fit",
        "dialogue_density_fit",
        "conversation_type_fit",
        "destination_coverage",
        "aggregate_destination_coverage",
        "destination_stop_reason",
        "quality_score",
        "warnings",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in placements:
            row = {field: item.get(field, "") for field in fields}
            row["warnings"] = ";".join(item.get("warnings", []))
            writer.writerow(row)


def _format_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "Cinelingus Performance Placement Report",
        "=======================================",
        f"Created: {report.get('creation_timestamp')}",
        f"Scheduling mode: {report.get('scheduling_mode')}",
        f"Placements: {report.get('placement_count')}",
        f"Average quality: {summary.get('average_quality_score')}",
        f"Average duration fit: {summary.get('average_duration_fit')}",
        f"Average density fit: {summary.get('average_density_fit')}",
        f"Warnings: {summary.get('warning_count')}",
        "",
        "Lowest Destination Fills",
    ]
    for item in report.get("destination_fills", [])[:10]:
        lines.append(
            f"  {item.get('destination_performance_id')}: coverage={item.get('coverage')} "
            f"quality={item.get('fill_quality')} stop={item.get('stop_reason')} "
            f"warnings={','.join(item.get('warnings', [])) or 'none'}"
        )
    lines.append("")
    lines.append("Lowest Quality Placements")
    for item in summary.get("lowest_quality", [])[:10]:
        lines.append(
            f"  {item.get('source_performance_id')} -> {item.get('destination_performance_id')}: "
            f"quality={item.get('quality_score')} warnings={','.join(item.get('warnings', [])) or 'none'}"
        )
    return "\n".join(lines) + "\n"


def _mappings_by_pair(mappings: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for mapping in mappings:
        source_id = str(mapping.get("source_performance_id") or "unknown_source_performance")
        destination_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id"))
        grouped.setdefault((source_id, destination_id), []).append(mapping)
    return grouped


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows if row.get("id")}


def _ratio_fit(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return max(0.0, min(left, right) / max(left, right))


def _average(values: list[Any]) -> float | None:
    numeric = [_float(value, None) for value in values]
    filtered = [value for value in numeric if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def _float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
