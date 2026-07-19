from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json


def build_problem_region_report(
    *,
    schedule: dict[str, Any],
    output_json: Path,
    output_csv: Path,
    output_txt: Path,
) -> dict[str, Any]:
    mappings = schedule.get("mappings", [])
    fallback_mappings = [
        _mapping_problem(index, mapping, "fallback_mapping")
        for index, mapping in enumerate(mappings)
        if mapping.get("enabled", True) and mapping.get("alignment_mode") != "speech_window_snap"
    ]
    underfilled = [
        _performance_problem(row)
        for row in schedule.get("destination_performance_fills", [])
        if _float(row.get("coverage"), 0.0) < _float(row.get("target_coverage"), 0.9)
    ]
    uncovered = [row for row in underfilled if int(row.get("uncovered_speech_window_count") or 0) > 0]
    undercovered_slots = _speech_slot_undercoverage_problems(schedule, minimum_coverage=0.8)
    low_fit = [
        _mapping_problem(index, mapping, "low_fit_mapping")
        for index, mapping in enumerate(mappings)
        if mapping.get("enabled", True)
        and (_float(mapping.get("score"), 1.0) < 0.55 or _float(mapping.get("visual_fit_score"), 1.0) < 0.75)
    ]
    rows = sorted(
        fallback_mappings + underfilled + undercovered_slots + low_fit,
        key=lambda row: (_float(row.get("start"), 0.0), row.get("problem_type", "")),
    )
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "problem_count": len(rows),
        "summary": {
            "fallback_mapping_count": len(fallback_mappings),
            "underfilled_performance_count": len(underfilled),
            "uncovered_speech_performance_count": len(uncovered),
            "undercovered_speech_window_count": len(undercovered_slots),
            "low_fit_mapping_count": len(low_fit),
        },
        "problems": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    _write_csv(rows, output_csv)
    output_txt.write_text(_format_text(report), encoding="utf-8")
    return report


def _mapping_problem(index: int, mapping: dict[str, Any], problem_type: str) -> dict[str, Any]:
    start = _float(mapping.get("destination_timestamp"), 0.0)
    duration = _float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration")), 0.0)
    return {
        "problem_type": problem_type,
        "severity": "medium" if problem_type == "fallback_mapping" else "low",
        "start": round(start, 3),
        "end": round(start + duration, 3),
        "duration": round(duration, 3),
        "mapping_indices": [index],
        "performance_id": mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id"),
        "window_id": mapping.get("window_id"),
        "clip_id": mapping.get("clip_id"),
        "score": mapping.get("score"),
        "visual_fit_score": mapping.get("visual_fit_score"),
        "alignment_mode": mapping.get("alignment_mode"),
        "reason": _mapping_reason(mapping, problem_type),
        "preview_hint": f"python run_cinelingus.py preview --mapping {index}",
        "transcript": mapping.get("source_transcript", ""),
    }


def _performance_problem(row: dict[str, Any]) -> dict[str, Any]:
    start = _float(row.get("start"), 0.0)
    duration = _float(row.get("duration"), 0.0)
    coverage = _float(row.get("coverage"), 0.0)
    target = _float(row.get("target_coverage"), 0.9)
    return {
        "problem_type": "underfilled_performance",
        "severity": "high" if coverage < target * 0.5 else "medium",
        "start": round(start, 3),
        "end": round(start + duration, 3),
        "duration": round(duration, 3),
        "mapping_indices": [],
        "performance_id": row.get("destination_performance_id"),
        "coverage": row.get("coverage"),
        "target_coverage": row.get("target_coverage"),
        "coverage_basis": row.get("coverage_basis"),
        "speech_window_count": row.get("speech_window_count", 0),
        "covered_speech_window_count": row.get("covered_speech_window_count", 0),
        "uncovered_speech_window_count": row.get("uncovered_speech_window_count", 0),
        "stop_reason": row.get("stop_reason"),
        "reason": f"coverage {coverage:.2f} below target {target:.2f}; {row.get('stop_reason', '')}",
        "preview_hint": "Review this timestamp range in the generated movie.",
    }


def _speech_slot_undercoverage_problems(schedule: dict[str, Any], *, minimum_coverage: float) -> list[dict[str, Any]]:
    mappings = [mapping for mapping in schedule.get("mappings", []) if mapping.get("enabled", True)]
    problems = []
    for row in schedule.get("destination_performance_fills", []):
        for slot in row.get("speech_windows", []):
            slot_id = str(slot.get("id"))
            start = _float(slot.get("start"), 0.0)
            end = _float(slot.get("end"), start + _float(slot.get("duration"), 0.0))
            duration = max(0.0, end - start)
            if duration <= 0.0:
                continue
            coverage_seconds = _slot_coverage_seconds(slot_id, start, end, mappings)
            coverage_ratio = min(1.0, coverage_seconds / duration)
            if coverage_ratio >= minimum_coverage:
                continue
            problems.append(
                {
                    "problem_type": "undercovered_speech_window",
                    "severity": "high" if coverage_ratio < 0.25 else "medium",
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(duration, 3),
                    "mapping_indices": _slot_mapping_indices(slot_id, mappings),
                    "performance_id": row.get("destination_performance_id"),
                    "window_id": slot_id,
                    "coverage": round(coverage_ratio, 4),
                    "target_coverage": minimum_coverage,
                    "coverage_basis": "individual_speech_window",
                    "covered_duration": round(min(duration, coverage_seconds), 3),
                    "uncovered_duration": round(max(0.0, duration - coverage_seconds), 3),
                    "alignment_mode": "speech_window_snap",
                    "reason": f"speech window coverage {coverage_ratio:.2f} below target {minimum_coverage:.2f}",
                    "preview_hint": "Review this speech-window timestamp in the generated movie.",
                }
            )
    return problems


def _slot_coverage_seconds(slot_id: str, start: float, end: float, mappings: list[dict[str, Any]]) -> float:
    total = 0.0
    for mapping in mappings:
        if slot_id not in {str(item) for item in mapping.get("alignment_source_window_ids", [])}:
            continue
        mapping_start = _float(mapping.get("destination_timestamp"), 0.0)
        mapping_end = mapping_start + _float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration")), 0.0)
        total += max(0.0, min(end, mapping_end) - max(start, mapping_start))
    return total


def _slot_mapping_indices(slot_id: str, mappings: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, mapping in enumerate(mappings)
        if slot_id in {str(item) for item in mapping.get("alignment_source_window_ids", [])}
    ]


def _mapping_reason(mapping: dict[str, Any], problem_type: str) -> str:
    if problem_type == "fallback_mapping":
        return "mapping could not be tied to a detected destination speech slot"
    parts = []
    if _float(mapping.get("score"), 1.0) < 0.55:
        parts.append("low duration/timing score")
    if _float(mapping.get("visual_fit_score"), 1.0) < 0.75:
        parts.append("low visual fit score")
    return "; ".join(parts) or "mapping marked as low fit"


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "problem_type",
        "severity",
        "start",
        "end",
        "duration",
        "performance_id",
        "window_id",
        "clip_id",
        "coverage",
        "target_coverage",
        "coverage_basis",
        "uncovered_speech_window_count",
        "covered_duration",
        "uncovered_duration",
        "score",
        "visual_fit_score",
        "alignment_mode",
        "reason",
        "preview_hint",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _format_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "Cinelingus Problem Regions Report",
        "==================================",
        f"Created: {report.get('creation_timestamp')}",
        "",
        "Summary",
        f"  fallback mappings: {summary.get('fallback_mapping_count')}",
        f"  underfilled performances: {summary.get('underfilled_performance_count')}",
        f"  performances with uncovered speech windows: {summary.get('uncovered_speech_performance_count')}",
        f"  undercovered speech windows: {summary.get('undercovered_speech_window_count')}",
        f"  low-fit mappings: {summary.get('low_fit_mapping_count')}",
        "",
        "Top Problems",
    ]
    for row in report.get("problems", [])[:25]:
        lines.append(
            f"  {row.get('start')}s {row.get('problem_type')} [{row.get('severity')}]: {row.get('reason')}"
        )
        if row.get("preview_hint"):
            lines.append(f"    {row.get('preview_hint')}")
    return "\n".join(lines) + "\n"


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
