from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json


def annotate_windows_with_shots(windows: list[dict[str, Any]], shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not shots:
        return [dict(window) for window in windows]
    annotated = []
    for window in windows:
        item = dict(window)
        context = shot_context_for_range(float(item.get("start", 0.0)), float(item.get("end", item.get("start", 0.0) + item.get("duration", 0.0))), shots)
        item.update(context)
        annotated.append(item)
    return annotated


def shot_context_for_range(start: float, end: float, shots: list[dict[str, Any]]) -> dict[str, Any]:
    duration = max(0.0, end - start)
    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for shot in shots:
        shot_start = float(shot.get("start", 0.0))
        shot_end = float(shot.get("end", shot_start))
        overlap = max(0.0, min(end, shot_end) - max(start, shot_start))
        if overlap > best_overlap:
            best = shot
            best_overlap = overlap
    if best is None:
        return {
            "primary_shot_id": None,
            "shot_id": None,
            "shot_start": None,
            "shot_end": None,
            "crosses_shot_boundary": False,
            "boundary_overlap_seconds": 0.0,
        }
    shot_start = float(best.get("start", 0.0))
    shot_end = float(best.get("end", shot_start))
    crosses = start < shot_start - 0.001 or end > shot_end + 0.001
    return {
        "primary_shot_id": best.get("id"),
        "shot_id": best.get("id"),
        "shot_start": round(shot_start, 3),
        "shot_end": round(shot_end, 3),
        "crosses_shot_boundary": crosses,
        "boundary_overlap_seconds": round(max(0.0, duration - best_overlap), 3),
    }


def predicted_render_timing(window: dict[str, Any], clip: dict[str, Any], max_time_stretch: float) -> dict[str, float | str]:
    window_duration = float(window["duration"])
    clip_duration = float(clip["duration"])
    min_factor = 1.0 - max_time_stretch
    max_factor = 1.0 + max_time_stretch
    factor = window_duration / clip_duration
    trim_duration = clip_duration
    if min_factor <= factor <= max_factor:
        stretch_factor = factor
        timing_strategy = "stretch_to_window"
    elif clip_duration <= window_duration:
        stretch_factor = 1.0
        timing_strategy = "pad_trailing_silence"
    else:
        trim_duration = window_duration
        stretch_factor = 1.0
        timing_strategy = "trim_to_window"
    rendered_duration = trim_duration * stretch_factor
    return {
        "stretch_factor": stretch_factor,
        "trim_duration": trim_duration,
        "rendered_duration": rendered_duration,
        "timing_strategy": timing_strategy,
    }


def visual_fit_for_candidate(
    window: dict[str, Any],
    clip: dict[str, Any],
    *,
    max_time_stretch: float,
    shot_boundary_mode: str,
) -> dict[str, Any]:
    if shot_boundary_mode == "off":
        return {"visual_fit_score": 1.0, "mapping_crosses_shot_boundary": False, "boundary_overrun_seconds": 0.0}
    if window.get("shot_end") is None:
        return {"visual_fit_score": 0.8, "mapping_crosses_shot_boundary": False, "boundary_overrun_seconds": 0.0}

    timing = predicted_render_timing(window, clip, max_time_stretch)
    start = float(window.get("start", 0.0))
    end = start + float(timing["rendered_duration"])
    shot_end = float(window["shot_end"])
    overrun = max(0.0, end - shot_end)
    crosses = overrun > 0.001
    if not crosses:
        score = 1.0
    elif window.get("crosses_shot_boundary"):
        score = max(0.35, 1.0 - overrun / max(float(window.get("duration", 0.001)), 0.001))
    elif shot_boundary_mode == "strict":
        score = 0.0
    else:
        score = max(0.1, 1.0 - overrun / max(float(timing["rendered_duration"]), 0.001))
    return {
        "visual_fit_score": round(score, 4),
        "mapping_crosses_shot_boundary": crosses,
        "boundary_overrun_seconds": round(overrun, 3),
    }


def build_visual_schedule_report(
    *,
    shots_artifact: dict[str, Any],
    timeline: dict[str, Any],
    schedule: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    shots = shots_artifact.get("shots", [])
    windows = timeline.get("windows", [])
    mappings = schedule.get("mappings", [])
    shot_rows: dict[str, dict[str, Any]] = {}
    for shot in shots:
        shot_id = str(shot.get("id"))
        shot_rows[shot_id] = {
            "shot_id": shot_id,
            "start": shot.get("start"),
            "end": shot.get("end"),
            "duration": shot.get("duration"),
            "speaking_windows": 0,
            "mapped_clips": 0,
            "crossing_mappings": 0,
        }
    for window in windows:
        shot_id = window.get("shot_id") or window.get("primary_shot_id")
        if shot_id in shot_rows:
            shot_rows[str(shot_id)]["speaking_windows"] += 1
    for mapping in mappings:
        shot_id = mapping.get("shot_id") or mapping.get("primary_shot_id")
        if shot_id in shot_rows:
            shot_rows[str(shot_id)]["mapped_clips"] += 1
            if mapping.get("mapping_crosses_shot_boundary"):
                shot_rows[str(shot_id)]["crossing_mappings"] += 1

    mapped_counts = [row["mapped_clips"] for row in shot_rows.values()]
    average_mapped = sum(mapped_counts) / len(mapped_counts) if mapped_counts else 0.0
    overloaded_threshold = max(3, average_mapped * 2.0)
    empty_dialogue_shots = [row["shot_id"] for row in shot_rows.values() if row["mapped_clips"] == 0]
    overloaded_shots = [row for row in shot_rows.values() if row["mapped_clips"] > overloaded_threshold]
    crossing_mappings = [
        {
            "window_id": mapping.get("window_id"),
            "clip_id": mapping.get("clip_id"),
            "shot_id": mapping.get("shot_id"),
            "destination_timestamp": mapping.get("destination_timestamp"),
            "boundary_overrun_seconds": mapping.get("boundary_overrun_seconds", 0.0),
            "visual_fit_score": mapping.get("visual_fit_score"),
        }
        for mapping in mappings
        if mapping.get("mapping_crosses_shot_boundary")
    ]
    highest_risk = sorted(
        [
            {
                "window_id": mapping.get("window_id"),
                "clip_id": mapping.get("clip_id"),
                "shot_id": mapping.get("shot_id"),
                "visual_fit_score": mapping.get("visual_fit_score", 1.0),
                "boundary_overrun_seconds": mapping.get("boundary_overrun_seconds", 0.0),
                "score": mapping.get("score"),
            }
            for mapping in mappings
        ],
        key=lambda item: (float(item.get("visual_fit_score") or 1.0), -float(item.get("boundary_overrun_seconds") or 0.0)),
    )[:20]
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": shots_artifact.get("media_hash", schedule.get("media_hash", "")),
        "creation_timestamp": utc_now(),
        "shot_boundary_mode": schedule.get("shot_boundary_mode", "off"),
        "total_shots": len(shots),
        "total_speaking_windows": len(windows),
        "total_mappings": len(mappings),
        "empty_dialogue_shots": empty_dialogue_shots,
        "overloaded_shots": overloaded_shots,
        "crossing_mappings": crossing_mappings,
        "longest_silent_visual_runs": _silent_runs(list(shot_rows.values())),
        "highest_risk_mappings": highest_risk,
        "shots": list(shot_rows.values()),
    }
    write_json(output_path, report)
    return report


def _silent_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if row.get("mapped_clips") == 0:
            current.append(row)
        elif current:
            runs.append(_run_summary(current))
            current = []
    if current:
        runs.append(_run_summary(current))
    return sorted(runs, key=lambda item: item["duration"], reverse=True)[:10]


def _run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = float(rows[0].get("start") or 0.0)
    end = float(rows[-1].get("end") or start)
    return {
        "start_shot_id": rows[0]["shot_id"],
        "end_shot_id": rows[-1]["shot_id"],
        "shot_count": len(rows),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
    }
