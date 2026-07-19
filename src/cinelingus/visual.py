from __future__ import annotations

import re
from pathlib import Path
from statistics import mean
from typing import Any

from . import __version__
from .tools import run
from .tools import ToolError
from .util import utc_now, write_json

PTS_RE = re.compile(r"pts_time:(?P<time>-?\d+(?:\.\d+)?)")
CORE_VISUAL_EVIDENCE_VERSION = "core_visual_evidence_v2"
BLACK_RE = re.compile(
    r"black_start:(?P<start>-?\d+(?:\.\d+)?)\s+black_end:(?P<end>-?\d+(?:\.\d+)?)\s+black_duration:(?P<duration>\d+(?:\.\d+)?)"
)
FRAME_DIFFERENCE_RE = re.compile(
    r"frame:\s*\d+.*?pts_time:(?P<time>-?\d+(?:\.\d+)?).*?lavfi\.signalstats\.YAVG=(?P<value>\d+(?:\.\d+)?)",
    flags=re.DOTALL,
)


def detect_shots(
    *,
    media_path: Path,
    media_hash: str,
    duration: float,
    output_path: Path,
    threshold: float,
    min_shot_duration: float,
    config_signature: str,
) -> dict[str, Any]:
    boundaries = detect_scene_boundaries(media_path, threshold)
    shots = build_shots_from_boundaries(boundaries, duration=duration, min_shot_duration=min_shot_duration, confidence=threshold)
    transitions, transition_status = _safe_transition_evidence(media_path, duration=duration)
    motion_samples, motion_status = _safe_frame_difference_samples(media_path)
    boundary_stability = build_boundary_stability(boundaries, motion_samples)
    stillness_intervals = build_stillness_intervals(motion_samples, duration=duration)
    gradual_candidates = build_gradual_transition_candidates(motion_samples, duration=duration)
    transitions.extend(gradual_candidates)
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "config_signature": config_signature,
        "detector": "ffmpeg_scene_change",
        "core_evidence_version": CORE_VISUAL_EVIDENCE_VERSION,
        "threshold": threshold,
        "min_shot_duration": min_shot_duration,
        "transitions": transitions,
        "boundary_stability": boundary_stability,
        "stillness_intervals": stillness_intervals,
        "core_evidence": {
            "transition_analysis": transition_status,
            "boundary_stability_analysis": motion_status,
            "frame_difference_sample_rate": 4.0,
            "frame_difference_threshold": 0.06,
            "stillness_threshold": 0.03,
            "gradual_transition_label": "GRADUAL_TRANSITION_CANDIDATE",
        },
        "shots": shots,
    }
    write_json(output_path, artifact)
    return artifact


def detect_scene_boundaries(media_path: Path, threshold: float) -> list[float]:
    filter_expr = f"select=gt(scene\\,{threshold}),showinfo"
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(media_path),
            "-vf",
            filter_expr,
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    text = "\n".join(part for part in [result.stdout, result.stderr] if part)
    points: list[float] = []
    for match in PTS_RE.finditer(text):
        value = float(match.group("time"))
        if value > 0:
            points.append(value)
    return sorted(set(round(point, 3) for point in points))


def parse_black_intervals(text: str, *, duration: float, guard_seconds: float = 0.25) -> list[dict[str, Any]]:
    intervals = []
    for match in BLACK_RE.finditer(str(text or "")):
        black_start = max(0.0, float(match.group("start")))
        black_end = min(float(duration), float(match.group("end")))
        intervals.append({
            "id": f"transition_{len(intervals) + 1:06d}",
            "kind": "BLACK_OR_FADE_GUARD",
            "start": round(max(0.0, black_start - guard_seconds), 3),
            "end": round(min(float(duration), black_end + guard_seconds), 3),
            "detected_black_start": round(black_start, 3),
            "detected_black_end": round(black_end, 3),
            "confidence": 0.7,
            "capability_tag": "CORE_HEURISTIC",
            "detector": "ffmpeg_blackdetect_guard_v1",
        })
    return intervals


def parse_frame_difference_samples(text: str) -> list[dict[str, float]]:
    return [
        {"time": round(float(match.group("time")), 3), "frame_difference": round(float(match.group("value")) / 255.0, 6)}
        for match in FRAME_DIFFERENCE_RE.finditer(str(text or ""))
    ]


def build_boundary_stability(
    boundaries: list[float],
    samples: list[dict[str, float]],
    *,
    threshold: float = 0.06,
    side_offset: float = 0.5,
    maximum_distance: float = 0.4,
) -> list[dict[str, Any]]:
    evidence = []
    for boundary in sorted(set(float(value) for value in boundaries)):
        selected = []
        for target in (boundary - side_offset, boundary + side_offset):
            if not samples:
                continue
            nearest = min(samples, key=lambda row: abs(float(row["time"]) - target))
            if abs(float(nearest["time"]) - target) <= maximum_distance:
                selected.append(nearest)
        if len(selected) != 2:
            evidence.append({
                "boundary": round(boundary, 3),
                "status": "UNAVAILABLE",
                "low_boundary_motion": None,
                "capability_tag": "FALLBACK_INFERENCE",
                "evidence_name": "BOUNDARY_STABILITY_UNAVAILABLE",
                "confidence": 0.0,
            })
            continue
        magnitude = sum(float(row["frame_difference"]) for row in selected) / len(selected)
        evidence.append({
            "boundary": round(boundary, 3),
            "status": "AVAILABLE",
            "sample_times": [float(row["time"]) for row in selected],
            "frame_difference_magnitude": round(magnitude, 6),
            "low_boundary_motion": magnitude <= threshold,
            "threshold": threshold,
            "capability_tag": "CORE_HEURISTIC",
            "evidence_name": "LOW_FRAME_DIFFERENCE_AT_BOUNDARY",
            "confidence": round(min(1.0, abs(magnitude - threshold) / max(threshold, 0.001)), 4),
        })
    return evidence


def build_stillness_intervals(
    samples: list[dict[str, float]],
    *,
    duration: float,
    threshold: float = 0.03,
    minimum_duration: float = 0.75,
    maximum_gap: float = 0.4,
) -> list[dict[str, Any]]:
    """Return literal sustained low-frame-difference intervals."""
    runs = _sample_runs(samples, predicate=lambda value: value <= threshold, maximum_gap=maximum_gap)
    rows = []
    for run in runs:
        start, end = _run_extent(run, duration=duration)
        if end - start < minimum_duration:
            continue
        rows.append({
            "id": f"stillness_{len(rows) + 1:06d}",
            "kind": "SUSTAINED_LOW_FRAME_DIFFERENCE",
            "start": start,
            "end": end,
            "mean_frame_difference": round(mean(float(row["frame_difference"]) for row in run), 6),
            "threshold": threshold,
            "capability_tag": "CORE_HEURISTIC",
            "detector": "ffmpeg_frame_difference_v2",
        })
    return rows


def build_gradual_transition_candidates(
    samples: list[dict[str, float]],
    *,
    duration: float,
    lower_threshold: float = 0.06,
    upper_threshold: float = 0.35,
    minimum_duration: float = 0.5,
    maximum_gap: float = 0.4,
) -> list[dict[str, Any]]:
    """Guard sustained change without asserting a dissolve or transition type."""
    runs = _sample_runs(samples, predicate=lambda value: lower_threshold <= value <= upper_threshold, maximum_gap=maximum_gap)
    rows = []
    for run in runs:
        start, end = _run_extent(run, duration=duration)
        if len(run) < 3 or end - start < minimum_duration:
            continue
        rows.append({
            "id": f"gradual_transition_{len(rows) + 1:06d}",
            "kind": "GRADUAL_TRANSITION_CANDIDATE",
            "start": start,
            "end": end,
            "mean_frame_difference": round(mean(float(row["frame_difference"]) for row in run), 6),
            "capability_tag": "CORE_HEURISTIC",
            "detector": "sustained_frame_difference_guard_v1",
            "confidence": 0.5,
        })
    return rows


def _sample_runs(samples: list[dict[str, float]], *, predicate: Any, maximum_gap: float) -> list[list[dict[str, float]]]:
    runs: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    for sample in sorted(samples, key=lambda row: float(row["time"])):
        if predicate(float(sample["frame_difference"])):
            if current and float(sample["time"]) - float(current[-1]["time"]) > maximum_gap:
                runs.append(current)
                current = []
            current.append(sample)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _run_extent(run: list[dict[str, float]], *, duration: float) -> tuple[float, float]:
    half_sample = 0.125
    return (round(max(0.0, float(run[0]["time"]) - half_sample), 3), round(min(float(duration), float(run[-1]["time"]) + half_sample), 3))


def _safe_transition_evidence(media_path: Path, *, duration: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        result = run(["ffmpeg", "-hide_banner", "-i", str(media_path), "-vf", "blackdetect=d=0.10:pix_th=0.10", "-an", "-f", "null", "-"])
        text = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return parse_black_intervals(text, duration=duration), {"status": "AVAILABLE", "backend": "ffmpeg_blackdetect", "backend_version": "v1", "device": "cpu", "fallback": False}
    except (ToolError, OSError) as exc:
        return [], {"status": "FAILED", "backend": "ffmpeg_blackdetect", "backend_version": "v1", "device": "cpu", "fallback": True, "detail": str(exc)}


def _safe_frame_difference_samples(media_path: Path) -> tuple[list[dict[str, float]], dict[str, Any]]:
    filters = "fps=4,tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG"
    try:
        result = run(["ffmpeg", "-hide_banner", "-i", str(media_path), "-vf", filters, "-an", "-f", "null", "-"])
        text = "\n".join(part for part in (result.stdout, result.stderr) if part)
        samples = parse_frame_difference_samples(text)
        status = "AVAILABLE" if samples else "LOW_CONFIDENCE"
        return samples, {"status": status, "backend": "ffmpeg_frame_difference", "backend_version": "v1", "device": "cpu", "fallback": not bool(samples), "sample_count": len(samples)}
    except (ToolError, OSError) as exc:
        return [], {"status": "FAILED", "backend": "ffmpeg_frame_difference", "backend_version": "v1", "device": "cpu", "fallback": True, "detail": str(exc), "sample_count": 0}


def build_shots_from_boundaries(
    boundaries: list[float],
    *,
    duration: float,
    min_shot_duration: float,
    confidence: float,
) -> list[dict[str, Any]]:
    clean_boundaries = [0.0]
    for point in sorted(boundaries):
        if point <= 0 or point >= duration:
            continue
        if point - clean_boundaries[-1] < min_shot_duration:
            continue
        clean_boundaries.append(point)
    if duration - clean_boundaries[-1] < min_shot_duration and len(clean_boundaries) > 1:
        clean_boundaries.pop()
    clean_boundaries.append(duration)

    shots = []
    for index, (start, end) in enumerate(zip(clean_boundaries, clean_boundaries[1:]), start=1):
        shot_duration = max(0.0, end - start)
        shots.append(
            {
                "id": f"shot_{index:06d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(shot_duration, 3),
                "confidence": round(confidence, 3),
            }
        )
    if not shots:
        shots.append({"id": "shot_000001", "start": 0.0, "end": round(duration, 3), "duration": round(duration, 3), "confidence": round(confidence, 3)})
    return shots


def build_visual_report(*, shots_artifact: dict[str, Any], movie: dict[str, Any], output_path: Path) -> dict[str, Any]:
    shots = shots_artifact.get("shots", [])
    durations = [float(shot.get("duration", 0.0)) for shot in shots]
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": shots_artifact["media_hash"],
        "creation_timestamp": utc_now(),
        "config_signature": shots_artifact.get("config_signature"),
        "detector": shots_artifact.get("detector"),
        "threshold": shots_artifact.get("threshold"),
        "min_shot_duration": shots_artifact.get("min_shot_duration"),
        "video_duration": float(movie.get("duration", 0.0)),
        "total_shots": len(shots),
        "average_shot_duration": round(mean(durations), 3) if durations else 0.0,
        "shortest_shot_duration": round(min(durations), 3) if durations else 0.0,
        "longest_shot_duration": round(max(durations), 3) if durations else 0.0,
        "shots_path": str(output_path.with_name("shots.json")),
    }
    write_json(output_path, report)
    return report
