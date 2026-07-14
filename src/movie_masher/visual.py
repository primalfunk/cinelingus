from __future__ import annotations

import re
from pathlib import Path
from statistics import mean
from typing import Any

from . import __version__
from .tools import run
from .util import utc_now, write_json

PTS_RE = re.compile(r"pts_time:(?P<time>-?\d+(?:\.\d+)?)")


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
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "config_signature": config_signature,
        "detector": "ffmpeg_scene_change",
        "threshold": threshold,
        "min_shot_duration": min_shot_duration,
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
