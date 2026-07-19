from __future__ import annotations

from typing import Any


def covered_speech_intervals(
    mappings: list[dict[str, Any]], speech_windows: list[dict[str, Any]]
) -> list[tuple[float, float]]:
    """Union rendered mapping intervals after clipping them to target speech."""
    targets = []
    for row in speech_windows:
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", start + float(row.get("duration", 0.0) or 0.0)) or start)
        if end > start:
            targets.append((start, end))
    intervals: list[tuple[float, float]] = []
    for mapping in mappings:
        start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        end = start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
        for target_start, target_end in targets or [(start, end)]:
            overlap_start = max(start, target_start)
            overlap_end = min(end, target_end)
            if overlap_end > overlap_start:
                intervals.append((overlap_start, overlap_end))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.001:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered_speech_duration(
    mappings: list[dict[str, Any]], speech_windows: list[dict[str, Any]]
) -> float:
    return sum(end - start for start, end in covered_speech_intervals(mappings, speech_windows))
