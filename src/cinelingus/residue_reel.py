from __future__ import annotations

from pathlib import Path
from typing import Any
import wave

from . import __version__
from .util import utc_now, write_json


def build_residue_verification_reel(
    *,
    input_wav: Path,
    regions: list[dict[str, Any]],
    output_wav: Path,
    output_map: Path,
    duration: float | None = None,
    context_padding: float = 0.18,
    merge_gap: float = 0.3,
    separator_seconds: float = 0.35,
) -> dict[str, Any]:
    source_regions = _merged_regions(
        regions,
        duration=duration,
        padding=context_padding,
        merge_gap=merge_gap,
    )
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    with wave.open(str(input_wav), "rb") as source:
        params = source.getparams()
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        silence_frames = max(0, round(separator_seconds * frame_rate))
        silence = b"\x00" * silence_frames * source.getnchannels() * source.getsampwidth()
        reel_cursor_frames = 0
        with wave.open(str(output_wav), "wb") as target:
            target.setparams(params)
            for index, (start, end) in enumerate(source_regions, start=1):
                start_frame = max(0, min(total_frames, round(start * frame_rate)))
                end_frame = max(start_frame, min(total_frames, round(end * frame_rate)))
                if end_frame <= start_frame:
                    continue
                if segments and silence_frames:
                    target.writeframes(silence)
                    reel_cursor_frames += silence_frames
                source.setpos(start_frame)
                frames = source.readframes(end_frame - start_frame)
                reel_start = reel_cursor_frames / frame_rate
                target.writeframes(frames)
                reel_cursor_frames += end_frame - start_frame
                reel_end = reel_cursor_frames / frame_rate
                segments.append({
                    "id": f"verification_segment_{index:06d}",
                    "original_start": round(start_frame / frame_rate, 6),
                    "original_end": round(end_frame / frame_rate, 6),
                    "reel_start": round(reel_start, 6),
                    "reel_end": round(reel_end, 6),
                    "duration": round((end_frame - start_frame) / frame_rate, 6),
                })
    original_duration = sum(row["duration"] for row in segments)
    reel_duration = segments[-1]["reel_end"] if segments else 0.0
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "input_wav": str(input_wav),
        "output_wav": str(output_wav),
        "region_count": len(source_regions),
        "segment_count": len(segments),
        "original_region_duration": round(original_duration, 3),
        "reel_duration": round(reel_duration, 3),
        "separator_seconds": round(separator_seconds, 3),
        "context_padding": round(context_padding, 3),
        "segments": segments,
    }
    write_json(output_map, artifact)
    return artifact


def rebase_reel_timeline(
    *,
    reel_timeline: dict[str, Any],
    reel_map: dict[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    windows = reel_timeline.get("windows") or reel_timeline.get("events") or []
    rebased = []
    for window_index, window in enumerate(windows, start=1):
        reel_start, reel_end = _bounds(window)
        candidates = []
        for segment_index, segment in enumerate(reel_map.get("segments", []), start=1):
            segment_start = float(segment.get("reel_start", 0.0) or 0.0)
            segment_end = float(segment.get("reel_end", segment_start) or segment_start)
            overlap_start = max(reel_start, segment_start)
            overlap_end = min(reel_end, segment_end)
            if overlap_end <= overlap_start:
                continue
            candidates.append((overlap_end - overlap_start, segment_index, segment, overlap_start, overlap_end))
        if not candidates:
            continue
        # Whisper can return a single window spanning a separator. Its transcript
        # cannot be divided reliably, so assign it once to the source segment with
        # the greatest overlap instead of cloning the same evidence across several
        # unrelated original timestamps.
        _, segment_index, segment, overlap_start, overlap_end = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        segment_start = float(segment.get("reel_start", 0.0) or 0.0)
        original_start = float(segment.get("original_start", 0.0) or 0.0) + overlap_start - segment_start
        original_end = float(segment.get("original_start", 0.0) or 0.0) + overlap_end - segment_start
        rebased.append({
            **window,
            "id": f"rv{window_index:06d}_{segment_index:04d}",
            "start": round(original_start, 3),
            "end": round(original_end, 3),
            "duration": round(original_end - original_start, 3),
            "confidence": float(window.get("confidence", 0.5) or 0.5),
            "verification_segment_id": segment.get("id"),
            "reel_start": round(overlap_start, 3),
            "reel_end": round(overlap_end, 3),
            "spanned_verification_segment_count": len(candidates),
        })
    artifact = {
        **{key: value for key, value in reel_timeline.items() if key not in {"windows", "events"}},
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "detector": "residue_verification_reel_rebase_v2_single_assignment",
        "verification_reel": reel_map.get("output_wav"),
        "verification_segment_count": len(reel_map.get("segments", [])),
        "windows": sorted(rebased, key=lambda row: (float(row["start"]), float(row["end"]))),
    }
    if output_path is not None:
        write_json(output_path, artifact)
    return artifact


def schedule_for_verification_regions(schedule: dict[str, Any], regions: list[dict[str, Any]]) -> dict[str, Any]:
    bounds = [_bounds(row) for row in regions]
    scoped = dict(schedule)
    scoped["destination_speech_regions"] = [
        row for row in schedule.get("destination_speech_regions", [])
        if any(_overlap(*_bounds(row), start, end) for start, end in bounds)
    ]
    return scoped


def _merged_regions(
    regions: list[dict[str, Any]],
    *,
    duration: float | None,
    padding: float,
    merge_gap: float,
) -> list[tuple[float, float]]:
    intervals = []
    for region in regions:
        start, end = _bounds(region)
        start = max(0.0, start - max(0.0, padding))
        end += max(0.0, padding)
        if duration is not None:
            end = min(float(duration), end)
        if end > start:
            intervals.append((start, end))
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + max(0.0, merge_gap):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _bounds(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row.get("start", row.get("destination_timestamp", 0.0)) or 0.0)
    end = row.get("end")
    if end is None:
        end = start + float(row.get("duration", row.get("planned_render_duration", 0.0)) or 0.0)
    return start, float(end)


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return min(left_end, right_end) > max(left_start, right_start)
