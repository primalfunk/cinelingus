from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json


def build_performance_library(
    *,
    media_hash: str,
    performances: dict[str, Any],
    clips: list[dict[str, Any]],
    output_path: Path,
    config_signature: str | None = None,
) -> dict[str, Any]:
    library_items = []
    for performance in performances.get("performances", []):
        start = float(performance.get("start", 0.0) or 0.0)
        end = float(performance.get("end", start) or start)
        contained_clips = [_clip_summary(clip) for clip in clips if _overlaps(clip, start, end)]
        library_items.append(
            {
                "id": performance.get("id"),
                "start": performance.get("start"),
                "end": performance.get("end"),
                "duration": performance.get("duration"),
                "conversation_type": performance.get("conversation_type", "unknown"),
                "signature": performance.get("signature", {}),
                "speaker_sequence": performance.get("speaker_sequence", []),
                "speaker_ids": performance.get("speaker_ids", []),
                "dominant_speaker_id": performance.get("dominant_speaker_id"),
                "speaker_pattern": performance.get("speaker_pattern", performance.get("turn_pattern", "")),
                "turn_pattern": performance.get("turn_pattern", ""),
                "clip_count": len(contained_clips),
                "clips": contained_clips,
                "review_history": [],
                "render_statistics": {
                    "scheduled_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                },
            }
        )
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "config_signature": config_signature or "",
        "performance_count": len(library_items),
        "performances": library_items,
    }
    write_json(output_path, artifact)
    return artifact


def _clip_summary(clip: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": clip.get("id"),
        "path": clip.get("path"),
        "movie_timestamp": clip.get("movie_timestamp"),
        "duration": clip.get("duration"),
        "transcript": clip.get("transcript", ""),
        "speaker": clip.get("speaker"),
        "speaker_id": clip.get("speaker_id") or clip.get("speaker"),
        "speech_rate": clip.get("speech_rate"),
        "average_loudness": clip.get("average_loudness"),
    }


def _overlaps(clip: dict[str, Any], start: float, end: float) -> bool:
    clip_start = float(clip.get("movie_timestamp", 0.0) or 0.0)
    clip_end = clip_start + float(clip.get("duration", 0.0) or 0.0)
    return max(start, clip_start) < min(end, clip_end)
