from __future__ import annotations

from pathlib import Path
import math
import warnings
import wave
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

from . import __version__
from .tools import run
from .util import rel, utc_now, write_json


class AudioProvenanceError(RuntimeError):
    pass


MINIMUM_ACTIVE_AUDIO_RATIO = 0.25
ACTIVITY_WINDOW_SECONDS = 0.1
ACTIVITY_THRESHOLD_DBFS = -35.0



def extract_audio_for_provenance(*, media_path: Path, output_path: Path, sample_rate: int = 48000, channels: int = 2) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(output_path),
        ]
    )
    return _audio_stats(output_path)


def compare_wav_audio(*, left_path: Path, right_path: Path) -> dict[str, Any]:
    with wave.open(str(left_path), "rb") as left, wave.open(str(right_path), "rb") as right:
        sample_width = left.getsampwidth()
        if sample_width != right.getsampwidth():
            raise AudioProvenanceError("Cannot compare audio with different sample widths.")
        frame_count = min(left.getnframes(), right.getnframes())
        left_frames = left.readframes(frame_count)
        right_frames = right.readframes(frame_count)
    diff = _subtract_pcm16(left_frames, right_frames, sample_width)
    return {
        "left": str(left_path),
        "right": str(right_path),
        "compared_frames": frame_count,
        "diff_rms": round(_rms(diff, sample_width), 3),
    }

def verify_audio_provenance(
    *,
    root: Path,
    destination_video: Path,
    destination_hash: str,
    source_dialogue: Path,
    source_hash: str,
    schedule: dict[str, Any],
    short_schedule: dict[str, Any],
    replacement_audio: Path,
    final_video: Path,
    visual_segment: Path,
    output_path: Path,
    final_audio_analysis: dict[str, Any] | None = None,
    original_segment_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enabled = [row for row in short_schedule.get("mappings", []) if row.get("enabled", True)]
    clip_roots = sorted({_clip_cache_root(row.get("clip_path")) for row in enabled if row.get("clip_path")})
    clip_roots = [value for value in clip_roots if value]
    source_lines = [
        {
            "destination_timestamp": row.get("destination_timestamp"),
            "source_movie_timestamp": row.get("source_movie_timestamp") or row.get("clip_movie_timestamp"),
            "clip_id": row.get("clip_id"),
            "clip_path": row.get("clip_path"),
            "source_transcript": row.get("source_transcript", ""),
        }
        for row in enabled[:20]
    ]
    replacement = _audio_stats(replacement_audio)
    final = final_audio_analysis or {}
    original = original_segment_analysis or {}
    checks = {
        "source_hash_matches_schedule": schedule.get("source_media_hash") == source_hash,
        "destination_hash_matches_schedule": schedule.get("destination_media_hash", schedule.get("media_hash")) == destination_hash,
        "all_clip_roots_match_source": bool(clip_roots) and set(clip_roots) == {source_hash},
        "replacement_audio_has_energy": float(replacement.get("rms", 0.0) or 0.0) > 20.0,
        "replacement_audio_has_sufficient_activity": float(replacement.get("active_ratio", 0.0) or 0.0) >= MINIMUM_ACTIVE_AUDIO_RATIO,
        "final_audio_matches_replacement": _diff_pass(final.get("diff_from_replacement_rms")),
        "final_audio_differs_from_original_segment": _original_diff_pass(
            final.get("diff_from_original_segment_rms"),
            original.get("rms"),
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "status": status,
        "checks": checks,
        "inputs": {
            "destination_video": str(destination_video),
            "destination_hash": destination_hash,
            "source_dialogue": str(source_dialogue),
            "source_hash": source_hash,
        },
        "outputs": {
            "final_video": str(final_video),
            "replacement_audio": str(replacement_audio),
            "visual_segment_original_audio": str(visual_segment),
            "audio_provenance": str(output_path),
        },
        "relative_outputs": {
            "final_video": rel(final_video, root),
            "replacement_audio": rel(replacement_audio, root),
            "visual_segment_original_audio": rel(visual_segment, root),
            "audio_provenance": rel(output_path, root),
        },
        "schedule": {
            "source_media_hash": schedule.get("source_media_hash"),
            "destination_media_hash": schedule.get("destination_media_hash", schedule.get("media_hash")),
            "mapping_count": len(enabled),
            "source_clip_cache_roots": clip_roots,
            "first_source_lines": source_lines,
        },
        "audio_analysis": {
            "replacement_audio": replacement,
            "final_audio": final,
            "original_segment_audio": original,
        },
    }
    write_json(output_path, report)
    if status != "pass":
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        raise AudioProvenanceError(f"Audio provenance failed: {failed}. See {output_path}")
    return report


def analyze_wav_activity(path: Path) -> dict[str, Any]:
    """Return the activity and silence measurements used by output acceptance gates."""
    return _audio_stats(path)


def _audio_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rms": 0.0, "duration": 0.0}
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        sample_width = handle.getsampwidth()
        rms = _rms(frames, sample_width)
        duration = handle.getnframes() / max(1, handle.getframerate())
        activity = _windowed_activity(
            frames,
            sample_width=sample_width,
            channels=handle.getnchannels(),
            sample_rate=handle.getframerate(),
            duration=duration,
        )
        return {
            "path": str(path),
            "exists": True,
            "duration": round(duration, 3),
            "rms": round(rms, 3),
            "channels": handle.getnchannels(),
            "sample_rate": handle.getframerate(),
            **activity,
        }


def _rms(frames: bytes, sample_width: int) -> float:
    if not frames or sample_width != 2:
        return 0.0
    count = len(frames) // 2
    if count <= 0:
        return 0.0
    total = 0
    for index in range(0, len(frames) - 1, 2):
        value = int.from_bytes(frames[index : index + 2], "little", signed=True)
        total += value * value
    return (total / count) ** 0.5


def _windowed_activity(
    frames: bytes,
    *,
    sample_width: int,
    channels: int,
    sample_rate: int,
    duration: float,
) -> dict[str, float]:
    if not frames or sample_width != 2 or channels <= 0 or sample_rate <= 0:
        return {"active_ratio": 0.0, "silent_ratio": 1.0, "active_duration": 0.0}
    bytes_per_window = max(sample_width * channels, int(sample_rate * ACTIVITY_WINDOW_SECONDS) * sample_width * channels)
    threshold = 32767.0 * math.pow(10.0, ACTIVITY_THRESHOLD_DBFS / 20.0)
    window_count = 0
    active_windows = 0
    for offset in range(0, len(frames), bytes_per_window):
        window = frames[offset : offset + bytes_per_window]
        if not window:
            continue
        window_count += 1
        if audioop.rms(window, sample_width) >= threshold:
            active_windows += 1
    active_ratio = active_windows / max(1, window_count)
    return {
        "active_ratio": round(active_ratio, 4),
        "silent_ratio": round(1.0 - active_ratio, 4),
        "active_duration": round(min(duration, active_windows * ACTIVITY_WINDOW_SECONDS), 3),
    }




def _subtract_pcm16(left: bytes, right: bytes, sample_width: int) -> bytes:
    if sample_width != 2:
        return b""
    size = min(len(left), len(right))
    output = bytearray()
    for index in range(0, size - 1, 2):
        left_value = int.from_bytes(left[index : index + 2], "little", signed=True)
        right_value = int.from_bytes(right[index : index + 2], "little", signed=True)
        diff = max(-32768, min(32767, left_value - right_value))
        output.extend(int(diff).to_bytes(2, "little", signed=True))
    return bytes(output)

def _clip_cache_root(clip_path: Any) -> str | None:
    if not clip_path:
        return None
    parts = Path(str(clip_path)).parts
    for index, part in enumerate(parts):
        if part == "cache" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _diff_pass(value: Any) -> bool:
    if value is None:
        return True
    return float(value or 0.0) <= 600.0


def _original_diff_pass(diff: Any, original_rms: Any) -> bool:
    if diff is None:
        return True
    baseline = max(200.0, float(original_rms or 0.0) * 0.25)
    return float(diff or 0.0) >= baseline
