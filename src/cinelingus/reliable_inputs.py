from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .tools import ToolError, ffprobe_json


class MediaPreflightError(ValueError):
    """Raised before a run when selected material cannot satisfy its contract."""


def default_input_directory(home: Path | None = None) -> Path:
    """Return the operator's preferred media folder with resilient fallbacks."""
    user_home = Path.home() if home is None else Path(home)
    candidates = (
        user_home / "Downloads" / "Music",
        user_home / "Downloads",
        user_home / "Music",
        user_home,
    )
    return next((path for path in candidates if path.is_dir()), user_home / "Downloads" / "Music")


def chooser_initial_directory(
    selected_path: str | Path | None,
    *,
    last_directory: Path | None,
    default_directory: Path,
) -> Path:
    """Prefer a selected file's folder, then the session folder, then the default."""
    if selected_path:
        selected = Path(selected_path).expanduser()
        if selected.is_file():
            return selected.parent
        if selected.is_dir():
            return selected
    if last_directory is not None and Path(last_directory).is_dir():
        return Path(last_directory)
    return Path(default_directory)


def preflight_media_inputs(
    films: Iterable[Path],
    *,
    output_dir: Path,
    probe: Callable[[Path], dict[str, Any]] = ffprobe_json,
) -> dict[str, Any]:
    """Probe complete film inputs and predict the audio-supported output duration."""
    paths = tuple(Path(path).expanduser().resolve() for path in films)
    if not paths:
        raise MediaPreflightError("Choose at least one film before activating the instrument.")

    reports = []
    for index, path in enumerate(paths):
        label = "Anchor Film" if index == 0 else f"Supporting Film {index}"
        if not path.is_file():
            raise MediaPreflightError(f"{label} does not exist or is not a file: {path}")
        try:
            media_probe = probe(path)
        except (ToolError, OSError, ValueError) as exc:
            raise MediaPreflightError(f"{label} cannot be examined by FFmpeg: {path}\n{exc}") from exc
        streams = list(media_probe.get("streams") or [])
        video_streams = [row for row in streams if row.get("codec_type") == "video"]
        audio_streams = [row for row in streams if row.get("codec_type") == "audio"]
        container_duration = _positive_duration((media_probe.get("format") or {}).get("duration"))
        video_duration = _stream_duration(video_streams, container_duration)
        audio_duration = _stream_duration(audio_streams, container_duration)
        if not video_streams:
            raise MediaPreflightError(f"{label} has no usable video stream: {path}")
        if video_duration is None:
            raise MediaPreflightError(f"{label} has no finite positive video duration: {path}")
        if index > 0 and not audio_streams:
            raise MediaPreflightError(f"{label} has no usable audio stream: {path}")
        if index > 0 and audio_duration is None:
            raise MediaPreflightError(f"{label} has no finite positive audio duration: {path}")
        reports.append({
            "role": "anchor" if index == 0 else "supporting_audio",
            "path": str(path),
            "duration": round(container_duration or video_duration, 3),
            "video_duration": round(video_duration, 3),
            "audio_duration": round(audio_duration, 3) if audio_duration is not None else None,
            "video_stream_count": len(video_streams),
            "audio_stream_count": len(audio_streams),
        })

    if len(reports) == 1 and reports[0]["audio_stream_count"] < 1:
        raise MediaPreflightError(f"Anchor Film has no usable audio stream for a one-film operation: {paths[0]}")
    if len(reports) == 1 and reports[0]["audio_duration"] is None:
        raise MediaPreflightError(f"Anchor Film has no finite positive audio duration: {paths[0]}")

    output = Path(output_dir).expanduser().resolve()
    _verify_output_directory(output)
    anchor_video_duration = float(reports[0]["video_duration"])
    supporting_audio_durations = [float(row["audio_duration"]) for row in reports[1:]]
    if not supporting_audio_durations:
        supporting_audio_durations = [float(reports[0]["audio_duration"])]
    predicted_duration = min([anchor_video_duration, *supporting_audio_durations])
    return {
        "status": "pass",
        "input_scope": "complete_media_files",
        "films": reports,
        "output_directory": str(output),
        "duration_policy": "FULL_SOURCE_TIMELINE_LIMITED_BY_SUPPORTING_AUDIO",
        "predicted_output_duration": round(predicted_duration, 3),
        "anchor_curtailed": predicted_duration + 0.001 < anchor_video_duration,
    }


def _stream_duration(streams: list[dict[str, Any]], fallback: float | None) -> float | None:
    durations = [_positive_duration(row.get("duration")) for row in streams]
    durations = [duration for duration in durations if duration is not None]
    return min(durations) if durations else fallback


def _positive_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _verify_output_directory(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".cinelingus-write-test-", dir=output_dir, delete=True):
            pass
    except OSError as exc:
        raise MediaPreflightError(f"Output folder is not writable: {output_dir}\n{exc}") from exc
