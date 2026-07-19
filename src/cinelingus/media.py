from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from . import __version__
from .tools import ffprobe_json
from .util import utc_now, write_json


def _rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    return float(Fraction(value))


def inspect_media(path: Path, media_hash: str, output_path: Path) -> dict:
    probe = ffprobe_json(path)
    streams = probe.get("streams", [])
    fmt = probe.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    width = video.get("width")
    height = video.get("height")
    data = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "path": str(path),
        "duration": float(fmt.get("duration") or 0.0),
        "resolution": f"{width}x{height}" if width and height else None,
        "frame_rate": _rate(video.get("avg_frame_rate")),
        "sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
        "channels": int(audio["channels"]) if audio.get("channels") else None,
        "codec": video.get("codec_name"),
        "streams": streams,
    }
    write_json(output_path, data)
    return data
