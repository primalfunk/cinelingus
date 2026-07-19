from __future__ import annotations

import re
from pathlib import Path

from . import __version__
from .tools import run
from .util import utc_now, write_json

SILENCE_START = re.compile(r"silence_start: (?P<time>[0-9.]+)")
SILENCE_END = re.compile(r"silence_end: (?P<time>[0-9.]+)")


def extract_analysis_audio(input_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
    )


def detect_voice_windows(
    audio_path: Path,
    duration: float,
    *,
    noise_db: int,
    min_silence: float,
    min_speech: float,
    merge_gap: float,
) -> list[dict]:
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    silence: list[tuple[float, float]] = []
    active_start: float | None = None
    for line in (result.stderr or "").splitlines():
        start = SILENCE_START.search(line)
        if start:
            active_start = float(start.group("time"))
            continue
        end = SILENCE_END.search(line)
        if end and active_start is not None:
            silence.append((active_start, float(end.group("time"))))
            active_start = None
    if active_start is not None:
        silence.append((active_start, duration))

    windows: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in silence:
        if start - cursor >= min_speech:
            windows.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= min_speech:
        windows.append((cursor, duration))

    merged: list[tuple[float, float]] = []
    for start, end in windows:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return [
        {
            "id": f"w{i:06d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "confidence": 0.35,
        }
        for i, (start, end) in enumerate(merged, start=1)
    ]


def write_dialogue_events(path: Path, media_hash: str, windows: list[dict]) -> dict:
    events = []
    for i, window in enumerate(windows, start=1):
        events.append(
            {
                "id": f"e{i:06d}",
                "start": window["start"],
                "end": window["end"],
                "duration": window["duration"],
                "transcript": "",
                "confidence": window["confidence"],
                "speaker": None,
            }
        )
    data = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "detector": "ffmpeg_silencedetect_fallback",
        "events": events,
    }
    write_json(path, data)
    return data


def write_timeline(path: Path, media_hash: str, windows: list[dict]) -> dict:
    data = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "detector": "ffmpeg_silencedetect_fallback",
        "windows": windows,
    }
    write_json(path, data)
    return data
