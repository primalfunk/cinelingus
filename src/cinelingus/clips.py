from __future__ import annotations

from pathlib import Path

from . import __version__
from .tools import run
from .util import utc_now, write_json

MAX_FALLBACK_CLIP_DURATION = 12.0
MIN_FALLBACK_CLIP_DURATION = 0.25
MAX_UTTERANCE_GAP = 0.5


def _segments_for_event(event: dict) -> list[tuple[float, float]]:
    start = float(event["start"])
    end = float(event["end"])
    segments = []
    cursor = start
    while cursor < end:
        duration = min(MAX_FALLBACK_CLIP_DURATION, end - cursor)
        if duration >= MIN_FALLBACK_CLIP_DURATION:
            segments.append((cursor, duration))
        cursor += duration
    return segments


def coalesce_dialogue_events(events: list[dict]) -> list[dict]:
    """Join adjacent transcription fragments until an utterance is complete."""
    ordered = sorted((dict(event) for event in events), key=lambda row: float(row.get("start", 0.0) or 0.0))
    utterances: list[dict] = []
    for event in ordered:
        event.setdefault("event_ids", [event.get("id")])
        event.setdefault("utterance_segment_count", 1)
        if utterances and _continues_utterance(utterances[-1], event):
            previous = utterances[-1]
            previous["end"] = event.get("end")
            previous["duration"] = round(float(previous["end"]) - float(previous["start"]), 3)
            previous["transcript"] = " ".join(
                part for part in (str(previous.get("transcript", "")).strip(), str(event.get("transcript", "")).strip()) if part
            )
            previous["event_ids"] = list(previous.get("event_ids", [])) + list(event.get("event_ids", []))
            previous["utterance_segment_count"] = int(previous.get("utterance_segment_count", 1) or 1) + 1
            confidences = [value for value in (previous.get("confidence"), event.get("confidence")) if value is not None]
            if confidences:
                previous["confidence"] = round(min(float(value) for value in confidences), 4)
            continue
        utterances.append(event)
    return utterances


def _continues_utterance(previous: dict, current: dict) -> bool:
    previous_end = float(previous.get("end", previous.get("start", 0.0)) or 0.0)
    current_start = float(current.get("start", 0.0) or 0.0)
    gap = current_start - previous_end
    combined_duration = float(current.get("end", current_start) or current_start) - float(previous.get("start", 0.0) or 0.0)
    if gap < -0.001 or gap > MAX_UTTERANCE_GAP or combined_duration > MAX_FALLBACK_CLIP_DURATION:
        return False
    previous_text = str(previous.get("transcript", "")).rstrip()
    current_text = str(current.get("transcript", "")).lstrip()
    if not previous_text or not current_text:
        return False
    return previous_text[-1] not in ".?!" or current_text[:1].islower()


def slice_clips(source_media: Path, media_hash: str, events: list[dict], output_dir: Path, library_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    clip_number = 1
    utterances = coalesce_dialogue_events(events)
    for event in utterances:
        for segment_start, segment_duration in _segments_for_event(event):
            clip_id = f"c{clip_number:06d}"
            clip_path = output_dir / f"{clip_number:06d}.wav"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{segment_start:.3f}",
                    "-t",
                    f"{segment_duration:.3f}",
                    "-i",
                    str(source_media),
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    str(clip_path),
                ]
            )
            text = event.get("transcript", "")
            words = len(text.split())
            speech_rate = words / segment_duration if segment_duration > 0 else 0.0
            clips.append(
                {
                    "id": clip_id,
                    "event_id": event["id"],
                    "event_ids": event.get("event_ids", [event["id"]]),
                    "path": str(clip_path),
                    "movie_timestamp": round(segment_start, 3),
                    "duration": round(segment_duration, 3),
                    "transcript": text,
                    "confidence": event.get("confidence"),
                    "usable": event.get("usable", True),
                    "reject_reason": event.get("reject_reason"),
                    "speaker": event.get("speaker"),
                    "speaker_id": event.get("speaker_id", event.get("speaker")),
                    "utterance_segment_count": event.get("utterance_segment_count", 1),
                    "speech_rate": round(speech_rate, 3),
                    "average_loudness": None,
                }
            )
            clip_number += 1
    data = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "utterance_count": len(utterances),
        "clips": clips,
    }
    write_json(library_path, data)
    return data
