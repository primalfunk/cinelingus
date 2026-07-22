from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json

PUNCT_ONLY = re.compile(r"^[\W_]+$")
REPEATED_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class FilterConfig:
    min_duration: float
    max_duration: float
    min_confidence: float
    min_chars_per_second: float
    max_chars_per_second: float
    repeated_text_window: int


def filter_dialogue_events(raw: dict[str, Any], config: FilterConfig, output_path: Path) -> dict[str, Any]:
    events, stats = _filter_rows(raw.get("events", []), config, id_prefix="e")
    data = _base_filtered_artifact(raw, stats)
    data["events"] = events
    write_json(output_path, data)
    return data


def filter_timeline(raw: dict[str, Any], config: FilterConfig, output_path: Path) -> dict[str, Any]:
    windows, stats = _filter_rows(raw.get("windows", []), config, id_prefix="w")
    data = _base_filtered_artifact(raw, stats)
    data["windows"] = windows
    write_json(output_path, data)
    return data


def usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("usable", True)]


def _base_filtered_artifact(raw: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    data = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": raw["media_hash"],
        "creation_timestamp": utc_now(),
        "source_artifact_detector": raw.get("detector"),
        "source_config_signature": raw.get("config_signature"),
        "detector": f"segment_filter:{__version__}",
        "filter_stats": stats,
    }
    for key in ("speaker_map_media_hash", "speaker_diarization_tool", "speaker_diagnostics", "speaker_warnings", "acoustic_activity_windows"):
        if key in raw:
            data[key] = raw[key]
    return data

def _filter_rows(rows: list[dict[str, Any]], config: FilterConfig, *, id_prefix: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered = []
    recent_texts: deque[str] = deque(maxlen=max(1, config.repeated_text_window))
    reasons = Counter()
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["id"] = f"{id_prefix}{index:06d}"
        usable, reason = _classify(item, recent_texts, config)
        item["usable"] = usable
        item["reject_reason"] = None if usable else reason
        if not usable:
            reasons[reason] += 1
        normalized = _normalize_text(item.get("transcript", ""))
        if normalized:
            recent_texts.append(normalized)
        filtered.append(item)
    usable_count = sum(1 for row in filtered if row["usable"])
    return filtered, {"raw_count": len(rows), "usable_count": usable_count, "rejected_count": len(rows) - usable_count, "reject_reasons": dict(reasons)}


def _classify(row: dict[str, Any], recent_texts: deque[str], config: FilterConfig) -> tuple[bool, str | None]:
    duration = float(row.get("duration", 0.0) or 0.0)
    confidence = float(row.get("confidence", 0.0) or 0.0)
    text = str(row.get("transcript", ""))
    normalized = _normalize_text(text)
    if duration < config.min_duration:
        return False, "too_short"
    if duration > config.max_duration:
        return False, "too_long"
    if confidence < config.min_confidence:
        return False, "low_confidence"
    if not normalized:
        return False, "empty_text"
    if PUNCT_ONLY.match(normalized):
        return False, "mostly_punctuation"
    chars_per_second = len(normalized) / max(duration, 0.001)
    if chars_per_second < config.min_chars_per_second:
        return False, "unrealistic_char_rate_low"
    if chars_per_second > config.max_chars_per_second:
        return False, "unrealistic_char_rate_high"
    if normalized in recent_texts:
        return False, "repeated_text"
    if _looks_like_hallucination(normalized):
        return False, "likely_hallucination"
    return True, None


def _normalize_text(text: str) -> str:
    return REPEATED_SPACES.sub(" ", text.strip().lower())


def _looks_like_hallucination(text: str) -> bool:
    suspicious = {
        "thank you for watching",
        "thanks for watching",
        "subscribe",
        "like and subscribe",
        "captions by",
        "subtitles by",
    }
    return any(phrase in text for phrase in suspicious)

