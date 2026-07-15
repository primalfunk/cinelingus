from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


TRANSPOSITION = "Movie Masher"
LEGACY_TRANSPOSITION = "Transposition"
SELF_SHUFFLE = "Self Shuffle"
LEGACY_SELF_SHUFFLE = "Self-Shuffle"


MODE_DESCRIPTIONS = {
    TRANSPOSITION: "Transfers spoken performances from one film into another.",
    "Self Shuffle": "Rearranges a film's own spoken performances against itself.",
    "Echo": "Reintroduces dialogue through altered repetition and recurrence.",
    "Drift": "Allows dialogue and image to separate gradually over time.",
    "Contagion": "Lets one cinematic pattern spread through another.",
    "Possession": "Allows one vocal identity to overtake another.",
    "Foreshadow": "Introduces later speech into earlier moments.",
    "Bloom": "Expands selected fragments into increasingly transformed structures.",
}

MODE_GLYPHS = {
    TRANSPOSITION: "◇",
    "Self Shuffle": "↻",
    "Echo": "≈",
    "Drift": "→",
    "Contagion": "✦",
    "Possession": "◉",
    "Foreshadow": "◁",
    "Bloom": "✺",
}


def display_mode_name(value: str | None) -> str:
    normalized = str(value or "").strip()
    if normalized in {LEGACY_TRANSPOSITION, "movie_masher", "transposition", TRANSPOSITION}:
        return TRANSPOSITION
    if normalized in {SELF_SHUFFLE, LEGACY_SELF_SHUFFLE, "self_shuffle"}:
        return SELF_SHUFFLE
    return normalized


def internal_mode_name(value: str | None) -> str:
    displayed = display_mode_name(value)
    if displayed == TRANSPOSITION:
        return TRANSPOSITION
    if displayed == SELF_SHUFFLE:
        return LEGACY_SELF_SHUFFLE
    return str(value or "").strip()


def migrate_mode_value(value: str | None) -> tuple[str, str | None]:
    migrated = display_mode_name(value)
    note = None
    if str(value or "").strip() in {LEGACY_TRANSPOSITION, "movie_masher", "transposition"}:
        note = f"{value} migrated to {TRANSPOSITION}"
    return migrated, note


@dataclass(frozen=True)
class OperatorMessage:
    event_id: str
    title: str
    message: str
    severity: str = "info"
    stage_key: str | None = None
    diagnostic_detail: str = ""
    journal: bool = True


STAGE_MESSAGES = {
    "inspect": OperatorMessage("inspect_media", "Cataloguing specimen", "The selected material is being catalogued.", stage_key="inspect"),
    "source_dialogue": OperatorMessage("source_transcription", "Transcribing spoken passages", "The spoken record is being examined for usable passages.", stage_key="source_dialogue"),
    "clips": OperatorMessage("clip_slicing", "Constructing the dialogue archive", "Usable spoken fragments are being catalogued.", stage_key="clips"),
    "destination_speech": OperatorMessage("speaker_diarization", "Examining recurring voices", "Recurring vocal identities are under examination.", stage_key="destination_speech"),
    "performances": OperatorMessage("performance_grouping", "Assembling related performances", "Related spoken performances are being assembled.", stage_key="performances"),
    "schedule": OperatorMessage("scheduling", "Arranging the experiment", "Possible exchanges are being compared and arranged.", stage_key="schedule"),
    "render_audio": OperatorMessage("rendering", "Reconstructing the specimen", "The selected spoken performances are being reconstructed.", stage_key="render_audio"),
    "render_video": OperatorMessage("muxing", "Completing the cinematic artifact", "Picture and reconstructed sound are being assembled.", stage_key="render_video"),
    "finalize": OperatorMessage("validation", "Examining the completed artifact", "The completed artifact is undergoing final examination.", stage_key="finalize"),
}

MAJOR_STAGE_KEYS = tuple(STAGE_MESSAGES)


def stage_message(stage_key: str) -> OperatorMessage:
    return STAGE_MESSAGES.get(stage_key, OperatorMessage("operation", "Continuing the experiment", "The current operation is continuing.", stage_key=stage_key, journal=False))


def operator_message_for_log(line: str) -> OperatorMessage | None:
    text = str(line or "").strip()
    if not text:
        return None
    lowered = text.lower()
    timeout_configuration = any(token in lowered for token in ("inactivity timeout:", "total timeout:"))
    actual_timeout = (
        "timed out" in lowered
        or "inference timeout after" in lowered
        or "made no progress for" in lowered
        or "total safety limit" in lowered
        or ("exceeded" in lowered and "time" in lowered)
    )
    if actual_timeout and not timeout_configuration:
        return OperatorMessage("timeout", "Observation period exceeded", "The current examination exceeded its allotted observation period.", "warning", diagnostic_detail=text)
    if "fallback" in lowered or "fall back" in lowered:
        return OperatorMessage("fallback", "Alternate method in use", "The operation is continuing by an alternate method.", "warning", diagnostic_detail=text)
    if "cuda" in lowered and any(token in lowered for token in ("unavailable", "not available", "disabled")):
        return OperatorMessage("accelerator_unavailable", "Accelerated examination unavailable", "The operation will continue by slower means.", "warning", diagnostic_detail=text)
    if "validation failed" in lowered or "acceptance failed" in lowered:
        return OperatorMessage("validation_failed", "Final examination did not pass", "The reconstructed artifact did not pass final examination.", "error", diagnostic_detail=text)
    if "cache" in lowered or lowered.startswith("reused "):
        return OperatorMessage("cache_recovered", "Previous observations recovered", "Compatible prior observations have been recovered.", diagnostic_detail=text)
    if "processing finished" in lowered or "dialogue reel complete" in lowered:
        return OperatorMessage("completed", "Artifact archived", "The resulting cinematic artifact has been archived.", diagnostic_detail=text)
    key = stage_key_for_diagnostic(text)
    if key:
        base = stage_message(key)
        return OperatorMessage(base.event_id, base.title, base.message, base.severity, key, text, base.journal)
    return None


def stage_key_for_diagnostic(line: str) -> str | None:
    lowered = str(line or "").lower()
    patterns: tuple[tuple[str, Iterable[str]], ...] = (
        ("inspect", ("inspect media", "inspecting", "media inspection", "source loading")),
        ("source_dialogue", ("extract analysis audio", "transcribing source", "source dialogue", "whisper transcription", "spoken dialogue")),
        ("clips", ("clip slicing", "slicing", "clip library", "dialogue clips")),
        ("destination_speech", ("speaker diarization", "diarization", "destination timeline", "destination speech", "speaking performances")),
        ("performances", ("scene analysis", "performance grouping", "performances")),
        ("schedule", ("mapping", "scheduling", "schedule", "matching", "scene pair scoring")),
        ("render_audio", ("rendering replacement", "rendering selected", "rendering new dialogue", "rendered audio", "vignette render")),
        ("render_video", ("muxing", "rendered video", "render/export", "concatenating")),
        ("finalize", ("final artifact validation", "validating final artifact", "filter acceptance", "wrote run reports", "finalizing artifact")),
    )
    for stage_key, tokens in patterns:
        if any(token in lowered for token in tokens):
            return stage_key
    return None


def operator_text_is_backend_free(message: OperatorMessage) -> bool:
    visible = f"{message.title} {message.message}".lower()
    return not any(raw in visible for raw in ("whisper", "pyannote", "ffmpeg", "cuda", "schema", "subprocess"))


def contains_traceback(text: str) -> bool:
    return bool(re.search(r"(^|\n)traceback \(most recent call last\):", str(text or ""), flags=re.IGNORECASE))


def journal_messages_for_lines(lines: Iterable[str]) -> list[OperatorMessage]:
    messages = []
    seen = set()
    for line in lines:
        if str(line or "").strip().lower().startswith("[heartbeat]"):
            continue
        event = operator_message_for_log(line)
        if event is None or not event.journal or event.event_id in seen:
            continue
        seen.add(event.event_id)
        messages.append(event)
    return messages
