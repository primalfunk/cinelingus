from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from . import __version__
from .util import utc_now, write_json


DEFAULT_MAX_PAUSE = 2.0


def build_performances(
    *,
    media_hash: str,
    role: str,
    output_path: Path,
    speaking_windows: list[dict[str, Any]],
    shots: list[dict[str, Any]] | None = None,
    dialogue_events: list[dict[str, Any]] | None = None,
    max_pause: float = DEFAULT_MAX_PAUSE,
    config_signature: str | None = None,
) -> dict[str, Any]:
    windows = sorted([dict(window) for window in speaking_windows if _duration(window) > 0], key=lambda item: float(item.get("start", 0.0)))
    shots = sorted([dict(shot) for shot in (shots or [])], key=lambda item: float(item.get("start", 0.0)))
    events = sorted([dict(event) for event in (dialogue_events or [])], key=lambda item: float(item.get("start", 0.0)))
    groups = _group_windows(windows, max_pause=max_pause)
    performances = [
        _build_performance(
            index=index,
            group=group,
            shots=shots,
            dialogue_events=events,
            role=role,
        )
        for index, group in enumerate(groups, start=1)
    ]
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "role": role,
        "max_pause": max_pause,
        "config_signature": config_signature or "",
        "performance_count": len(performances),
        "performances": performances,
    }
    write_json(output_path, artifact)
    return artifact


def performance_windows(performance_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for performance in performance_artifact.get("performances", []):
        windows.append(
            {
                "id": performance["id"],
                "start": performance["start"],
                "end": performance["end"],
                "duration": performance["duration"],
                "confidence": performance.get("confidence", 0.7),
                "performance_id": performance["id"],
                "performance_type": performance.get("conversation_type", "unknown"),
                "performance_type_v2": performance.get("performance_type", performance.get("conversation_type", "unknown")),
                "dialogue_density": performance.get("dialogue_density", 0.0),
                "words_per_second": performance.get("words_per_second", 0.0),
                "estimated_energy": performance.get("estimated_energy", 0.0),
                "speech_continuity": performance.get("speech_continuity", 0.0),
                "silence_ratio": performance.get("silence_ratio", performance.get("pause_ratio", 0.0)),
                "response_delay": performance.get("response_delay", 0.0),
                "speaking_window_ids": list(performance.get("speaking_window_ids", [])),
                "visible_speaking_window_count": len(performance.get("speaking_window_ids", [])),
                "shot_count": len(performance.get("shot_ids", [])),
                "signature": performance.get("signature", {}),
                "speaker_sequence": performance.get("speaker_sequence", []),
                "turn_pattern": performance.get("turn_pattern", ""),
            }
        )
    return windows


def _group_windows(windows: list[dict[str, Any]], *, max_pause: float) -> list[list[dict[str, Any]]]:
    if not windows:
        return []
    groups: list[list[dict[str, Any]]] = []
    current = [windows[0]]
    current_end = _end(windows[0])
    for window in windows[1:]:
        start = float(window.get("start", 0.0))
        gap = start - current_end
        if gap <= max_pause:
            current.append(window)
            current_end = max(current_end, _end(window))
        else:
            groups.append(current)
            current = [window]
            current_end = _end(window)
    groups.append(current)
    return groups


def _build_performance(
    *,
    index: int,
    group: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    dialogue_events: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    start = min(float(item.get("start", 0.0)) for item in group)
    end = max(_end(item) for item in group)
    duration = max(0.001, end - start)
    pauses = _pauses(group)
    contained_shots = _contained_ids(shots, start, end)
    contained_event_rows = _contained_items(dialogue_events, start, end)
    contained_events = [str(item.get("id")) for item in contained_event_rows]
    speech_duration = sum(_duration(item) for item in group)
    density = max(0.0, min(1.0, speech_duration / duration))
    turn_count = _estimated_turn_count(group)
    confidence_values = [_float(item.get("confidence"), 0.7) for item in group]
    confidence = mean(confidence_values) if confidence_values else 0.7
    speaker_count = _estimated_speaker_count(group)
    speaker_sequence = _speaker_sequence(group, speaker_count=speaker_count)
    conversation_type = _conversation_type(duration=duration, density=density, turn_count=turn_count, speaker_count=speaker_count)
    performance_type = _performance_type_v2(
        duration=duration,
        density=density,
        turn_count=turn_count,
        speaker_count=speaker_count,
        pauses=pauses,
        group=group,
    )
    pause_stats = {
        "count": len(pauses),
        "average": round(mean(pauses), 3) if pauses else 0.0,
        "max": round(max(pauses), 3) if pauses else 0.0,
        "total": round(sum(pauses), 3),
    }
    word_count = _word_count(contained_event_rows)
    words_per_second = word_count / max(speech_duration, 0.001) if word_count else 0.0
    pause_ratio = max(0.0, min(1.0, sum(pauses) / duration))
    speech_continuity = _speech_continuity(group, duration)
    response_delay = _response_delay(pauses)
    interruptions_detected = _interruptions_detected(group)
    shot_lengths = [_duration(item) for item in _contained_items(shots, start, end)]
    average_shot_length = mean(shot_lengths) if shot_lengths else duration
    shot_change_rate = len(contained_shots) / max(duration, 0.001)
    energy = _estimated_energy(
        density=density,
        turn_count=turn_count,
        duration=duration,
        pause_ratio=pause_ratio,
        words_per_second=words_per_second,
    )
    signature = _performance_signature(
        performance_id=f"p{index:06d}",
        duration=duration,
        group=group,
        speaker_count=speaker_count,
        speaker_sequence=speaker_sequence,
        pauses=pauses,
        pause_stats=pause_stats,
        density=density,
        words_per_second=words_per_second,
        estimated_energy=energy,
        speech_continuity=speech_continuity,
        response_delay=response_delay,
        interruptions_detected=interruptions_detected,
        average_shot_length=average_shot_length,
        shot_change_rate=shot_change_rate,
        confidence=confidence,
        shot_count=len(contained_shots),
        conversation_type=conversation_type,
        performance_type=performance_type,
    )
    return {
        "id": f"p{index:06d}",
        "role": role,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "speaking_window_ids": [str(item.get("id")) for item in group],
        "dialogue_event_ids": contained_events,
        "shot_ids": contained_shots,
        "estimated_speaker_count": speaker_count,
        "estimated_turn_count": turn_count,
        "speaker_sequence": speaker_sequence,
        "turn_pattern": " ".join(speaker_sequence),
        "pause_statistics": pause_stats,
        "dialogue_density": round(density, 4),
        "pause_ratio": round(pause_ratio, 4),
        "silence_ratio": round(pause_ratio, 4),
        "words_per_second": round(words_per_second, 4),
        "estimated_energy": round(energy, 4),
        "speech_continuity": round(speech_continuity, 4),
        "response_delay": round(response_delay, 3),
        "interruptions_detected": interruptions_detected,
        "conversation_type": conversation_type,
        "performance_type": performance_type,
        "average_shot_length": round(average_shot_length, 3),
        "shot_change_rate": round(shot_change_rate, 4),
        "render_history": [],
        "review_history": [],
        "signature": signature,
        "confidence": round(confidence, 4),
    }


def _speaker_sequence(group: list[dict[str, Any]], *, speaker_count: int) -> list[str]:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    explicit: dict[str, str] = {}
    sequence: list[str] = []
    for index, item in enumerate(group):
        speaker = item.get("speaker")
        if speaker:
            key = str(speaker)
            if key not in explicit:
                explicit[key] = labels[min(len(explicit), len(labels) - 1)]
            sequence.append(explicit[key])
        elif speaker_count <= 1:
            sequence.append("A")
        else:
            sequence.append(labels[index % min(speaker_count, len(labels))])
    compact: list[str] = []
    for label in sequence:
        if not compact or compact[-1] != label:
            compact.append(label)
    return compact or ["A"]


def _performance_signature(
    *,
    performance_id: str,
    duration: float,
    group: list[dict[str, Any]],
    speaker_count: int,
    speaker_sequence: list[str],
    pauses: list[float],
    pause_stats: dict[str, Any],
    density: float,
    words_per_second: float,
    estimated_energy: float,
    speech_continuity: float,
    response_delay: float,
    interruptions_detected: bool,
    average_shot_length: float,
    shot_change_rate: float,
    confidence: float,
    shot_count: int,
    conversation_type: str,
    performance_type: str,
) -> dict[str, Any]:
    turn_durations = [_duration(item) for item in group if _duration(item) > 0]
    turn_count = max(1, len(speaker_sequence))
    pause_ratio = max(0.0, min(1.0, float(pause_stats.get("total", 0.0)) / max(duration, 0.001)))
    return {
        "performance_id": performance_id,
        "signature_version": "2.0",
        "performance_type": performance_type,
        "conversation_type": conversation_type,
        "duration": round(duration, 3),
        "turn_count": turn_count,
        "speaker_count": speaker_count,
        "speaker_sequence": speaker_sequence,
        "turn_pattern": " ".join(speaker_sequence),
        "average_turn_duration": round(mean(turn_durations), 3) if turn_durations else 0.0,
        "minimum_turn_duration": round(min(turn_durations), 3) if turn_durations else 0.0,
        "maximum_turn_duration": round(max(turn_durations), 3) if turn_durations else 0.0,
        "pause_count": int(pause_stats.get("count", 0)),
        "average_pause_duration": round(float(pause_stats.get("average", 0.0) or 0.0), 3),
        "longest_pause": round(float(pause_stats.get("max", 0.0) or 0.0), 3),
        "dialogue_density": round(density, 4),
        "words_per_second": round(words_per_second, 4),
        "estimated_energy": round(estimated_energy, 4),
        "shot_count": shot_count,
        "average_shot_length": round(average_shot_length, 3),
        "shot_change_rate": round(shot_change_rate, 4),
        "speech_continuity": round(speech_continuity, 4),
        "interruptions_detected": bool(interruptions_detected),
        "response_delay": round(response_delay, 3),
        "silence_ratio": round(pause_ratio, 4),
        "speaker_lifetime": round(duration, 3),
        "speaker_confidence": round(confidence, 4),
        "speaker_continuity": round(_speaker_continuity(speaker_sequence), 4),
        "speaker_participation": _speaker_participation(speaker_sequence),
        "coverage": 1.0,
        "confidence": round(confidence, 4),
        "review_score": 0.0,
        "render_history": [],
        "review_history": [],
    }


def _conversation_type(*, duration: float, density: float, turn_count: int, speaker_count: int) -> str:
    if density < 0.18:
        return "background_speech"
    if speaker_count <= 1 and turn_count <= 2:
        return "monologue"
    if turn_count >= 8 and duration <= 20:
        return "rapid_exchange"
    if speaker_count >= 3:
        return "group_discussion"
    return "exchange"


def _performance_type_v2(
    *,
    duration: float,
    density: float,
    turn_count: int,
    speaker_count: int,
    pauses: list[float],
    group: list[dict[str, Any]],
) -> str:
    if duration < 1.0 or turn_count <= 0:
        return "fragment"
    if density < 0.18:
        return "background_conversation"
    if _interruptions_detected(group):
        return "interrupted_conversation"
    if speaker_count >= 3:
        return "group_conversation"
    if turn_count >= 8 and duration <= 20:
        return "rapid_exchange"
    if turn_count >= 6 and pauses and mean(pauses) < 0.35 and density > 0.55:
        return "argument"
    if speaker_count <= 1 and turn_count <= 2:
        return "monologue"
    if speaker_count >= 2 or turn_count > 2:
        return "dialogue_exchange"
    return "unknown"


def _estimated_speaker_count(group: list[dict[str, Any]]) -> int:
    speakers = {item.get("speaker") for item in group if item.get("speaker")}
    if speakers:
        return len(speakers)
    if len(group) >= 6:
        return 2
    return 1


def _estimated_turn_count(group: list[dict[str, Any]]) -> int:
    if not group:
        return 0
    speakers = [item.get("speaker") for item in group]
    if any(speakers):
        turns = 1
        previous = speakers[0]
        for speaker in speakers[1:]:
            if speaker != previous:
                turns += 1
                previous = speaker
        return turns
    return len(group)


def _pauses(group: list[dict[str, Any]]) -> list[float]:
    pauses = []
    previous_end = _end(group[0]) if group else 0.0
    for item in group[1:]:
        start = float(item.get("start", 0.0))
        pauses.append(round(max(0.0, start - previous_end), 3))
        previous_end = max(previous_end, _end(item))
    return pauses


def _word_count(events: list[dict[str, Any]]) -> int:
    total = 0
    for event in events:
        transcript = str(event.get("transcript", "") or "")
        total += len([word for word in transcript.replace("\n", " ").split(" ") if word.strip()])
    return total


def _speech_continuity(group: list[dict[str, Any]], duration: float) -> float:
    if not group:
        return 0.0
    longest = max((_duration(item) for item in group), default=0.0)
    return max(0.0, min(1.0, longest / max(duration, 0.001)))


def _response_delay(pauses: list[float]) -> float:
    responsive = [pause for pause in pauses if pause <= 3.0]
    return mean(responsive) if responsive else 0.0


def _interruptions_detected(group: list[dict[str, Any]]) -> bool:
    if len(group) < 2:
        return False
    previous_end = _end(group[0])
    for item in group[1:]:
        start = float(item.get("start", 0.0))
        if start < previous_end - 0.05:
            return True
        previous_end = max(previous_end, _end(item))
    return False


def _estimated_energy(*, density: float, turn_count: int, duration: float, pause_ratio: float, words_per_second: float) -> float:
    turn_rate = turn_count / max(duration, 0.001)
    speech_rate_score = min(1.0, words_per_second / 3.5) if words_per_second > 0 else 0.5
    energy = (density * 0.42) + (min(1.0, turn_rate / 0.5) * 0.24) + ((1.0 - pause_ratio) * 0.18) + (speech_rate_score * 0.16)
    return max(0.0, min(1.0, energy))


def _speaker_continuity(sequence: list[str]) -> float:
    if len(sequence) < 2:
        return 1.0
    repeats = sum(1 for left, right in zip(sequence, sequence[1:]) if left == right)
    return repeats / (len(sequence) - 1)


def _speaker_participation(sequence: list[str]) -> dict[str, float]:
    if not sequence:
        return {}
    total = len(sequence)
    labels = sorted(set(sequence))
    return {label: round(sequence.count(label) / total, 4) for label in labels}


def _contained_ids(items: list[dict[str, Any]], start: float, end: float) -> list[str]:
    ids = []
    for item in items:
        item_start = float(item.get("start", 0.0))
        item_end = _end(item)
        if max(start, item_start) < min(end, item_end):
            ids.append(str(item.get("id")))
    return ids


def _contained_items(items: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    contained = []
    for item in items:
        item_start = float(item.get("start", 0.0))
        item_end = _end(item)
        if max(start, item_start) < min(end, item_end):
            contained.append(item)
    return contained


def _end(item: dict[str, Any]) -> float:
    return float(item.get("end", float(item.get("start", 0.0)) + _duration(item)))


def _duration(item: dict[str, Any]) -> float:
    return max(0.0, float(item.get("duration", 0.0)))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
