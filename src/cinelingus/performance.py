from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from . import __version__
from .util import utc_now, write_json
from .filters import usable_rows


DEFAULT_MAX_PAUSE = 2.0


def build_performances(
    *,
    media_hash: str,
    role: str,
    output_path: Path,
    speaking_windows: list[dict[str, Any]],
    shots: list[dict[str, Any]] | None = None,
    dialogue_events: list[dict[str, Any]] | None = None,
    visual_observations: list[dict[str, Any]] | None = None,
    max_pause: float = DEFAULT_MAX_PAUSE,
    config_signature: str | None = None,
) -> dict[str, Any]:
    windows = sorted([dict(window) for window in speaking_windows if _duration(window) > 0], key=lambda item: float(item.get("start", 0.0)))
    shots = sorted([dict(shot) for shot in (shots or [])], key=lambda item: float(item.get("start", 0.0)))
    events = sorted([dict(event) for event in (dialogue_events or [])], key=lambda item: float(item.get("start", 0.0)))
    visual_observations = sorted([dict(row) for row in (visual_observations or [])], key=lambda item: float(item.get("start", 0.0)))
    groups = _group_windows(windows, max_pause=max_pause)
    performances = [
        _build_performance(
            index=index,
            group=group,
            shots=shots,
            dialogue_events=events,
            visual_observations=visual_observations,
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
                "speaker_ids": list(performance.get("speaker_ids", [])),
                "dominant_speaker_id": performance.get("dominant_speaker_id"),
                "speaker_pattern": performance.get("speaker_pattern", performance.get("turn_pattern", "")),
                "ordered_turns": list(performance.get("ordered_turns", [])),
                "speech_intervals": list(performance.get("speech_intervals", [])),
                "silence_intervals": list(performance.get("silence_intervals", [])),
                "cadence": dict(performance.get("cadence", {})),
                "interruption_frequency": performance.get("interruption_frequency", 0.0),
                "scene_category": performance.get("scene_category", performance.get("conversation_type", "unknown")),
                "source_shot_boundaries": list(performance.get("source_shot_boundaries", [])),
                "adaptability": dict(performance.get("adaptability", {})),
                "audio": dict(performance.get("audio", {})),
                "visual": dict(performance.get("visual", {})),
                "conversation": dict(performance.get("conversation", {})),
                "editing": dict(performance.get("editing", {})),
                "movement": dict(performance.get("movement", {})),
                "emotion": dict(performance.get("emotion", {})),
                "metadata": dict(performance.get("metadata", {})),
                "cinematic_intent": dict(performance.get("visual", {}).get("cinematic_intent", {})),
                "performance_model_version": performance.get("performance_model_version", "legacy"),
            }
        )
    return windows


def attach_performance_speech_windows(
    performance_rows: list[dict[str, Any]], timeline_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach canonical speech-window evidence to scheduler performance windows."""
    all_timeline_by_id = {str(window.get("id")): window for window in timeline_windows}
    usable_timeline_by_id = {str(window.get("id")): window for window in usable_rows(timeline_windows)}
    enriched = []
    for row in performance_rows:
        item = dict(row)
        speech_windows = []
        for window_id in item.get("speaking_window_ids", []):
            key = str(window_id)
            source = usable_timeline_by_id.get(key)
            source_kind = "detected_speech_window"
            if source is None:
                source = all_timeline_by_id.get(key)
                source_kind = "recovered_filtered_speech_window"
            if not source:
                continue
            start = float(source.get("start", 0.0) or 0.0)
            duration = float(source.get("duration", 0.0) or 0.0)
            end = float(source.get("end", start + duration) or start + duration)
            duration = max(0.0, end - start)
            if duration <= 0.0:
                continue
            speech_windows.append({
                "id": str(source.get("id")), "start": round(start, 3), "end": round(end, 3),
                "duration": round(duration, 3),
                "transcript": str(source.get("transcript") or source.get("text") or ""),
                "confidence": source.get("confidence", item.get("confidence", 0.7)),
                "speaker_id": source.get("speaker_id") or source.get("speaker"),
                "speaker": source.get("speaker") or source.get("speaker_id"),
                "speaker_confidence": source.get("speaker_confidence"), "source_kind": source_kind,
                "recovered": source_kind != "detected_speech_window", "reject_reason": source.get("reject_reason"),
            })
        if speech_windows:
            item["speech_windows"] = speech_windows
        enriched.append(item)
    return enriched


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
    visual_observations: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    start = min(float(item.get("start", 0.0)) for item in group)
    end = max(_end(item) for item in group)
    duration = max(0.001, end - start)
    pauses = _pauses(group)
    contained_shot_rows = _contained_items(shots, start, end)
    contained_shots = [str(item.get("id")) for item in contained_shot_rows]
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
    ordered_turns = [_performance_turn(item, index) for index, item in enumerate(group, start=1)]
    speech_intervals = [
        {"start": turn["start"], "end": turn["end"], "duration": turn["duration"], "turn_id": turn["id"]}
        for turn in ordered_turns
    ]
    silence_intervals = _silence_intervals(group)
    interruption_frequency = sum(
        1 for left, right in zip(group, group[1:]) if float(right.get("start", 0.0)) < _end(left) - 0.05
    ) / max(1, len(group) - 1)
    adaptability = {
        "stretchable": confidence >= 0.5 and duration >= 0.5,
        "compressible": confidence >= 0.5 and duration >= 1.0,
        "splittable": len(ordered_turns) > 1,
        "mergeable": any(
            left.get("speaker_id") == right.get("speaker_id") and left.get("speaker_id") is not None
            for left, right in zip(ordered_turns, ordered_turns[1:])
        ),
    }
    unified = _unified_sections(
        group=group,
        visual_rows=_contained_items(visual_observations, start, end),
        shot_rows=contained_shot_rows,
        duration=duration,
        speech_duration=speech_duration,
        words_per_second=words_per_second,
        pauses=pauses,
        ordered_turns=ordered_turns,
        confidence=confidence,
        energy=energy,
        interruption_frequency=interruption_frequency,
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
        "source_shot_boundaries": [
            {
                "shot_id": str(item.get("id")),
                "start": round(float(item.get("start", 0.0)), 3),
                "end": round(_end(item), 3),
            }
            for item in contained_shot_rows
        ],
        "ordered_turns": ordered_turns,
        "speech_intervals": speech_intervals,
        "silence_intervals": silence_intervals,
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
        "interruption_frequency": round(interruption_frequency, 4),
        "cadence": {
            "turns_per_second": round(turn_count / duration, 4),
            "words_per_second": round(words_per_second, 4),
            "average_pause": pause_stats["average"],
            "response_delay": round(response_delay, 3),
        },
        "conversation_type": conversation_type,
        "performance_type": performance_type,
        "scene_category": conversation_type,
        "adaptability": adaptability,
        "average_shot_length": round(average_shot_length, 3),
        "shot_change_rate": round(shot_change_rate, 4),
        "render_history": [],
        "review_history": [],
        "signature": signature,
        "confidence": round(confidence, 4),
        **unified,
    }


def _unified_sections(
    *,
    group: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    shot_rows: list[dict[str, Any]],
    duration: float,
    speech_duration: float,
    words_per_second: float,
    pauses: list[float],
    ordered_turns: list[dict[str, Any]],
    confidence: float,
    energy: float,
    interruption_frequency: float,
) -> dict[str, Any]:
    participants = sorted({str(row.get("speaker_id")) for row in ordered_turns if row.get("speaker_id")})
    participation_seconds: dict[str, float] = {}
    for row in ordered_turns:
        speaker = row.get("speaker_id")
        if speaker:
            participation_seconds[str(speaker)] = participation_seconds.get(str(speaker), 0.0) + float(row.get("duration", 0.0) or 0.0)
    dominant_ratio = max(participation_seconds.values(), default=speech_duration) / max(speech_duration, 0.001)
    overlaps = [
        max(0.0, _end(left) - float(right.get("start", 0.0) or 0.0))
        for left, right in zip(group, group[1:])
        if float(right.get("start", 0.0) or 0.0) < _end(left)
    ]
    visual_confidence = _mean_field(visual_rows, "overall_confidence", 0.0)
    intent_keys = sorted({key for row in visual_rows for key in row.get("cinematic_intent", {})})
    intentions = {key: round(mean(float(row.get("cinematic_intent", {}).get(key, 0.0) or 0.0) for row in visual_rows), 4) for key in intent_keys} if visual_rows else {}
    return {
        "performance_model_version": "unified_performance_v1",
        "audio": {
            "speech_timing": [{"start": row["start"], "end": row["end"], "duration": row["duration"]} for row in ordered_turns],
            "transcript": " ".join(str(row.get("transcript", "") or "") for row in ordered_turns).strip(),
            "speaking_rate": round(words_per_second, 4),
            "pauses": [round(value, 3) for value in pauses],
            "rhythm": {"turns_per_second": round(len(ordered_turns) / max(duration, 0.001), 4), "average_pause": round(mean(pauses), 3) if pauses else 0.0},
            "silence_ratio": round(max(0.0, 1.0 - speech_duration / max(duration, 0.001)), 4),
            "confidence": round(confidence, 4),
        },
        "visual": {
            "shot_observation_ids": [row.get("shot_id") for row in visual_rows],
            "faces": round(_mean_nested(visual_rows, "visible_face_count", "estimate"), 4),
            "mouth_activity": round(_mean_field(visual_rows, "mouth_activity_probability", 0.0), 4),
            "body_motion": round(_mean_field(visual_rows, "subject_motion_probability", 0.0), 4),
            "gaze": "probabilistic_unknown" if not visual_rows else "per_shot_observations",
            "framing": {"close_up": round(_mean_field(visual_rows, "close_up_probability", 0.0), 4), "wide": round(_mean_field(visual_rows, "wide_shot_probability", 0.0), 4)},
            "camera_motion": round(_mean_field(visual_rows, "camera_motion_probability", 0.0), 4),
            "cinematic_intent": intentions,
            "confidence": round(visual_confidence, 4),
        },
        "conversation": {
            "participants": participants,
            "participant_count": len(participants) if participants else 1,
            "interaction_pattern": "probabilistic_turn_profile",
            "interruptions": len(overlaps),
            "speaker_overlap": round(sum(overlaps) / max(duration, 0.001), 4),
            "response_latency": round(mean(pauses), 3) if pauses else 0.0,
            "turn_density": round(len(ordered_turns) / max(duration, 0.001), 4),
            "average_utterance_length": round(mean(float(row.get("duration", 0.0) or 0.0) for row in ordered_turns), 3) if ordered_turns else 0.0,
            "dominant_speaker_ratio": round(min(1.0, dominant_ratio), 4),
            "conversation_tempo": round((len(ordered_turns) + len(pauses)) / max(duration, 0.001), 4),
            "interruption_frequency": round(interruption_frequency, 4),
        },
        "editing": {
            "shot_boundaries": [{"shot_id": row.get("id"), "start": row.get("start"), "end": row.get("end")} for row in shot_rows],
            "cut_timing": round(len(shot_rows) / max(duration, 0.001), 4),
            "transitions": [],
            "reaction_alignment": round(intentions.get("reaction", 0.0), 4),
            "continuity": round(max(0.0, 1.0 - max(0, len(shot_rows) - 1) / max(duration, 1.0)), 4),
        },
        "movement": {
            "action_intensity": round(_mean_field(visual_rows, "action_level", 0.0), 4),
            "stillness": round(_mean_field(visual_rows, "stillness_probability", 0.0), 4),
            "motion_vectors": {"magnitude": round(_mean_field(visual_rows, "optical_motion_magnitude", 0.0), 4), "kind": "aggregate"},
        },
        "emotion": {
            "energy": round(energy, 4),
            "intensity": round(min(1.0, energy * 0.65 + interruption_frequency * 0.35), 4),
            "conversational_tension": round(min(1.0, interruption_frequency * 0.55 + (1.0 - (mean(pauses) if pauses else 0.0) / 2.0) * 0.45), 4),
            "method": "heuristic",
        },
        "metadata": {
            "confidence": round(mean([confidence, visual_confidence]) if visual_rows else confidence * 0.65, 4),
            "source_references": {"speaking_window_ids": [str(row.get("id")) for row in group], "shot_ids": [str(row.get("id")) for row in shot_rows]},
        },
    }


def _mean_field(rows: list[dict[str, Any]], field: str, default: float) -> float:
    values = [float(row.get(field, default) or 0.0) for row in rows]
    return mean(values) if values else default


def _mean_nested(rows: list[dict[str, Any]], field: str, child: str) -> float:
    values = [float(row.get(field, {}).get(child, 0.0) or 0.0) for row in rows]
    return mean(values) if values else 0.0


def _performance_turn(item: dict[str, Any], index: int) -> dict[str, Any]:
    start = float(item.get("start", 0.0))
    end = _end(item)
    speaker_id = item.get("speaker_id") or item.get("speaker")
    return {
        "id": str(item.get("id") or f"turn_{index:04d}"),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
        "speaker_id": str(speaker_id) if speaker_id is not None else None,
        "transcript": str(item.get("transcript", "") or ""),
        "confidence": round(_float(item.get("confidence"), 0.7), 4),
    }


def _silence_intervals(group: list[dict[str, Any]]) -> list[dict[str, float]]:
    intervals = []
    for left, right in zip(group, group[1:]):
        start = _end(left)
        end = float(right.get("start", start))
        if end > start:
            intervals.append({"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)})
    return intervals


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
