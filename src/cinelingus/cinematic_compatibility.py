from __future__ import annotations

from typing import Any


COMPATIBILITY_VERSION = "cinematic_compatibility_v1"


def score_cinematic_compatibility(
    *,
    source: dict[str, Any],
    destination: dict[str, Any],
    repetition_count: int = 0,
) -> dict[str, Any]:
    source_signature = source.get("source_performance_signature") or source.get("signature") or {}
    destination_signature = destination.get("signature") or destination.get("destination_performance_signature") or {}
    source_rate = _number(source_signature.get("words_per_second"), source.get("speech_rate", 0.0))
    destination_rate = _number(destination_signature.get("words_per_second"), destination.get("words_per_second", 0.0))
    source_duration = _number(source.get("duration"), source_signature.get("duration", 0.0))
    destination_duration = _number(destination.get("duration"), destination_signature.get("duration", 0.0))
    visual = destination.get("visual") or {}
    conversation = destination.get("conversation") or {}
    editing = destination.get("editing") or {}
    movement = destination.get("movement") or {}
    emotion = destination.get("emotion") or {}
    intent = destination.get("cinematic_intent") or visual.get("cinematic_intent") or {}
    source_participants = _number(source_signature.get("speaker_count"), 1.0)
    destination_participants = _number(conversation.get("participant_count"), destination_signature.get("speaker_count", 1.0))
    source_turn_density = _number(source_signature.get("turn_count"), 1.0) / max(source_duration, 0.001)
    destination_turn_density = _number(conversation.get("turn_density"), _number(destination_signature.get("turn_count"), 1.0) / max(destination_duration, 0.001))

    audio = {
        "transcript_completeness": _transcript_completeness(source, source_duration),
        "speech_clarity": _clamp(_number(source.get("confidence"), 0.7)),
        "speaking_rate": _ratio(source_rate + 0.25, destination_rate + 0.25),
        "timing_fit": _ratio(source_duration, destination_duration),
    }
    visual_components = {
        "mouth_agreement": _clamp(_number(visual.get("mouth_activity"), 0.5)),
        "visible_speaking_compatibility": _clamp(_number(intent.get("dialogue"), 0.5) * 0.65 + _number(intent.get("reaction"), 0.0) * 0.35),
        "action_conflict": _clamp(1.0 - _number(movement.get("action_intensity"), 0.5)),
        "face_visibility": _clamp(_number(visual.get("faces"), 0.0) / 2.0),
        "shot_intent": _clamp(max(_number(intent.get("dialogue"), 0.0), _number(intent.get("reaction"), 0.0), _number(intent.get("listening"), 0.0))),
    }
    conversation_components = {
        "interaction_similarity": _ratio(source_participants, destination_participants),
        "turn_compatibility": _ratio(source_turn_density + 0.05, destination_turn_density + 0.05),
    }
    editing_components = {
        "cut_alignment": _clamp(1.0 - _number(destination.get("boundary_overlap_seconds"), 0.0) / max(destination_duration, 0.001)),
        "transition_cleanliness": _clamp(_number(editing.get("continuity"), 0.75)),
        "reaction_timing": _clamp(_number(editing.get("reaction_alignment"), _number(intent.get("reaction"), 0.0))),
    }
    performance = {
        "rhythm_similarity": _ratio(source_turn_density + 0.05, destination_turn_density + 0.05),
        "energy_similarity": 1.0 - min(1.0, abs(_number(source_signature.get("estimated_energy"), 0.5) - _number(emotion.get("energy"), _number(destination_signature.get("estimated_energy"), 0.5)))),
    }
    surprise = _clamp(abs(_number(source_signature.get("estimated_energy"), 0.5) - _number(emotion.get("energy"), 0.5)) * 0.55 + _number(intent.get("reaction"), 0.0) * 0.45)
    novelty = {
        "surprise": surprise,
        "uniqueness": 1.0 if repetition_count <= 0 else 1.0 / (1.0 + repetition_count),
        "repetition_penalty": _clamp(1.0 - repetition_count * 0.35),
    }
    aggregate_confidence = _clamp((_number(source.get("confidence"), 0.7) + _number(visual.get("confidence"), 0.0) + _number(destination.get("metadata", {}).get("confidence"), 0.5)) / 3.0)
    categories = {
        "audio": _average(audio),
        "visual": _average(visual_components),
        "conversation": _average(conversation_components),
        "editing": _average(editing_components),
        "performance": _average(performance),
        "novelty": _average(novelty),
        "confidence": aggregate_confidence,
    }
    realism = _weighted(categories, {"audio": 0.28, "visual": 0.30, "conversation": 0.18, "editing": 0.14, "performance": 0.10})
    comedy = _clamp(surprise * 0.42 + novelty["uniqueness"] * 0.22 + editing_components["reaction_timing"] * 0.20 + categories["audio"] * 0.16)
    compatibility = _weighted(categories, {"audio": 0.22, "visual": 0.23, "conversation": 0.17, "editing": 0.14, "performance": 0.14, "novelty": 0.05, "confidence": 0.05})
    axes = {
        "realism": round(realism, 4),
        "comedy": round(comedy, 4),
        "surprise": round(surprise, 4),
        "novelty": round(categories["novelty"], 4),
        "compatibility": round(compatibility, 4),
        "confidence": round(aggregate_confidence, 4),
    }
    creative_exception_bonus = max(0.0, comedy - realism) * 0.18
    score = _clamp(compatibility + creative_exception_bonus)
    observations = _structured_observations(audio, visual_components, conversation_components, editing_components, novelty, aggregate_confidence)
    return {
        "version": COMPATIBILITY_VERSION,
        "score": round(score, 4),
        "categories": {key: round(value, 4) for key, value in categories.items()},
        "components": {
            "audio": {key: round(value, 4) for key, value in audio.items()},
            "visual": {key: round(value, 4) for key, value in visual_components.items()},
            "conversation": {key: round(value, 4) for key, value in conversation_components.items()},
            "editing": {key: round(value, 4) for key, value in editing_components.items()},
            "performance": {key: round(value, 4) for key, value in performance.items()},
            "novelty": {key: round(value, 4) for key, value in novelty.items()},
        },
        "axes": axes,
        "creative_exception_bonus": round(creative_exception_bonus, 4),
        "observations": observations,
        "explanation": _explanation(categories, axes),
    }


def _transcript_completeness(source: dict[str, Any], duration: float) -> float:
    transcript = str(source.get("transcript") or source.get("source_transcript") or "").strip()
    words = [word for word in transcript.split() if any(char.isalnum() for char in word)]
    if not words:
        return 0.0
    rate = len(words) / max(duration, 0.001)
    punctuation = 1.0 if transcript[-1:] in {".", "!", "?", "…"} else 0.65
    rate_fit = 1.0 if rate <= 3.4 else max(0.0, 1.0 - (rate - 3.4) / 3.0)
    return _clamp(punctuation * 0.45 + rate_fit * 0.55)


def _structured_observations(*groups: Any) -> list[dict[str, Any]]:
    *component_groups, confidence = groups
    rows = []
    domain_names = ("audio", "visual", "conversation", "editing", "novelty")
    for domain, components in zip(domain_names, component_groups):
        for label, score in components.items():
            rows.append({
                "domain": domain,
                "label": label,
                "status": "pass" if score >= 0.68 else "warning" if score >= 0.4 else "fail",
                "score": round(score, 4),
                "confidence": round(float(confidence), 4),
            })
    return rows


def _explanation(categories: dict[str, float], axes: dict[str, float]) -> str:
    strongest = max(categories, key=categories.get)
    weakest = min(categories, key=categories.get)
    return f"strongest={strongest}; weakest={weakest}; realism={axes['realism']:.3f}; comedy={axes['comedy']:.3f}"


def _weighted(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(values[key] * weight for key, weight in weights.items()) / max(sum(weights.values()), 0.001)


def _average(values: dict[str, float]) -> float:
    return sum(values.values()) / max(len(values), 1)


def _ratio(left: float, right: float) -> float:
    left, right = max(0.001, left), max(0.001, right)
    return _clamp(1.0 - abs(left - right) / max(left, right))


def _number(value: Any, fallback: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
