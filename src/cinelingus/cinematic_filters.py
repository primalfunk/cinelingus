from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCORING_DIMENSIONS = (
    "duration",
    "speaker_count",
    "turn_count",
    "average_turn_duration",
    "pause",
    "dialogue_density",
    "energy",
    "shot_rate",
    "conversation_type",
    "performance_type",
    "speaker_pattern",
    "speech_continuity",
    "response_delay",
    "silence_ratio",
    "words_per_second",
    "interruptions",
    "reuse_penalty",
    "contrast_bonus",
    "silence_penalty",
    "review_feedback",
)


@dataclass(frozen=True)
class CinematicFilter:
    id: str
    display_name: str
    description: str
    weights: dict[str, float]
    notes: str

    def score(self, components: dict[str, float], source_signature: dict[str, Any], destination_signature: dict[str, Any]) -> dict[str, Any]:
        adjusted = dict(components)
        adjusted["contrast_bonus"] = _contrast_bonus(source_signature, destination_signature)
        adjusted.setdefault("reuse_penalty", 1.0)
        adjusted.setdefault("silence_penalty", 1.0)
        adjusted.setdefault("review_feedback", 0.5)
        adjusted = self._apply_candidate_modifiers(adjusted, source_signature, destination_signature)
        total_weight = sum(max(0.0, self.weights.get(key, 0.0)) for key in adjusted) or 1.0
        score = sum(adjusted[key] * max(0.0, self.weights.get(key, 0.0)) for key in adjusted) / total_weight
        return {
            "score": round(max(0.0, min(1.0, score)), 4),
            "components": {key: round(value, 4) for key, value in adjusted.items()},
            "weights": {key: round(self.weights.get(key, 0.0), 4) for key in adjusted},
            "explanation": self.explain(adjusted),
        }

    def _apply_candidate_modifiers(
        self,
        components: dict[str, float],
        source_signature: dict[str, Any],
        destination_signature: dict[str, Any],
    ) -> dict[str, float]:
        source_density = _float(source_signature.get("dialogue_density"), 0.0)
        source_energy = _float(source_signature.get("estimated_energy"), 0.0)
        source_pause = _float(source_signature.get("average_pause_duration"), 0.0)
        source_continuity = _float(source_signature.get("speech_continuity"), 0.0)
        source_wps = _float(source_signature.get("words_per_second"), 0.0)
        source_silence = _float(source_signature.get("silence_ratio"), 0.0)
        destination_density = _float(destination_signature.get("dialogue_density"), 0.0)
        destination_response = _float(destination_signature.get("response_delay"), 0.0)
        if self.id == "dense_comedy":
            components["dialogue_density"] = max(components.get("dialogue_density", 0.0), min(1.0, source_density))
            components["energy"] = max(components.get("energy", 0.0), min(1.0, source_energy + 0.1))
            components["speech_continuity"] = max(components.get("speech_continuity", 0.0), min(1.0, source_continuity + 0.1))
            components["words_per_second"] = max(components.get("words_per_second", 0.0), min(1.0, source_wps / 3.5))
            components["silence_penalty"] = max(0.0, 1.0 - source_pause / 2.0)
        elif self.id == "deadpan":
            pause_preference = min(1.0, source_pause / 1.5)
            low_density = 1.0 - min(1.0, source_density)
            components["pause"] = max(components.get("pause", 0.0), pause_preference)
            components["response_delay"] = max(components.get("response_delay", 0.0), _ratio(source_pause + 0.25, destination_response + 0.25))
            components["silence_ratio"] = max(components.get("silence_ratio", 0.0), min(1.0, source_silence + 0.2))
            components["dialogue_density"] = max(components.get("dialogue_density", 0.0), low_density)
            components["energy"] = max(components.get("energy", 0.0), 1.0 - source_energy)
        elif self.id == "contrast":
            components["contrast_bonus"] = max(components.get("contrast_bonus", 0.0), _contrast_bonus(source_signature, destination_signature))
        elif self.id == "minimal_reuse":
            components["reuse_penalty"] = 1.0
        elif self.id == "chaos":
            components["contrast_bonus"] = max(components.get("contrast_bonus", 0.0), 0.75)
            components["reuse_penalty"] = 1.0
            components["silence_penalty"] = 1.0
        elif self.id == "volatile":
            source_interruptions = 1.0 if bool(source_signature.get("interruptions_detected")) else 0.0
            source_turn_count = _float(source_signature.get("turn_count"), 0.0)
            components["energy"] = max(components.get("energy", 0.0), source_energy)
            components["dialogue_density"] = max(components.get("dialogue_density", 0.0), source_density)
            components["interruptions"] = max(components.get("interruptions", 0.0), source_interruptions)
            components["speaker_pattern"] = max(components.get("speaker_pattern", 0.0), min(1.0, source_wps / 3.5))
            components["turn_count"] = max(components.get("turn_count", 0.0), min(1.0, source_turn_count / 7.0))
        elif self.id == "rhythm":
            components["dialogue_density"] = 1.0 - min(1.0, abs(source_density - destination_density))
            components["response_delay"] = max(components.get("response_delay", 0.0), _ratio(source_pause + 0.25, destination_response + 0.25))
        return components

    def explain(self, components: dict[str, float]) -> str:
        strongest = sorted(components.items(), key=lambda item: item[1] * self.weights.get(item[0], 0.0), reverse=True)[:3]
        weakest = sorted(components.items(), key=lambda item: item[1])[:2]
        return f"{self.display_name}: favored " + ", ".join(key for key, _value in strongest) + "; weakest " + ", ".join(key for key, _value in weakest)


BASELINE_WEIGHTS = {
    "duration": 0.16,
    "speaker_count": 0.08,
    "turn_count": 0.11,
    "average_turn_duration": 0.09,
    "pause": 0.07,
    "dialogue_density": 0.09,
    "energy": 0.07,
    "shot_rate": 0.035,
    "conversation_type": 0.055,
    "performance_type": 0.04,
    "speaker_pattern": 0.07,
    "speech_continuity": 0.025,
    "response_delay": 0.035,
    "silence_ratio": 0.025,
    "words_per_second": 0.025,
    "interruptions": 0.025,
    "reuse_penalty": 0.0,
    "contrast_bonus": 0.0,
    "silence_penalty": 0.0,
    "review_feedback": 0.0,
}

FILTERS: dict[str, CinematicFilter] = {
    "balanced": CinematicFilter(
        id="balanced",
        display_name="Balanced",
        description="General-purpose dramatic fit with moderate pacing and repetition discipline.",
        weights={**BASELINE_WEIGHTS, "reuse_penalty": 0.04, "contrast_bonus": 0.02, "review_feedback": 0.04},
        notes="Balanced preserves broad performance shape while avoiding unnecessary repetition.",
    ),
    "rhythm": CinematicFilter(
        id="rhythm",
        display_name="Rhythm",
        description="Prioritize timing, turn duration, pause rhythm, and dialogue density.",
        weights={**BASELINE_WEIGHTS, "duration": 0.25, "average_turn_duration": 0.18, "pause": 0.16, "dialogue_density": 0.14, "speaker_pattern": 0.11},
        notes="Rhythm favors candidates whose dramatic timing resembles the destination performance.",
    ),
    "dense_comedy": CinematicFilter(
        id="dense_comedy",
        display_name="Dense Comedy",
        description="Favor energetic, dense, fast-moving replacements with less dead air.",
        weights={**BASELINE_WEIGHTS, "dialogue_density": 0.24, "energy": 0.2, "turn_count": 0.16, "silence_penalty": 0.12, "reuse_penalty": 0.01},
        notes="Dense Comedy prefers energetic performances and tolerates mild repetition when pacing improves.",
    ),
    "deadpan": CinematicFilter(
        id="deadpan",
        display_name="Deadpan",
        description="Favor slower, drier, pause-heavy replacements.",
        weights={**BASELINE_WEIGHTS, "pause": 0.24, "dialogue_density": 0.2, "energy": 0.16, "duration": 0.12, "silence_penalty": 0.02},
        notes="Deadpan prefers underplayed pacing, lower density, and awkward pauses.",
    ),
    "contrast": CinematicFilter(
        id="contrast",
        display_name="Contrast",
        description="Prefer structurally safe candidates that oppose the destination's energy or density.",
        weights={**BASELINE_WEIGHTS, "contrast_bonus": 0.28, "duration": 0.15, "speaker_pattern": 0.11, "dialogue_density": 0.04, "energy": 0.04},
        notes="Contrast rewards surprising opposition while preserving basic structure.",
    ),
    "minimal_reuse": CinematicFilter(
        id="minimal_reuse",
        display_name="Minimal Reuse",
        description="Avoid repeated source material; prefer restraint over excessive reuse.",
        weights={**BASELINE_WEIGHTS, "reuse_penalty": 0.28, "duration": 0.18, "speaker_pattern": 0.12, "contrast_bonus": 0.0},
        notes="Minimal Reuse strongly protects source variety.",
    ),
    "chaos": CinematicFilter(
        id="chaos",
        display_name="Chaos",
        description="Encourage unusual choices while retaining technical safety and whole-line preservation.",
        weights={**BASELINE_WEIGHTS, "contrast_bonus": 0.24, "energy": 0.13, "dialogue_density": 0.12, "duration": 0.08, "speaker_pattern": 0.04, "reuse_penalty": 0.0},
        notes="Chaos relaxes taste penalties and rewards destabilizing contrast.",
    ),
    "volatile": CinematicFilter(
        id="volatile",
        display_name="Volatile",
        description="Favor high energy, rapid alternation, interruptions, and dense dialogue.",
        weights={**BASELINE_WEIGHTS, "energy": 0.24, "dialogue_density": 0.2, "interruptions": 0.18, "speaker_pattern": 0.16, "turn_count": 0.14, "pause": 0.02},
        notes="Volatile prefers compressed, interruptive exchanges without relaxing structural safety.",
    ),
    "structural": CinematicFilter(
        id="structural",
        display_name="Structural",
        description="Prioritize participant structure, turn sequence, scene category, and duration.",
        weights={**BASELINE_WEIGHTS, "speaker_count": 0.24, "turn_count": 0.22, "speaker_pattern": 0.2, "conversation_type": 0.16, "performance_type": 0.16, "duration": 0.2, "energy": 0.01},
        notes="Structural gives the destination performance's participant and turn geometry first priority.",
    ),
}

FILTER_CHOICES = tuple(FILTERS.keys())
FILTER_DISPLAY_NAMES = {key: value.display_name for key, value in FILTERS.items()}


def get_filter(filter_id: str | None) -> CinematicFilter:
    key = filter_id or "balanced"
    if key not in FILTERS:
        raise ValueError(f"Unknown cinematic filter: {key}")
    return FILTERS[key]


def _contrast_bonus(source_signature: dict[str, Any], destination_signature: dict[str, Any]) -> float:
    source_density = _float(source_signature.get("dialogue_density"), 0.0)
    destination_density = _float(destination_signature.get("dialogue_density"), 0.0)
    source_energy = _float(source_signature.get("estimated_energy"), 0.0)
    destination_energy = _float(destination_signature.get("estimated_energy"), 0.0)
    density_contrast = abs(source_density - destination_density)
    energy_contrast = abs(source_energy - destination_energy)
    return max(0.0, min(1.0, density_contrast * 0.55 + energy_contrast * 0.45))


def _ratio(left: float, right: float) -> float:
    left = max(0.001, float(left))
    right = max(0.001, float(right))
    return max(0.0, min(1.0, 1.0 - abs(left - right) / max(left, right)))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
