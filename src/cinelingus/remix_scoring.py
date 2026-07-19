from __future__ import annotations

from typing import Any


SCORE_KEYS = (
    "realism_score",
    "humor_score",
    "coherence_score",
    "technical_score",
    "novelty_score",
)

PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "best_realism": {
        "realism_score": 0.38,
        "humor_score": 0.12,
        "coherence_score": 0.2,
        "technical_score": 0.24,
        "novelty_score": 0.06,
    },
    "funniest_result": {
        "realism_score": 0.16,
        "humor_score": 0.36,
        "coherence_score": 0.16,
        "technical_score": 0.18,
        "novelty_score": 0.14,
    },
    "balanced": {
        "realism_score": 0.26,
        "humor_score": 0.22,
        "coherence_score": 0.18,
        "technical_score": 0.24,
        "novelty_score": 0.1,
    },
}


def normalize_scoring_profile(profile: str | None) -> str:
    if profile in {"realism", "best_realism"}:
        return "best_realism"
    if profile in {"funniest", "funny", "funniest_result"}:
        return "funniest_result"
    return "balanced"


def build_candidate_score(
    *,
    realism: float,
    humor: float,
    coherence: float,
    technical: float,
    novelty: float,
    profile: str = "balanced",
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_id = normalize_scoring_profile(profile)
    weights = PROFILE_WEIGHTS[profile_id]
    scores = {
        "realism_score": _clamp(realism),
        "humor_score": _clamp(humor),
        "coherence_score": _clamp(coherence),
        "technical_score": _clamp(technical),
        "novelty_score": _clamp(novelty),
    }
    total_weight = sum(weights.values()) or 1.0
    combined = sum(scores[key] * weights[key] for key in SCORE_KEYS) / total_weight
    return {
        **{key: round(value, 4) for key, value in scores.items()},
        "combined_score": round(_clamp(combined), 4),
        "scoring_profile": profile_id,
        "weights": {key: round(value, 4) for key, value in weights.items()},
        "components": components or {},
    }


def score_from_candidate_fields(candidate: dict[str, Any], *, profile: str = "balanced") -> dict[str, Any]:
    risk = _float(candidate.get("technical_risk_score"), 0.0)
    breakdown = candidate.get("scoring_breakdown") or {}
    return build_candidate_score(
        realism=_float(candidate.get("estimated_realism_score"), 0.0),
        humor=_float(candidate.get("estimated_humor_novelty_score"), 0.0),
        coherence=_average(
            [
                breakdown.get("scene_continuity"),
                breakdown.get("length_compatibility"),
                breakdown.get("speech_density"),
                candidate.get("target_window_speech_coverage"),
            ]
        ),
        technical=max(0.0, 1.0 - risk),
        novelty=_float(breakdown.get("comedic_potential"), _float(candidate.get("estimated_humor_novelty_score"), 0.0)),
        profile=profile,
        components=breakdown,
    )


def _average(values: list[Any]) -> float:
    numeric = [_float(value, None) for value in values]
    rows = [value for value in numeric if value is not None]
    return sum(rows) / len(rows) if rows else 0.0


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
