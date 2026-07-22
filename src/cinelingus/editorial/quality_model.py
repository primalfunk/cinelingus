from __future__ import annotations

from typing import Any


DEFAULT_QUALITY_WEIGHTS = {
    "render_completeness": 0.22,
    "sentence_integrity": 0.13,
    "word_coverage": 0.15,
    "compatibility": 0.12,
    "performance_fit": 0.10,
    "speaker_confidence": 0.07,
    "visual_confidence": 0.06,
    "editing_continuity": 0.05,
    "reuse_integrity": 0.04,
    "timing_integrity": 0.04,
    "residue_integrity": 0.02,
}


def placement_quality(
    *,
    mapping: dict[str, Any],
    verification: dict[str, Any] | None,
    residue_failed: bool = False,
    weights: dict[str, float] | None = None,
    max_time_stretch: float = 0.1,
) -> dict[str, Any]:
    """Combine existing, inspectable evidence; unavailable evidence is omitted."""
    evidence = dict(verification or {})
    values: dict[str, float | None] = {
        "render_completeness": _coverage(evidence),
        "sentence_integrity": _sentence_integrity(evidence),
        "word_coverage": _coverage(evidence),
        "compatibility": _number(mapping.get("cinematic_compatibility_score")),
        "performance_fit": _number(mapping.get("performance_similarity_score")),
        "speaker_confidence": _speaker_confidence(mapping),
        "visual_confidence": _nested(mapping, "cinematic_compatibility_categories", "visual"),
        "editing_continuity": _nested(mapping, "cinematic_compatibility_categories", "editing"),
        "reuse_integrity": _reuse_integrity(mapping),
        "timing_integrity": _timing_integrity(mapping, max_time_stretch=max_time_stretch),
        "residue_integrity": 0.0 if residue_failed else 1.0,
    }
    configured = dict(DEFAULT_QUALITY_WEIGHTS)
    configured.update(weights or {})
    available = {key: _clamp(value) for key, value in values.items() if value is not None}
    active_weights = {key: max(0.0, float(configured.get(key, 0.0))) for key in available}
    denominator = sum(active_weights.values())
    score = (
        sum(available[key] * active_weights[key] for key in available) / denominator
        if denominator > 0 else 0.0
    )
    return {
        "score": round(score, 4),
        "contributors": {key: round(value, 4) for key, value in available.items()},
        "weights": {key: round(active_weights[key] / denominator, 4) for key in available} if denominator else {},
        "unavailable_contributors": sorted(set(configured) - set(available)),
        "model": "normalized_existing_evidence_v1",
    }


def _coverage(row: dict[str, Any]) -> float | None:
    value = _number(row.get("word_coverage_percentage"))
    return value / 100.0 if value is not None else None


def _sentence_integrity(row: dict[str, Any]) -> float | None:
    if not row:
        return None
    missing = int(bool(row.get("missing_sentence_beginning"))) + int(bool(row.get("missing_sentence_ending")))
    return 0.0 if row.get("mid_word_cut") else 1.0 - missing * 0.5


def _speaker_confidence(mapping: dict[str, Any]) -> float | None:
    signature = mapping.get("destination_performance_signature") or {}
    explicit = _number(signature.get("speaker_confidence"))
    if explicit is not None:
        return explicit
    if mapping.get("speaker_match_preserved") is True:
        return 1.0
    if mapping.get("speaker_match_preserved") is False:
        return 0.35
    return None


def _reuse_integrity(mapping: dict[str, Any]) -> float | None:
    components = mapping.get("performance_similarity_components") or {}
    value = _number(components.get("reuse_penalty"))
    if value is not None:
        return value
    count = _number(mapping.get("source_reuse_count"))
    return None if count is None else 1.0 / max(1.0, count)


def _timing_integrity(mapping: dict[str, Any], *, max_time_stretch: float) -> float | None:
    factor = _number(mapping.get("stretch_factor"))
    if factor is None:
        return None
    allowance = max(0.001, float(max_time_stretch))
    return 1.0 - min(1.0, abs(factor - 1.0) / allowance)


def _nested(row: dict[str, Any], parent: str, child: str) -> float | None:
    value = row.get(parent)
    return _number(value.get(child)) if isinstance(value, dict) else None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float | None) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))
