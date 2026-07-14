from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json

DEFAULT_TASTE_PROFILE = {
    "preferred_similarity": 0.72,
    "preferred_dialogue_density": 0.62,
    "preferred_pause_ratio": 0.22,
    "preferred_reuse_level": 0.15,
    "preferred_performance_length": 8.0,
    "preferred_conversation_types": ["dialogue_exchange", "exchange", "rapid_exchange", "argument"],
    "preferred_energy_balance": 0.58,
    "awkwardness_tolerance": 0.45,
    "contrast_preference": 0.35,
    "surprise_preference": 0.42,
}

POSITIVE_REVIEW_LABELS = {
    "excellent",
    "unexpectedly_convincing",
    "very_funny",
    "beautifully_awkward",
    "great_timing",
    "good_speaker_match",
    "voice_mismatch_funny",
    "good",
}
NEGATIVE_REVIEW_LABELS = {
    "poor_match",
    "awkward_pause",
    "wrong_rhythm",
    "repeated_line",
    "needs_better_fit",
    "wrong_speaker",
    "speaker_unclear",
    "voice_mismatch_distracting",
    "bad_duration_fit",
    "line_too_long",
    "performance_mismatch",
    "bad_shot_crossing",
    "wrong_energy_pacing",
    "disable",
}


def default_taste_profile(*, output_path: Path | None = None) -> dict[str, Any]:
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "profile_name": "default_absurd_but_watchable",
        "parameters": dict(DEFAULT_TASTE_PROFILE),
        "notes": "The scheduler decides whether something can happen. The Taste Engine decides whether it is worth happening.",
    }
    if output_path is not None:
        write_json(output_path, artifact)
    return artifact


def build_editorial_highlights(
    *,
    schedule: dict[str, Any],
    performance_diagnostics: dict[str, Any] | None = None,
    taste_profile: dict[str, Any] | None = None,
    output_path: Path,
) -> dict[str, Any]:
    profile = taste_profile or default_taste_profile()
    rows = _editorial_rows(schedule=schedule, diagnostics=performance_diagnostics or {}, profile=profile)
    highlights = {
        "most_convincing": _top(rows, "believability", reverse=True),
        "funniest": _top(rows, "comedic_potential", reverse=True),
        "most_awkward": _top(rows, "awkwardness", reverse=True),
        "most_improved_matches": _top(rows, "improvement_potential", reverse=True),
        "needs_attention": _top(rows, "review_priority", reverse=True),
    }
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": schedule.get("media_hash", ""),
        "creation_timestamp": utc_now(),
        "taste_profile": profile,
        "summary": {
            "evaluated_performances": len(rows),
            "average_editorial_score": _average([row.get("editorial_score") for row in rows]),
            "needs_review_count": sum(1 for row in rows if row.get("editorial_label") == "Needs Review"),
            "positive_highlight_count": len(highlights["most_convincing"]) + len(highlights["funniest"]),
        },
        "highlights": highlights,
        "performances": sorted(rows, key=lambda row: (-float(row.get("editorial_score", 0.0)), str(row.get("performance_id", "")))),
    }
    write_json(output_path, artifact)
    return artifact


def _editorial_rows(*, schedule: dict[str, Any], diagnostics: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostic_by_id = {str(row.get("destination_performance_id")): row for row in diagnostics.get("diagnostics", [])}
    mappings_by_destination: dict[str, list[dict[str, Any]]] = {}
    for mapping in schedule.get("mappings", []):
        if not mapping.get("enabled", True):
            continue
        destination_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id"))
        mappings_by_destination.setdefault(destination_id, []).append(mapping)

    fill_by_id = {str(row.get("destination_performance_id")): row for row in schedule.get("destination_performance_fills", [])}
    destination_ids = sorted(set(mappings_by_destination) | set(fill_by_id), key=_natural_key)
    rows = []
    for destination_id in destination_ids:
        mappings = mappings_by_destination.get(destination_id, [])
        fill = fill_by_id.get(destination_id, {})
        diagnostic = diagnostic_by_id.get(destination_id, {})
        components = _score_components(mappings=mappings, fill=fill, diagnostic=diagnostic, profile=profile)
        editorial_score = _weighted_editorial_score(components)
        label = _editorial_label(components, editorial_score)
        rows.append(
            {
                "performance_id": destination_id,
                "source_performance_ids": sorted({str(mapping.get("source_performance_id")) for mapping in mappings if mapping.get("source_performance_id")}),
                "start": fill.get("start", _first(mappings, "destination_timestamp")),
                "duration": fill.get("duration", _sum(mappings, "planned_render_duration")),
                "coverage": _float(fill.get("coverage"), 0.0),
                "scheduler_score": _average([mapping.get("score") for mapping in mappings]),
                "performance_similarity_score": _average([mapping.get("performance_similarity_score") for mapping in mappings]),
                "editorial_score": round(editorial_score, 4),
                "editorial_label": label,
                "highlight_status": _highlight_status(label, components),
                "review_status": _review_status(mappings),
                "final_selection_reason": _selection_reason(mappings, components, label),
                "components": components,
                "review_labels": sorted({str(mapping.get("review_label")) for mapping in mappings if mapping.get("review_label")}),
                "warnings": diagnostic.get("warnings", []),
                "mapping_indices": [index for index, mapping in enumerate(schedule.get("mappings", [])) if mapping in mappings],
            }
        )
    return rows


def _score_components(*, mappings: list[dict[str, Any]], fill: dict[str, Any], diagnostic: dict[str, Any], profile: dict[str, Any]) -> dict[str, float]:
    params = profile.get("parameters", profile)
    similarity = _float(diagnostic.get("average_similarity_score"), _average([m.get("performance_similarity_score") for m in mappings]) or 0.0)
    scheduler = _average([m.get("score") for m in mappings]) or 0.0
    coverage = _float(fill.get("coverage"), 0.0)
    density = _average([m.get("performance_dialogue_density") for m in mappings]) or _float(params.get("preferred_dialogue_density"), 0.62)
    density_fit = 1.0 - min(1.0, abs(density - _float(params.get("preferred_dialogue_density"), 0.62)))
    stretch_delta = _float(diagnostic.get("highest_stretch_delta"), max([abs(_float(m.get("stretch_factor"), 1.0) - 1.0) for m in mappings], default=0.0))
    reuse_count = int(diagnostic.get("reuse_count", 0) or 0)
    reuse_penalty = min(1.0, reuse_count / max(1.0, len(mappings)))
    contrast = _average([m.get("performance_similarity_components", {}).get("contrast_bonus") for m in mappings]) or 0.0
    pause_mismatch = 1.0 - (_average([m.get("performance_similarity_components", {}).get("pause") for m in mappings]) or 1.0)
    review_boost, review_penalty = _review_adjustments(mappings)
    believability = _clamp(similarity * 0.42 + scheduler * 0.24 + coverage * 0.2 + density_fit * 0.14 - stretch_delta * 0.25 - review_penalty * 0.2)
    rhythmic_integrity = _clamp((1.0 - pause_mismatch) * 0.42 + similarity * 0.32 + coverage * 0.18 - stretch_delta * 0.18)
    continuity = _clamp(coverage * 0.45 + scheduler * 0.24 + similarity * 0.2 - reuse_penalty * 0.16)
    novelty = _clamp((contrast * 0.45) + _float(params.get("surprise_preference"), 0.42) * 0.3 + (1.0 - reuse_penalty) * 0.25)
    awkwardness = _clamp(pause_mismatch * 0.35 + stretch_delta * 0.25 + contrast * 0.25 + _float(params.get("awkwardness_tolerance"), 0.45) * 0.15)
    comedic_potential = _clamp(novelty * 0.35 + awkwardness * 0.28 + density_fit * 0.17 + contrast * 0.2 + review_boost * 0.15)
    interestingness = _clamp(novelty * 0.34 + comedic_potential * 0.26 + believability * 0.18 + awkwardness * 0.16 + review_boost * 0.18)
    review_priority = _clamp((1.0 - believability) * 0.35 + stretch_delta * 0.25 + reuse_penalty * 0.18 + (1.0 - coverage) * 0.22 + review_penalty * 0.25)
    improvement_potential = _clamp((similarity - scheduler) * 0.5 + interestingness * 0.3 + (1.0 - review_priority) * 0.2)
    return {
        "believability": round(believability, 4),
        "interestingness": round(interestingness, 4),
        "rhythmic_integrity": round(rhythmic_integrity, 4),
        "continuity": round(continuity, 4),
        "novelty": round(novelty, 4),
        "awkwardness": round(awkwardness, 4),
        "comedic_potential": round(comedic_potential, 4),
        "review_priority": round(review_priority, 4),
        "improvement_potential": round(improvement_potential, 4),
    }


def _weighted_editorial_score(components: dict[str, float]) -> float:
    return _clamp(
        components["believability"] * 0.22
        + components["interestingness"] * 0.2
        + components["rhythmic_integrity"] * 0.18
        + components["continuity"] * 0.16
        + components["novelty"] * 0.14
        + (1.0 - components["review_priority"]) * 0.1
    )


def _editorial_label(components: dict[str, float], editorial_score: float) -> str:
    if components["review_priority"] >= 0.72:
        return "Needs Review"
    if components["believability"] >= 0.82 and components["novelty"] >= 0.55:
        return "Surprisingly Convincing"
    if components["believability"] >= 0.78:
        return "Convincing"
    if components["comedic_potential"] >= 0.72:
        return "Comedic"
    if components["awkwardness"] >= 0.72 and components["interestingness"] >= 0.58:
        return "Beautifully Awkward"
    if editorial_score < 0.42:
        return "Distracting"
    if components["continuity"] < 0.42:
        return "Confusing"
    return "Neutral"


def _highlight_status(label: str, components: dict[str, float]) -> list[str]:
    statuses = []
    if label in {"Convincing", "Surprisingly Convincing"}:
        statuses.append("most_convincing")
    if label == "Comedic" or components.get("comedic_potential", 0.0) >= 0.7:
        statuses.append("funniest")
    if label == "Beautifully Awkward" or components.get("awkwardness", 0.0) >= 0.72:
        statuses.append("most_awkward")
    if label in {"Needs Review", "Distracting", "Confusing"}:
        statuses.append("needs_attention")
    return statuses or ["general"]


def _review_status(mappings: list[dict[str, Any]]) -> str:
    labels = {str(mapping.get("review_label")) for mapping in mappings if mapping.get("review_label")}
    if labels & POSITIVE_REVIEW_LABELS:
        return "positively_reviewed"
    if labels & NEGATIVE_REVIEW_LABELS:
        return "needs_revision"
    return "unreviewed"


def _selection_reason(mappings: list[dict[str, Any]], components: dict[str, float], label: str) -> str:
    reasons = [str(mapping.get("selection_reason")) for mapping in mappings if mapping.get("selection_reason")]
    base = reasons[0] if reasons else "scheduled_candidate"
    strongest = max(components.items(), key=lambda item: item[1])[0]
    return f"{base}; editorial label {label}; strongest component {strongest}"


def _review_adjustments(mappings: list[dict[str, Any]]) -> tuple[float, float]:
    labels = {str(mapping.get("review_label")) for mapping in mappings if mapping.get("review_label")}
    positive = len(labels & POSITIVE_REVIEW_LABELS)
    negative = len(labels & NEGATIVE_REVIEW_LABELS)
    return min(1.0, positive / 2.0), min(1.0, negative / 2.0)


def _top(rows: list[dict[str, Any]], component: str, *, reverse: bool) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: (float(row.get("components", {}).get(component, 0.0)), float(row.get("editorial_score", 0.0))), reverse=reverse)[:10]
    return [_highlight_summary(row, component) for row in selected]


def _highlight_summary(row: dict[str, Any], component: str) -> dict[str, Any]:
    return {
        "performance_id": row.get("performance_id"),
        "start": row.get("start"),
        "duration": row.get("duration"),
        "editorial_score": row.get("editorial_score"),
        "editorial_label": row.get("editorial_label"),
        "component": component,
        "component_score": row.get("components", {}).get(component),
        "review_status": row.get("review_status"),
        "mapping_indices": row.get("mapping_indices", []),
    }


def _average(values: list[Any]) -> float | None:
    numeric = [_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def _first(mappings: list[dict[str, Any]], key: str) -> Any:
    for mapping in mappings:
        value = mapping.get(key)
        if value not in {None, ""}:
            return value
    return None


def _sum(mappings: list[dict[str, Any]], key: str) -> float:
    return round(sum(_float(mapping.get(key), 0.0) for mapping in mappings), 3)


def _float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float | int | None) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _natural_key(value: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return (prefix, int(digits or 0))
