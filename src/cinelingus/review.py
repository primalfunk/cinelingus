from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json

FILTER_ALL = "All mappings"
FILTER_ENABLED = "Enabled only"
FILTER_DISABLED = "Disabled only"
FILTER_CROSS_SHOT = "Cross-shot mappings"
FILTER_LOW_SCORE = "Low score"
FILTER_LOW_VISUAL_FIT = "Low visual fit"
FILTER_RISKY = "Risky mappings"
FILTER_REVIEWED = "Reviewed mappings"
FILTER_UNREVIEWED = "Unreviewed mappings"

REVIEW_LABEL_UNREVIEWED = "unreviewed"
REVIEW_LABEL_GOOD = "good"
REVIEW_LABEL_EXCELLENT = "excellent"
REVIEW_LABEL_UNEXPECTEDLY_CONVINCING = "unexpectedly_convincing"
REVIEW_LABEL_VERY_FUNNY = "very_funny"
REVIEW_LABEL_BEAUTIFULLY_AWKWARD = "beautifully_awkward"
REVIEW_LABEL_GREAT_TIMING = "great_timing"
REVIEW_LABEL_POOR_MATCH = "poor_match"
REVIEW_LABEL_WRONG_RHYTHM = "wrong_rhythm"
REVIEW_LABEL_REPEATED_LINE = "repeated_line"
REVIEW_LABEL_NEEDS_BETTER_FIT = "needs_better_fit"
REVIEW_LABEL_WRONG_SPEAKER = "wrong_speaker"
REVIEW_LABEL_GOOD_SPEAKER_MATCH = "good_speaker_match"
REVIEW_LABEL_SPEAKER_UNCLEAR = "speaker_unclear"
REVIEW_LABEL_VOICE_MISMATCH_FUNNY = "voice_mismatch_funny"
REVIEW_LABEL_VOICE_MISMATCH_DISTRACTING = "voice_mismatch_distracting"
REVIEW_LABEL_TOO_EARLY = "too_early"
REVIEW_LABEL_TOO_LATE = "too_late"
REVIEW_LABEL_BAD_DURATION = "bad_duration_fit"
REVIEW_LABEL_LINE_TOO_LONG = "line_too_long"
REVIEW_LABEL_AWKWARD_PAUSE = "awkward_pause"
REVIEW_LABEL_PERFORMANCE_MISMATCH = "performance_mismatch"
REVIEW_LABEL_BAD_SHOT = "bad_shot_crossing"
REVIEW_LABEL_WRONG_ENERGY = "wrong_energy_pacing"
REVIEW_LABEL_DISABLE = "disable"

REVIEW_LABELS = (
    REVIEW_LABEL_UNREVIEWED,
    REVIEW_LABEL_GOOD,
    REVIEW_LABEL_EXCELLENT,
    REVIEW_LABEL_UNEXPECTEDLY_CONVINCING,
    REVIEW_LABEL_VERY_FUNNY,
    REVIEW_LABEL_BEAUTIFULLY_AWKWARD,
    REVIEW_LABEL_GREAT_TIMING,
    REVIEW_LABEL_POOR_MATCH,
    REVIEW_LABEL_WRONG_RHYTHM,
    REVIEW_LABEL_REPEATED_LINE,
    REVIEW_LABEL_NEEDS_BETTER_FIT,
    REVIEW_LABEL_WRONG_SPEAKER,
    REVIEW_LABEL_GOOD_SPEAKER_MATCH,
    REVIEW_LABEL_SPEAKER_UNCLEAR,
    REVIEW_LABEL_VOICE_MISMATCH_FUNNY,
    REVIEW_LABEL_VOICE_MISMATCH_DISTRACTING,
    REVIEW_LABEL_TOO_EARLY,
    REVIEW_LABEL_TOO_LATE,
    REVIEW_LABEL_BAD_DURATION,
    REVIEW_LABEL_LINE_TOO_LONG,
    REVIEW_LABEL_AWKWARD_PAUSE,
    REVIEW_LABEL_PERFORMANCE_MISMATCH,
    REVIEW_LABEL_BAD_SHOT,
    REVIEW_LABEL_WRONG_ENERGY,
    REVIEW_LABEL_DISABLE,
)

REVIEW_FILTERS = (
    FILTER_ALL,
    FILTER_RISKY,
    FILTER_CROSS_SHOT,
    FILTER_LOW_VISUAL_FIT,
    FILTER_LOW_SCORE,
    FILTER_DISABLED,
    FILTER_REVIEWED,
    FILTER_UNREVIEWED,
    FILTER_ENABLED,
)


def mapping_matches_filter(mapping: dict[str, Any], filter_name: str) -> bool:
    if filter_name == FILTER_ALL:
        return True
    if filter_name == FILTER_ENABLED:
        return bool(mapping.get("enabled", True))
    if filter_name == FILTER_DISABLED:
        return not bool(mapping.get("enabled", True))
    if filter_name == FILTER_CROSS_SHOT:
        return bool(mapping.get("mapping_crosses_shot_boundary"))
    if filter_name == FILTER_LOW_SCORE:
        return _float(mapping.get("score"), 1.0) < 0.55
    if filter_name == FILTER_LOW_VISUAL_FIT:
        return _float(mapping.get("visual_fit_score"), 1.0) < 0.75
    if filter_name == FILTER_REVIEWED:
        return mapping.get("review_label", REVIEW_LABEL_UNREVIEWED) != REVIEW_LABEL_UNREVIEWED
    if filter_name == FILTER_UNREVIEWED:
        return mapping.get("review_label", REVIEW_LABEL_UNREVIEWED) == REVIEW_LABEL_UNREVIEWED
    if filter_name == FILTER_RISKY:
        return (
            bool(mapping.get("mapping_crosses_shot_boundary"))
            or _float(mapping.get("visual_fit_score"), 1.0) < 0.75
            or _float(mapping.get("score"), 1.0) < 0.55
            or not bool(mapping.get("enabled", True))
            or mapping.get("review_label") in {REVIEW_LABEL_POOR_MATCH, REVIEW_LABEL_WRONG_RHYTHM, REVIEW_LABEL_REPEATED_LINE, REVIEW_LABEL_NEEDS_BETTER_FIT, REVIEW_LABEL_WRONG_SPEAKER, REVIEW_LABEL_VOICE_MISMATCH_DISTRACTING, REVIEW_LABEL_BAD_DURATION, REVIEW_LABEL_LINE_TOO_LONG, REVIEW_LABEL_AWKWARD_PAUSE, REVIEW_LABEL_PERFORMANCE_MISMATCH, REVIEW_LABEL_BAD_SHOT, REVIEW_LABEL_TOO_EARLY, REVIEW_LABEL_TOO_LATE, REVIEW_LABEL_WRONG_ENERGY, REVIEW_LABEL_DISABLE}
        )
    return True


def filtered_mapping_indices(mappings: list[dict[str, Any]], filter_name: str) -> list[int]:
    return [index for index, mapping in enumerate(mappings) if mapping_matches_filter(mapping, filter_name)]


def review_summary(mappings: list[dict[str, Any]], visible_count: int | None = None) -> str:
    total = len(mappings)
    enabled = sum(1 for mapping in mappings if mapping.get("enabled", True))
    cross = sum(1 for mapping in mappings if mapping.get("mapping_crosses_shot_boundary"))
    low_visual = sum(1 for mapping in mappings if _float(mapping.get("visual_fit_score"), 1.0) < 0.75)
    visible = total if visible_count is None else visible_count
    reviewed = sum(1 for mapping in mappings if mapping.get("review_label", REVIEW_LABEL_UNREVIEWED) != REVIEW_LABEL_UNREVIEWED)
    return f"{visible} shown / {enabled} enabled / {total} total / {reviewed} reviewed / {cross} cross-shot / {low_visual} low visual fit"


def review_row_values(mapping: dict[str, Any]) -> tuple[Any, ...]:
    return (
        "yes" if mapping.get("enabled", True) else "no",
        mapping.get("window_id", ""),
        mapping.get("clip_id", ""),
        _fmt(mapping.get("destination_timestamp")),
        _fmt(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", ""))),
        _fmt(mapping.get("score")),
        mapping.get("shot_id") or mapping.get("primary_shot_id") or "",
        _fmt(mapping.get("visual_fit_score")),
        "yes" if mapping.get("mapping_crosses_shot_boundary") else "no",
        _fmt(mapping.get("boundary_overrun_seconds")),
        mapping.get("timing_strategy", ""),
        mapping.get("source_speaker_id", ""),
        mapping.get("destination_speaker_id", ""),
        _speaker_match_text(mapping),
        mapping.get("speaker_fallback_reason") or "",
        mapping.get("review_label", REVIEW_LABEL_UNREVIEWED),
        mapping.get("source_transcript", ""),
    )


PERFORMANCE_FILTER_ALL = "All performances"
PERFORMANCE_FILTER_RISKY = "Risky performances"
PERFORMANCE_FILTER_LOW_COVERAGE = "Low coverage"
PERFORMANCE_FILTER_REUSED = "Has reuse"
PERFORMANCE_FILTER_REVIEWED = "Reviewed performances"
PERFORMANCE_FILTER_UNREVIEWED = "Unreviewed performances"

PERFORMANCE_REVIEW_FILTERS = (
    PERFORMANCE_FILTER_ALL,
    PERFORMANCE_FILTER_RISKY,
    PERFORMANCE_FILTER_LOW_COVERAGE,
    PERFORMANCE_FILTER_REUSED,
    PERFORMANCE_FILTER_REVIEWED,
    PERFORMANCE_FILTER_UNREVIEWED,
)


def build_performance_review_rows(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    mappings = schedule.get("mappings", [])
    mappings_by_performance: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, mapping in enumerate(mappings):
        performance_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id") or "unknown")
        mappings_by_performance.setdefault(performance_id, []).append((index, mapping))

    fill_by_id = {
        str(row.get("destination_performance_id")): row
        for row in schedule.get("destination_performance_fills", [])
    }
    ordered_ids = list(fill_by_id)
    for performance_id in mappings_by_performance:
        if performance_id not in fill_by_id:
            ordered_ids.append(performance_id)

    rows = []
    for performance_id in ordered_ids:
        grouped = mappings_by_performance.get(performance_id, [])
        fill = fill_by_id.get(performance_id, {})
        grouped_mappings = [mapping for _index, mapping in grouped]
        mapping_indices = [index for index, _mapping in grouped]
        scores = [_float(mapping.get("score"), 0.0) for mapping in grouped_mappings]
        labels = {
            str(mapping.get("review_label", REVIEW_LABEL_UNREVIEWED))
            for mapping in grouped_mappings
            if mapping.get("review_label", REVIEW_LABEL_UNREVIEWED) != REVIEW_LABEL_UNREVIEWED
        }
        transcripts = [str(mapping.get("source_transcript", "")).strip() for mapping in grouped_mappings if str(mapping.get("source_transcript", "")).strip()]
        reuse_count = sum(1 for mapping in grouped_mappings if mapping.get("rescue_reused_clip") or mapping.get("reuse_allowed_reason"))
        disabled_count = sum(1 for mapping in grouped_mappings if not mapping.get("enabled", True))
        reviewed_count = sum(1 for mapping in grouped_mappings if mapping.get("review_label", REVIEW_LABEL_UNREVIEWED) != REVIEW_LABEL_UNREVIEWED)
        speaker_aware = [mapping for mapping in grouped_mappings if mapping.get("source_speaker_id") or mapping.get("destination_speaker_id")]
        speaker_matches = sum(1 for mapping in speaker_aware if mapping.get("speaker_match_preserved"))
        speaker_match_rate = round(speaker_matches / len(speaker_aware), 4) if speaker_aware else None
        speaker_fallbacks = sorted({str(mapping.get("speaker_fallback_reason")) for mapping in grouped_mappings if mapping.get("speaker_fallback_reason")})
        coverage = _float(fill.get("coverage"), 0.0)
        average_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        rows.append(
            {
                "performance_id": performance_id,
                "performance_type": fill.get("destination_performance_type") or _first(grouped_mappings, "performance_type"),
                "start": fill.get("start", _first(grouped_mappings, "destination_timestamp")),
                "duration": fill.get("duration", _sum(grouped_mappings, "planned_render_duration")),
                "coverage": coverage,
                "target_coverage": fill.get("target_coverage"),
                "mapping_count": len(grouped),
                "enabled_count": len(grouped) - disabled_count,
                "disabled_count": disabled_count,
                "reviewed_count": reviewed_count,
                "reuse_count": reuse_count,
                "source_speaker_ids": sorted({str(mapping.get("source_speaker_id")) for mapping in grouped_mappings if mapping.get("source_speaker_id")}),
                "destination_speaker_ids": sorted({str(mapping.get("destination_speaker_id")) for mapping in grouped_mappings if mapping.get("destination_speaker_id")}),
                "speaker_match_rate": speaker_match_rate,
                "speaker_fallbacks": speaker_fallbacks,
                "average_score": average_score,
                "stop_reason": fill.get("stop_reason", ""),
                "source_performance_ids": fill.get("source_performance_ids") or sorted({str(mapping.get("source_performance_id")) for mapping in grouped_mappings if mapping.get("source_performance_id")}),
                "review_labels": sorted(labels),
                "mapping_indices": mapping_indices,
                "transcript_preview": " / ".join(transcripts[:4]),
                "risky": coverage < _float(fill.get("target_coverage"), 0.9) or reuse_count > 0 or disabled_count > 0 or average_score < 0.55,
            }
        )
    rows.sort(key=lambda row: _float(row.get("start"), 0.0))
    return rows


def performance_matches_filter(row: dict[str, Any], filter_name: str) -> bool:
    if filter_name == PERFORMANCE_FILTER_ALL:
        return True
    if filter_name == PERFORMANCE_FILTER_RISKY:
        return bool(row.get("risky"))
    if filter_name == PERFORMANCE_FILTER_LOW_COVERAGE:
        return _float(row.get("coverage"), 0.0) < _float(row.get("target_coverage"), 0.9)
    if filter_name == PERFORMANCE_FILTER_REUSED:
        return int(row.get("reuse_count") or 0) > 0
    if filter_name == PERFORMANCE_FILTER_REVIEWED:
        return int(row.get("reviewed_count") or 0) > 0
    if filter_name == PERFORMANCE_FILTER_UNREVIEWED:
        return int(row.get("reviewed_count") or 0) == 0
    return True


def filtered_performance_rows(schedule: dict[str, Any], filter_name: str) -> list[dict[str, Any]]:
    return [row for row in build_performance_review_rows(schedule) if performance_matches_filter(row, filter_name)]


def performance_review_row_values(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("performance_id", ""),
        row.get("performance_type", ""),
        _fmt(row.get("start")),
        _fmt(row.get("duration")),
        _fmt(row.get("coverage")),
        row.get("mapping_count", 0),
        row.get("reuse_count", 0),
        _fmt(row.get("average_score")),
        row.get("reviewed_count", 0),
        _fmt(row.get("speaker_match_rate")) if row.get("speaker_match_rate") is not None else "",
        ",".join(row.get("source_speaker_ids") or []),
        ",".join(row.get("destination_speaker_ids") or []),
        ",".join(row.get("speaker_fallbacks") or []),
        ",".join(row.get("review_labels") or []),
        row.get("stop_reason", ""),
        row.get("transcript_preview", ""),
    )


def performance_review_summary(rows: list[dict[str, Any]], visible_count: int | None = None) -> str:
    total = len(rows)
    visible = total if visible_count is None else visible_count
    risky = sum(1 for row in rows if row.get("risky"))
    reused = sum(1 for row in rows if int(row.get("reuse_count") or 0) > 0)
    reviewed = sum(1 for row in rows if int(row.get("reviewed_count") or 0) > 0)
    return f"{visible} shown / {total} performances / {reviewed} reviewed / {risky} risky / {reused} reused"


def apply_performance_review_label(schedule: dict[str, Any], performance_ids: list[str], label: str) -> None:
    if label not in REVIEW_LABELS:
        raise ValueError(f"Unknown review label: {label}")
    selected = set(performance_ids)
    for mapping in schedule.get("mappings", []):
        performance_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id") or "unknown")
        if performance_id in selected:
            mapping["review_label"] = label
            if label == REVIEW_LABEL_DISABLE:
                mapping["enabled"] = False


def performance_mapping_indices(schedule: dict[str, Any], performance_ids: list[str]) -> list[int]:
    selected = set(performance_ids)
    indices = []
    for index, mapping in enumerate(schedule.get("mappings", [])):
        performance_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id") or "unknown")
        if performance_id in selected:
            indices.append(index)
    return indices


def _speaker_match_text(mapping: dict[str, Any]) -> str:
    if not mapping.get("source_speaker_id") and not mapping.get("destination_speaker_id"):
        return ""
    return "yes" if mapping.get("speaker_match_preserved") else "no"


def _first(mappings: list[dict[str, Any]], key: str) -> Any:
    for mapping in mappings:
        value = mapping.get(key)
        if value not in {None, ""}:
            return value
    return ""


def _sum(mappings: list[dict[str, Any]], key: str) -> float:
    return round(sum(_float(mapping.get(key), 0.0) for mapping in mappings), 3)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    return value



def apply_review_label(mappings: list[dict[str, Any]], indices: list[int], label: str) -> None:
    if label not in REVIEW_LABELS:
        raise ValueError(f"Unknown review label: {label}")
    for index in indices:
        mappings[index]["review_label"] = label
        if label == REVIEW_LABEL_DISABLE:
            mappings[index]["enabled"] = False


def build_review_notes(schedule: dict[str, Any], *, schedule_path: Path | None = None) -> dict[str, Any]:
    mappings = schedule.get("mappings", [])
    notes = []
    label_counts = {label: 0 for label in REVIEW_LABELS}
    for index, mapping in enumerate(mappings):
        label = mapping.get("review_label", REVIEW_LABEL_UNREVIEWED)
        if label not in label_counts:
            label = REVIEW_LABEL_UNREVIEWED
        label_counts[label] += 1
        note_text = str(mapping.get("review_note", "")).strip()
        if label != REVIEW_LABEL_UNREVIEWED or note_text:
            notes.append(
                {
                    "mapping_index": index,
                    "window_id": mapping.get("window_id"),
                    "clip_id": mapping.get("clip_id"),
                    "performance_id": mapping.get("performance_id"),
                    "performance_type": mapping.get("performance_type"),
                    "review_label": label,
                    "review_note": note_text,
                    "enabled": bool(mapping.get("enabled", True)),
                    "destination_timestamp": mapping.get("destination_timestamp"),
                    "score": mapping.get("score"),
                    "visual_fit_score": mapping.get("visual_fit_score"),
                    "mapping_crosses_shot_boundary": bool(mapping.get("mapping_crosses_shot_boundary")),
                }
            )
    return {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": schedule.get("media_hash", ""),
        "creation_timestamp": utc_now(),
        "schedule_path": str(schedule_path) if schedule_path else "",
        "total_mappings": len(mappings),
        "reviewed_mappings": len(notes),
        "label_counts": label_counts,
        "notes": notes,
    }


def write_review_notes(schedule: dict[str, Any], output_path: Path, *, schedule_path: Path | None = None) -> dict[str, Any]:
    data = build_review_notes(schedule, schedule_path=schedule_path)
    write_json(output_path, data)
    return data
