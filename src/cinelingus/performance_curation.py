from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import __version__
from .util import utc_now, write_json


STRATA = (
    "coupled_performance",
    "adapted_performance",
    "multi_turn_whole_line_recovery",
    "micro_utterance",
    "suppressed_unreplaced",
    "residue_corrected",
)

REVIEW_RUBRIC = (
    "cadence_and_turn_timing",
    "dialogue_intelligibility",
    "emotional_and_semantic_fit",
    "voice_or_line_repetition",
    "ambience_continuity",
    "editorial_intentionality",
)

DEFAULT_REVIEW_CONTEXT = 0.45
MIN_REVIEW_DURATION = 2.5
MAX_REVIEW_DURATION = 8.0


def build_performance_curation_manifest(
    *,
    schedule: dict[str, Any],
    source_video: Path,
    output_path: Path,
    max_per_stratum: int = 2,
) -> dict[str, Any]:
    mappings = [dict(row, mapping_index=index) for index, row in enumerate(schedule.get("mappings", [])) if row.get("enabled", True)]
    fills = list(schedule.get("destination_performance_fills", []))
    fills_by_id = {str(row.get("destination_performance_id")): row for row in fills}
    mappings_by_destination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mapping in mappings:
        mappings_by_destination[_destination_id(mapping)].append(mapping)

    candidates: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    for decision in schedule.get("performance_decisions", []):
        tier = int(decision.get("scheduler_tier", 0) or 0)
        destination_id = str(decision.get("destination_performance_id"))
        fill = fills_by_id.get(destination_id, {})
        if tier == 1:
            candidates["coupled_performance"].append(_performance_candidate("coupled_performance", destination_id, fill, mappings_by_destination.get(destination_id, []), "tier_1_complete_performance"))
        elif tier == 2:
            candidates["adapted_performance"].append(_performance_candidate("adapted_performance", destination_id, fill, mappings_by_destination.get(destination_id, []), "tier_2_adapted_performance"))
        elif tier == 5:
            candidates["suppressed_unreplaced"].append(_performance_candidate("suppressed_unreplaced", destination_id, fill, [], "no_valid_donor_original_dialogue_suppressed"))

    for destination_id, rows in mappings_by_destination.items():
        spanning = [row for row in rows if int(row.get("scheduler_tier", 0) or 0) == 4 and row.get("alignment_spans_speech_windows")]
        if spanning:
            candidates["multi_turn_whole_line_recovery"].append(
                _performance_candidate(
                    "multi_turn_whole_line_recovery",
                    destination_id,
                    fills_by_id.get(destination_id, {}),
                    spanning,
                    "whole_line_spans_multiple_short_destination_turns",
                )
            )

    for region in schedule.get("destination_speech_regions", []):
        start, end = _bounds(region)
        if 0.0 < end - start <= 1.0:
            overlapping = [row for row in mappings if _overlap(start, end, *_bounds(row))]
            candidates["micro_utterance"].append({
                "stratum": "micro_utterance",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "destination_performance_id": _destination_id(overlapping[0]) if overlapping else None,
                "mapping_indices": [row["mapping_index"] for row in overlapping],
                "reason": "destination_speech_window_at_or_below_one_second",
                "transcript": region.get("transcript", ""),
            })

    for region in schedule.get("residue_correction_regions", []):
        start, end = _bounds(region)
        candidates["residue_corrected"].append({
            "stratum": "residue_corrected",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "destination_performance_id": region.get("destination_performance_id"),
            "mapping_indices": [],
            "reason": region.get("evidence_kind") or "post_render_residue_correction",
        })

    selected = []
    selected_keys: set[tuple[Any, ...]] = set()
    for stratum in STRATA:
        rows = _select_distributed(candidates[stratum], maximum=max(0, max_per_stratum))
        for row in rows:
            key = _selection_identity(row)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(dict(row, review_rubric=list(REVIEW_RUBRIC)))

    density = analyze_dialogue_density(schedule)
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "source_video": str(source_video),
        "strata": list(STRATA),
        "review_rubric": list(REVIEW_RUBRIC),
        "max_per_stratum": max(0, max_per_stratum),
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
        "selected_count": len(selected),
        "selected": sorted(selected, key=lambda row: (float(row.get("start", 0.0) or 0.0), str(row.get("stratum")))),
        "dialogue_density_diagnostics": density,
        "residue_curation_checkpoint": schedule.get("residue_curation_checkpoint", {}),
    }
    write_json(output_path, artifact)
    return artifact


def build_reviewed_seed_refinement_manifest(
    *,
    schedule: dict[str, Any],
    reviewed_manifest: dict[str, Any],
    positive_indices: list[int],
    source_video: Path,
    output_path: Path,
    max_per_seed: int = 3,
    context: float = DEFAULT_REVIEW_CONTEXT,
    exclude_mapping_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Turn human-approved broad samples into mapping-level follow-up clips."""
    selected_by_index = {
        int(row.get("index", index)): row
        for index, row in enumerate(reviewed_manifest.get("selected", []), start=1)
    }
    mappings = [
        dict(row, mapping_index=index)
        for index, row in enumerate(schedule.get("mappings", []))
        if row.get("enabled", True)
    ]
    seeds = []
    rejected = []
    used_clip_ids: set[str] = set()
    excluded_indices = {int(value) for value in (exclude_mapping_indices or [])}
    for review_index in positive_indices:
        review = selected_by_index.get(int(review_index))
        if review is None:
            rejected.append({"review_index": review_index, "reason": "review_index_not_found"})
            continue
        seed_start = float(review.get("preview_start", review.get("start", 0.0)) or 0.0)
        seed_end = float(review.get("preview_end", review.get("end", seed_start)) or seed_start)
        rows = []
        for mapping in mappings:
            if mapping["mapping_index"] in excluded_indices:
                continue
            start, end = _bounds(mapping)
            if not _overlap(seed_start, seed_end, start, end):
                continue
            qualification = _spoken_mapping_qualification(mapping)
            if not qualification["qualified"]:
                rejected.append({
                    "review_index": review_index,
                    "mapping_index": mapping["mapping_index"],
                    "clip_id": mapping.get("clip_id"),
                    "reason": qualification["reason"],
                })
                continue
            clip_id = str(mapping.get("clip_id") or "")
            if clip_id and clip_id in used_clip_ids:
                rejected.append({
                    "review_index": review_index,
                    "mapping_index": mapping["mapping_index"],
                    "clip_id": clip_id,
                    "reason": "repeated_source_clip",
                })
                continue
            highlight_start, highlight_end = _review_bounds(
                start=start,
                end=end,
                seed_start=seed_start,
                seed_end=seed_end,
                context=context,
            )
            rows.append({
                "review_index": review_index,
                "source_stratum": review.get("stratum"),
                "start": round(highlight_start, 3),
                "end": round(highlight_end, 3),
                "duration": round(highlight_end - highlight_start, 3),
                "mapping_index": mapping["mapping_index"],
                "mapping_indices": [mapping["mapping_index"]],
                "clip_id": mapping.get("clip_id"),
                "source_transcript": mapping.get("source_transcript", ""),
                "destination_performance_id": _destination_id(mapping),
                "sentence_complete": qualification["sentence_complete"],
                "words_per_second": qualification["words_per_second"],
                "speaker_turn_risk": qualification["speaker_turn_risk"],
                "speech_verified": True,
                "selection_reason": "human_positive_seed_spoken_mapping_refinement",
                "review_rubric": list(REVIEW_RUBRIC),
            })
        rows.sort(key=lambda row: (-_refinement_score(row), float(row["start"])))
        chosen = _nonoverlapping(rows, maximum=max(0, max_per_seed))
        for row in chosen:
            clip_id = str(row.get("clip_id") or "")
            if clip_id:
                used_clip_ids.add(clip_id)
            seeds.append(row)

    seeds.sort(key=lambda row: (int(row["review_index"]), float(row["start"])))
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "source_video": str(source_video),
        "positive_review_indices": [int(value) for value in positive_indices],
        "negative_review_indices": sorted(
            int(row.get("index", index))
            for index, row in enumerate(reviewed_manifest.get("selected", []), start=1)
            if int(row.get("index", index)) not in {int(value) for value in positive_indices}
        ),
        "selection_policy": {
            "speech_required": True,
            "complete_sentence_preferred": True,
            "source_clip_reuse_forbidden": True,
            "cross_seed_deduplication": True,
            "minimum_review_duration": MIN_REVIEW_DURATION,
            "maximum_review_duration": MAX_REVIEW_DURATION,
            "context_seconds": round(max(0.0, context), 3),
            "maximum_words_per_second": 3.4,
            "single_voice_multi_speaker_density_gate": True,
        },
        "excluded_mapping_indices": sorted(excluded_indices),
        "selected_count": len(seeds),
        "selected": seeds,
        "rejected": rejected,
    }
    write_json(output_path, artifact)
    return artifact


def analyze_dialogue_density(schedule: dict[str, Any]) -> dict[str, Any]:
    mappings = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mapping in mappings:
        grouped[_destination_id(mapping)].append(mapping)
    fills = {str(row.get("destination_performance_id")): row for row in schedule.get("destination_performance_fills", [])}
    rows = []
    for destination_id in sorted(set(grouped) | set(fills)):
        placement_rows = grouped.get(destination_id, [])
        fill = fills.get(destination_id, {})
        performance_start = float(fill.get("start", min((_bounds(row)[0] for row in placement_rows), default=0.0)) or 0.0)
        performance_duration = float(fill.get("duration", 0.0) or 0.0)
        performance_end = performance_start + performance_duration
        intervals = []
        for mapping in placement_rows:
            start, end = _bounds(mapping)
            if performance_duration > 0:
                start, end = max(start, performance_start), min(end, performance_end)
            if end > start:
                intervals.append((start, end))
        audible_duration = _union_duration(intervals)
        summed_duration = sum(end - start for start, end in intervals)
        speech_duration = float(fill.get("speech_duration", 0.0) or 0.0)
        speech_window_count = int(fill.get("speech_window_count", 0) or 0)
        replacement_to_speech_ratio = audible_duration / speech_duration if speech_duration > 0 else 0.0
        stacking_ratio = summed_duration / audible_duration if audible_duration > 0 else 0.0
        mapping_per_window = len(placement_rows) / speech_window_count if speech_window_count > 0 else 0.0
        warnings = []
        if replacement_to_speech_ratio > 1.15:
            warnings.append("replacement_overdensity")
        if stacking_ratio > 1.05:
            warnings.append("overlapping_replacement_dialogue")
        if mapping_per_window > 1.5:
            warnings.append("fragmented_micro_line_density")
        rows.append({
            "destination_performance_id": destination_id,
            "start": round(performance_start, 3),
            "duration": round(performance_duration, 3),
            "speech_duration": round(speech_duration, 3),
            "mapping_count": len(placement_rows),
            "speech_window_count": speech_window_count,
            "audible_replacement_duration": round(audible_duration, 3),
            "replacement_to_speech_ratio": round(replacement_to_speech_ratio, 4),
            "overlap_stacking_ratio": round(stacking_ratio, 4),
            "mappings_per_speech_window": round(mapping_per_window, 4),
            "warnings": warnings,
        })

    clip_counts = Counter(str(row.get("clip_id")) for row in mappings if row.get("clip_id") is not None)
    repeated = {clip_id: count for clip_id, count in sorted(clip_counts.items()) if count > 1}
    return {
        "performance_count": len(rows),
        "warning_performance_count": sum(1 for row in rows if row["warnings"]),
        "overdense_performance_count": sum(1 for row in rows if "replacement_overdensity" in row["warnings"]),
        "overlapping_dialogue_performance_count": sum(1 for row in rows if "overlapping_replacement_dialogue" in row["warnings"]),
        "fragmented_micro_line_performance_count": sum(1 for row in rows if "fragmented_micro_line_density" in row["warnings"]),
        "repeated_source_clip_count": len(repeated),
        "repeated_source_clips": repeated,
        "performances": rows,
    }


def _performance_candidate(stratum: str, destination_id: str, fill: dict[str, Any], mappings: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    start = float(fill.get("start", min((_bounds(row)[0] for row in mappings), default=0.0)) or 0.0)
    duration = float(fill.get("duration", 0.0) or 0.0)
    end = start + duration if duration > 0 else max((_bounds(row)[1] for row in mappings), default=start)
    return {
        "stratum": stratum,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
        "destination_performance_id": destination_id,
        "mapping_indices": [row["mapping_index"] for row in mappings],
        "reason": reason,
        "coverage": fill.get("coverage"),
    }


def _select_distributed(rows: list[dict[str, Any]], *, maximum: int) -> list[dict[str, Any]]:
    valid = [row for row in rows if float(row.get("end", 0.0) or 0.0) > float(row.get("start", 0.0) or 0.0)]
    if maximum <= 0 or not valid:
        return []
    ordered = sorted(valid, key=lambda row: float(row.get("start", 0.0) or 0.0))
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return [ordered[len(ordered) // 2]]
    indices = {round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)}
    return [ordered[index] for index in sorted(indices)]


def _selection_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    indices = tuple(sorted(int(value) for value in row.get("mapping_indices", []) if value is not None))
    if indices:
        return ("mappings", indices)
    return (
        "bounds",
        round(float(row.get("start", 0.0) or 0.0), 1),
        round(float(row.get("end", 0.0) or 0.0), 1),
    )


def _spoken_mapping_qualification(mapping: dict[str, Any]) -> dict[str, Any]:
    transcript = str(mapping.get("source_transcript") or "").strip()
    words = [word for word in transcript.replace("—", " ").replace("-", " ").split() if any(char.isalnum() for char in word)]
    if not transcript or not words:
        return {"qualified": False, "reason": "no_verified_spoken_transcript", "sentence_complete": False, "words_per_second": 0.0, "speaker_turn_risk": False}
    if len(words) < 2 and transcript.rstrip()[-1:] not in {"!", "?", "."}:
        return {"qualified": False, "reason": "nonlexical_or_incomplete_audio", "sentence_complete": False, "words_per_second": 0.0, "speaker_turn_risk": False}
    sentence_complete = transcript.rstrip()[-1:] in {".", "!", "?", "…"}
    duration = float(mapping.get("planned_render_duration", 0.0) or 0.0)
    words_per_second = len(words) / max(duration, 0.001)
    source_speakers = {str(value) for value in mapping.get("source_speaker_sequence", []) if value is not None}
    destination_speakers = {str(value) for value in mapping.get("destination_speaker_sequence", []) if value is not None}
    speaker_turn_risk = len(source_speakers) == 1 and len(destination_speakers) > 1
    if duration < 0.65:
        return {"qualified": False, "reason": "spoken_fragment_too_short", "sentence_complete": sentence_complete, "words_per_second": round(words_per_second, 3), "speaker_turn_risk": speaker_turn_risk}
    if speaker_turn_risk and words_per_second > 3.4:
        return {"qualified": False, "reason": "single_voice_over_multi_speaker_exchange", "sentence_complete": sentence_complete, "words_per_second": round(words_per_second, 3), "speaker_turn_risk": True}
    if words_per_second > 3.4:
        return {"qualified": False, "reason": "improbable_audible_line_completeness", "sentence_complete": sentence_complete, "words_per_second": round(words_per_second, 3), "speaker_turn_risk": speaker_turn_risk}
    return {"qualified": True, "reason": "verified_spoken_mapping", "sentence_complete": sentence_complete, "words_per_second": round(words_per_second, 3), "speaker_turn_risk": speaker_turn_risk}


def _review_bounds(*, start: float, end: float, seed_start: float, seed_end: float, context: float) -> tuple[float, float]:
    left = max(seed_start, start - max(0.0, context))
    right = min(seed_end, end + max(0.0, context))
    if right - left < MIN_REVIEW_DURATION:
        center = (start + end) / 2.0
        left = max(seed_start, center - MIN_REVIEW_DURATION / 2.0)
        right = min(seed_end, left + MIN_REVIEW_DURATION)
        left = max(seed_start, right - MIN_REVIEW_DURATION)
    if right - left > MAX_REVIEW_DURATION:
        center = (start + end) / 2.0
        left = max(seed_start, center - MAX_REVIEW_DURATION / 2.0)
        right = min(seed_end, left + MAX_REVIEW_DURATION)
    return left, max(left, right)


def _refinement_score(row: dict[str, Any]) -> float:
    duration = float(row.get("duration", 0.0) or 0.0)
    transcript = str(row.get("source_transcript") or "")
    return (1.0 if row.get("sentence_complete") else 0.0) + min(len(transcript.split()), 12) / 24.0 - abs(duration - 4.0) / 20.0


def _nonoverlapping(rows: list[dict[str, Any]], *, maximum: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if any(
            _overlap(float(row["start"]), float(row["end"]), float(existing["start"]), float(existing["end"]))
            for existing in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= maximum:
            break
    return selected


def _destination_id(row: dict[str, Any]) -> str:
    return str(row.get("destination_performance_id") or row.get("performance_id") or row.get("window_id"))


def _bounds(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row.get("start", row.get("destination_timestamp", 0.0)) or 0.0)
    end = row.get("end")
    if end is None:
        end = start + float(row.get("planned_render_duration", row.get("duration", 0.0)) or 0.0)
    return start, float(end)


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return min(left_end, right_end) > max(left_start, right_start)


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = [list(interval) for interval in sorted(intervals)]
    compact = [merged[0]]
    for start, end in merged[1:]:
        if start <= compact[-1][1]:
            compact[-1][1] = max(compact[-1][1], end)
        else:
            compact.append([start, end])
    return sum(end - start for start, end in compact)
