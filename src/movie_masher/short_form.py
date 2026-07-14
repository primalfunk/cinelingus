from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from . import __version__
from .remix_modes import RemixMode
from .remix_scoring import build_candidate_score, normalize_scoring_profile, score_from_candidate_fields
from .util import utc_now, write_json


def build_short_remix_candidates(
    *,
    schedule: dict[str, Any],
    target_duration_seconds: float,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    preference: str = "balanced",
    mode: RemixMode | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    mode_id = mode.mode_id if mode else "best_short_remix"
    scoring_profile = mode.scoring_profile if mode else preference
    mappings = [dict(row) for row in schedule.get("mappings", []) if row.get("enabled", True)]
    mappings.sort(key=lambda row: float(row.get("destination_timestamp", 0.0) or 0.0))
    grouped = _performance_groups(mappings)
    candidates = []
    for rows in grouped:
        if rows:
            candidates.append(_candidate(rows, target_duration_seconds, minimum_duration_seconds, maximum_duration_seconds, preference, scoring_profile=scoring_profile))
    candidates.extend(_rolling_candidates(mappings, target_duration_seconds, minimum_duration_seconds, maximum_duration_seconds, preference, scoring_profile=scoring_profile))
    candidates = _dedupe_candidates(candidates)
    sequence = _best_sequence_candidate(candidates, target_duration_seconds, minimum_duration_seconds, maximum_duration_seconds, preference, scoring_profile=scoring_profile)
    if sequence is not None:
        candidates.append(sequence)
    _apply_mode_specific_candidate_contracts(candidates, schedule)
    candidates.sort(key=lambda row: row["final_combined_score"], reverse=True)
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "selected_mode": mode_id,
        "mode": _mode_summary(mode),
        "target_duration_seconds": round(float(target_duration_seconds), 3),
        "minimum_duration_seconds": round(float(minimum_duration_seconds), 3),
        "maximum_duration_seconds": round(float(maximum_duration_seconds), 3),
        "preference": preference,
        "scoring_profile": normalize_scoring_profile(scoring_profile),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    if output_path:
        write_json(output_path, artifact)
    return artifact



def _apply_mode_specific_candidate_contracts(candidates: list[dict[str, Any]], schedule: dict[str, Any]) -> None:
    if not _is_self_shuffle_schedule(schedule):
        return
    minimum_coverage = 0.25
    for candidate in candidates:
        coverage = float(candidate.get("target_window_speech_coverage", 0.0) or 0.0)
        if coverage >= minimum_coverage:
            continue
        flags = list(candidate.get("suitability_flags", []))
        if "insufficient_self_shuffle_coverage" not in flags:
            flags.append("insufficient_self_shuffle_coverage")
        candidate["suitability_flags"] = flags
        candidate["suitability_status"] = "risky"
        candidate["final_combined_score"] = round(_clamp(float(candidate.get("final_combined_score", 0.0) or 0.0) - 0.18), 4)
        candidate["reason_summary"] = _reason(
            int(candidate.get("mapping_count", 0) or 0),
            _speaker_rate_from_candidate(candidate),
            float(candidate.get("confidence_score", 0.0) or 0.0),
            float(candidate.get("technical_risk_score", 0.0) or 0.0),
            float(candidate.get("estimated_humor_novelty_score", 0.0) or 0.0),
            flags,
        )


def _is_self_shuffle_schedule(schedule: dict[str, Any]) -> bool:
    return schedule.get("mutation_id") == "self_shuffle" or schedule.get("transformation_name") in {"self_shuffle", "mutation_self_shuffle"}


def _speaker_rate_from_candidate(candidate: dict[str, Any]) -> float:
    components = candidate.get("scoring_breakdown") or {}
    return float(components.get("speaker_match_confidence", 0.0) or 0.0)

def select_best_short_candidate(candidates: dict[str, Any]) -> dict[str, Any]:
    rows = candidates.get("candidates", [])
    if not rows:
        raise ValueError("No short-form remix candidates are available.")
    ranked = [dict(row) for row in rows]
    top = ranked[0]
    strong = [row for row in ranked if row.get("suitability_status") == "strong"]
    selected = top
    strategy = "highest_score"
    if top.get("suitability_status") != "strong" and strong:
        best_strong = strong[0]
        if float(best_strong.get("final_combined_score", 0.0) or 0.0) >= float(top.get("final_combined_score", 0.0) or 0.0) - 0.12:
            selected = best_strong
            strategy = "near_top_strong_candidate"
    elif top.get("candidate_type") == "best_sequence":
        strategy = "best_sequence_fallback"
    selected["selection_strategy"] = strategy
    if selected is not top:
        selected["selection_reason"] = "Skipped a slightly higher-risk candidate in favor of cleaner short-form suitability."
    else:
        selected["selection_reason"] = "Selected highest-ranked candidate after short-form suitability scoring."
    return selected


def build_short_remix_schedule(
    schedule: dict[str, Any],
    candidate: dict[str, Any],
    *,
    padding: float = 1.0,
    start_time: float | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    mapping_indices = {int(index) for index in candidate.get("mapping_indices", [])}
    start = float(start_time) if start_time is not None else max(0.0, float(candidate.get("destination_start", 0.0) or 0.0) - max(0.0, padding))
    end = start + float(duration) if duration is not None else None
    rows = []
    for index, mapping in enumerate(schedule.get("mappings", [])):
        if index not in mapping_indices:
            continue
        mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        mapping_duration = float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)) or 0.0)
        mapping_end = mapping_start + mapping_duration
        if end is not None and (mapping_end <= start or mapping_start >= end):
            continue
        item = dict(mapping)
        item["enabled"] = True
        if end is not None:
            _clip_mapping_to_segment(item, start, end)
        _rebase_destination_times(item, start)
        rows.append(item)
    short_schedule = dict(schedule)
    short_schedule["mappings"] = rows
    short_schedule["selected_mode"] = "best_short_remix"
    short_schedule["source_candidate_id"] = candidate.get("id")
    return short_schedule


def _rebase_destination_times(mapping: dict[str, Any], start: float) -> None:
    for field in (
        "destination_timestamp",
        "alignment_slot_start",
        "alignment_slot_end",
        "shot_start",
        "shot_end",
    ):
        if mapping.get(field) is None:
            continue
        mapping[field] = round(max(0.0, float(mapping.get(field, 0.0) or 0.0) - start), 3)


def _clip_mapping_to_segment(mapping: dict[str, Any], segment_start: float, segment_end: float) -> None:
    mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
    planned_duration = float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)) or 0.0)
    mapping_end = mapping_start + planned_duration
    overlap_start = max(mapping_start, segment_start)
    overlap_end = min(mapping_end, segment_end)
    overlap_duration = max(0.0, overlap_end - overlap_start)
    trimmed_from_start = max(0.0, segment_start - mapping_start)
    if trimmed_from_start > 0:
        mapping["destination_timestamp"] = round(overlap_start, 3)
        mapping["clip_trim_start"] = round(float(mapping.get("clip_trim_start", 0.0) or 0.0) + trimmed_from_start, 3)
    if planned_duration > 0:
        mapping["planned_render_duration"] = round(overlap_duration, 3)
    if mapping.get("clip_trim_duration") is not None:
        mapping["clip_trim_duration"] = round(min(float(mapping.get("clip_trim_duration", 0.0) or 0.0), overlap_duration), 3)


def build_short_remix_report(
    *,
    selected_mode: str,
    target_duration_seconds: float,
    actual_duration_seconds: float,
    candidate: dict[str, Any],
    candidates: dict[str, Any],
    output_video: Path,
    output_audio: Path,
    total_processing_time_seconds: float,
    output_path: Path,
    mode: RemixMode | None = None,
    candidate_rankings_output: Path | None = None,
    latest_report_output: Path | None = None,
    audio_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rejected = [row for row in candidates.get("candidates", [])[1:6]]
    shared_score = candidate.get("shared_score") or score_from_candidate_fields(candidate, profile=candidates.get("scoring_profile", selected_mode))
    ranking_path = candidate_rankings_output or output_path.with_name("candidate_rankings.json")
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "selected_mode": selected_mode,
        "mode": _mode_summary(mode) or candidates.get("mode", {}),
        "target_duration": round(float(target_duration_seconds), 3),
        "actual_duration": round(float(actual_duration_seconds), 3),
        "selected_scenes": [candidate],
        "selection_summary": {
            "candidate_id": candidate.get("id"),
            "final_score": candidate.get("final_combined_score"),
            "shared_score": shared_score,
            "suitability_status": candidate.get("suitability_status"),
            "suitability_flags": candidate.get("suitability_flags", []),
            "reason": candidate.get("reason_summary"),
            "selection_strategy": candidate.get("selection_strategy"),
            "selection_reason": candidate.get("selection_reason"),
            "candidate_type": candidate.get("candidate_type"),
            "mapping_count": candidate.get("mapping_count"),
            "scheduled_speech_duration": candidate.get("scheduled_speech_duration"),
            "target_window_speech_coverage": candidate.get("target_window_speech_coverage"),
        },
        "scoring_breakdown": candidate.get("scoring_breakdown", {}),
        "shared_scoring": shared_score,
        "speaker_labels_used": sorted(set(candidate.get("source_speaker_labels", []) + candidate.get("destination_speaker_labels", []))),
        "rejected_top_candidates": [
            {
                "id": row.get("id"),
                "score": row.get("final_combined_score"),
                "shared_score": row.get("shared_score") or score_from_candidate_fields(row, profile=candidates.get("scoring_profile", selected_mode)),
                "suitability_status": row.get("suitability_status"),
                "suitability_flags": row.get("suitability_flags", []),
                "reason": row.get("reason_summary"),
            }
            for row in rejected
        ],
        "total_processing_time": round(float(total_processing_time_seconds), 3),
        "outputs": {
            "video": str(output_video),
            "audio": str(output_audio),
            "report": str(output_path),
            "latest_report": str(latest_report_output) if latest_report_output else str(output_path),
            "candidate_rankings": str(ranking_path),
            "audio_provenance": ((audio_provenance or {}).get("outputs") or {}).get("audio_provenance"),
        },
        "audio_provenance": audio_provenance or {},
    }
    write_json(output_path, report)
    if latest_report_output:
        write_json(latest_report_output, report)
    write_candidate_rankings(candidates=candidates, selected_candidate=candidate, output_path=ranking_path, mode=mode)
    return report


def write_candidate_rankings(
    *,
    candidates: dict[str, Any],
    selected_candidate: dict[str, Any] | None,
    output_path: Path,
    mode: RemixMode | None = None,
) -> dict[str, Any]:
    selected_id = selected_candidate.get("id") if selected_candidate else None
    profile = mode.scoring_profile if mode else candidates.get("scoring_profile", candidates.get("preference", "balanced"))
    rows = []
    for row in candidates.get("candidates", []):
        rows.append(
            {
                "rank": row.get("rank"),
                "id": row.get("id"),
                "candidate_type": row.get("candidate_type"),
                "selected": row.get("id") == selected_id,
                "duration": row.get("duration"),
                "mapping_count": row.get("mapping_count"),
                "suitability_status": row.get("suitability_status"),
                "suitability_flags": row.get("suitability_flags", []),
                "reason": row.get("reason_summary"),
                "legacy_final_combined_score": row.get("final_combined_score"),
                "score": row.get("shared_score") or score_from_candidate_fields(row, profile=profile),
            }
        )
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "selected_mode": candidates.get("selected_mode") or (mode.mode_id if mode else "best_short_remix"),
        "mode": _mode_summary(mode) or candidates.get("mode", {}),
        "selected_candidate_id": selected_id,
        "candidate_count": len(rows),
        "candidates": rows,
    }
    write_json(output_path, artifact)
    return artifact


def expanded_short_window(
    *,
    candidate: dict[str, Any],
    destination_duration: float,
    target_duration_seconds: float,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    padding: float = 1.0,
) -> tuple[float, float]:
    candidate_start = max(0.0, float(candidate.get("destination_start", 0.0) or 0.0) - max(0.0, padding))
    candidate_end = max(candidate_start, float(candidate.get("destination_end", candidate_start) or candidate_start) + max(0.0, padding))
    available = max(0.001, float(destination_duration or 0.001))
    target = max(float(minimum_duration_seconds), min(float(target_duration_seconds), float(maximum_duration_seconds), available))
    candidate_duration = max(0.001, candidate_end - candidate_start)
    window_duration = min(available, max(target, min(float(maximum_duration_seconds), candidate_duration)))
    center = (candidate_start + candidate_end) / 2.0
    start = center - window_duration / 2.0
    start = max(0.0, min(start, max(0.0, available - window_duration)))
    end = min(available, start + window_duration)
    if end - start < min(float(minimum_duration_seconds), available):
        end = min(available, start + min(float(minimum_duration_seconds), available))
        start = max(0.0, end - min(float(minimum_duration_seconds), available))
    return round(start, 3), round(max(start + 0.001, end), 3)


def _best_sequence_candidate(
    candidates: list[dict[str, Any]],
    target: float,
    minimum: float,
    maximum: float,
    preference: str,
    scoring_profile: str = "balanced",
) -> dict[str, Any] | None:
    if len(candidates) < 2:
        return None
    selected = []
    used_indices: set[int] = set()
    pool = [row for row in candidates if row.get("candidate_type") == "performance_scene"] or candidates
    for candidate in sorted(pool, key=lambda row: row.get("final_combined_score", 0.0), reverse=True):
        indices = {int(index) for index in candidate.get("mapping_indices", [])}
        if not indices or indices & used_indices:
            continue
        selected.append(candidate)
        used_indices.update(indices)
        speech = sum(float(row.get("scheduled_speech_duration", 0.0) or 0.0) for row in selected)
        if len(selected) >= 2 and (speech >= max(4.0, float(target) * 0.04) or len(selected) >= 4):
            break
    if len(selected) < 2:
        return None
    selected.sort(key=lambda row: float(row.get("destination_start", 0.0) or 0.0))
    internal_gaps = [
        max(0.0, float(selected[index].get("destination_start", 0.0) or 0.0) - float(selected[index - 1].get("destination_end", 0.0) or 0.0))
        for index in range(1, len(selected))
    ]
    max_internal_gap = max(internal_gaps, default=0.0)
    if max_internal_gap > 12.0:
        return None
    start = min(float(row.get("destination_start", 0.0) or 0.0) for row in selected)
    end = max(float(row.get("destination_end", 0.0) or 0.0) for row in selected)
    mapping_indices = sorted({int(index) for row in selected for index in row.get("mapping_indices", [])})
    scheduled_speech = sum(float(row.get("scheduled_speech_duration", 0.0) or 0.0) for row in selected)
    duration = max(0.001, end - start)
    if duration > float(maximum) + 0.001:
        return None
    target_window = max(float(minimum), min(float(target), float(maximum)))
    risk = _average([row.get("technical_risk_score") for row in selected])
    realism = _average([row.get("estimated_realism_score") for row in selected])
    humor = _average([row.get("estimated_humor_novelty_score") for row in selected])
    confidence = _average([row.get("confidence_score") for row in selected])
    coverage = _clamp(scheduled_speech / max(target_window, 0.001))
    speech_density = _clamp(scheduled_speech / max(duration, 0.001))
    flags = []
    if coverage < 0.04:
        flags.append("thin_dialogue_for_target_window")
    if speech_density < 0.04:
        flags.append("sparse_speech_density")
    if risk > 0.5:
        flags.append("high_technical_risk")
    shared_score = build_candidate_score(
        realism=realism,
        humor=humor,
        coherence=_average([coverage, speech_density, confidence]),
        technical=1.0 - risk,
        novelty=humor,
        profile=scoring_profile,
        components={
            "sequence_member_count": len(selected),
            "timing_fit": round(confidence, 4),
            "speech_density": round(speech_density, 4),
            "target_window_speech_coverage": round(coverage, 4),
            "comedic_potential": round(humor, 4),
            "low_technical_risk": round(1.0 - risk, 4),
        },
    )
    final = _clamp(shared_score["combined_score"] + min(0.12, 0.03 * len(selected)) - min(0.1, 0.03 * len(flags)))
    speakers_source = sorted({speaker for row in selected for speaker in row.get("source_speaker_labels", [])})
    speakers_destination = sorted({speaker for row in selected for speaker in row.get("destination_speaker_labels", [])})
    return {
        "id": f"candidate_sequence_{int(start * 1000):09d}_{int(end * 1000):09d}_{len(mapping_indices)}",
        "candidate_type": "best_sequence",
        "sequence_candidate_ids": [row.get("id") for row in selected],
        "destination_start": round(start, 3),
        "destination_end": round(end, 3),
        "duration": round(duration, 3),
        "source_segment_start": min(float(row.get("source_segment_start", 0.0) or 0.0) for row in selected),
        "source_segment_end": max(float(row.get("source_segment_end", 0.0) or 0.0) for row in selected),
        "replacement_dialogue_source_start": min(float(row.get("replacement_dialogue_source_start", 0.0) or 0.0) for row in selected),
        "replacement_dialogue_source_end": max(float(row.get("replacement_dialogue_source_end", 0.0) or 0.0) for row in selected),
        "mapping_indices": mapping_indices,
        "mapping_count": len(mapping_indices),
        "max_internal_gap": round(max_internal_gap, 3),
        "scheduled_speech_duration": round(scheduled_speech, 3),
        "speech_density": round(speech_density, 4),
        "target_window_speech_coverage": round(coverage, 4),
        "suitability_flags": flags,
        "suitability_status": "strong" if not flags else "risky",
        "source_speaker_labels": speakers_source,
        "destination_speaker_labels": speakers_destination,
        "confidence_score": round(confidence, 4),
        "estimated_realism_score": round(realism, 4),
        "estimated_humor_novelty_score": round(humor, 4),
        "technical_risk_score": round(risk, 4),
        "final_combined_score": round(final, 4),
        "shared_score": shared_score,
        "scoring_breakdown": {
            "sequence_member_count": len(selected),
            "timing_fit": round(confidence, 4),
            "speech_density": round(speech_density, 4),
            "target_window_speech_coverage": round(coverage, 4),
            "comedic_potential": round(humor, 4),
            "low_technical_risk": round(1.0 - risk, 4),
        },
        "reason_summary": f"Best sequence: {len(selected)} strong moments combined for better short-form coverage.",
    }


def _performance_groups(mappings: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback: list[list[dict[str, Any]]] = []
    for index, mapping in enumerate(mappings):
        mapping["_schedule_index"] = int(mapping.get("_schedule_index", index))
        key = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or "")
        if key:
            groups[key].append(mapping)
        else:
            fallback.append([mapping])
    return list(groups.values()) + fallback


def _rolling_candidates(
    mappings: list[dict[str, Any]],
    target: float,
    minimum: float,
    maximum: float,
    preference: str,
    scoring_profile: str = "balanced",
) -> list[dict[str, Any]]:
    rows = []
    for start_index in range(len(mappings)):
        bucket = []
        for mapping in mappings[start_index:]:
            bucket.append(mapping)
            duration = _bounds(bucket)[1] - _bounds(bucket)[0]
            if duration >= minimum * 0.75:
                rows.append(_candidate(bucket, target, minimum, maximum, preference, candidate_type="rolling_sequence", scoring_profile=scoring_profile))
            if duration >= maximum:
                break
    return rows


def _candidate(
    rows: list[dict[str, Any]],
    target: float,
    minimum: float,
    maximum: float,
    preference: str,
    candidate_type: str = "performance_scene",
    scoring_profile: str = "balanced",
) -> dict[str, Any]:
    start, end = _bounds(rows)
    duration = max(0.001, end - start)
    target_window_duration = max(float(minimum), min(float(target), float(maximum)))
    scheduled_speech_duration = sum(float(row.get("planned_render_duration", row.get("clip_trim_duration", 0.0)) or 0.0) for row in rows)
    speech_density = _clamp(scheduled_speech_duration / max(duration, 0.001))
    target_window_speech_coverage = _clamp(scheduled_speech_duration / max(target_window_duration, 0.001))
    avg_score = _average([row.get("score") for row in rows])
    speaker_evidence = [row for row in rows if row.get("source_speaker_id") or row.get("destination_speaker_id")]
    speaker_rate = _average([1.0 if row.get("speaker_match_preserved") else 0.0 for row in speaker_evidence])
    visual = _average([row.get("visual_fit_score") for row in rows])
    stretch = _average([abs(float(row.get("stretch_factor", 1.0) or 1.0) - 1.0) for row in rows])
    cross_rate = _average([1.0 if row.get("mapping_crosses_shot_boundary") or row.get("crosses_shot_boundary") else 0.0 for row in rows])
    length = _length_score(duration, target, minimum, maximum)
    density = min(1.0, len(rows) / max(1.0, duration / 12.0))
    novelty = _average([0.7 if not row.get("speaker_match_preserved") else 0.45 for row in rows])
    risk = min(1.0, cross_rate * 0.45 + stretch * 2.0 + (1.0 - visual) * 0.35)
    realism = _clamp(avg_score * 0.35 + speaker_rate * 0.25 + visual * 0.2 + length * 0.2 - risk * 0.25)
    humor = _clamp(novelty * 0.35 + density * 0.25 + avg_score * 0.2 + (1.0 - risk) * 0.2)
    shared_score = build_candidate_score(
        realism=realism,
        humor=humor,
        coherence=_average([length, 1.0 - cross_rate, speech_density, target_window_speech_coverage]),
        technical=1.0 - risk,
        novelty=novelty,
        profile=normalize_scoring_profile(scoring_profile or preference),
        components={
            "speaker_match_confidence": round(speaker_rate, 4),
            "timing_fit": round(avg_score, 4),
            "dialogue_clarity": round(avg_score, 4),
            "length_compatibility": round(length, 4),
            "scene_continuity": round(1.0 - cross_rate, 4),
            "density_of_good_swaps": round(density, 4),
            "speech_density": round(speech_density, 4),
            "target_window_speech_coverage": round(target_window_speech_coverage, 4),
            "comedic_potential": round(humor, 4),
            "low_technical_risk": round(1.0 - risk, 4),
        },
    )
    final = _clamp(shared_score["combined_score"])
    suitability_flags = _suitability_flags(
        mapping_count=len(rows),
        duration=duration,
        minimum=minimum,
        risk=risk,
        avg_score=avg_score,
        speech_density=speech_density,
        target_window_speech_coverage=target_window_speech_coverage,
        speaker_rate=speaker_rate,
        has_speaker_evidence=bool(speaker_evidence),
    )
    suitability_penalty = min(0.35, 0.06 * len(suitability_flags))
    final = _clamp(final - suitability_penalty)
    speakers_source = sorted({str(row.get("source_speaker_id")) for row in rows if row.get("source_speaker_id")})
    speakers_destination = sorted({str(row.get("destination_speaker_id")) for row in rows if row.get("destination_speaker_id")})
    return {
        "id": f"candidate_{int(start * 1000):09d}_{int(end * 1000):09d}_{len(rows)}",
        "candidate_type": candidate_type,
        "destination_start": round(start, 3),
        "destination_end": round(end, 3),
        "duration": round(duration, 3),
        "source_segment_start": round(min(_source_timestamp(row) for row in rows), 3),
        "source_segment_end": round(max(_source_timestamp(row) + float(row.get("clip_trim_duration", row.get("planned_render_duration", 0.0)) or 0.0) for row in rows), 3),
        "replacement_dialogue_source_start": round(min(float(row.get("clip_trim_start", 0.0) or 0.0) for row in rows), 3),
        "replacement_dialogue_source_end": round(max(float(row.get("clip_trim_start", 0.0) or 0.0) + float(row.get("clip_trim_duration", 0.0) or 0.0) for row in rows), 3),
        "mapping_indices": [int(row.get("_schedule_index", 0)) for row in rows],
        "mapping_count": len(rows),
        "scheduled_speech_duration": round(scheduled_speech_duration, 3),
        "speech_density": round(speech_density, 4),
        "target_window_speech_coverage": round(target_window_speech_coverage, 4),
        "suitability_flags": suitability_flags,
        "suitability_status": "strong" if not suitability_flags else "risky",
        "source_speaker_labels": speakers_source,
        "destination_speaker_labels": speakers_destination,
        "confidence_score": round(avg_score, 4),
        "estimated_realism_score": round(realism, 4),
        "estimated_humor_novelty_score": round(humor, 4),
        "technical_risk_score": round(risk, 4),
        "final_combined_score": round(final, 4),
        "shared_score": shared_score,
        "scoring_breakdown": {
            "speaker_match_confidence": round(speaker_rate, 4),
            "timing_fit": round(avg_score, 4),
            "dialogue_clarity": round(avg_score, 4),
            "length_compatibility": round(length, 4),
            "scene_continuity": round(1.0 - cross_rate, 4),
            "density_of_good_swaps": round(density, 4),
            "speech_density": round(speech_density, 4),
            "target_window_speech_coverage": round(target_window_speech_coverage, 4),
            "comedic_potential": round(humor, 4),
            "low_technical_risk": round(1.0 - risk, 4),
        },
        "reason_summary": _reason(len(rows), speaker_rate, avg_score, risk, humor, suitability_flags),
    }


def _source_timestamp(row: dict[str, Any]) -> float:
    for field in ("clip_movie_timestamp", "source_movie_timestamp", "movie_timestamp"):
        if row.get(field) is not None:
            return float(row.get(field, 0.0) or 0.0)
    return 0.0


def _suitability_flags(
    *,
    mapping_count: int,
    duration: float,
    minimum: float,
    risk: float,
    avg_score: float,
    speech_density: float,
    target_window_speech_coverage: float,
    speaker_rate: float,
    has_speaker_evidence: bool = False,
) -> list[str]:
    flags = []
    if mapping_count < 2:
        flags.append("too_few_swaps")
    if duration < minimum * 0.5:
        flags.append("scene_too_short")
    if target_window_speech_coverage < 0.04:
        flags.append("thin_dialogue_for_target_window")
    if speech_density < 0.08:
        flags.append("sparse_speech_density")
    if avg_score < 0.45:
        flags.append("weak_timing_fit")
    if has_speaker_evidence and speaker_rate < 0.35:
        flags.append("weak_speaker_consistency")
    if risk > 0.5:
        flags.append("high_technical_risk")
    return flags


def _bounds(rows: list[dict[str, Any]]) -> tuple[float, float]:
    start = min(float(row.get("destination_timestamp", 0.0) or 0.0) for row in rows)
    end = max(float(row.get("destination_timestamp", 0.0) or 0.0) + float(row.get("planned_render_duration", row.get("clip_trim_duration", 0.0)) or 0.0) for row in rows)
    return start, end


def _length_score(duration: float, target: float, minimum: float, maximum: float) -> float:
    if minimum <= duration <= maximum:
        return _clamp(1.0 - abs(duration - target) / max(target, 1.0) * 0.5)
    if duration < minimum:
        return _clamp(duration / max(minimum, 1.0) * 0.7)
    return _clamp(maximum / max(duration, 1.0) * 0.7)


def _reason(count: int, speaker_rate: float, timing: float, risk: float, humor: float, suitability_flags: list[str]) -> str:
    parts = [f"{count} mapped line{'s' if count != 1 else ''}"]
    parts.append("strong speaker consistency" if speaker_rate >= 0.7 else "limited speaker evidence")
    parts.append("clean timing" if timing >= 0.7 else "usable timing")
    parts.append("low technical risk" if risk <= 0.25 else "some technical risk")
    if humor >= 0.65:
        parts.append("high absurd contrast")
    prefix = "Strong candidate" if not suitability_flags else "Risky candidate"
    suffix = f" Cautions: {', '.join(suitability_flags)}." if suitability_flags else ""
    return f"{prefix}: " + ", ".join(parts) + "." + suffix


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for candidate in candidates:
        key = tuple(candidate.get("mapping_indices", []))
        if key in seen:
            continue
        seen.add(key)
        rows.append(candidate)
    return rows


def _average(values: list[Any]) -> float:
    nums = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(nums) / len(nums) if nums else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))



def _mode_summary(mode: RemixMode | None) -> dict[str, Any]:
    if mode is None:
        return {}
    return {
        "mode_id": mode.mode_id,
        "display_name": mode.display_name,
        "short_description": mode.short_description,
        "candidate_generation_strategy": mode.candidate_generation_strategy,
        "scoring_profile": mode.scoring_profile,
        "assembly_strategy": mode.assembly_strategy,
        "ui_visibility": mode.ui_visibility,
    }
