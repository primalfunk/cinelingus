from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from . import __version__
from .short_form import build_short_remix_schedule
from .util import utc_now, write_json

MINIMUM_DIALOGUE_COVERAGE = 0.6
VIGNETTE_EDGE_PADDING_SECONDS = 0.5


def build_dialogue_scene_artifact(
    *,
    media_hash: str,
    role: str,
    performances: dict[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    scenes = [_scene_from_performance(row, role=role, index=index) for index, row in enumerate(performances.get("performances", []), start=1)]
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "media_hash": media_hash,
        "role": role,
        "scene_count": len(scenes),
        "scenes": scenes,
    }
    if output_path:
        write_json(output_path, artifact)
    return artifact


def build_scene_pair_candidates(
    *,
    schedule: dict[str, Any],
    source_scenes: dict[str, Any],
    destination_scenes: dict[str, Any],
    self_shuffle: bool = False,
    minimum_temporal_separation: float = 30.0,
    minimum_dialogue_coverage: float = MINIMUM_DIALOGUE_COVERAGE,
    output_path: Path | None = None,
) -> dict[str, Any]:
    source_by_id = {str(row.get("scene_id")): row for row in source_scenes.get("scenes", [])}
    destination_by_id = {str(row.get("scene_id")): row for row in destination_scenes.get("scenes", [])}
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    destination_rows: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    rejected = []
    for index, mapping in enumerate(schedule.get("mappings", [])):
        if not mapping.get("enabled", True):
            continue
        destination_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id") or "")
        raw_source_id = str(mapping.get("source_performance_id") or "")
        source_id = _resolve_source_scene_id(mapping, source_by_id)
        if not destination_id or not source_id:
            rejected.append(
                {
                    "candidate_id": f"unresolved_mapping_{index + 1:06d}",
                    "destination_scene_id": destination_id,
                    "source_scene_id": raw_source_id,
                    "reason_rejected": "missing_destination_scene_id" if not destination_id else "source_scene_not_found",
                    "mapping_index": index,
                }
            )
            continue
        item = dict(mapping)
        item["source_scene_resolution"] = "direct" if source_id == raw_source_id else "source_timestamp"
        grouped[(destination_id, source_id)].append((index, item))
        destination_rows[destination_id].append((index, item))

    candidates = []
    for rank, ((destination_id, source_id), indexed_rows) in enumerate(grouped.items(), start=1):
        destination = destination_by_id.get(destination_id)
        source = source_by_id.get(source_id)
        if not destination or not source:
            rejected.append(
                {
                    "candidate_id": f"scene_pair_{rank:06d}",
                    "destination_scene_id": destination_id,
                    "source_scene_id": source_id,
                    "reason_rejected": "destination_scene_not_found" if not destination else "source_scene_not_found",
                    "mapping_count": len(indexed_rows),
                }
            )
            continue
        pair_rows = [row for _, row in indexed_rows]
        render_indexed_rows = destination_rows[destination_id]
        render_rows = [row for _, row in render_indexed_rows]
        temporal_gap = abs(float(destination.get("start_time", 0.0) or 0.0) - float(source.get("start_time", 0.0) or 0.0))
        if self_shuffle and temporal_gap < minimum_temporal_separation:
            rejected.append(
                {
                    "candidate_id": f"scene_pair_{rank:06d}",
                    "destination_scene_id": destination_id,
                    "source_scene_id": source_id,
                    "reason_rejected": "self_shuffle_temporal_separation",
                    "temporal_gap_seconds": round(temporal_gap, 3),
                    "minimum_temporal_separation": round(minimum_temporal_separation, 3),
                }
            )
            continue
        scene_start = _float(destination.get("start_time"), 0.0)
        scene_end = _float(destination.get("end_time"), scene_start)
        mapped_duration = _mapped_coverage_seconds(render_rows, scene_start, scene_end)
        scene_duration = max(0.001, scene_end - scene_start)
        dialogue_coverage = min(1.0, mapped_duration / scene_duration)
        if dialogue_coverage < max(0.0, minimum_dialogue_coverage):
            rejected.append(
                {
                    "candidate_id": f"scene_pair_{rank:06d}",
                    "destination_scene_id": destination_id,
                    "source_scene_id": source_id,
                    "reason_rejected": "insufficient_dialogue_coverage",
                    "dialogue_coverage": round(dialogue_coverage, 4),
                    "minimum_dialogue_coverage": round(max(0.0, minimum_dialogue_coverage), 4),
                    "mapping_count": len(render_rows),
                }
            )
            continue
        render_start, render_end = _mapped_render_window(
            render_rows,
            scene_start=scene_start,
            scene_end=scene_end,
            padding=VIGNETTE_EDGE_PADDING_SECONDS,
        )
        components = _score_components(destination, source, render_rows)
        overall = _weighted_score(components)
        candidates.append(
            {
                "id": f"scene_pair_{rank:06d}",
                "candidate_type": "dialogue_scene_pair",
                "destination_scene_id": destination_id,
                "source_scene_id": source_id,
                "destination_start": round(render_start, 3),
                "destination_end": round(render_end, 3),
                "destination_duration": round(max(0.0, render_end - render_start), 3),
                "destination_scene_start": destination.get("start_time"),
                "destination_scene_end": destination.get("end_time"),
                "destination_scene_duration": destination.get("conversation_duration"),
                "source_start": source.get("start_time"),
                "source_end": source.get("end_time"),
                "source_duration": source.get("conversation_duration"),
                # The source-scene pair ranks the vignette, but rendering must
                # retain every scheduled line in the destination scene.
                "mapping_indices": [index for index, _ in render_indexed_rows],
                "mapping_count": len(render_rows),
                "pair_mapping_indices": [index for index, _ in indexed_rows],
                "pair_mapping_count": len(pair_rows),
                "dialogue_duration": round(mapped_duration, 3),
                "dialogue_coverage": round(dialogue_coverage, 4),
                "destination_speakers": destination.get("speaker_sequence", []),
                "source_speakers": source.get("speaker_sequence", []),
                "source_scene_resolution": sorted({str(row.get("source_scene_resolution") or "direct") for row in pair_rows}),
                "speaker_mapping": _speaker_mapping_summary(pair_rows),
                "component_scores": components,
                "overall_score": overall,
                "reason_selected": _selection_reason(components, render_rows),
                "temporal_gap_seconds": round(temporal_gap, 3),
            }
        )
    candidates.sort(key=lambda row: (-float(row.get("overall_score", 0.0) or 0.0), float(row.get("destination_start", 0.0) or 0.0)))
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "candidate_count": len(candidates),
        "minimum_temporal_separation": round(minimum_temporal_separation, 3),
        "minimum_dialogue_coverage": round(max(0.0, minimum_dialogue_coverage), 4),
        "candidates": candidates,
        "rejected_candidates": rejected,
    }
    if output_path:
        write_json(output_path, artifact)
    return artifact


def _resolve_source_scene_id(mapping: dict[str, Any], source_by_id: dict[str, dict[str, Any]]) -> str:
    source_id = str(mapping.get("source_performance_id") or "")
    if source_id in source_by_id:
        return source_id
    source_timestamp = mapping.get("clip_movie_timestamp", mapping.get("source_movie_timestamp"))
    try:
        timestamp = float(source_timestamp)
    except (TypeError, ValueError):
        return ""
    matches = []
    for scene_id, scene in source_by_id.items():
        start = _float(scene.get("start_time"), 0.0)
        end = _float(scene.get("end_time"), start)
        if start - 0.001 <= timestamp <= end + 0.001:
            matches.append((max(0.0, end - start), scene_id))
    return min(matches)[1] if matches else ""


def select_vignette_reel(
    *,
    candidates: dict[str, Any],
    target_duration_seconds: float,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    minimum_dialogue_coverage: float = MINIMUM_DIALOGUE_COVERAGE,
    output_path: Path | None = None,
) -> dict[str, Any]:
    selected = []
    rejected = []
    used_destination: set[str] = set()
    used_source: set[str] = set()
    total = 0.0
    for candidate in candidates.get("candidates", []):
        coverage = candidate.get("dialogue_coverage")
        if coverage is not None and float(coverage or 0.0) < max(0.0, minimum_dialogue_coverage):
            rejected.append(_reject(candidate, "insufficient_dialogue_coverage"))
            continue
        duration = float(candidate.get("destination_duration", candidate.get("dialogue_duration", 0.0)) or 0.0)
        if duration <= 0:
            rejected.append(_reject(candidate, "zero_duration"))
            continue
        if total + duration > maximum_duration_seconds:
            rejected.append(_reject(candidate, "would_exceed_maximum_duration"))
            continue
        destination_id = str(candidate.get("destination_scene_id"))
        source_id = str(candidate.get("source_scene_id"))
        if destination_id in used_destination:
            rejected.append(_reject(candidate, "duplicate_destination_scene"))
            continue
        if source_id in used_source:
            rejected.append(_reject(candidate, "duplicate_donor_scene"))
            continue
        if _overlaps_selected(candidate, selected):
            rejected.append(_reject(candidate, "overlapping_destination_time"))
            continue
        chosen = dict(candidate)
        chosen["vignette_index"] = len(selected) + 1
        selected.append(chosen)
        used_destination.add(destination_id)
        used_source.add(source_id)
        total += duration
        if total >= target_duration_seconds:
            break
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "target_duration_seconds": round(target_duration_seconds, 3),
        "minimum_duration_seconds": round(minimum_duration_seconds, 3),
        "maximum_duration_seconds": round(maximum_duration_seconds, 3),
        "actual_scene_duration_seconds": round(total, 3),
        "selected_count": len(selected),
        "selected_vignettes": selected,
        "rejected_candidates": rejected[:50],
        "selection_status": "multi_vignette" if len(selected) > 1 else "single_vignette_fallback",
    }
    if output_path:
        write_json(output_path, artifact)
    return artifact


def build_vignette_schedule(schedule: dict[str, Any], vignette: dict[str, Any], *, padding: float = 0.5) -> dict[str, Any]:
    start = max(0.0, float(vignette.get("destination_start", 0.0) or 0.0) - max(0.0, padding))
    duration = float(vignette.get("destination_duration", 0.0) or 0.0) + max(0.0, padding) * 2
    return build_short_remix_schedule(schedule, vignette, padding=padding, start_time=start, duration=duration)


def offset_vignette_schedule(schedule: dict[str, Any], *, offset_seconds: float) -> dict[str, Any]:
    # Place rebased vignette timings at their position in the concatenated reel.
    offset = max(0.0, float(offset_seconds or 0.0))
    rows = []
    for mapping in schedule.get('mappings', []):
        item = dict(mapping)
        for field in ('destination_timestamp', 'alignment_slot_start', 'alignment_slot_end', 'shot_start', 'shot_end'):
            if item.get(field) is not None:
                item[field] = round(float(item.get(field, 0.0) or 0.0) + offset, 3)
        rows.append(item)
    result = dict(schedule)
    result['mappings'] = rows
    result['reel_offset_seconds'] = round(offset, 3)
    return result


def build_vignette_reel_report(
    *,
    reel: dict[str, Any],
    candidates: dict[str, Any],
    destination_scenes: dict[str, Any],
    source_scenes: dict[str, Any],
    output_video: Path,
    output_audio: Path,
    output_path: Path,
    vignette_outputs: list[dict[str, Any]],
    audio_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = list(reel.get("selected_vignettes", []))
    first = selected[0] if selected else {}
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "selected_mode": "dialogue_reel",
        "selection_status": reel.get("selection_status"),
        "target_duration": reel.get("target_duration_seconds"),
        "actual_duration": reel.get("actual_scene_duration_seconds", 0.0),
        "actual_scene_duration": reel.get("actual_scene_duration_seconds"),
        "selected_scenes": selected,
        "selection_summary": {
            "candidate_id": first.get("id"),
            "vignette_count": len(selected),
            "selection_strategy": "greedy_ranked_scene_pairs",
            "selection_reason": "Selected highest-scoring non-reused dialogue scene pairs for a hard-cut reel.",
            "first_destination_timestamp": first.get("destination_start"),
            "first_donor_timestamp": first.get("source_start"),
        },
        "scoring_breakdown": first.get("component_scores", {}),
        "speaker_labels_used": sorted(
            {
                str(label)
                for row in selected
                for label in list(row.get("destination_speakers", [])) + list(row.get("source_speakers", []))
                if label
            }
        ),
        "rejected_top_candidates": reel.get("rejected_candidates", [])[:5],
        "total_processing_time": 0.0,
        "selected_vignettes": selected,
        "rejected_candidates": reel.get("rejected_candidates", []),
        "candidate_count": candidates.get("candidate_count", 0),
        "dialogue_scenes": {
            "destination_count": destination_scenes.get("scene_count", 0),
            "source_count": source_scenes.get("scene_count", 0),
        },
        "vignette_outputs": vignette_outputs,
        "outputs": {
            "video": str(output_video),
            "audio": str(output_audio),
            "report": str(output_path),
            "audio_provenance": ((audio_provenance or {}).get("outputs") or {}).get("audio_provenance"),
        },
        "audio_provenance": audio_provenance or {},
    }
    write_json(output_path, report)
    return report

def _scene_from_performance(performance: dict[str, Any], *, role: str, index: int) -> dict[str, Any]:
    duration = _float(performance.get("duration"), 0.0)
    pause_stats = performance.get("pause_statistics") or {}
    return {
        "scene_id": str(performance.get("id") or f"scene_{index:06d}"),
        "role": role,
        "start_time": round(_float(performance.get("start"), 0.0), 3),
        "end_time": round(_float(performance.get("end"), _float(performance.get("start"), 0.0) + duration), 3),
        "speaker_sequence": list(performance.get("speaker_sequence", [])),
        "speaker_order": list(performance.get("speaker_sequence", [])),
        "speaker_count": int(performance.get("estimated_speaker_count", len(performance.get("speaker_ids", [])) or 1) or 1),
        "speaker_transitions": _transitions(list(performance.get("speaker_sequence", []))),
        "dialogue_clips": list(performance.get("dialogue_event_ids", performance.get("speaking_window_ids", []))),
        "pause_timing": pause_stats,
        "conversation_duration": round(duration, 3),
        "average_speaking_rate": round(_float(performance.get("words_per_second"), 0.0), 4),
        "estimated_emotional_intensity": round(_float(performance.get("estimated_energy"), 0.0), 4),
        "estimated_overlap": round(1.0 if performance.get("interruptions_detected") else 0.0, 4),
        "silence_percentage": round(_float(performance.get("silence_ratio"), _float(performance.get("pause_ratio"), 0.0)) * 100.0, 3),
        "conversation_type": performance.get("conversation_type"),
        "performance_type": performance.get("performance_type"),
        "turn_pattern": performance.get("turn_pattern", ""),
        "confidence": _float(performance.get("confidence"), 0.7),
    }


def _score_components(destination: dict[str, Any], source: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    dest_duration = _float(destination.get("conversation_duration"), 0.0)
    source_duration = _float(source.get("conversation_duration"), 0.0)
    duration_score = 1.0 - min(1.0, abs(dest_duration - source_duration) / max(dest_duration, source_duration, 0.001))
    speaker_score = 1.0 if int(destination.get("speaker_count", 1) or 1) == int(source.get("speaker_count", 1) or 1) else 0.45
    turn_score = 1.0 if destination.get("speaker_sequence") == source.get("speaker_sequence") else _sequence_similarity(destination.get("speaker_sequence", []), source.get("speaker_sequence", []))
    pause_score = 1.0 - min(1.0, abs(_float((destination.get("pause_timing") or {}).get("average"), 0.0) - _float((source.get("pause_timing") or {}).get("average"), 0.0)) / 2.0)
    rate_score = 1.0 - min(1.0, abs(_float(destination.get("average_speaking_rate"), 0.0) - _float(source.get("average_speaking_rate"), 0.0)) / 4.0)
    continuity = min(1.0, sum(_duration(row) for row in rows) / max(dest_duration, 0.001))
    mapping_scores = [_float(row.get("score"), 0.7) for row in rows]
    return {
        "timing_compatibility": round(duration_score, 4),
        "speaker_compatibility": round(speaker_score, 4),
        "turn_taking_similarity": round(turn_score, 4),
        "pause_similarity": round(pause_score, 4),
        "speaking_rate_similarity": round(rate_score, 4),
        "dialogue_continuity": round(continuity, 4),
        "mapping_quality": round(mean(mapping_scores) if mapping_scores else 0.0, 4),
        "scene_completeness": round(min(1.0, len(rows) / max(1, int(destination.get("speaker_count", 1) or 1))), 4),
    }


def _mapped_coverage_seconds(rows: list[dict[str, Any]], scene_start: float, scene_end: float) -> float:
    intervals = []
    for row in rows:
        start = max(scene_start, _float(row.get("destination_timestamp"), scene_start))
        end = min(scene_end, start + _duration(row))
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _mapped_render_window(
    rows: list[dict[str, Any]],
    *,
    scene_start: float,
    scene_end: float,
    padding: float,
) -> tuple[float, float]:
    starts = []
    ends = []
    for row in rows:
        start = _float(row.get("destination_timestamp"), scene_start)
        end = start + _duration(row)
        if end <= scene_start or start >= scene_end:
            continue
        starts.append(max(scene_start, start))
        ends.append(min(scene_end, end))
    if not starts:
        return scene_start, scene_end
    return max(scene_start, min(starts) - max(0.0, padding)), min(scene_end, max(ends) + max(0.0, padding))


def _weighted_score(components: dict[str, float]) -> float:
    weights = {
        "timing_compatibility": 0.2,
        "speaker_compatibility": 0.2,
        "turn_taking_similarity": 0.15,
        "pause_similarity": 0.1,
        "speaking_rate_similarity": 0.1,
        "dialogue_continuity": 0.1,
        "mapping_quality": 0.1,
        "scene_completeness": 0.05,
    }
    return round(sum(components.get(key, 0.0) * weight for key, weight in weights.items()), 4)


def _selection_reason(components: dict[str, float], rows: list[dict[str, Any]]) -> str:
    strengths = [key for key, value in components.items() if value >= 0.75]
    if strengths:
        return "strong " + ", ".join(strengths[:3])
    if rows:
        return "usable mapped dialogue scene"
    return "candidate generated from scene pair"


def _speaker_mapping_summary(rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping = {}
    for row in rows:
        source = row.get("source_speaker_id")
        destination = row.get("destination_speaker_id")
        if source and destination:
            mapping[str(destination)] = str(source)
    return mapping


def _reject(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("id"),
        "destination_scene_id": candidate.get("destination_scene_id"),
        "source_scene_id": candidate.get("source_scene_id"),
        "overall_score": candidate.get("overall_score"),
        "reason_rejected": reason,
    }


def _overlaps_selected(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    start = _float(candidate.get("destination_start"), 0.0)
    end = _float(candidate.get("destination_end"), start)
    for row in selected:
        other_start = _float(row.get("destination_start"), 0.0)
        other_end = _float(row.get("destination_end"), other_start)
        if start < other_end and end > other_start:
            return True
    return False


def _transitions(sequence: list[Any]) -> list[str]:
    return [f"{sequence[index]}->{sequence[index + 1]}" for index in range(len(sequence) - 1)]


def _sequence_similarity(left: list[Any], right: list[Any]) -> float:
    if not left or not right:
        return 0.0
    matches = sum(1 for l_item, r_item in zip(left, right) if l_item == r_item)
    return round(matches / max(len(left), len(right)), 4)


def _duration(row: dict[str, Any]) -> float:
    return _float(row.get("planned_render_duration", row.get("clip_trim_duration", row.get("duration"))), 0.0)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


