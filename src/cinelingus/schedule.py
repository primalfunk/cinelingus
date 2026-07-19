from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .cinematic_filters import get_filter
from .intervals import covered_speech_duration
from .shot_context import predicted_render_timing, visual_fit_for_candidate
from .transformation_verbs import translation_transformation_plan, select_dialogue_clips, select_speaking_windows
from .util import utc_now, write_json


def build_schedule(
    *,
    clips: list[dict],
    windows: list[dict],
    source_hash: str,
    destination_hash: str,
    max_time_stretch: float,
    output_path: Path,
    scheduling_mode: str = "strict_order",
    best_fit_lookahead: int = 8,
    shot_boundary_mode: str = "off",
    transformation_name: str = "translation",
    transformation_history: list[dict[str, Any]] | None = None,
    source_performances: dict[str, Any] | None = None,
    cinematic_filter: str = "balanced",
    allow_source_reuse: bool = False,
) -> dict:
    if scheduling_mode not in {"strict_order", "best_fit", "window_fill", "whole_line_fill", "performance_fill"}:
        raise ValueError(f"Unsupported scheduling mode: {scheduling_mode}")
    if shot_boundary_mode not in {"off", "soft", "strict"}:
        raise ValueError(f"Unsupported shot boundary mode: {shot_boundary_mode}")

    active_filter = get_filter(cinematic_filter)
    mappings = []
    clip_index = 0
    usable_clips = _annotate_clips_with_source_performances(select_dialogue_clips(clips), source_performances)
    selected_windows = select_speaking_windows(windows)
    if scheduling_mode == "whole_line_fill":
        selected_windows = _merge_adjacent_windows(selected_windows, max_gap=2.0)
    max_fill_window_duration = max((float(window.get("duration", 0.0)) for window in selected_windows), default=0.0)

    if scheduling_mode == "performance_fill":
        clip_groups = _source_performance_clip_groups(usable_clips, source_performances)
        group_index = 0
        for window in selected_windows:
            group_index = _append_performance_fill_mappings(
                mappings=mappings,
                clip_groups=clip_groups,
                group_index=group_index,
                window=window,
                max_time_stretch=max_time_stretch,
                shot_boundary_mode=shot_boundary_mode,
                best_fit_lookahead=best_fit_lookahead,
                cinematic_filter=cinematic_filter,
            )
            if group_index >= len(clip_groups):
                break
    else:
        for window in selected_windows:
            if clip_index >= len(usable_clips):
                break

            if scheduling_mode in {"window_fill", "whole_line_fill"}:
                clip_index = _append_window_fill_mappings(
                    mappings=mappings,
                    usable_clips=usable_clips,
                    clip_index=clip_index,
                    window=window,
                    max_time_stretch=max_time_stretch,
                    shot_boundary_mode=shot_boundary_mode,
                    scheduling_mode=scheduling_mode,
                    max_window_duration=max_fill_window_duration,
                    cinematic_filter=cinematic_filter,
                )
                continue

            if scheduling_mode == "best_fit":
                clip, chosen_index, score_data = _choose_best_fit(
                    usable_clips,
                    start_index=clip_index,
                    window=window,
                    lookahead=best_fit_lookahead,
                    max_time_stretch=max_time_stretch,
                    shot_boundary_mode=shot_boundary_mode,
                )
                skipped = chosen_index - clip_index
                clip_index = chosen_index + 1
                selection_reason = "best_duration_fit_within_lookahead"
            else:
                clip = usable_clips[clip_index]
                clip_index += 1
                skipped = 0
                score_data = _score_candidate(window, clip, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
                selection_reason = "next_source_clip_in_order"

            mapping = _build_mapping(
                window=window,
                clip=clip,
                max_time_stretch=max_time_stretch,
                score_data=score_data,
                scheduling_mode=scheduling_mode,
                shot_boundary_mode=shot_boundary_mode,
                selection_reason=selection_reason,
                skipped_source_clips=skipped,
                cinematic_filter=cinematic_filter,
            )
            mappings.append(mapping)

    if scheduling_mode in {"whole_line_fill", "window_fill"}:
        _append_short_performance_rescues(
            mappings=mappings,
            usable_clips=usable_clips,
            windows=selected_windows,
            max_time_stretch=max_time_stretch,
            shot_boundary_mode=shot_boundary_mode,
            cinematic_filter=cinematic_filter,
            allow_source_reuse=allow_source_reuse,
        )
        if allow_source_reuse:
            _append_source_exhaustion_reuse_fill(
                mappings=mappings,
                usable_clips=usable_clips,
                windows=selected_windows,
                max_time_stretch=max_time_stretch,
                shot_boundary_mode=shot_boundary_mode,
                cinematic_filter=cinematic_filter,
            )
            _append_undercovered_speech_slot_fill(
                mappings=mappings,
                usable_clips=usable_clips,
                windows=selected_windows,
                max_time_stretch=max_time_stretch,
                shot_boundary_mode=shot_boundary_mode,
                cinematic_filter=cinematic_filter,
            )
        destination_performance_fills = _destination_performance_fills(
            selected_windows,
            mappings,
            source_exhausted=len(mappings) > 0 and len({mapping["clip_id"] for mapping in mappings}) >= len(usable_clips),
            target_coverage=0.9,
        )
        _reanchor_single_slot_mappings_to_speech_start(
            mappings=mappings,
            fills=destination_performance_fills,
        )

    destination_performance_fills = _destination_performance_fills(
        selected_windows,
        mappings,
        source_exhausted=len(mappings) > 0 and len({mapping["clip_id"] for mapping in mappings}) >= len(usable_clips),
        target_coverage=0.9 if scheduling_mode in {"whole_line_fill", "window_fill"} else 0.75,
    )

    clip_use_counts = {
        clip_id: sum(1 for mapping in mappings if str(mapping.get("clip_id")) == clip_id)
        for clip_id in sorted({str(mapping.get("clip_id")) for mapping in mappings})
    }
    repeated_clip_counts = {clip_id: count for clip_id, count in clip_use_counts.items() if count > 1}
    data = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": destination_hash,
        "source_media_hash": source_hash,
        "destination_media_hash": destination_hash,
        "creation_timestamp": utc_now(),
        "scheduling_mode": scheduling_mode,
        "active_filter": active_filter.id,
        "active_filter_display_name": active_filter.display_name,
        "active_filter_notes": active_filter.notes,
        "shot_boundary_mode": shot_boundary_mode,
        "max_time_stretch": max_time_stretch,
        "best_fit_lookahead": best_fit_lookahead if scheduling_mode in {"best_fit", "performance_fill"} else 0,
        "selected_window_count": len(selected_windows),
        "scheduled_window_count": len({mapping["window_id"] for mapping in mappings}),
        "used_clip_count": len({mapping["clip_id"] for mapping in mappings}),
        "source_reuse_policy": "allowed" if allow_source_reuse else "forbidden",
        "reused_clip_placement_count": sum(count - 1 for count in repeated_clip_counts.values()),
        "source_clip_reuse_counts": repeated_clip_counts,
        "source_exhausted": len(mappings) > 0 and len({mapping["clip_id"] for mapping in mappings}) >= len(usable_clips),
        "transformation_name": transformation_name,
        "transformation_history": transformation_history or translation_transformation_plan(),
        "performance_placements": _performance_placements(mappings),
        "destination_performance_fills": destination_performance_fills,
        "mappings": mappings,
    }
    write_json(output_path, data)
    return data


def _annotate_clips_with_source_performances(
    clips: list[dict],
    source_performances: dict[str, Any] | None,
) -> list[dict]:
    groups = _source_performance_clip_groups(clips, source_performances)
    metadata_by_clip_id: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups):
        group_clips = group.get("clips", [])
        for clip in group_clips:
            metadata_by_clip_id[str(clip.get("id"))] = {
                "source_performance_id": group.get("id"),
                "source_performance_type": group.get("conversation_type"),
                "source_performance_clip_count": len(group_clips),
                "source_performance_duration": round(float(group.get("duration", 0.0) or 0.0), 3),
                "source_performance_group_index": group_index,
                "source_performance_turn_count": group.get("estimated_turn_count"),
                "source_performance_dialogue_density": group.get("dialogue_density"),
                "source_performance_signature": group.get("signature", {}),
                "source_speaker_sequence": group.get("speaker_sequence", []),
                "source_turn_pattern": group.get("turn_pattern", ""),
            }
    annotated = []
    for clip in clips:
        item = dict(clip)
        item.update(metadata_by_clip_id.get(str(clip.get("id")), {}))
        annotated.append(item)
    return annotated


def _source_performance_clip_groups(clips: list[dict], source_performances: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not source_performances:
        return [
            {
                "id": f"source_group_{index:06d}",
                "conversation_type": "unknown",
                "clips": [clip],
                "duration": float(clip.get("duration", 0.0) or 0.0),
            }
            for index, clip in enumerate(clips, start=1)
        ]

    remaining = list(clips)
    groups: list[dict[str, Any]] = []
    for performance in source_performances.get("performances", []):
        start = float(performance.get("start", 0.0))
        end = float(performance.get("end", start))
        group_clips = []
        next_remaining = []
        for clip in remaining:
            clip_start = float(clip.get("movie_timestamp", 0.0) or 0.0)
            clip_end = clip_start + float(clip.get("duration", 0.0) or 0.0)
            if max(start, clip_start) < min(end, clip_end):
                group_clips.append(clip)
            else:
                next_remaining.append(clip)
        remaining = next_remaining
        if group_clips:
            groups.append(
                {
                    "id": str(performance.get("id", f"source_group_{len(groups) + 1:06d}")),
                    "conversation_type": performance.get("conversation_type", "unknown"),
                    "clips": group_clips,
                    "duration": round(sum(float(clip.get("duration", 0.0) or 0.0) for clip in group_clips), 3),
                    "source_start": performance.get("start"),
                    "source_end": performance.get("end"),
                    "estimated_turn_count": performance.get("estimated_turn_count"),
                    "dialogue_density": performance.get("dialogue_density"),
                    "signature": performance.get("signature", {}),
                    "speaker_sequence": performance.get("speaker_sequence", []),
                    "turn_pattern": performance.get("turn_pattern", ""),
                }
            )
    for clip in remaining:
        groups.append(
            {
                "id": f"ungrouped_{clip.get('id', len(groups) + 1)}",
                "conversation_type": "unknown",
                "clips": [clip],
                "duration": float(clip.get("duration", 0.0) or 0.0),
            }
        )
    return groups


def _append_performance_fill_mappings(
    *,
    mappings: list[dict],
    clip_groups: list[dict[str, Any]],
    group_index: int,
    window: dict,
    max_time_stretch: float,
    shot_boundary_mode: str,
    best_fit_lookahead: int,
    cinematic_filter: str,
) -> int:
    if group_index >= len(clip_groups):
        return group_index
    chosen_index = _choose_performance_group(
        clip_groups,
        start_index=group_index,
        window=window,
        lookahead=best_fit_lookahead,
        cinematic_filter=cinematic_filter,
    )
    if chosen_index is None:
        return group_index

    skipped_groups = chosen_index - group_index
    skipped_clips = sum(len(group.get("clips", [])) for group in clip_groups[group_index:chosen_index])
    group = clip_groups[chosen_index]
    cursor = float(window.get("start", 0.0))
    window_end = cursor + float(window.get("duration", 0.0) or 0.0)
    for clip_offset, clip in enumerate(group.get("clips", [])):
        duration = float(clip.get("duration", 0.0) or 0.0)
        if duration <= 0 or cursor + duration > window_end + 0.001:
            break
        subwindow = dict(window)
        subwindow["start"] = round(cursor, 3)
        subwindow["duration"] = round(duration, 3)
        subwindow["end"] = round(cursor + duration, 3)
        score_data = _score_candidate(subwindow, clip, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
        mapping = _build_mapping(
            window=subwindow,
            clip=clip,
            max_time_stretch=max_time_stretch,
            score_data=score_data,
            scheduling_mode="performance_fill",
            shot_boundary_mode=shot_boundary_mode,
            selection_reason="source_performance_to_destination_performance",
            skipped_source_clips=skipped_clips if clip_offset == 0 else 0,
            cinematic_filter=cinematic_filter,
        )
        mapping["source_performance_id"] = group.get("id")
        mapping["source_performance_type"] = group.get("conversation_type")
        mapping["source_performance_clip_count"] = len(group.get("clips", []))
        mapping["source_performance_duration"] = round(float(group.get("duration", 0.0) or 0.0), 3)
        similarity = _score_performance_signature_match(group, window, cinematic_filter=cinematic_filter)
        mapping["destination_performance_id"] = window.get("performance_id") or window.get("id")
        mapping["performance_group_index"] = chosen_index
        mapping["performance_similarity_score"] = similarity["score"]
        mapping["performance_similarity_components"] = similarity["components"]
        mapping["speaker_pattern_match"] = similarity["components"].get("speaker_pattern", 0.0)
        mapping["matching_rationale"] = similarity["rationale"]
        mapping["skipped_source_performances"] = skipped_groups if clip_offset == 0 else 0
        mapping["trailing_silence"] = 0.0
        mappings.append(mapping)
        cursor += duration
    return chosen_index + 1


def _choose_performance_group(
    groups: list[dict[str, Any]],
    *,
    start_index: int,
    window: dict,
    lookahead: int,
    cinematic_filter: str,
) -> int | None:
    window_duration = float(window.get("duration", 0.0) or 0.0)
    end_index = min(len(groups), start_index + max(1, lookahead))
    best_index: int | None = None
    best_score = -1.0
    for index in range(start_index, end_index):
        group = groups[index]
        duration = float(group.get("duration", 0.0) or 0.0)
        if duration <= 0 or duration > window_duration + 0.001:
            continue
        score_data = _score_performance_signature_match(group, window, cinematic_filter=cinematic_filter)
        score = score_data["score"]
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _score_clip_signature_match(clip: dict[str, Any], window: dict[str, Any], *, cinematic_filter: str = "balanced") -> dict[str, Any]:
    source_sig = clip.get("source_performance_signature") or {}
    destination_sig = window.get("signature") or {}
    if not source_sig or not destination_sig:
        return {"score": 0.0, "components": {}, "rationale": "signature unavailable"}
    return _score_performance_signature_match(
        {
            "duration": source_sig.get("duration", clip.get("source_performance_duration", clip.get("duration", 0.0))),
            "conversation_type": clip.get("source_performance_type"),
            "signature": source_sig,
            "speaker_sequence": clip.get("source_speaker_sequence", []),
        },
        window,
        cinematic_filter=cinematic_filter,
    )


def _score_performance_signature_match(group: dict[str, Any], window: dict[str, Any], *, cinematic_filter: str = "balanced") -> dict[str, Any]:
    source_sig = group.get("signature") or {}
    destination_sig = window.get("signature") or {}
    source_duration = _float(source_sig.get("duration"), _float(group.get("duration"), 0.0))
    destination_duration = _float(destination_sig.get("duration"), _float(window.get("duration"), 0.0))
    source_type = source_sig.get("performance_type") or group.get("performance_type") or group.get("conversation_type")
    destination_type = destination_sig.get("performance_type") or window.get("performance_type_v2") or window.get("performance_type")
    source_conversation = group.get("conversation_type") or source_sig.get("conversation_type")
    destination_conversation = window.get("performance_type") or destination_sig.get("conversation_type")
    components = {
        "duration": _ratio_similarity(source_duration, destination_duration),
        "speaker_count": _ratio_similarity(_float(source_sig.get("speaker_count"), 1.0), _float(destination_sig.get("speaker_count"), 1.0)),
        "turn_count": _ratio_similarity(_float(source_sig.get("turn_count"), 1.0), _float(destination_sig.get("turn_count"), 1.0)),
        "average_turn_duration": _ratio_similarity(_float(source_sig.get("average_turn_duration"), source_duration), _float(destination_sig.get("average_turn_duration"), destination_duration)),
        "pause": _ratio_similarity(_float(source_sig.get("average_pause_duration"), 0.0) + 0.25, _float(destination_sig.get("average_pause_duration"), 0.0) + 0.25),
        "dialogue_density": 1.0 - min(1.0, abs(_float(source_sig.get("dialogue_density"), 0.0) - _float(destination_sig.get("dialogue_density"), 0.0))),
        "energy": 1.0 - min(1.0, abs(_float(source_sig.get("estimated_energy"), 0.0) - _float(destination_sig.get("estimated_energy"), 0.0))),
        "shot_rate": _ratio_similarity(_float(source_sig.get("shot_change_rate"), 0.0) + 0.05, _float(destination_sig.get("shot_change_rate"), 0.0) + 0.05),
        "conversation_type": _type_similarity(source_conversation, destination_conversation),
        "performance_type": _type_similarity(source_type, destination_type),
        "speaker_pattern": _speaker_pattern_similarity(
            list(source_sig.get("speaker_sequence") or group.get("speaker_sequence") or []),
            list(destination_sig.get("speaker_sequence") or window.get("speaker_sequence") or []),
        ),
        "speech_continuity": _ratio_similarity(_float(source_sig.get("speech_continuity"), 1.0), _float(destination_sig.get("speech_continuity"), 1.0)),
        "response_delay": _ratio_similarity(_float(source_sig.get("response_delay"), _float(source_sig.get("average_pause_duration"), 0.0)) + 0.25, _float(destination_sig.get("response_delay"), _float(destination_sig.get("average_pause_duration"), 0.0)) + 0.25),
        "silence_ratio": 1.0 - min(1.0, abs(_float(source_sig.get("silence_ratio"), 0.0) - _float(destination_sig.get("silence_ratio"), 0.0))),
        "words_per_second": _ratio_similarity(_float(source_sig.get("words_per_second"), 0.0) + 0.25, _float(destination_sig.get("words_per_second"), 0.0) + 0.25),
    }
    baseline_weights = {
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
    }
    baseline_score = sum(components[key] * baseline_weights[key] for key in baseline_weights)
    active_filter = get_filter(cinematic_filter)
    filtered = active_filter.score(components, source_sig, destination_sig)
    rationale = _signature_rationale(components)
    return {
        "score": filtered["score"],
        "baseline_score": round(baseline_score, 4),
        "filter_id": active_filter.id,
        "components": filtered["components"],
        "filter_weights": filtered["weights"],
        "rationale": f"{rationale}; {filtered['explanation']}",
    }


def _ratio_similarity(left: float, right: float) -> float:
    left = max(0.001, float(left))
    right = max(0.001, float(right))
    return max(0.0, min(1.0, 1.0 - abs(left - right) / max(left, right)))


def _type_similarity(source: Any, destination: Any) -> float:
    if not source or not destination:
        return 0.75
    source_value = str(source)
    destination_value = str(destination)
    if source_value == destination_value:
        return 1.0
    compatible = {
        frozenset({"exchange", "dialogue_exchange"}),
        frozenset({"rapid_exchange", "argument"}),
        frozenset({"background_speech", "background_conversation"}),
        frozenset({"group_discussion", "group_conversation"}),
    }
    if frozenset({source_value, destination_value}) in compatible:
        return 0.85
    return 0.55


def _speaker_pattern_similarity(source: list[str], destination: list[str]) -> float:
    if not source or not destination:
        return 0.5
    max_len = max(len(source), len(destination))
    min_len = min(len(source), len(destination))
    matches = sum(1 for index in range(min_len) if source[index] == destination[index])
    length_score = min_len / max_len
    positional = matches / max_len
    alternation_score = 1.0 - min(1.0, abs(_alternation_rate(source) - _alternation_rate(destination)))
    return max(0.0, min(1.0, positional * 0.45 + length_score * 0.25 + alternation_score * 0.3))


def _alternation_rate(sequence: list[str]) -> float:
    if len(sequence) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(sequence, sequence[1:]) if left != right)
    return changes / (len(sequence) - 1)


def _signature_rationale(components: dict[str, float]) -> str:
    strongest = sorted(components.items(), key=lambda item: item[1], reverse=True)[:3]
    weakest = sorted(components.items(), key=lambda item: item[1])[:2]
    return "strong " + ", ".join(key for key, _value in strongest) + "; weak " + ", ".join(key for key, _value in weakest)


def _merge_adjacent_windows(windows: list[dict], *, max_gap: float) -> list[dict]:
    if not windows:
        return []
    merged: list[dict] = []
    current = dict(windows[0])
    current_ids = [str(current.get("id"))]
    current_start = float(current.get("start", 0.0))
    current_end = float(current.get("end", current_start + float(current.get("duration", 0.0))))

    for window in windows[1:]:
        item = dict(window)
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start + float(item.get("duration", 0.0))))
        gap = start - current_end
        if gap <= max_gap:
            current_ids.append(str(item.get("id")))
            current_end = max(current_end, end)
            current["id"] = f"{current_ids[0]}..{current_ids[-1]}"
            current["source_window_ids"] = list(current_ids)
            current["speaking_window_ids"] = list(current.get("speaking_window_ids", [])) + list(item.get("speaking_window_ids", []))
            current["speech_windows"] = list(current.get("speech_windows", [])) + list(item.get("speech_windows", []))
            current["start"] = round(current_start, 3)
            current["end"] = round(current_end, 3)
            current["duration"] = round(max(0.0, current_end - current_start), 3)
            current["crosses_shot_boundary"] = True
        else:
            merged.append(current)
            current = item
            current_ids = [str(current.get("id"))]
            current_start = start
            current_end = end
    merged.append(current)
    return merged


def _append_window_fill_mappings(
    *,
    mappings: list[dict],
    usable_clips: list[dict],
    clip_index: int,
    window: dict,
    max_time_stretch: float,
    shot_boundary_mode: str,
    scheduling_mode: str,
    max_window_duration: float,
    cinematic_filter: str,
) -> int:
    window_start = float(window["start"])
    window_end = window_start + max(float(window.get("duration", 0.0)), 0.0)
    cursor = window_start
    min_remaining = 0.05
    slots = _alignment_slots(window)

    while clip_index < len(usable_clips) and cursor < window_end - min_remaining:
        spans = _alignment_slot_spans(slots, cursor, window_end)
        if not spans:
            break
        chosen_index = None
        slot_start, slot_end, slot_ids = spans[0]
        remaining = 0.0
        for candidate_start, candidate_end, candidate_ids in spans:
            candidate_cursor = max(cursor, candidate_start)
            candidate_remaining = max(0.0, candidate_end - candidate_cursor)
            candidate_index = _find_next_whole_clip_that_fits(
                usable_clips,
                start_index=clip_index,
                remaining=candidate_remaining,
                max_time_stretch=max_time_stretch,
                allow_skip=scheduling_mode == "window_fill" or candidate_remaining >= max_window_duration - 0.001,
                max_window_duration=max_window_duration,
            )
            if candidate_index is not None:
                chosen_index = candidate_index
                cursor = candidate_cursor
                slot_start, slot_end, slot_ids = candidate_start, candidate_end, candidate_ids
                remaining = candidate_remaining
                break
        if chosen_index is None:
            cursor = spans[0][1] + 0.001
            continue

        skipped = chosen_index - clip_index
        clip = usable_clips[chosen_index]
        clip_index = chosen_index + 1
        subwindow = dict(window)
        subwindow["start"] = round(cursor, 3)
        subwindow["duration"] = round(remaining, 3)
        if "end" in subwindow:
            subwindow["end"] = round(slot_end, 3)
        if slot_ids:
            subwindow["alignment_mode"] = "speech_window_snap"
            subwindow["alignment_source_window_ids"] = slot_ids
            subwindow["alignment_source_kind"] = _alignment_source_kind(slots, slot_ids)
            subwindow["alignment_slot_start"] = round(slot_start, 3)
            subwindow["alignment_slot_end"] = round(slot_end, 3)
            subwindow["alignment_spans_speech_windows"] = len(slot_ids) > 1
        else:
            subwindow["alignment_mode"] = "performance_fill_fallback"
            subwindow["alignment_source_window_ids"] = []
            subwindow["alignment_slot_start"] = round(cursor, 3)
            subwindow["alignment_slot_end"] = round(slot_end, 3)
            subwindow["alignment_spans_speech_windows"] = False

        score_data = _score_candidate(subwindow, clip, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
        mapping = _build_mapping(
            window=subwindow,
            clip=clip,
            max_time_stretch=max_time_stretch,
            score_data=score_data,
            scheduling_mode=scheduling_mode,
            shot_boundary_mode=shot_boundary_mode,
            selection_reason="whole_line_fill_destination_window",
            skipped_source_clips=skipped,
            cinematic_filter=cinematic_filter,
        )
        if mapping["timing_strategy"] == "trim_to_window":
            raise RuntimeError("whole-line fill selected a clip that would be trimmed")
        rendered = max(float(mapping.get("planned_render_duration", 0.0)), min_remaining)
        if rendered < remaining and clip_index < len(usable_clips):
            mapping["trailing_silence"] = 0.0
        mappings.append(mapping)
        cursor += rendered

    return clip_index


def _alignment_slots(window: dict) -> list[dict[str, Any]]:
    raw_slots = window.get("speech_windows") or []
    slots = []
    for raw in raw_slots:
        start = float(raw.get("start", 0.0) or 0.0)
        duration = float(raw.get("duration", 0.0) or 0.0)
        end = float(raw.get("end", start + duration) or start + duration)
        if end > start:
            slots.append({"id": str(raw.get("id")), "start": start, "end": end, "source_kind": raw.get("source_kind", "detected_speech_window")})
    slots.sort(key=lambda item: item["start"])
    return slots


def _next_alignment_slot(
    slots: list[dict[str, Any]],
    cursor: float,
    window_end: float,
) -> tuple[float, float, list[str]] | None:
    spans = _alignment_slot_spans(slots, cursor, window_end, max_gap=0.0, max_span_duration=0.0)
    return spans[0] if spans else None


def _alignment_slot_spans(
    slots: list[dict[str, Any]],
    cursor: float,
    window_end: float,
    *,
    max_gap: float = 1.25,
    max_span_duration: float = 8.0,
) -> list[tuple[float, float, list[str]]]:
    if not slots:
        return [(cursor, window_end, [])] if cursor < window_end else []

    start_index = None
    for index, slot in enumerate(slots):
        if float(slot["end"]) <= cursor + 0.001:
            continue
        start_index = index
        break
    if start_index is None:
        return []

    first = slots[start_index]
    span_start = max(cursor, float(first["start"]))
    span_end = min(window_end, float(first["end"]))
    ids = [str(first["id"])]
    spans = [(span_start, span_end, list(ids))]
    if max_gap <= 0.0 or max_span_duration <= 0.0:
        return spans

    previous_end = span_end
    for slot in slots[start_index + 1:]:
        next_start = float(slot["start"])
        next_end = min(window_end, float(slot["end"]))
        if next_end <= next_start:
            continue
        gap = next_start - previous_end
        if gap > max_gap:
            break
        candidate_end = max(span_end, next_end)
        if candidate_end - span_start > max_span_duration:
            break
        ids.append(str(slot["id"]))
        span_end = candidate_end
        previous_end = next_end
        spans.append((span_start, span_end, list(ids)))
        if span_end >= window_end - 0.001:
            break
    return spans


def _alignment_source_kind(slots: list[dict[str, Any]], slot_ids: list[str]) -> str:
    kinds = {
        str(slot.get("source_kind", "detected_speech_window"))
        for slot in slots
        if str(slot.get("id")) in {str(slot_id) for slot_id in slot_ids}
    }
    if not kinds:
        return "none"
    if kinds == {"detected_speech_window"}:
        return "detected_speech_window"
    if "synthetic_speech_slot" in kinds:
        return "synthetic_speech_slot"
    if "recovered_filtered_speech_window" in kinds:
        return "recovered_filtered_speech_window"
    return sorted(kinds)[0]


def _append_short_performance_rescues(
    *,
    mappings: list[dict],
    usable_clips: list[dict],
    windows: list[dict],
    max_time_stretch: float,
    shot_boundary_mode: str,
    max_rescue_duration: float = 8.0,
    cinematic_filter: str = "balanced",
    allow_source_reuse: bool = False,
) -> None:
    used_clip_ids = {str(mapping.get("clip_id")) for mapping in mappings}
    covered_window_ids = {str(mapping.get("window_id")) for mapping in mappings if float(mapping.get("planned_render_duration", 0.0) or 0.0) > 0.0}
    for window in windows:
        window_id = str(window.get("id"))
        window_duration = float(window.get("duration", 0.0) or 0.0)
        if window_id in covered_window_ids or window_duration <= 0.0 or window_duration > max_rescue_duration:
            continue
        rescue_window, candidate, reused_clip = _speech_slot_rescue_candidate(
            window=window,
            clips=usable_clips,
            used_clip_ids=used_clip_ids,
            max_time_stretch=max_time_stretch,
            allow_source_reuse=allow_source_reuse,
        )
        if candidate is None:
            candidate, reused_clip = _find_rescue_clip(
                usable_clips,
                used_clip_ids=used_clip_ids,
                window_duration=window_duration,
                max_time_stretch=max_time_stretch,
                allow_reuse=allow_source_reuse,
            )
            rescue_window = dict(window)
        if candidate is None:
            continue
        score_data = _score_candidate(rescue_window, candidate, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
        mapping = _build_mapping(
            window=rescue_window,
            clip=candidate,
            max_time_stretch=max_time_stretch,
            score_data=score_data,
            scheduling_mode="whole_line_fill_rescue",
            shot_boundary_mode=shot_boundary_mode,
            selection_reason="short_performance_rescue",
            skipped_source_clips=0,
            cinematic_filter=cinematic_filter,
        )
        if mapping["timing_strategy"] == "trim_to_window":
            cursor = max(cursor + 0.001, slot_end + 0.001)
            continue
        mapping["rescue_allowed_reason"] = "otherwise_empty_short_destination_performance"
        mapping["rescue_window_duration"] = round(window_duration, 3)
        mapping["rescue_reused_clip"] = reused_clip
        if reused_clip:
            mapping["selection_reason"] = "short_performance_rescue_reuse"
        mappings.append(mapping)
        used_clip_ids.add(str(candidate.get("id")))
        covered_window_ids.add(window_id)


def _speech_slot_rescue_candidate(
    *,
    window: dict,
    clips: list[dict],
    used_clip_ids: set[str],
    max_time_stretch: float,
    allow_source_reuse: bool = False,
) -> tuple[dict, dict | None, bool]:
    slots = _alignment_slots(window)
    if not slots:
        return dict(window), None, False
    window_start = float(window.get("start", 0.0) or 0.0)
    window_end = window_start + float(window.get("duration", 0.0) or 0.0)
    spans = _alignment_slot_spans(slots, window_start, window_end)
    for slot_start, slot_end, slot_ids in spans:
        span_duration = max(0.0, slot_end - slot_start)
        candidate, reused_clip = _find_rescue_clip(
            clips,
            used_clip_ids=used_clip_ids,
            window_duration=span_duration,
            max_time_stretch=max_time_stretch,
            allow_reuse=allow_source_reuse,
        )
        if candidate is None:
            continue
        rescue_window = dict(window)
        rescue_window["start"] = round(slot_start, 3)
        rescue_window["duration"] = round(span_duration, 3)
        rescue_window["end"] = round(slot_end, 3)
        rescue_window["alignment_mode"] = "speech_window_snap"
        rescue_window["alignment_source_window_ids"] = list(slot_ids)
        rescue_window["alignment_source_kind"] = _alignment_source_kind(slots, slot_ids)
        rescue_window["alignment_slot_start"] = round(slot_start, 3)
        rescue_window["alignment_slot_end"] = round(slot_end, 3)
        rescue_window["alignment_spans_speech_windows"] = len(slot_ids) > 1
        return rescue_window, candidate, reused_clip
    return dict(window), None, False


def _find_rescue_clip(
    clips: list[dict],
    *,
    used_clip_ids: set[str],
    window_duration: float,
    max_time_stretch: float,
    allow_reuse: bool = False,
) -> tuple[dict | None, bool]:
    unused = _rescue_candidates(
        clips,
        used_clip_ids=used_clip_ids,
        window_duration=window_duration,
        max_time_stretch=max_time_stretch,
        include_used=False,
    )
    if unused:
        return unused[0][2], False
    if not allow_reuse:
        return None, False
    reused = _rescue_candidates(
        clips,
        used_clip_ids=used_clip_ids,
        window_duration=window_duration,
        max_time_stretch=max_time_stretch,
        include_used=True,
    )
    if not reused:
        return None, False
    return reused[0][2], True


def _rescue_candidates(
    clips: list[dict],
    *,
    used_clip_ids: set[str],
    window_duration: float,
    max_time_stretch: float,
    include_used: bool,
) -> list[tuple[float, float, dict]]:
    minimum_fit_factor = max(0.001, 1.0 - max_time_stretch)
    candidates = []
    for clip in clips:
        clip_id = str(clip.get("id"))
        if not include_used and clip_id in used_clip_ids:
            continue
        if include_used and clip_id not in used_clip_ids:
            continue
        clip_duration = max(float(clip.get("duration", 0.0) or 0.0), 0.001)
        if clip_duration * minimum_fit_factor <= window_duration + 0.001:
            duration_delta = abs(window_duration - clip_duration)
            candidates.append((duration_delta, clip_duration, clip))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates


def _append_source_exhaustion_reuse_fill(
    *,
    mappings: list[dict],
    usable_clips: list[dict],
    windows: list[dict],
    max_time_stretch: float,
    shot_boundary_mode: str,
    target_coverage: float = 0.9,
    min_destination_duration: float = 4.0,
    cinematic_filter: str = "balanced",
) -> None:
    if not mappings or not usable_clips:
        return
    latest_original_index = max(
        (
            int(mapping.get("source_performance_group_index") or 0)
            for mapping in mappings
            if not mapping.get("rescue_reused_clip")
        ),
        default=-1,
    )
    if latest_original_index < 0:
        return
    used_clip_ids = {str(mapping.get("clip_id")) for mapping in mappings}
    recent_performance_ids = [str(mapping.get("source_performance_id")) for mapping in mappings[-5:] if mapping.get("source_performance_id")]
    fills = _destination_performance_fills(windows, mappings, source_exhausted=True, target_coverage=target_coverage)
    for fill in fills:
        if fill.get("coverage", 0.0) >= target_coverage:
            continue
        if fill.get("stop_reason") not in {"source_dialogue_exhausted", "no_source_line_fit_destination_performance"}:
            continue
        if float(fill.get("duration", 0.0) or 0.0) < min_destination_duration:
            continue
        window = _window_by_destination_id(windows, str(fill["destination_performance_id"]))
        if window is None:
            continue
        if window.get("speech_windows"):
            _append_source_exhaustion_reuse_fill_speech_slots(
                mappings=mappings,
                usable_clips=usable_clips,
                window=window,
                max_time_stretch=max_time_stretch,
                shot_boundary_mode=shot_boundary_mode,
                target_coverage=target_coverage,
                recent_performance_ids=recent_performance_ids,
                used_clip_ids=used_clip_ids,
                cinematic_filter=cinematic_filter,
            )
            continue
        cursor = float(window.get("start", 0.0) or 0.0) + float(fill.get("scheduled_duration", 0.0) or 0.0)
        window_end = float(window.get("start", 0.0) or 0.0) + float(window.get("duration", 0.0) or 0.0)
        safety = 0
        while cursor < window_end - 0.05 and safety < max(10, len(usable_clips) * 20):
            safety += 1
            remaining = window_end - cursor
            candidate = _find_reuse_fill_clip(
                usable_clips,
                remaining=remaining,
                max_time_stretch=max_time_stretch,
                recent_performance_ids=recent_performance_ids,
            )
            if candidate is None:
                break
            subwindow = dict(window)
            subwindow["start"] = round(cursor, 3)
            subwindow["duration"] = round(remaining, 3)
            subwindow["end"] = round(window_end, 3)
            score_data = _score_candidate(subwindow, candidate, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
            mapping = _build_mapping(
                window=subwindow,
                clip=candidate,
                max_time_stretch=max_time_stretch,
                score_data=score_data,
                scheduling_mode="whole_line_fill_reuse",
                shot_boundary_mode=shot_boundary_mode,
                selection_reason="source_exhaustion_reuse_fill",
                skipped_source_clips=0,
                cinematic_filter=cinematic_filter,
            )
            if mapping["timing_strategy"] == "trim_to_window":
                break
            mapping["rescue_reused_clip"] = str(candidate.get("id")) in used_clip_ids
            mapping["reuse_allowed_reason"] = "source_dialogue_exhausted"
            mapping["reuse_source_performance_id"] = candidate.get("source_performance_id")
            mapping["reuse_distance_seconds"] = _reuse_distance_seconds(candidate, mappings, cursor)
            mappings.append(mapping)
            used_clip_ids.add(str(candidate.get("id")))
            if candidate.get("source_performance_id"):
                recent_performance_ids.append(str(candidate.get("source_performance_id")))
                recent_performance_ids = recent_performance_ids[-5:]
            cursor += max(float(mapping.get("planned_render_duration", 0.0) or 0.0), 0.05)
            coverage = (cursor - float(window.get("start", 0.0) or 0.0)) / max(float(window.get("duration", 0.0) or 0.0), 0.001)
            if coverage >= target_coverage:
                break


def _reanchor_single_slot_mappings_to_speech_start(*, mappings: list[dict], fills: list[dict]) -> None:
    slots_by_id = {slot["id"]: slot for fill in fills for slot in fill.get("speech_windows", [])}
    for mapping in mappings:
        if not mapping.get("enabled", True):
            continue
        slot_ids = [str(item) for item in mapping.get("alignment_source_window_ids", [])]
        if not slot_ids:
            continue
        target_slot_id = slot_ids[-1]
        slot = slots_by_id.get(target_slot_id)
        if not slot:
            continue
        slot_start = float(slot["start"])
        if len(slot_ids) > 1 and not _earlier_span_slots_are_covered(slot_ids[:-1], slots_by_id, mappings, exclude=mapping):
            continue
        slot_end = float(slot["end"])
        current_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        duration = float(mapping.get("planned_render_duration", 0.0) or 0.0)
        if duration <= 0.0 or current_start >= slot_start - 0.001:
            continue
        old_coverage = max(0.0, min(slot_end, current_start + duration) - max(slot_start, current_start))
        new_coverage = max(0.0, min(slot_end, slot_start + duration) - slot_start)
        if new_coverage <= old_coverage + 0.001:
            continue
        if _would_overlap_destination_mapping(mapping, mappings, slot_start, slot_start + duration):
            continue
        mapping["destination_timestamp"] = round(slot_start, 3)
        mapping["alignment_source_window_ids"] = [target_slot_id]
        mapping["alignment_slot_start"] = round(slot_start, 3)
        mapping["alignment_slot_end"] = round(slot_end, 3)
        mapping["alignment_spillover_seconds"] = round(max(0.0, slot_start + duration - slot_end), 3)
        mapping["alignment_spans_speech_windows"] = False
        mapping["selection_reason"] = f"{mapping.get('selection_reason', 'speech_slot')}_speech_start_reanchored"
        _update_delay_operation(mapping, slot_start)


def _earlier_span_slots_are_covered(
    slot_ids: list[str],
    slots_by_id: dict[str, dict],
    mappings: list[dict],
    *,
    exclude: dict,
    minimum_coverage: float = 0.8,
) -> bool:
    for slot_id in slot_ids:
        slot = slots_by_id.get(slot_id)
        if not slot:
            return False
        slot_start = float(slot.get("start", 0.0) or 0.0)
        slot_end = float(slot.get("end", slot_start) or slot_start)
        duration = max(0.0, slot_end - slot_start)
        if duration <= 0.0:
            return False
        covered = _speech_slot_covered_duration_excluding(slot_id, slot_start, slot_end, mappings, exclude=exclude)
        if covered / duration < minimum_coverage:
            return False
    return True


def _speech_slot_covered_duration_excluding(
    slot_id: str,
    slot_start: float,
    slot_end: float,
    mappings: list[dict],
    *,
    exclude: dict,
) -> float:
    total = 0.0
    for mapping in mappings:
        if mapping is exclude or not mapping.get("enabled", True):
            continue
        if slot_id not in {str(item) for item in mapping.get("alignment_source_window_ids", [])}:
            continue
        mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        mapping_end = mapping_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
        total += max(0.0, min(slot_end, mapping_end) - max(slot_start, mapping_start))
    return total


def _would_overlap_destination_mapping(target: dict, mappings: list[dict], start: float, end: float) -> bool:
    destination_id = str(target.get("destination_performance_id") or target.get("performance_id") or target.get("window_id"))
    for mapping in mappings:
        if mapping is target or not mapping.get("enabled", True):
            continue
        mapping_destination_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id"))
        if mapping_destination_id != destination_id:
            continue
        mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        mapping_end = mapping_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
        if min(end, mapping_end) - max(start, mapping_start) > 0.001:
            return True
    return False


def _update_delay_operation(mapping: dict, seconds: float) -> None:
    for operation in mapping.get("render_operations", []):
        if operation.get("operation") == "delay":
            operation["seconds"] = round(seconds, 3)
            return


def _append_undercovered_speech_slot_fill(
    *,
    mappings: list[dict],
    usable_clips: list[dict],
    windows: list[dict],
    max_time_stretch: float,
    shot_boundary_mode: str,
    cinematic_filter: str,
    minimum_slot_coverage: float = 0.8,
) -> None:
    if not usable_clips:
        return
    used_clip_ids = {str(mapping.get("clip_id")) for mapping in mappings}
    recent_performance_ids = [
        str(mapping.get("source_performance_id"))
        for mapping in mappings[-10:]
        if mapping.get("source_performance_id")
    ]
    for window in windows:
        destination_id = str(window.get("performance_id") or window.get("id"))
        slots = _alignment_windows(window)
        for slot in slots:
            slot_start = float(slot["start"])
            slot_end = float(slot["end"])
            slot_duration = max(0.0, slot_end - slot_start)
            if slot_duration <= 0.05:
                continue
            safety = 0
            while safety < max(10, len(usable_clips) * 2):
                safety += 1
                covered = _speech_slot_covered_duration(str(slot["id"]), slot_start, slot_end, mappings)
                if covered / slot_duration >= minimum_slot_coverage:
                    break
                occupied = _occupied_segments_for_slot(destination_id, slot_start, slot_end, mappings)
                free_start, free_end = _first_free_segment(slot_start, slot_end, occupied)
                remaining = max(0.0, free_end - free_start)
                if remaining <= 0.05:
                    break
                candidate = _find_reuse_fill_clip(
                    usable_clips,
                    remaining=remaining,
                    max_time_stretch=max_time_stretch,
                    recent_performance_ids=recent_performance_ids,
                )
                if candidate is None:
                    break
                subwindow = dict(window)
                subwindow["id"] = str(slot["id"])
                subwindow["start"] = round(free_start, 3)
                subwindow["duration"] = round(remaining, 3)
                subwindow["end"] = round(free_end, 3)
                subwindow["alignment_mode"] = "speech_window_snap"
                subwindow["alignment_source_window_ids"] = [str(slot["id"])]
                subwindow["alignment_source_kind"] = slot.get("source_kind", "detected_speech_window")
                subwindow["alignment_slot_start"] = round(slot_start, 3)
                subwindow["alignment_slot_end"] = round(slot_end, 3)
                subwindow["alignment_spans_speech_windows"] = False
                score_data = _score_candidate(subwindow, candidate, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
                mapping = _build_mapping(
                    window=subwindow,
                    clip=candidate,
                    max_time_stretch=max_time_stretch,
                    score_data=score_data,
                    scheduling_mode="speech_slot_reuse_fill",
                    shot_boundary_mode=shot_boundary_mode,
                    selection_reason="undercovered_speech_slot_reuse_fill",
                    skipped_source_clips=0,
                    cinematic_filter=cinematic_filter,
                )
                if mapping["timing_strategy"] == "trim_to_window":
                    break
                mapping["rescue_reused_clip"] = str(candidate.get("id")) in used_clip_ids
                mapping["reuse_allowed_reason"] = "undercovered_speech_slot"
                mapping["reuse_source_performance_id"] = candidate.get("source_performance_id")
                mapping["reuse_distance_seconds"] = _reuse_distance_seconds(candidate, mappings, free_start)
                mappings.append(mapping)
                used_clip_ids.add(str(candidate.get("id")))
                if candidate.get("source_performance_id"):
                    recent_performance_ids.append(str(candidate.get("source_performance_id")))
                    del recent_performance_ids[:-5]


def _speech_slot_covered_duration(slot_id: str, slot_start: float, slot_end: float, mappings: list[dict]) -> float:
    total = 0.0
    for mapping in mappings:
        if not mapping.get("enabled", True):
            continue
        if slot_id not in {str(item) for item in mapping.get("alignment_source_window_ids", [])}:
            continue
        mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        mapping_end = mapping_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
        total += max(0.0, min(slot_end, mapping_end) - max(slot_start, mapping_start))
    return total


def _occupied_segments_for_slot(destination_id: str, slot_start: float, slot_end: float, mappings: list[dict]) -> list[tuple[float, float]]:
    occupied = []
    for mapping in mappings:
        if not mapping.get("enabled", True):
            continue
        mapping_destination_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id"))
        if mapping_destination_id != destination_id:
            continue
        mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        mapping_end = mapping_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
        overlap_start = max(slot_start, mapping_start)
        overlap_end = min(slot_end, mapping_end)
        if overlap_end > overlap_start:
            occupied.append((overlap_start, overlap_end))
    return occupied


def _append_source_exhaustion_reuse_fill_speech_slots(
    *,
    mappings: list[dict],
    usable_clips: list[dict],
    window: dict,
    max_time_stretch: float,
    shot_boundary_mode: str,
    target_coverage: float,
    recent_performance_ids: list[str],
    used_clip_ids: set[str],
    cinematic_filter: str,
) -> None:
    destination_id = str(window.get("performance_id") or window.get("id"))
    occupied = [
        (
            float(mapping.get("destination_timestamp", 0.0) or 0.0),
            float(mapping.get("destination_timestamp", 0.0) or 0.0) + float(mapping.get("planned_render_duration", 0.0) or 0.0),
        )
        for mapping in mappings
        if str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id")) == destination_id
    ]
    speech_duration = max(_alignment_target_duration(window), 0.001)
    scheduled = sum(max(0.0, end - start) for start, end in occupied)
    slots = _alignment_slots(window)
    cursor = float(window.get("start", 0.0) or 0.0)
    window_end = float(window.get("start", 0.0) or 0.0) + float(window.get("duration", 0.0) or 0.0)
    safety = 0
    max_iterations = max(10, len(usable_clips) * max(1, len(slots)) * 4)
    while cursor < window_end - 0.05 and safety < max_iterations:
        safety += 1
        if scheduled / speech_duration >= target_coverage:
            break
        spans = _alignment_slot_spans(slots, cursor, window_end)
        if not spans:
            break
        candidate = None
        slot_start, slot_end, slot_ids = spans[0]
        remaining = 0.0
        for candidate_start, candidate_end, candidate_ids in spans:
            free_start, free_end = _first_free_segment(candidate_start, candidate_end, occupied)
            candidate_remaining = max(0.0, free_end - free_start)
            if candidate_remaining <= 0.05:
                continue
            candidate_clip = _find_reuse_fill_clip(
                usable_clips,
                remaining=candidate_remaining,
                max_time_stretch=max_time_stretch,
                recent_performance_ids=recent_performance_ids,
            )
            if candidate_clip is not None:
                candidate = candidate_clip
                slot_start, slot_end, slot_ids = free_start, free_end, candidate_ids
                remaining = candidate_remaining
                break
        if candidate is None:
            cursor = spans[0][1] + 0.001
            continue
        subwindow = dict(window)
        subwindow["start"] = round(slot_start, 3)
        subwindow["duration"] = round(remaining, 3)
        subwindow["end"] = round(slot_end, 3)
        subwindow["alignment_mode"] = "speech_window_snap"
        subwindow["alignment_source_window_ids"] = list(slot_ids)
        subwindow["alignment_source_kind"] = _alignment_source_kind(slots, slot_ids)
        subwindow["alignment_slot_start"] = round(slot_start, 3)
        subwindow["alignment_slot_end"] = round(slot_end, 3)
        subwindow["alignment_spans_speech_windows"] = len(slot_ids) > 1
        score_data = _score_candidate(subwindow, candidate, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
        mapping = _build_mapping(
            window=subwindow,
            clip=candidate,
            max_time_stretch=max_time_stretch,
            score_data=score_data,
            scheduling_mode="whole_line_fill_reuse",
            shot_boundary_mode=shot_boundary_mode,
            selection_reason="source_exhaustion_reuse_fill",
            skipped_source_clips=0,
            cinematic_filter=cinematic_filter,
        )
        if mapping["timing_strategy"] == "trim_to_window":
            cursor = max(cursor + 0.001, slot_end + 0.001)
            continue
        mapping["rescue_reused_clip"] = str(candidate.get("id")) in used_clip_ids
        mapping["reuse_allowed_reason"] = "source_dialogue_exhausted"
        mapping["reuse_source_performance_id"] = candidate.get("source_performance_id")
        mapping["reuse_distance_seconds"] = _reuse_distance_seconds(candidate, mappings, slot_start)
        mappings.append(mapping)
        used_clip_ids.add(str(candidate.get("id")))
        if candidate.get("source_performance_id"):
            recent_performance_ids.append(str(candidate.get("source_performance_id")))
            del recent_performance_ids[:-5]
        rendered = max(float(mapping.get("planned_render_duration", 0.0) or 0.0), 0.05)
        occupied.append((slot_start, slot_start + rendered))
        scheduled += rendered
        cursor = slot_start + rendered


def _first_free_segment(
    start: float,
    end: float,
    occupied: list[tuple[float, float]],
    *,
    min_gap: float = 0.05,
) -> tuple[float, float]:
    cursor = float(start)
    for occupied_start, occupied_end in sorted(occupied, key=lambda item: item[0]):
        if occupied_end <= cursor + min_gap:
            continue
        if occupied_start <= cursor + min_gap:
            cursor = max(cursor, occupied_end)
            if cursor >= end - min_gap:
                return end, end
            continue
        if occupied_start < end - min_gap:
            return cursor, min(end, occupied_start)
        break
    return cursor, end


def _window_by_destination_id(windows: list[dict], destination_id: str) -> dict | None:
    for window in windows:
        if str(window.get("performance_id") or window.get("id")) == destination_id:
            return window
    return None


def _find_reuse_fill_clip(
    clips: list[dict],
    *,
    remaining: float,
    max_time_stretch: float,
    recent_performance_ids: list[str],
) -> dict | None:
    minimum_fit_factor = max(0.001, 1.0 - max_time_stretch)
    candidates = []
    for clip in clips:
        clip_duration = max(float(clip.get("duration", 0.0) or 0.0), 0.001)
        if clip_duration * minimum_fit_factor > remaining + 0.001:
            continue
        source_performance_id = str(clip.get("source_performance_id") or "")
        recent_penalty = 1 if source_performance_id in recent_performance_ids[-2:] else 0
        duration_delta = abs(remaining - clip_duration)
        candidates.append((recent_penalty, duration_delta, -clip_duration, clip))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _reuse_distance_seconds(candidate: dict, mappings: list[dict], destination_timestamp: float) -> float | None:
    previous = [
        abs(destination_timestamp - float(mapping.get("destination_timestamp", 0.0) or 0.0))
        for mapping in mappings
        if mapping.get("clip_id") == candidate.get("id")
    ]
    if not previous:
        return None
    return round(min(previous), 3)


def _find_next_whole_clip_that_fits(
    clips: list[dict],
    *,
    start_index: int,
    remaining: float,
    max_time_stretch: float,
    allow_skip: bool,
    max_window_duration: float,
) -> int | None:
    minimum_fit_factor = max(0.001, 1.0 - max_time_stretch)
    current_duration = max(float(clips[start_index].get("duration", 0.0)), 0.001)
    if current_duration * minimum_fit_factor <= remaining + 0.001:
        return start_index
    if not allow_skip and current_duration * minimum_fit_factor <= max_window_duration + 0.001:
        return None
    if not allow_skip:
        return start_index + 1 if start_index + 1 < len(clips) else None
    for index in range(start_index + 1, len(clips)):
        clip_duration = max(float(clips[index].get("duration", 0.0)), 0.001)
        if clip_duration * minimum_fit_factor <= remaining + 0.001:
            return index
    return None


def _choose_best_fit(
    clips: list[dict],
    *,
    start_index: int,
    window: dict,
    lookahead: int,
    max_time_stretch: float,
    shot_boundary_mode: str,
) -> tuple[dict, int, dict[str, Any]]:
    end_index = min(len(clips), start_index + max(1, lookahead))
    best_clip = clips[start_index]
    best_index = start_index
    best_score = _score_candidate(window, best_clip, max_time_stretch, shot_boundary_mode=shot_boundary_mode)

    for index in range(start_index + 1, end_index):
        candidate = clips[index]
        candidate_score = _score_candidate(window, candidate, max_time_stretch, shot_boundary_mode=shot_boundary_mode)
        if candidate_score["score"] > best_score["score"]:
            best_clip = candidate
            best_index = index
            best_score = candidate_score

    return best_clip, best_index, best_score


def _build_mapping(
    *,
    window: dict,
    clip: dict,
    max_time_stretch: float,
    score_data: dict[str, Any],
    scheduling_mode: str,
    shot_boundary_mode: str,
    selection_reason: str,
    skipped_source_clips: int,
    cinematic_filter: str = "balanced",
) -> dict:
    window_duration = float(window["duration"])
    timing = predicted_render_timing(window, clip, max_time_stretch)
    stretch_factor = float(timing["stretch_factor"])
    trim_duration = float(timing["trim_duration"])
    rendered_duration = float(timing["rendered_duration"])
    timing_strategy = str(timing["timing_strategy"])

    if shot_boundary_mode == "strict" and window.get("shot_end") is not None and not window.get("crosses_shot_boundary"):
        available = max(0.0, float(window["shot_end"]) - float(window.get("start", 0.0)))
        if available > 0.0 and rendered_duration > available + 0.001:
            trim_duration = min(trim_duration, available / max(stretch_factor, 0.001))
            rendered_duration = trim_duration * stretch_factor
            timing_strategy = f"{timing_strategy}_shot_limited"

    trailing = max(0.0, window_duration - rendered_duration)
    render_operations = [
        {"operation": "trim", "start": 0.0, "duration": round(trim_duration, 3)},
    ]
    if abs(stretch_factor - 1.0) > 0.001:
        render_operations.append({"operation": "time_stretch", "factor": round(stretch_factor, 4), "preserve_pitch": True})
    render_operations.extend(
        [
            {"operation": "normalize_loudness", "target_lufs": None},
            {"operation": "fade_in_out", "duration": None},
            {"operation": "delay", "seconds": round(float(window["start"]), 3)},
            {"operation": "limit", "peak_limit": 0.95},
        ]
    )
    mapping_start = float(window.get("start", 0.0))
    mapping_end = mapping_start + rendered_duration
    shot_end = window.get("shot_end")
    boundary_overrun = max(0.0, mapping_end - float(shot_end)) if shot_end is not None else 0.0
    mapping_crosses = bool(boundary_overrun > 0.001)
    performance_similarity = _score_clip_signature_match(clip, window, cinematic_filter=cinematic_filter)
    return {
        "window_id": window["id"],
        "performance_id": window.get("performance_id"),
        "performance_type": window.get("performance_type"),
        "performance_dialogue_density": window.get("dialogue_density"),
        "performance_visible_windows": window.get("visible_speaking_window_count"),
        "performance_shot_count": window.get("shot_count"),
        "source_performance_id": clip.get("source_performance_id"),
        "source_performance_type": clip.get("source_performance_type"),
        "source_performance_clip_count": clip.get("source_performance_clip_count"),
        "source_performance_duration": clip.get("source_performance_duration"),
        "source_performance_group_index": clip.get("source_performance_group_index"),
        "source_performance_turn_count": clip.get("source_performance_turn_count"),
        "source_performance_dialogue_density": clip.get("source_performance_dialogue_density"),
        "source_performance_signature": clip.get("source_performance_signature"),
        "source_speaker_sequence": clip.get("source_speaker_sequence"),
        "source_turn_pattern": clip.get("source_turn_pattern"),
        "destination_performance_signature": window.get("signature"),
        "destination_speaker_sequence": window.get("speaker_sequence"),
        "destination_turn_pattern": window.get("turn_pattern"),
        "source_speaker_id": clip.get("speaker_id") or clip.get("speaker"),
        "destination_speaker_id": window.get("speaker_id") or window.get("speaker") or window.get("dominant_speaker_id"),
        "speaker_match_preserved": bool((clip.get("speaker_id") or clip.get("speaker")) and (window.get("speaker_id") or window.get("speaker") or window.get("dominant_speaker_id")) and (clip.get("speaker_id") or clip.get("speaker")) == (window.get("speaker_id") or window.get("speaker") or window.get("dominant_speaker_id"))),
        "speaker_fallback_reason": None if not (clip.get("speaker_id") or clip.get("speaker")) or not (window.get("speaker_id") or window.get("speaker") or window.get("dominant_speaker_id")) or (clip.get("speaker_id") or clip.get("speaker")) == (window.get("speaker_id") or window.get("speaker") or window.get("dominant_speaker_id")) else "timing_fit_overrode_speaker",
        "destination_performance_id": window.get("performance_id") or window.get("id"),
        "clip_id": clip["id"],
        "clip_path": clip["path"],
        "clip_movie_timestamp": clip.get("movie_timestamp"),
        "source_movie_timestamp": clip.get("movie_timestamp"),
        "enabled": True,
        "destination_timestamp": window["start"],
        "stretch_factor": round(stretch_factor, 4),
        "clip_trim_start": 0.0,
        "clip_trim_duration": round(trim_duration, 3),
        "leading_silence": 0.0,
        "trailing_silence": round(trailing, 3),
        "planned_render_duration": round(rendered_duration, 3),
        "render_operations": render_operations,
        "score": score_data["score"],
        "score_components": score_data["components"],
        "selection_reason": selection_reason,
        "scheduling_mode": scheduling_mode,
        "shot_boundary_mode": shot_boundary_mode,
        "timing_strategy": timing_strategy,
        "skipped_source_clips": skipped_source_clips,
        "source_transcript": clip.get("transcript", ""),
        "primary_shot_id": window.get("primary_shot_id") or window.get("shot_id"),
        "shot_id": window.get("shot_id") or window.get("primary_shot_id"),
        "shot_start": window.get("shot_start"),
        "shot_end": window.get("shot_end"),
        "crosses_shot_boundary": bool(window.get("crosses_shot_boundary", False)),
        "boundary_overlap_seconds": round(float(window.get("boundary_overlap_seconds") or 0.0), 3),
        "mapping_crosses_shot_boundary": mapping_crosses,
        "boundary_overrun_seconds": round(boundary_overrun, 3),
        "visual_fit_score": score_data["components"].get("visual_fit", 1.0),
        "baseline_similarity_score": performance_similarity.get("baseline_score", performance_similarity["score"]),
        "performance_similarity_score": performance_similarity["score"],
        "active_filter": performance_similarity.get("filter_id", cinematic_filter),
        "performance_similarity_components": performance_similarity["components"],
        "filter_weights": performance_similarity.get("filter_weights", {}),
        "speaker_pattern_match": performance_similarity["components"].get("speaker_pattern", 0.0),
        "matching_rationale": performance_similarity["rationale"],
        "alignment_mode": window.get("alignment_mode", "performance_fill_fallback"),
        "alignment_source_window_ids": window.get("alignment_source_window_ids", []),
        "alignment_source_kind": window.get("alignment_source_kind", "none"),
        "alignment_slot_start": window.get("alignment_slot_start"),
        "alignment_slot_end": window.get("alignment_slot_end"),
        "alignment_spillover_seconds": round(max(0.0, mapping_end - float(window.get("alignment_slot_end", mapping_end) or mapping_end)), 3),
        "alignment_spans_speech_windows": bool(window.get("alignment_spans_speech_windows", False)),
    }


def _destination_performance_fills(
    windows: list[dict],
    mappings: list[dict],
    *,
    source_exhausted: bool,
    target_coverage: float = 0.75,
) -> list[dict[str, Any]]:
    mappings_by_destination: dict[str, list[dict]] = {}
    for mapping in mappings:
        destination_id = str(mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id"))
        mappings_by_destination.setdefault(destination_id, []).append(mapping)

    rows = []
    for window in windows:
        destination_id = str(window.get("performance_id") or window.get("id"))
        destination_mappings = mappings_by_destination.get(destination_id, [])
        duration = max(0.0, float(window.get("duration", 0.0) or 0.0))
        speech_duration = _alignment_target_duration(window)
        speech_windows = _alignment_windows(window)
        speech_window_ids = [str(slot["id"]) for slot in speech_windows]
        covered_speech_window_ids = sorted(
            {
                str(window_id)
                for mapping in destination_mappings
                for window_id in mapping.get("alignment_source_window_ids", [])
            }
        )
        uncovered_speech_window_ids = sorted(set(speech_window_ids) - set(covered_speech_window_ids))
        target_duration = speech_duration if speech_duration > 0 else duration
        scheduled = round(covered_speech_duration(destination_mappings, speech_windows), 3)
        coverage = min(1.0, scheduled / target_duration) if target_duration > 0 else 0.0
        if coverage >= target_coverage:
            stop_reason = "target_coverage_met"
        elif not destination_mappings:
            stop_reason = "no_source_line_fit_destination_performance"
        elif source_exhausted:
            stop_reason = "source_dialogue_exhausted"
        else:
            stop_reason = "remaining_gap_has_no_fitting_whole_line"
        rows.append(
            {
                "destination_performance_id": destination_id,
                "destination_performance_type": window.get("performance_type"),
                "start": window.get("start"),
                "duration": round(duration, 3),
                "speech_duration": round(speech_duration, 3),
                "speech_windows": speech_windows,
                "speech_window_ids": speech_window_ids,
                "speech_window_count": len(speech_window_ids),
                "covered_speech_window_count": len(covered_speech_window_ids),
                "covered_speech_window_ids": covered_speech_window_ids,
                "uncovered_speech_window_ids": uncovered_speech_window_ids,
                "uncovered_speech_window_count": len(uncovered_speech_window_ids),
                "coverage_basis": "speech_windows" if speech_duration > 0 else "performance_duration",
                "scheduled_duration": scheduled,
                "coverage": round(coverage, 4),
                "target_coverage": target_coverage,
                "mapping_count": len(destination_mappings),
                "source_performance_ids": sorted(
                    {str(mapping.get("source_performance_id")) for mapping in destination_mappings if mapping.get("source_performance_id")}
                ),
                "stop_reason": stop_reason,
            }
        )
    return rows


def _alignment_target_duration(window: dict) -> float:
    total = 0.0
    for slot in window.get("speech_windows") or []:
        start = float(slot.get("start", 0.0) or 0.0)
        duration = float(slot.get("duration", 0.0) or 0.0)
        end = float(slot.get("end", start + duration) or start + duration)
        total += max(0.0, end - start)
    return round(total, 3)


def _alignment_window_ids(window: dict) -> list[str]:
    return [str(slot["id"]) for slot in _alignment_windows(window)]


def _alignment_windows(window: dict) -> list[dict[str, Any]]:
    rows = []
    for slot in window.get("speech_windows") or []:
        slot_id = slot.get("id")
        if slot_id is None:
            continue
        start = _float(slot.get("start"), 0.0)
        duration = _float(slot.get("duration"), 0.0)
        end = _float(slot.get("end"), start + duration)
        duration = max(0.0, end - start)
        if duration <= 0.0:
            continue
        rows.append(
            {
                "id": str(slot_id),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "source_kind": slot.get("source_kind", "detected_speech_window"),
            }
        )
    return rows


def _performance_placements(mappings: list[dict]) -> list[dict[str, Any]]:
    placements: dict[tuple[str, str], dict[str, Any]] = {}
    for mapping in mappings:
        source_id = mapping.get("source_performance_id") or "unknown_source_performance"
        destination_id = mapping.get("destination_performance_id") or mapping.get("performance_id") or mapping.get("window_id")
        key = (str(source_id), str(destination_id))
        row = placements.setdefault(
            key,
            {
                "source_performance_id": source_id,
                "source_performance_type": mapping.get("source_performance_type"),
                "destination_performance_id": destination_id,
                "destination_performance_type": mapping.get("performance_type"),
                "clip_ids": [],
                "mapping_count": 0,
                "scheduled_duration": 0.0,
            },
        )
        row["clip_ids"].append(mapping.get("clip_id"))
        row["mapping_count"] += 1
        row["scheduled_duration"] = round(row["scheduled_duration"] + float(mapping.get("planned_render_duration", 0.0) or 0.0), 3)
    return list(placements.values())


def _score_candidate(window: dict, clip: dict, max_time_stretch: float, *, shot_boundary_mode: str = "off") -> dict[str, Any]:
    window_duration = max(float(window.get("duration", 0.0)), 0.001)
    clip_duration = max(float(clip.get("duration", 0.0)), 0.001)
    duration_delta = abs(window_duration - clip_duration)
    duration_similarity = max(0.0, 1.0 - (duration_delta / max(window_duration, clip_duration)))

    required_factor = window_duration / clip_duration
    stretch_overage = max(0.0, abs(required_factor - 1.0) - max_time_stretch)
    stretch_fit = max(0.0, 1.0 - (stretch_overage / max(1.0, max_time_stretch)))

    trim_ratio = max(0.0, clip_duration - window_duration) / clip_duration
    trim_fit = max(0.0, 1.0 - trim_ratio)

    confidence = _bounded_float(clip.get("confidence"), default=0.7)
    speech_rate = float(clip.get("speech_rate") or 0.0)
    speech_rate_fit = 1.0 if 0.5 <= speech_rate <= 5.0 else 0.75 if speech_rate > 0 else 0.6
    loudness_fit = 1.0 if clip.get("average_loudness") is not None else 0.8
    visual_fit = visual_fit_for_candidate(
        window,
        clip,
        max_time_stretch=max_time_stretch,
        shot_boundary_mode=shot_boundary_mode,
    )["visual_fit_score"]

    components = {
        "duration_similarity": round(duration_similarity, 4),
        "stretch_fit": round(stretch_fit, 4),
        "trim_fit": round(trim_fit, 4),
        "confidence": round(confidence, 4),
        "speech_rate_fit": round(speech_rate_fit, 4),
        "loudness_fit": round(loudness_fit, 4),
        "visual_fit": round(float(visual_fit), 4),
        "performance_duration": round(float(window.get("duration", 0.0)), 3),
        "performance_density": round(float(window.get("dialogue_density", 0.0) or 0.0), 4),
        "performance_turns": int(window.get("visible_speaking_window_count", 1) or 1),
    }
    if shot_boundary_mode == "off":
        score = (
            duration_similarity * 0.42
            + stretch_fit * 0.22
            + trim_fit * 0.14
            + confidence * 0.12
            + speech_rate_fit * 0.06
            + loudness_fit * 0.04
        )
    else:
        score = (
            duration_similarity * 0.25
            + stretch_fit * 0.10
            + trim_fit * 0.08
            + confidence * 0.08
            + speech_rate_fit * 0.03
            + loudness_fit * 0.02
            + float(visual_fit) * 0.44
        )
    return {"score": round(score, 4), "components": components}


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

