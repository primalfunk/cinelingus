from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .filters import usable_rows
from .render import mux_video, render_dialogue_wav, render_schedule_over_original_audio
from .schedule import build_schedule
from .speakers import speaker_preservation_summary
from .transformation_verbs import self_shuffle_transformation_plan, shuffle_selection
from .util import utc_now, write_json
from .filter_lab.registry import default_filter_registry


@dataclass(frozen=True)
class MutationDefinition:
    id: str
    display_name: str
    description: str
    default_parameters: dict[str, Any]


MUTATIONS: dict[str, MutationDefinition] = {
    definition.implementation_key: MutationDefinition(
        id=definition.implementation_key,
        display_name=definition.name,
        description=definition.summary,
        default_parameters=definition.parameter_defaults,
    )
    for definition in default_filter_registry().definitions(implemented_only=True)
    if definition.implementation_key and definition.family_id != "multiworld"
}

MUTATION_CHOICES = tuple(MUTATIONS.keys())
MUTATION_DISPLAY_NAMES = {key: value.display_name for key, value in MUTATIONS.items()}


def get_mutation(mutation_id: str) -> MutationDefinition:
    if mutation_id not in MUTATIONS:
        choices = ", ".join(MUTATION_CHOICES)
        raise ValueError(f"Unknown mutation '{mutation_id}'. Available mutations: {choices}")
    return MUTATIONS[mutation_id]


def build_mutation_plan(
    *,
    mutation_id: str,
    source_media_hash: str,
    source_path: Path,
    selected_objects: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    render_strategy: dict[str, Any],
    expected_output_path: Path,
    output_path: Path,
    parameters: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    definition = get_mutation(mutation_id)
    plan = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "mutation_id": definition.id,
        "mutation_name": definition.display_name,
        "source_media_hash": source_media_hash,
        "source_path": str(source_path),
        "parameters": parameters,
        "selected_objects": selected_objects,
        "operation_list": operations,
        "placement_plan": placements,
        "render_strategy": render_strategy,
        "warnings": warnings or [],
        "expected_output_path": str(expected_output_path),
    }
    write_json(output_path, plan)
    return plan


def build_mutation_report(
    *,
    mutation_id: str,
    source_path: Path,
    source_media_hash: str,
    parameters: dict[str, Any],
    plan_path: Path,
    output_video: Path,
    output_audio: Path,
    schedule: dict[str, Any],
    output_path: Path,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    mappings = schedule.get("mappings", [])
    unchanged = [mapping for mapping in mappings if mapping.get("self_shuffle_unchanged_line")]
    definition = get_mutation(mutation_id)
    filter_definition = default_filter_registry().get(mutation_id)
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "source_film": str(source_path),
        "source_media_hash": source_media_hash,
        "selected_filter": {
            "id": filter_definition.id,
            "name": filter_definition.name,
            "family": filter_definition.family_id,
            "version": filter_definition.version,
            "reads_dimensions": [item.value for item in filter_definition.reads_dimensions],
            "changes_dimensions": [item.value for item in filter_definition.changes_dimensions],
        },
        "mutation_filter": {
            "id": definition.id,
            "display_name": definition.display_name,
            "description": definition.description,
        },
        "parameters": parameters,
        "counts": {
            "selected_objects": len(mappings),
            "moved_objects": sum(1 for mapping in mappings if mapping.get("mutation_operation") in {"drift", "self_shuffle"}),
            "unchanged_objects": len(unchanged),
            "repeated_objects": sum(1 for mapping in mappings if mapping.get("mutation_operation") == "echo"),
            "altered_pauses": 0,
        },
        "outputs": {
            "audio": str(output_audio),
            "video": str(output_video),
            "output_duration": schedule.get("render_duration"),
        },
        "artifacts": {
            "mutation_plan": str(plan_path),
            "mutation_report": str(output_path),
            "filter_recipe": str(schedule.get("filter_recipe_path") or ""),
            "filter_plan": str(schedule.get("filter_plan_path") or ""),
            "filter_acceptance": str(schedule.get("filter_acceptance_path") or ""),
        },
        "warnings": warnings or [],
        "unchanged_line_policy": {
            "self_shuffle_requires_changed_lines": mutation_id == "self_shuffle",
            "unchanged_line_count": len(unchanged),
            "disabled_unchanged_line_count": sum(1 for mapping in unchanged if not mapping.get("enabled", True)),
        },
        "speaker_summary": speaker_preservation_summary(schedule),
        "review_summary": {},
        "analysis_artifacts_used": list(filter_definition.required_artifacts),
        "requested_analysis_backends": schedule.get("requested_analysis_backends", {}),
        "actual_analysis_backends": schedule.get("actual_analysis_backends", {}),
        "filter_metrics": schedule.get("filter_metrics", {}),
        "filter_validation": schedule.get("filter_validation", {}),
        "transformation_summary": schedule.get("filter_summary") or f"{filter_definition.name} created {len(mappings)} dialogue mappings.",
    }
    write_json(output_path, report)
    return report


def build_echo_schedule(*, clips: list[dict[str, Any]], duration: float, parameters: dict[str, Any]) -> dict[str, Any]:
    delay = float(parameters.get("delay_seconds", 18.0))
    frequency = max(1, int(parameters.get("repeat_frequency", 5)))
    max_repeats = max(0, int(parameters.get("max_repeats", 24)))
    selected = [clip for index, clip in enumerate(clips) if index % frequency == 0 and float(clip.get("duration", 0.0) or 0.0) > 0]
    mappings = []
    for clip in selected[:max_repeats]:
        source_start = float(clip.get("movie_timestamp", 0.0) or 0.0)
        clip_duration = float(clip.get("duration", 0.0) or 0.0)
        destination = source_start + delay
        if destination + min(clip_duration, 0.1) > duration:
            continue
        mappings.append(_mutation_mapping(clip, destination, clip_duration, "echo", "repeat_line_after_delay"))
    schedule = _mutation_schedule("echo", duration, mappings)
    schedule["filter_validation"] = {
        "passed": bool(mappings),
        "echo_delay_matches_parameter": all(abs(float(row["destination_timestamp"]) - float(row["source_movie_timestamp"]) - delay) < 0.002 for row in mappings),
        "repeat_limit_is_respected": len(mappings) <= max_repeats,
    }
    schedule["filter_metrics"] = {"repeated_objects": len(mappings), "configured_delay": delay}
    return schedule


def build_drift_schedule(*, clips: list[dict[str, Any]], duration: float, parameters: dict[str, Any]) -> dict[str, Any]:
    start_offset = float(parameters.get("starting_offset", 0.25))
    max_offset = float(parameters.get("maximum_offset", 8.0))
    mappings = []
    for clip in clips:
        source_start = float(clip.get("movie_timestamp", 0.0) or 0.0)
        clip_duration = float(clip.get("duration", 0.0) or 0.0)
        if clip_duration <= 0:
            continue
        progress = min(1.0, max(0.0, source_start / max(duration, 0.001)))
        offset = start_offset + (max_offset - start_offset) * progress
        destination = min(max(0.0, source_start + offset), max(0.0, duration - clip_duration))
        mappings.append(_mutation_mapping(clip, destination, clip_duration, "drift", "progressive_dialogue_delay"))
    schedule = _mutation_schedule("drift", duration, mappings)
    offsets = [float(row["destination_timestamp"]) - float(row["source_movie_timestamp"]) for row in mappings]
    non_decreasing = all(right + 0.002 >= left for left, right in zip(offsets, offsets[1:]))
    schedule["filter_validation"] = {
        "passed": bool(mappings) and non_decreasing,
        "offset_is_non_decreasing": non_decreasing,
        "source_line_identity_is_preserved": all(row.get("clip_id") for row in mappings),
    }
    schedule["filter_metrics"] = {
        "average_temporal_displacement": round(sum(offsets) / len(offsets), 3) if offsets else 0.0,
        "maximum_temporal_displacement": round(max(offsets), 3) if offsets else 0.0,
    }
    return schedule


def build_self_shuffle_schedule(
    *,
    clips: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    media_hash: str,
    max_time_stretch: float,
    output_path: Path,
    seed: int,
    best_fit_lookahead: int,
    cinematic_filter: str,
    source_performances: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usable_windows = usable_rows(windows)
    shuffled = speaker_aware_shuffle_selection(clips, usable_windows, seed=seed)
    schedule = build_schedule(
        clips=shuffled,
        windows=usable_windows,
        source_hash=media_hash,
        destination_hash=media_hash,
        max_time_stretch=max_time_stretch,
        output_path=output_path,
        scheduling_mode="whole_line_fill",
        best_fit_lookahead=best_fit_lookahead,
        transformation_name="mutation_self_shuffle",
        transformation_history=self_shuffle_transformation_plan(),
        cinematic_filter=cinematic_filter,
        source_performances=source_performances,
    )
    schedule["mutation_id"] = "self_shuffle"
    schedule["self_shuffle_render_strategy"] = "dialogue_only_v1"
    schedule["render_duration"] = None
    for mapping in schedule.get("mappings", []):
        mapping["mutation_operation"] = "self_shuffle"
    _mark_speaker_shuffle_fallbacks(schedule)
    enforce_self_shuffle_changed_lines(schedule=schedule, clips=clips)
    return schedule


def speaker_aware_shuffle_selection(clips: list[dict[str, Any]], windows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    shuffled = shuffle_selection(clips, seed=seed)
    if not any(clip.get("speaker_id") or clip.get("speaker") for clip in shuffled):
        return shuffled
    remaining = list(shuffled)
    selected: list[dict[str, Any]] = []
    for window in windows:
        destination_speaker = window.get("speaker_id") or window.get("speaker") or window.get("dominant_speaker_id")
        chosen_index = None
        if destination_speaker:
            for index, clip in enumerate(remaining):
                if (clip.get("speaker_id") or clip.get("speaker")) == destination_speaker:
                    chosen_index = index
                    break
        if chosen_index is None and remaining:
            chosen_index = 0
        if chosen_index is None:
            break
        selected.append(remaining.pop(chosen_index))
    selected.extend(remaining)
    return selected


def enforce_self_shuffle_changed_lines(*, schedule: dict[str, Any], clips: list[dict[str, Any]]) -> dict[str, Any]:
    """Self Shuffle must not leave a spoken line in its original spoken slot."""
    mappings = schedule.get("mappings", [])
    clips_by_id = {str(clip.get("id")): clip for clip in clips}
    used_clip_ids = {str(mapping.get("clip_id")) for mapping in mappings if mapping.get("enabled", True)}
    repaired = 0
    disabled = 0
    for mapping in mappings:
        if not mapping.get("enabled", True):
            continue
        if not _self_shuffle_mapping_unchanged(mapping, clips_by_id.get(str(mapping.get("clip_id")))):
            mapping["self_shuffle_unchanged_line"] = False
            mapping["self_shuffle_policy"] = "changed_line_required"
            continue
        replacement = _find_changed_line_replacement(mapping, clips, used_clip_ids)
        if replacement is None:
            mapping["enabled"] = False
            mapping["self_shuffle_unchanged_line"] = True
            mapping["selection_reason"] = f"{mapping.get('selection_reason', 'self_shuffle')}_disabled_original_line"
            mapping["self_shuffle_policy"] = "disabled_because_original_line_would_remain"
            disabled += 1
            continue
        old_clip_id = str(mapping.get("clip_id"))
        used_clip_ids.discard(old_clip_id)
        used_clip_ids.add(str(replacement.get("id")))
        _replace_mapping_clip(mapping, replacement)
        mapping["self_shuffle_unchanged_line"] = False
        mapping["self_shuffle_policy"] = "changed_line_required"
        mapping["selection_reason"] = f"{mapping.get('selection_reason', 'self_shuffle')}_changed_line_repair"
        repaired += 1
    schedule["self_shuffle_policy"] = {
        "requires_changed_lines": True,
        "repaired_unchanged_mappings": repaired,
        "disabled_unchanged_mappings": disabled,
    }
    enabled = [row for row in mappings if row.get("enabled", True)]
    schedule["filter_validation"] = {
        "passed": bool(enabled) and not any(row.get("self_shuffle_unchanged_line") for row in enabled),
        "no_enabled_line_remains_in_original_slot": not any(row.get("self_shuffle_unchanged_line") for row in enabled),
        "all_clips_come_from_same_film": True,
    }
    return schedule


def _find_changed_line_replacement(mapping: dict[str, Any], clips: list[dict[str, Any]], used_clip_ids: set[str]) -> dict[str, Any] | None:
    destination_speaker = mapping.get("destination_speaker_id")
    candidates = []
    for clip in clips:
        clip_id = str(clip.get("id"))
        if clip_id in used_clip_ids:
            continue
        if _self_shuffle_mapping_unchanged(mapping, clip):
            continue
        speaker = clip.get("speaker_id") or clip.get("speaker")
        speaker_match = bool(destination_speaker and speaker == destination_speaker)
        duration_fit = _duration_fit(float(clip.get("duration", 0.0) or 0.0), float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)) or 0.0))
        candidates.append((speaker_match, duration_fit, clip))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _replace_mapping_clip(mapping: dict[str, Any], clip: dict[str, Any]) -> None:
    clip_duration = float(clip.get("duration", 0.0) or 0.0)
    planned = float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", clip_duration)) or clip_duration)
    trim_duration = min(clip_duration, planned) if clip_duration > 0 else planned
    mapping["clip_id"] = clip.get("id")
    mapping["clip_path"] = clip.get("path")
    mapping["clip_trim_start"] = 0.0
    mapping["clip_trim_duration"] = round(trim_duration, 3)
    mapping["source_transcript"] = clip.get("transcript", "")
    mapping["source_speaker_id"] = clip.get("speaker_id") or clip.get("speaker")
    mapping["speaker_match_preserved"] = bool(mapping.get("source_speaker_id") and mapping.get("destination_speaker_id") and mapping.get("source_speaker_id") == mapping.get("destination_speaker_id"))
    mapping["clip_movie_timestamp"] = clip.get("movie_timestamp")
    mapping["source_movie_timestamp"] = clip.get("movie_timestamp")


def _self_shuffle_mapping_unchanged(mapping: dict[str, Any], clip: dict[str, Any] | None) -> bool:
    if clip is None:
        return False
    source_start = _float(clip.get("movie_timestamp", mapping.get("clip_movie_timestamp")))
    source_duration = _float(clip.get("duration", mapping.get("clip_trim_duration")))
    destination_start = _float(mapping.get("alignment_slot_start", mapping.get("destination_timestamp")))
    slot_end = mapping.get("alignment_slot_end")
    if slot_end is None:
        destination_end = destination_start + _float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration")))
    else:
        destination_end = _float(slot_end)
    source_end = source_start + max(0.0, source_duration)
    overlap = max(0.0, min(source_end, destination_end) - max(source_start, destination_start))
    threshold = min(max(0.05, source_duration * 0.5), max(0.05, destination_end - destination_start))
    return overlap >= threshold


def _duration_fit(source_duration: float, destination_duration: float) -> float:
    if source_duration <= 0 or destination_duration <= 0:
        return 0.0
    return min(source_duration, destination_duration) / max(source_duration, destination_duration)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mark_speaker_shuffle_fallbacks(schedule: dict[str, Any]) -> None:
    for mapping in schedule.get("mappings", []):
        source = mapping.get("source_speaker_id")
        destination = mapping.get("destination_speaker_id")
        if not source or not destination:
            mapping["speaker_fallback_reason"] = "speaker_unavailable"
        elif source == destination:
            mapping["speaker_match_preserved"] = True
            mapping["speaker_fallback_reason"] = None
            mapping["selection_reason"] = f"{mapping.get('selection_reason', 'self_shuffle')}_same_speaker"
        else:
            mapping["speaker_match_preserved"] = False
            mapping["speaker_fallback_reason"] = mapping.get("speaker_fallback_reason") or "no_same_speaker_fit"


def render_mutation_media(
    *,
    original_media: Path,
    schedule: dict[str, Any],
    duration: float,
    audio_output: Path,
    video_output: Path,
    sample_rate: int,
    channels: int,
    target_lufs: float,
    fade_duration: float,
    mute_regions: list[dict[str, Any]] | None = None,
) -> None:
    schedule["render_duration"] = round(duration, 3)
    if schedule.get("mutation_id") == "self_shuffle":
        schedule["self_shuffle_render_strategy"] = "dialogue_only_v1"
        render_dialogue_wav(
            schedule=schedule,
            duration=duration,
            output_path=audio_output,
            sample_rate=sample_rate,
            channels=channels,
            target_lufs=target_lufs,
            fade_duration=fade_duration,
        )
    else:
        render_schedule_over_original_audio(
            original_media=original_media,
            schedule=schedule,
            duration=duration,
            output_path=audio_output,
            sample_rate=sample_rate,
            channels=channels,
            target_lufs=target_lufs,
            fade_duration=fade_duration,
            mute_regions=mute_regions,
        )
    mux_video(destination_video=original_media, dialogue_wav=audio_output, output_path=video_output)


def _mutation_schedule(mutation_id: str, duration: float, mappings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "transformation_name": f"mutation_{mutation_id}",
        "mutation_id": mutation_id,
        "render_duration": round(duration, 3),
        "mappings": mappings,
    }


def _mutation_mapping(clip: dict[str, Any], destination_timestamp: float, clip_duration: float, operation: str, reason: str) -> dict[str, Any]:
    return {
        "window_id": f"mutation_{operation}_{clip.get('id')}",
        "clip_id": clip.get("id"),
        "clip_path": clip.get("path"),
        "enabled": True,
        "destination_timestamp": round(destination_timestamp, 3),
        "stretch_factor": 1.0,
        "clip_trim_start": 0.0,
        "clip_trim_duration": round(clip_duration, 3),
        "leading_silence": 0.0,
        "trailing_silence": 0.0,
        "planned_render_duration": round(clip_duration, 3),
        "score": 1.0,
        "score_components": {},
        "selection_reason": reason,
        "scheduling_mode": f"mutation_{operation}",
        "timing_strategy": "whole_line_preserved",
        "render_operations": [],
        "shot_boundary_mode": "off",
        "visual_fit_score": 1.0,
        "mutation_operation": operation,
        "source_transcript": clip.get("transcript", ""),
        "source_movie_timestamp": round(float(clip.get("movie_timestamp", 0.0) or 0.0), 3),
        "clip_movie_timestamp": round(float(clip.get("movie_timestamp", 0.0) or 0.0), 3),
    }
