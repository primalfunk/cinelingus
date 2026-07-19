from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import rel, utc_now, write_json

OBJECT_MOVIE = "movie"
OBJECT_AUDIO_STREAM = "audio_stream"
OBJECT_VIDEO_STREAM = "video_stream"
OBJECT_DIALOGUE_EVENT = "dialogue_event"
OBJECT_DIALOGUE_CLIP = "dialogue_clip"
OBJECT_SPEAKING_WINDOW = "speaking_window"
OBJECT_SCHEDULE_MAPPING = "schedule_mapping"
OBJECT_RENDER_OPERATION = "render_operation"
OBJECT_TRANSFORMATION_STEP = "transformation_step"
OBJECT_SHOT = "shot"
OBJECT_VISUAL_REPORT = "visual_report"
OBJECT_VISUAL_SCHEDULE_REPORT = "visual_schedule_report"
OBJECT_REVIEW_NOTE = "review_note"
OBJECT_REVIEW_ANALYSIS = "review_analysis"
OBJECT_TRANSFORMATION_REPORT = "transformation_report"
OBJECT_TRANSFORMATION_PLAN = "transformation_plan"
OBJECT_PERFORMANCE = "performance"

CIR_OBJECT_TYPES = [
    OBJECT_MOVIE,
    OBJECT_AUDIO_STREAM,
    OBJECT_VIDEO_STREAM,
    OBJECT_DIALOGUE_EVENT,
    OBJECT_DIALOGUE_CLIP,
    OBJECT_SPEAKING_WINDOW,
    OBJECT_SCHEDULE_MAPPING,
    OBJECT_RENDER_OPERATION,
    OBJECT_TRANSFORMATION_STEP,
    OBJECT_SHOT,
    OBJECT_VISUAL_REPORT,
    OBJECT_VISUAL_SCHEDULE_REPORT,
    OBJECT_REVIEW_NOTE,
    OBJECT_REVIEW_ANALYSIS,
    OBJECT_TRANSFORMATION_REPORT,
    OBJECT_TRANSFORMATION_PLAN,
    OBJECT_PERFORMANCE,
]


def artifact_entry(
    *,
    artifact_type: str,
    cir_object_type: str,
    path: Path,
    root: Path,
    count: int | None = None,
    media_hash: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "artifact_type": artifact_type,
        "cir_object_type": cir_object_type,
        "path": rel(path, root),
        "exists": path.exists(),
    }
    if count is not None:
        entry["count"] = count
    if media_hash is not None:
        entry["media_hash"] = media_hash
    if role is not None:
        entry["role"] = role
    return entry


def build_cinematic_index(
    *,
    root: Path,
    output_path: Path,
    destination_movie: dict[str, Any],
    source_movie: dict[str, Any],
    source_events: dict[str, Any],
    filtered_source_events: dict[str, Any],
    clip_library: dict[str, Any],
    destination_timeline: dict[str, Any],
    filtered_destination_timeline: dict[str, Any],
    schedule: dict[str, Any],
    audio_output: Path,
    video_output: Path,
    run_report_json: Path,
    schedule_report_csv: Path,
    destination_cache: Path,
    source_cache: Path,
    shots: dict[str, Any] | None = None,
    visual_report: dict[str, Any] | None = None,
    visual_schedule_report: dict[str, Any] | None = None,
    review_notes: dict[str, Any] | None = None,
    review_analysis: dict[str, Any] | None = None,
    source_performances: dict[str, Any] | None = None,
    destination_performances: dict[str, Any] | None = None,
    transformation_report: Path | None = None,
    transformation_plan: Path | None = None,
) -> dict[str, Any]:
    destination_hash = str(destination_movie.get("media_hash", ""))
    source_hash = str(source_movie.get("media_hash", ""))
    mappings = schedule.get("mappings", [])
    render_operation_count = sum(len(mapping.get("render_operations", [])) for mapping in mappings)
    visual_shots = shots.get("shots", []) if shots else []
    transformation_report_path = transformation_report or (root / "output" / "transformation_report.json")
    has_transformation_report = transformation_report_path.exists()
    transformation_plan_path = transformation_plan or (root / "output" / "transformation_plan.json")
    has_transformation_plan = transformation_plan_path.exists()

    index = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "cir_version": "0.1",
        "object_types": CIR_OBJECT_TYPES,
        "media": {
            "destination_video": {
                "media_hash": destination_hash,
                "path": destination_movie.get("path"),
                "duration": destination_movie.get("duration"),
                "resolution": destination_movie.get("resolution"),
            },
            "source_dialogue": {
                "media_hash": source_hash,
                "path": source_movie.get("path"),
                "duration": source_movie.get("duration"),
                "resolution": source_movie.get("resolution"),
            },
        },
        "artifacts": [
            artifact_entry(
                artifact_type="movie",
                cir_object_type=OBJECT_MOVIE,
                path=destination_cache / "movie.json",
                root=root,
                count=1,
                media_hash=destination_hash,
                role="destination_video",
            ),
            artifact_entry(
                artifact_type="movie",
                cir_object_type=OBJECT_MOVIE,
                path=source_cache / "movie.json",
                root=root,
                count=1,
                media_hash=source_hash,
                role="source_dialogue",
            ),
            artifact_entry(
                artifact_type="dialogue_events",
                cir_object_type=OBJECT_DIALOGUE_EVENT,
                path=source_cache / "dialogue_events.json",
                root=root,
                count=len(source_events.get("events", [])),
                media_hash=source_hash,
                role="source_dialogue",
            ),
            artifact_entry(
                artifact_type="filtered_dialogue_events",
                cir_object_type=OBJECT_DIALOGUE_EVENT,
                path=source_cache / "filtered_dialogue_events.json",
                root=root,
                count=int(filtered_source_events.get("filter_stats", {}).get("usable_count", 0)),
                media_hash=source_hash,
                role="source_dialogue",
            ),
            artifact_entry(
                artifact_type="clip_library",
                cir_object_type=OBJECT_DIALOGUE_CLIP,
                path=source_cache / "clip_library.json",
                root=root,
                count=len(clip_library.get("clips", [])),
                media_hash=source_hash,
                role="source_dialogue",
            ),
            artifact_entry(
                artifact_type="timeline",
                cir_object_type=OBJECT_SPEAKING_WINDOW,
                path=destination_cache / "timeline.json",
                root=root,
                count=len(destination_timeline.get("windows", [])),
                media_hash=destination_hash,
                role="destination_video",
            ),
            artifact_entry(
                artifact_type="filtered_timeline",
                cir_object_type=OBJECT_SPEAKING_WINDOW,
                path=destination_cache / "filtered_timeline.json",
                root=root,
                count=int(filtered_destination_timeline.get("filter_stats", {}).get("usable_count", 0)),
                media_hash=destination_hash,
                role="destination_video",
            ),
            artifact_entry(
                artifact_type="source_performance",
                cir_object_type=OBJECT_PERFORMANCE,
                path=source_cache / "performance.json",
                root=root,
                count=len((source_performances or {}).get("performances", [])),
                media_hash=source_hash,
                role="source_dialogue",
            ),
            artifact_entry(
                artifact_type="destination_performance",
                cir_object_type=OBJECT_PERFORMANCE,
                path=destination_cache / "performance.json",
                root=root,
                count=len((destination_performances or {}).get("performances", [])),
                media_hash=destination_hash,
                role="destination_video",
            ),
            artifact_entry(
                artifact_type="replacement_schedule",
                cir_object_type=OBJECT_SCHEDULE_MAPPING,
                path=destination_cache / "replacement_schedule.json",
                root=root,
                count=len(mappings),
                media_hash=destination_hash,
                role="transformation",
            ),
            artifact_entry(
                artifact_type="render_operations",
                cir_object_type=OBJECT_RENDER_OPERATION,
                path=destination_cache / "replacement_schedule.json",
                root=root,
                count=render_operation_count,
                media_hash=destination_hash,
                role="transformation",
            ),
            artifact_entry(
                artifact_type="transformation_history",
                cir_object_type=OBJECT_TRANSFORMATION_STEP,
                path=destination_cache / "replacement_schedule.json",
                root=root,
                count=len(schedule.get("transformation_history", [])),
                media_hash=destination_hash,
                role="transformation",
            ),
            artifact_entry(
                artifact_type="shots",
                cir_object_type=OBJECT_SHOT,
                path=destination_cache / "shots.json",
                root=root,
                count=len(visual_shots),
                media_hash=destination_hash,
                role="destination_video",
            ),
            artifact_entry(
                artifact_type="visual_report",
                cir_object_type=OBJECT_VISUAL_REPORT,
                path=destination_cache / "visual_report.json",
                root=root,
                count=1 if visual_report else 0,
                media_hash=destination_hash,
                role="destination_video",
            ),
            artifact_entry(
                artifact_type="visual_schedule_report",
                cir_object_type=OBJECT_VISUAL_SCHEDULE_REPORT,
                path=destination_cache / "visual_schedule_report.json",
                root=root,
                count=1 if visual_schedule_report else 0,
                media_hash=destination_hash,
                role="destination_video",
            ),
            artifact_entry(
                artifact_type="review_notes",
                cir_object_type=OBJECT_REVIEW_NOTE,
                path=destination_cache / "review_notes.json",
                root=root,
                count=int(review_notes.get("reviewed_mappings", 0)) if review_notes else 0,
                media_hash=destination_hash,
                role="review",
            ),
            artifact_entry(
                artifact_type="review_analysis",
                cir_object_type=OBJECT_REVIEW_ANALYSIS,
                path=destination_cache / "review_analysis.json",
                root=root,
                count=int(review_analysis.get("reviewed_mappings", 0)) if review_analysis else 0,
                media_hash=destination_hash,
                role="review",
            ),
            artifact_entry(
                artifact_type="transformation_report",
                cir_object_type=OBJECT_TRANSFORMATION_REPORT,
                path=transformation_report_path,
                root=root,
                count=1 if has_transformation_report else 0,
                media_hash=destination_hash,
                role="transformation",
            ),
            artifact_entry(
                artifact_type="transformation_plan",
                cir_object_type=OBJECT_TRANSFORMATION_PLAN,
                path=transformation_plan_path,
                root=root,
                count=1 if has_transformation_plan else 0,
                media_hash=destination_hash,
                role="transformation",
            ),
        ],
        "transformation": {
            "name": schedule.get("transformation_name", "translation"),
            "history": schedule.get("transformation_history", []),
            "schedule_path": rel(destination_cache / "replacement_schedule.json", root),
        },
        "outputs": {
            "audio": rel(audio_output, root),
            "video": rel(video_output, root),
            "run_report_json": rel(run_report_json, root),
            "schedule_report_csv": rel(schedule_report_csv, root),
            "cinematic_index": rel(output_path, root),
            "transformation_report": rel(transformation_report_path, root),
            "transformation_plan": rel(transformation_plan_path, root),
        },
        "counts": {
            "movies": 2 if source_hash != destination_hash else 1,
            "raw_dialogue_events": len(source_events.get("events", [])),
            "usable_dialogue_events": int(filtered_source_events.get("filter_stats", {}).get("usable_count", 0)),
            "dialogue_clips": len(clip_library.get("clips", [])),
            "raw_speaking_windows": len(destination_timeline.get("windows", [])),
            "usable_speaking_windows": int(filtered_destination_timeline.get("filter_stats", {}).get("usable_count", 0)),
            "source_performances": len((source_performances or {}).get("performances", [])),
            "destination_performances": len((destination_performances or {}).get("performances", [])),
            "schedule_mappings": len(mappings),
            "enabled_schedule_mappings": sum(1 for mapping in mappings if mapping.get("enabled", True)),
            "render_operations": render_operation_count,
            "transformation_steps": len(schedule.get("transformation_history", [])),
            "shots": len(visual_shots),
            "average_shot_duration": visual_report.get("average_shot_duration", 0.0) if visual_report else 0.0,
            "shot_crossing_mappings": len(visual_schedule_report.get("crossing_mappings", [])) if visual_schedule_report else 0,
            "empty_dialogue_shots": len(visual_schedule_report.get("empty_dialogue_shots", [])) if visual_schedule_report else 0,
            "reviewed_mappings": int(review_notes.get("reviewed_mappings", 0)) if review_notes else 0,
            "review_bad_mappings": int(review_analysis.get("bad_mappings", 0)) if review_analysis else 0,
            "review_good_mappings": int(review_analysis.get("good_mappings", 0)) if review_analysis else 0,
            "transformation_reports": 1 if has_transformation_report else 0,
            "transformation_plans": 1 if has_transformation_plan else 0,
        },
    }
    write_json(output_path, index)
    return index
