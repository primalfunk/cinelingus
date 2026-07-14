from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .operations import ReplaceOperation, StretchOperation
from .placement import PlaceIntoPerformances
from .selections import SelectDialogue, SelectPerformances, SelectShots, SelectTimeline
from .util import rel, utc_now, write_json
from .validation import validate_artifact

VOCABULARY_VERSION = "1.0"

VERB_SELECT = "SELECT"
VERB_PLACE = "PLACE"
VERB_REPLACE = "REPLACE"
VERB_MOVE = "MOVE"
VERB_REMOVE = "REMOVE"
VERB_REPEAT = "REPEAT"
VERB_SHUFFLE = "SHUFFLE"
VERB_STRETCH = "STRETCH"
VERB_COMPRESS = "COMPRESS"
VERB_RENDER = "RENDER"
VERB_REPORT = "REPORT"

TRANSFORMATION_VOCABULARY = [
    VERB_SELECT,
    VERB_PLACE,
    VERB_REPLACE,
    VERB_MOVE,
    VERB_REMOVE,
    VERB_REPEAT,
    VERB_SHUFFLE,
    VERB_STRETCH,
    VERB_COMPRESS,
    VERB_RENDER,
    VERB_REPORT,
]

MOVIE_MASHER_LIFECYCLE = [VERB_SELECT, "TRANSFORM", "VALIDATE", VERB_PLACE, VERB_RENDER, VERB_REPORT]


def build_movie_masher_plan(
    *,
    root: Path,
    destination_movie: dict[str, Any],
    source_movie: dict[str, Any],
    clip_library: dict[str, Any],
    destination_timeline: dict[str, Any],
    visual: dict[str, Any],
    source_performances: dict[str, Any],
    destination_performances: dict[str, Any],
    output_dir: Path,
    max_time_stretch: float,
) -> dict[str, Any]:
    source_dialogue = SelectDialogue(
        role="source_dialogue",
        source_artifact="source_dialogue.clip_library",
        criteria={"usable": True, "duration_gt": 0},
    ).select(clip_library)
    source_performance_selection = SelectPerformances(
        role="source_dialogue",
        source_artifact="source_dialogue.performance",
        criteria={"usable": True, "duration_gt": 0},
    ).select(source_performances)
    destination_performance_selection = SelectPerformances(
        role="destination_video",
        source_artifact="destination_video.performance",
        criteria={"usable": True, "duration_gt": 0},
    ).select(destination_performances)
    destination_timeline_selection = SelectTimeline(
        role="destination_video",
        source_artifact="destination_video.filtered_timeline",
        criteria={"usable": True, "duration_gt": 0},
    ).select(destination_timeline)
    shot_selection = SelectShots(
        role="destination_video",
        source_artifact="destination_video.shots",
        criteria={},
    ).select(visual.get("shots", {}))

    replace_result = ReplaceOperation().apply(source_dialogue.objects, destination_performance_selection.objects)
    stretch_result = StretchOperation(max_factor=max_time_stretch).apply(source_dialogue.objects)[1]
    placement = PlaceIntoPerformances().plan(source_dialogue.objects, destination_performance_selection.objects)
    warnings = replace_result.warnings + stretch_result.warnings + placement.warnings

    output_video = output_dir / "movie_masher_output.mp4"
    output_audio = output_dir / "replacement_dialogue.wav"

    return {
        "schema_version": "1.0",
        "tool_version": __version__,
        "vocabulary_version": VOCABULARY_VERSION,
        "creation_timestamp": utc_now(),
        "transformation": {
            "id": "movie_masher",
            "display_name": "Transposition",
            "description": "Replace destination dialogue using dialogue extracted from another film.",
            "lifecycle": MOVIE_MASHER_LIFECYCLE,
        },
        "inputs": {
            "destination_video": {
                "path": destination_movie.get("path", ""),
                "media_hash": destination_movie.get("media_hash", ""),
                "duration": destination_movie.get("duration", 0.0),
            },
            "source_dialogue": {
                "path": source_movie.get("path", ""),
                "media_hash": source_movie.get("media_hash", ""),
                "duration": source_movie.get("duration", 0.0),
            },
        },
        "vocabulary": TRANSFORMATION_VOCABULARY,
        "selection": [
            source_dialogue.to_plan_entry(),
            source_performance_selection.to_plan_entry(),
            destination_performance_selection.to_plan_entry(),
            destination_timeline_selection.to_plan_entry(),
            shot_selection.to_plan_entry(),
        ],
        "operations": [
            {
                "verb": VERB_REPLACE,
                "description": "Replace destination performance dialogue with selected source dialogue.",
                **replace_result.to_plan_entry(),
            },
            {
                "verb": VERB_STRETCH,
                "description": "Allow bounded timing adjustment while preserving pitch.",
                **stretch_result.to_plan_entry(),
            },
        ],
        "placement": {
            "verb": VERB_PLACE,
            "description": "Place selected source dialogue into destination performances.",
            **placement.to_plan_entry(),
        },
        "render": {
            "verb": VERB_RENDER,
            "description": "Render a dialogue-only WAV and mux it with destination video.",
            "audio_output": rel(output_audio, root),
            "video_output": rel(output_video, root),
            "audio_policy": "dialogue_only_silence_elsewhere",
        },
        "validation_rules": [
            "source dialogue selection is non-empty",
            "destination performance selection is non-empty",
            "placements do not reuse source clips",
            "render duration follows available transformed audio/video length policy",
        ],
        "warnings": warnings,
        "report": {
            "expected_artifacts": [
                "replacement_schedule.json",
                "replacement_dialogue.wav",
                "movie_masher_output.mp4",
                "transformation_report.json",
            ]
        },
    }


def write_transformation_plan(
    *,
    plan: dict[str, Any],
    output_path: Path,
    latest_path: Path | None = None,
    schemas_dir: Path | None = None,
) -> Path:
    write_json(output_path, plan)
    if latest_path is not None:
        write_json(latest_path, plan)
    if schemas_dir is not None:
        validate_artifact("transformation_plan", output_path, schemas_dir)
        if latest_path is not None:
            validate_artifact("transformation_plan", latest_path, schemas_dir)
    return output_path
