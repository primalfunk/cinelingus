from __future__ import annotations

from copy import deepcopy
from typing import Any


def transformation_step(verb: str, *, description: str, inputs: list[str], outputs: list[str]) -> dict[str, Any]:
    return {
        "verb": verb,
        "description": description,
        "inputs": inputs,
        "outputs": outputs,
    }


def movie_masher_transformation_plan() -> list[dict[str, Any]]:
    return [
        transformation_step(
            "select",
            description="select dialogue clips from source_dialogue",
            inputs=["source_dialogue.clip_library"],
            outputs=["selected_dialogue_clips"],
        ),
        transformation_step(
            "select",
            description="select speaking windows from destination_video",
            inputs=["destination_video.filtered_timeline"],
            outputs=["selected_speaking_windows"],
        ),
        transformation_step(
            "place",
            description="place source dialogue clips into destination speaking windows",
            inputs=["selected_dialogue_clips", "selected_speaking_windows"],
            outputs=["replacement_schedule.mappings"],
        ),
        transformation_step(
            "replace",
            description="replace destination audio with the rendered dialogue-only soundtrack",
            inputs=["destination_video.video", "replacement_dialogue.wav"],
            outputs=["movie_masher_output.mp4"],
        ),
        transformation_step(
            "render",
            description="render destination video with transformed dialogue soundtrack",
            inputs=["replacement_schedule", "destination_video.video"],
            outputs=["movie_masher_output.mp4"],
        ),
    ]



def self_shuffle_transformation_plan() -> list[dict[str, Any]]:
    return [
        transformation_step(
            "select",
            description="select dialogue clips from the destination film",
            inputs=["film.clip_library"],
            outputs=["selected_dialogue_clips"],
        ),
        transformation_step(
            "shuffle",
            description="shuffle selected dialogue clips within the same film",
            inputs=["selected_dialogue_clips"],
            outputs=["shuffled_dialogue_clips"],
        ),
        transformation_step(
            "select",
            description="select speaking windows from the same film",
            inputs=["film.filtered_timeline"],
            outputs=["selected_speaking_windows"],
        ),
        transformation_step(
            "place",
            description="place shuffled dialogue clips into speaking windows",
            inputs=["shuffled_dialogue_clips", "selected_speaking_windows"],
            outputs=["self_shuffle_schedule.mappings"],
        ),
        transformation_step(
            "render",
            description="render the same film with shuffled dialogue soundtrack",
            inputs=["self_shuffle_schedule", "film.video"],
            outputs=["self_shuffle_output.mp4"],
        ),
    ]

def select_dialogue_clips(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(clip) for clip in clips if float(clip.get("duration", 0.0)) > 0 and clip.get("usable", True)]


def select_speaking_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(window) for window in windows if float(window.get("duration", 0.0)) > 0 and window.get("usable", True)]


def remove_disabled_mappings(mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(mapping) for mapping in mappings if mapping.get("enabled", True)]


def repeat_selection(items: list[dict[str, Any]], *, times: int) -> list[dict[str, Any]]:
    if times < 1:
        return []
    repeated = []
    for _ in range(times):
        repeated.extend(deepcopy(items))
    return repeated


def shuffle_selection(items: list[dict[str, Any]], *, seed: int | None = None) -> list[dict[str, Any]]:
    import random

    shuffled = deepcopy(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def stretch_duration(value: float, *, factor: float) -> float:
    return round(float(value) * float(factor), 3)


def compress_duration(value: float, *, factor: float) -> float:
    if factor == 0:
        raise ValueError("Compression factor cannot be zero.")
    return round(float(value) / float(factor), 3)
