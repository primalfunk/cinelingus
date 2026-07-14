from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from movie_masher.util import read_json, write_json
from movie_masher.validation import validate_artifact

from .artifacts import materialize_required_artifacts
from .plan import plan_from_schedule, write_filter_plan
from .recipe import FilterRecipe, save_recipe
from .registry import default_filter_registry
from .strategies import get_strategy_spec


def build_strategy_schedule(
    filter_id: str,
    *,
    clips: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    duration: float,
    parameters: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    spec = get_strategy_spec(filter_id)
    seed = int(parameters.get("seed", 1))
    stages = spec.progress_stages
    if progress_callback:
        progress_callback(f"{filter_id}: {stages[0]}; {stages[1]}")
    schedule = spec.builder(clips=clips, windows=windows, duration=duration, parameters=parameters, seed=seed)
    schedule["filter_progress_stages"] = list(stages)
    if progress_callback:
        progress_callback(f"{filter_id}: {stages[2]}; {stages[3]}")
    return schedule


def write_filter_artifacts(
    *,
    pipeline: Any,
    filter_id: str,
    parameters: dict[str, Any],
    schedule: dict[str, Any],
    output_dir: Path,
    output_form: str,
    target_duration: float | None,
) -> dict[str, Path]:
    registry = default_filter_registry()
    definition = registry.get(filter_id)
    known_parameters = {item.id for item in definition.parameters}
    normalized_input = {key: value for key, value in parameters.items() if key in known_parameters}
    if "allow_line_reuse" in known_parameters and "allow_reuse" in parameters:
        normalized_input.setdefault("allow_line_reuse", parameters["allow_reuse"])
    roles = _input_roles(pipeline, definition.required_inputs)
    requested, actual = _analysis_backends(pipeline)
    recipe = FilterRecipe.create(
        definition.id,
        input_media_roles=roles,
        parameters=normalized_input,
        output_settings={"form": output_form, "output_directory": str(output_dir)},
        random_seed=int(parameters.get("seed", 1)),
        target_duration=target_duration,
        requested_analysis_backends=requested,
        actual_analysis_backends=actual,
        registry=registry,
    )
    recipe_path = output_dir / "filter_recipe.json"
    plan_path = output_dir / "filter_plan.json"
    save_recipe(recipe, recipe_path)
    plan = plan_from_schedule(definition=definition, schedule=schedule, seed=recipe.random_seed)
    plan.compatibility_decisions = list(recipe.compatibility_decisions)
    write_filter_plan(plan, plan_path)
    validate_artifact("filter_recipe", recipe_path, pipeline.schemas_dir)
    validate_artifact("filter_plan", plan_path, pipeline.schemas_dir)
    artifacts = {"filter_recipe": recipe_path, "filter_plan": plan_path}
    analysis_artifacts = materialize_required_artifacts(pipeline=pipeline, required_artifacts=definition.required_artifacts, output_dir=output_dir, schedule=schedule)
    artifacts.update({f"analysis_{key}": value for key, value in analysis_artifacts.items()})
    if schedule.get("speaker_graph"):
        graph_path = output_dir / "speaker_graph.json"
        write_json(graph_path, {"schema_version": "1.0", "speaker_graph": schedule["speaker_graph"]})
        artifacts["speaker_graph"] = graph_path
    if schedule.get("infection_timeline"):
        timeline_path = output_dir / "infection_timeline.json"
        write_json(timeline_path, {"schema_version": "1.0", "infection_timeline": schedule["infection_timeline"]})
        artifacts["infection_timeline"] = timeline_path
    if schedule.get("filter_metrics", {}).get("bloom_profile"):
        profile_path = output_dir / "bloom_profile.json"
        write_json(profile_path, {"schema_version": "1.0", "bloom_profile": schedule["filter_metrics"]["bloom_profile"]})
        artifacts["bloom_profile"] = profile_path
    schedule.update({
        "filter_id": definition.id,
        "filter_version": definition.version,
        "filter_family": definition.family_id,
        "filter_recipe_path": str(recipe_path),
        "filter_plan_path": str(plan_path),
        "filter_recipe_signature": recipe.deterministic_signature(),
        "filter_summary": plan.summary or schedule.get("filter_summary", ""),
        "requested_analysis_backends": requested,
        "actual_analysis_backends": actual,
        "filter_specific_artifacts": {key: str(value) for key, value in artifacts.items() if key not in {"filter_recipe", "filter_plan"}},
    })
    return artifacts


def _input_roles(pipeline: Any, required_inputs: tuple[str, ...]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for role in required_inputs:
        if role == "film":
            roles[role] = str(pipeline.config.destination_video)
        elif role == "destination_video":
            roles[role] = str(pipeline.config.destination_video)
        elif role == "source_dialogue":
            roles[role] = str(pipeline.config.source_dialogue)
    return roles


def _analysis_backends(pipeline: Any) -> tuple[dict[str, str], dict[str, str]]:
    requested = {
        "diarization": str(pipeline.config.speaker_diarization_backend),
        "transcription": str(pipeline.config.whisper_model),
    }
    actual: dict[str, str] = {}
    speaker_path = pipeline.source.cache_dir / "speaker_map.json"
    dialogue_path = pipeline.source.cache_dir / "dialogue_events.json"
    if speaker_path.exists():
        speaker_map = read_json(speaker_path)
        actual["diarization"] = str(speaker_map.get("actual_backend") or speaker_map.get("diarization_tool") or "unknown")
    if dialogue_path.exists():
        dialogue = read_json(dialogue_path)
        actual["transcription"] = str(dialogue.get("whisper_model") or dialogue.get("speech_backend") or "unknown")
        if dialogue.get("whisper_device"):
            actual["transcription_device"] = str(dialogue["whisper_device"])
    return requested, actual
