from __future__ import annotations

from pathlib import Path
from typing import Any

from movie_masher.util import write_json

from .models import FilterDefinition, TransformationPlan


def plan_from_schedule(
    *,
    definition: FilterDefinition,
    schedule: dict[str, Any],
    seed: int,
    rejected_candidates: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    summary: str = "",
) -> TransformationPlan:
    mappings = [dict(item) for item in schedule.get("mappings", [])]
    regions = [
        {
            "window_id": item.get("window_id"),
            "start": item.get("destination_timestamp", item.get("alignment_slot_start")),
            "duration": item.get("planned_render_duration", item.get("clip_trim_duration")),
            "speaker_id": item.get("destination_speaker_id"),
        }
        for item in mappings if item.get("enabled", True)
    ]
    speaker_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[Any, Any]] = set()
    time_relationships: list[dict[str, Any]] = []
    progression: list[dict[str, Any]] = []
    for item in mappings:
        pair = (item.get("source_speaker_id"), item.get("destination_speaker_id"))
        if any(pair) and pair not in seen_pairs:
            speaker_pairs.append({"source_speaker_id": pair[0], "destination_speaker_id": pair[1]})
            seen_pairs.add(pair)
        source_time = item.get("source_movie_timestamp", item.get("clip_movie_timestamp"))
        destination_time = item.get("destination_timestamp", item.get("alignment_slot_start"))
        if source_time is not None and destination_time is not None:
            time_relationships.append({
                "mapping_id": item.get("window_id"),
                "source_start": source_time,
                "destination_start": destination_time,
                "displacement": round(float(source_time) - float(destination_time), 3),
            })
        if item.get("progression_value") is not None:
            progression.append({"mapping_id": item.get("window_id"), "value": item.get("progression_value")})
    return TransformationPlan(
        filter_id=definition.id,
        filter_version=definition.version,
        family_id=definition.family_id,
        deterministic_seed=seed,
        selected_destination_regions=regions,
        mappings=mappings,
        speaker_mappings=speaker_pairs,
        time_relationships=time_relationships,
        progression_values=progression,
        rejected_candidates=rejected_candidates or list(schedule.get("rejected_candidates", [])),
        warnings=warnings or list(schedule.get("warnings", [])),
        analysis_artifacts_used=list(definition.required_artifacts),
        validation=validation or dict(schedule.get("filter_validation", {})),
        metrics=metrics or dict(schedule.get("filter_metrics", {})),
        summary=summary or str(schedule.get("filter_summary", "")),
        cinematic_law=definition.cinematic_law,
        anchor_behavior=definition.anchor_behavior,
        film_count={"minimum": definition.minimum_films, "maximum": definition.maximum_films},
        inputs=list(definition.required_inputs),
        outputs=list(definition.output_artifacts),
        affected_artifacts=list(definition.affected_artifacts),
        intermediate_products=list(definition.intermediate_products),
    )


def write_filter_plan(plan: TransformationPlan, output_path: Path) -> Path:
    write_json(output_path, plan.to_dict())
    return output_path
