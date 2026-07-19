from __future__ import annotations

from pathlib import Path
from typing import Any

from cinelingus.util import read_json, rel, utc_now, write_json
from cinelingus.filter_lab.registry import default_filter_registry

from .base import TransformationMetadata, TransformationResult


def write_transformation_report(
    *,
    metadata: TransformationMetadata,
    pipeline: Any,
    result: TransformationResult,
) -> Path:
    scoped_output_path = pipeline.config.output_dir / metadata.id / "transformation_report.json"
    latest_output_path = pipeline.config.output_dir / "transformation_report.json"
    filter_definition = default_filter_registry().get(metadata.id)
    recipe_path = result.artifacts.get("filter_recipe")
    recipe_data = read_json(recipe_path) if recipe_path and recipe_path.exists() else {}
    data = {
        "schema_version": "1.0",
        "transformation": {
            "id": metadata.id,
            "display_name": metadata.display_name,
            "description": metadata.description,
            "version": metadata.version,
            "required_inputs": list(metadata.required_inputs),
            "generated_outputs": list(metadata.generated_outputs),
            "supported_modes": list(metadata.supported_modes),
            "filter_id": filter_definition.id,
            "family_id": filter_definition.family_id,
            "reads_dimensions": [item.value for item in filter_definition.reads_dimensions],
            "changes_dimensions": [item.value for item in filter_definition.changes_dimensions],
        },
        "creation_timestamp": utc_now(),
        "inputs": {
            "destination_video": str(pipeline.config.destination_video),
            "source_dialogue": str(pipeline.config.source_dialogue),
            "destination_hash": pipeline.destination.media_hash,
            "source_hash": pipeline.source.media_hash,
        },
        "outputs": {key: rel(path, pipeline.config.root) for key, path in result.outputs.items()},
        "artifacts": {key: rel(path, pipeline.config.root) for key, path in result.artifacts.items()},
        "warnings": result.warnings,
        "errors": result.errors,
        "analysis_artifacts_used": list(filter_definition.required_artifacts),
        "requested_analysis_backends": recipe_data.get("requested_analysis_backends", {}),
        "actual_analysis_backends": recipe_data.get("actual_analysis_backends", {}),
        "transformation_summary": f"{filter_definition.name} completed successfully and produced {len(result.outputs)} rendered outputs.",
    }
    write_json(scoped_output_path, data)
    write_json(latest_output_path, data)
    return scoped_output_path
