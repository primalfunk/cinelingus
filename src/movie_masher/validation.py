from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .util import read_json

SCHEMA_MAP = {
    "movie": "movie.schema.json",
    "dialogue_events": "dialogue_events.schema.json",
    "filtered_dialogue_events": "filtered_dialogue_events.schema.json",
    "timeline": "timeline.schema.json",
    "filtered_timeline": "filtered_timeline.schema.json",
    "replacement_schedule": "replacement_schedule.schema.json",
    "clip_library": "clip_library.schema.json",
    "cinematic_index": "cinematic_index.schema.json",
    "shots": "shots.schema.json",
    "visual_report": "visual_report.schema.json",
    "visual_schedule_report": "visual_schedule_report.schema.json",
    "review_notes": "review_notes.schema.json",
    "review_analysis": "review_analysis.schema.json",
    "transformation_report": "transformation_report.schema.json",
    "transformation_plan": "transformation_plan.schema.json",
    "performance": "performance.schema.json",
    "performance_library": "performance_library.schema.json",
    "performance_diagnostics": "performance_diagnostics.schema.json",
    "speaker_map": "speaker_map.schema.json",
    "speaker_mapping": "speaker_mapping.schema.json",
    "performance_placement_report": "performance_placement_report.schema.json",
    "taste_profile": "taste_profile.schema.json",
    "editorial_highlights": "editorial_highlights.schema.json",
    "mutation_plan": "mutation_plan.schema.json",
    "mutation_report": "mutation_report.schema.json",
    "short_remix_report": "short_remix_report.schema.json",
    "filter_recipe": "filter_recipe.schema.json",
    "filter_plan": "filter_plan.schema.json",
    "filter_contract": "filter_contract.schema.json",
    "filter_acceptance": "filter_acceptance.schema.json",
}


class ValidationError(ValueError):
    pass


def validate_artifact(artifact_type: str, artifact_path: Path, schemas_dir: Path) -> dict[str, Any]:
    schema_name = SCHEMA_MAP[artifact_type]
    schema = read_json(schemas_dir / schema_name)
    data = read_json(artifact_path)
    _validate_object(data, schema, str(artifact_path))
    return data


def _validate_object(data: Any, schema: dict[str, Any], location: str) -> None:
    if "const" in schema and data != schema["const"]:
        raise ValidationError(f"{location} must equal {schema['const']!r}")
    if "enum" in schema and data not in schema["enum"]:
        choices = ", ".join(repr(item) for item in schema["enum"])
        raise ValidationError(f"{location} must be one of: {choices}")
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(data, dict):
            raise ValidationError(f"{location} should be an object")
        missing = [key for key in schema.get("required", []) if key not in data]
        if missing:
            raise ValidationError(f"{location} is missing required fields: {', '.join(missing)}")
        properties = schema.get("properties", {})
        if len(data) < int(schema.get("minProperties", 0)):
            raise ValidationError(f"{location} has too few properties")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(data) - set(properties))
            if unknown:
                raise ValidationError(f"{location} has unknown fields: {', '.join(unknown)}")
        for key, child_schema in properties.items():
            if key in data:
                _validate_object(data[key], child_schema, f"{location}.{key}")
    elif expected_type == "array":
        if not isinstance(data, list):
            raise ValidationError(f"{location} should be an array")
        item_schema = schema.get("items")
        if len(data) < int(schema.get("minItems", 0)):
            raise ValidationError(f"{location} has too few items")
        if schema.get("uniqueItems") and len({repr(item) for item in data}) != len(data):
            raise ValidationError(f"{location} must contain unique items")
        if item_schema:
            for index, item in enumerate(data):
                _validate_object(item, item_schema, f"{location}[{index}]")
    elif isinstance(expected_type, list):
        if not any(_matches_type(data, item_type) for item_type in expected_type):
            raise ValidationError(f"{location} has wrong type")
    elif isinstance(expected_type, str):
        if not _matches_type(data, expected_type):
            raise ValidationError(f"{location} has wrong type")
        if expected_type == "string":
            if len(data) < int(schema.get("minLength", 0)):
                raise ValidationError(f"{location} is too short")
            if schema.get("pattern") and re.fullmatch(str(schema["pattern"]), data) is None:
                raise ValidationError(f"{location} has invalid format")
        if expected_type in {"number", "integer"}:
            if "minimum" in schema and data < schema["minimum"]:
                raise ValidationError(f"{location} is below its minimum")
            if "maximum" in schema and data > schema["maximum"]:
                raise ValidationError(f"{location} exceeds its maximum")


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True
