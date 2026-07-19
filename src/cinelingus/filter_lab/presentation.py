from __future__ import annotations

from typing import Any

from .models import FilterDefinition, RelationshipDimension
from .multiworld import film_label
from .registry import FilterRegistry, default_filter_registry


DIMENSION_LABELS = {
    RelationshipDimension.DIALOGUE: "Dialogue",
    RelationshipDimension.PERFORMANCE: "Performance",
    RelationshipDimension.IDENTITY: "Identity",
    RelationshipDimension.TIME: "Time",
    RelationshipDimension.SCENE_ORDER: "Scene order",
    RelationshipDimension.EMOTION: "Emotion",
    RelationshipDimension.MUSIC: "Music",
    RelationshipDimension.SOUNDSCAPE: "Soundscape",
    RelationshipDimension.SHOT_SELECTION: "Shot selection",
    RelationshipDimension.NARRATION: "Narration",
    RelationshipDimension.GENRE: "Genre",
}


def relationship_summary(definition: FilterDefinition) -> str:
    lines = []
    visible_dimensions = tuple(dict.fromkeys((*definition.reads_dimensions, *definition.changes_dimensions)))
    for dimension in visible_dimensions:
        label = DIMENSION_LABELS[dimension]
        if dimension in definition.changes_dimensions:
            state = "Changed"
        elif dimension in definition.reads_dimensions:
            state = definition.preserves.get(dimension.value, "Read and preserved where possible")
        else:
            state = definition.preserves.get(dimension.value, "Preserved")
        lines.append(f"{label}: {state}")
    return "  |  ".join(lines)


def input_field_ids(definition: FilterDefinition) -> tuple[str, ...]:
    fields = ["anchor_film"]
    fields.extend(f"film_{index}" for index in range(2, definition.minimum_films + 1))
    if definition.maximum_films is None or definition.maximum_films > definition.minimum_films:
        fields.append("additional_films")
    fields.append("output")
    return tuple(fields)


def film_selector_spec(definition: FilterDefinition, selected_count: int | None = None) -> dict[str, Any]:
    count = max(definition.minimum_films, int(selected_count or definition.minimum_films))
    if definition.maximum_films is not None:
        count = min(count, definition.maximum_films)
    rows = [
        {
            "index": index,
            "id": f"film_{index + 1}",
            "label": f"Film {film_label(index)}" + (" (Anchor)" if index == 0 else ""),
            "is_anchor": index == 0,
            "required": index < definition.minimum_films,
            "removable": index >= definition.minimum_films,
        }
        for index in range(count)
    ]
    return {
        "rows": rows,
        "can_add": definition.maximum_films is None or count < definition.maximum_films,
        "minimum_films": definition.minimum_films,
        "maximum_films": definition.maximum_films,
    }


def detail_text(definition: FilterDefinition, *, registry: FilterRegistry | None = None) -> str:
    registry = registry or default_filter_registry()
    family = registry.family(definition.family_id)
    status = "Available" if definition.implemented else "This filter is not yet implemented"
    maximum = "unlimited" if definition.maximum_films is None else str(definition.maximum_films)
    requirements = f"{definition.minimum_films}-{maximum} films; Film A is the anchor"
    limitations = " ".join(definition.known_limitations)
    text = (
        f"{family.name} / {status}\n"
        f"{definition.creative_description}\n"
        f"{definition.operational_description}\n"
        f"Law: {definition.cinematic_law}. Requires: {requirements}. "
        f"Affected elements: {', '.join(definition.affected_elements)}. Outputs: {', '.join(definition.output_artifacts)}."
    )
    return f"{text}\nKnown limitation: {limitations}" if limitations else text


def parameter_help(definition: FilterDefinition, values: dict[str, Any] | None = None) -> str:
    values = values or definition.parameter_defaults
    return "\n".join(f"{item.label} ({values.get(item.id, item.default)}): {item.description}" for item in definition.parameters)
