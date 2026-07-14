from __future__ import annotations

from typing import Any

from .models import FilterDefinition, RelationshipDimension
from .registry import FilterRegistry, default_filter_registry


DIMENSION_LABELS = {
    RelationshipDimension.DIALOGUE: "Dialogue",
    RelationshipDimension.PERFORMANCE: "Performance",
    RelationshipDimension.IDENTITY: "Identity",
    RelationshipDimension.TIME: "Time",
}


def relationship_summary(definition: FilterDefinition) -> str:
    lines = []
    for dimension in RelationshipDimension:
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
    if "source_dialogue" in definition.required_inputs:
        return ("destination", "source", "output")
    return ("film", "output")


def detail_text(definition: FilterDefinition, *, registry: FilterRegistry | None = None) -> str:
    registry = registry or default_filter_registry()
    family = registry.family(definition.family_id)
    status = "Available" if definition.implemented else "In Development"
    requirements = " + ".join(definition.required_inputs).replace("_", " ")
    limitations = " ".join(definition.known_limitations)
    text = (
        f"{family.name} / {status}\n"
        f"{definition.creative_description}\n"
        f"{definition.operational_description}\n"
        f"Requires: {requirements}. Outputs: {', '.join(definition.supported_output_forms) if definition.implemented else 'not yet available'}."
    )
    return f"{text}\nKnown limitation: {limitations}" if limitations else text


def parameter_help(definition: FilterDefinition, values: dict[str, Any] | None = None) -> str:
    values = values or definition.parameter_defaults
    return "\n".join(f"{item.label} ({values.get(item.id, item.default)}): {item.description}" for item in definition.parameters)
