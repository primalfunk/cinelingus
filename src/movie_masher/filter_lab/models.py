from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RelationshipDimension(StrEnum):
    DIALOGUE = "dialogue"
    PERFORMANCE = "performance"
    IDENTITY = "identity"
    TIME = "time"


@dataclass(frozen=True)
class FilterFamilyDefinition:
    id: str
    name: str
    description: str
    order: int


@dataclass(frozen=True)
class FilterParameter:
    id: str
    label: str
    kind: str
    default: Any
    description: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    advanced: bool = False

    def validate(self, value: Any) -> Any:
        if self.kind in {"integer", "float"}:
            converted = int(value) if self.kind == "integer" else float(value)
            if self.minimum is not None and converted < self.minimum:
                raise ValueError(f"{self.label} must be at least {self.minimum}.")
            if self.maximum is not None and converted > self.maximum:
                raise ValueError(f"{self.label} must be at most {self.maximum}.")
            return converted
        if self.kind == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"{self.label} must be one of: {', '.join(self.choices)}")
        return value


@dataclass(frozen=True)
class FilterDefinition:
    id: str
    name: str
    family_id: str
    summary: str
    creative_description: str
    operational_description: str
    reads_dimensions: tuple[RelationshipDimension, ...]
    changes_dimensions: tuple[RelationshipDimension, ...]
    preserves: dict[str, str]
    required_inputs: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    supported_output_forms: tuple[str, ...]
    parameters: tuple[FilterParameter, ...]
    implemented: bool
    experimental: bool
    version: str
    implementation_key: str | None = None
    implementation_class: str = "F"
    execution_mode: str = "unavailable"
    sparse_schedule: bool = False
    requires_speaker_identity: bool = False
    requires_output_acceptance: bool = False
    legacy_aliases: tuple[str, ...] = ()
    supports_preview: bool = False
    supports_stacking: bool = False
    destructive: bool = True
    requires_original_dialogue: bool = True
    can_precede: tuple[str, ...] = ()
    can_follow: tuple[str, ...] = ()
    incompatible_filters: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()

    @property
    def parameter_defaults(self) -> dict[str, Any]:
        return {parameter.id: parameter.default for parameter in self.parameters}

    def normalize_parameters(self, supplied: dict[str, Any] | None = None) -> dict[str, Any]:
        values = self.parameter_defaults
        values.update(supplied or {})
        known = {parameter.id: parameter for parameter in self.parameters}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"Unknown parameters for {self.name}: {', '.join(unknown)}")
        return {key: known[key].validate(value) for key, value in values.items()}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reads_dimensions"] = [item.value for item in self.reads_dimensions]
        data["changes_dimensions"] = [item.value for item in self.changes_dimensions]
        return data


@dataclass
class FilterExecutionContext:
    source_media: Path | None = None
    destination_media: Path | None = None
    source_dialogue: dict[str, Any] | None = None
    destination_dialogue: dict[str, Any] | None = None
    source_performances: dict[str, Any] | None = None
    destination_performances: dict[str, Any] | None = None
    source_speakers: dict[str, Any] | None = None
    destination_speakers: dict[str, Any] | None = None
    scenes: dict[str, Any] | None = None
    shots: dict[str, Any] | None = None
    semantic_features: dict[str, Any] | None = None
    emotional_features: dict[str, Any] | None = None
    target_duration: float | None = None
    output_form: str = "best_short"
    random_seed: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)
    cache_references: dict[str, str] = field(default_factory=dict)


@dataclass
class TransformationPlan:
    filter_id: str
    filter_version: str
    family_id: str
    deterministic_seed: int
    selected_destination_regions: list[dict[str, Any]] = field(default_factory=list)
    mappings: list[dict[str, Any]] = field(default_factory=list)
    speaker_mappings: list[dict[str, Any]] = field(default_factory=list)
    time_relationships: list[dict[str, Any]] = field(default_factory=list)
    progression_values: list[dict[str, Any]] = field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compatibility_decisions: list[dict[str, Any]] = field(default_factory=list)
    analysis_artifacts_used: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformationPlan":
        return cls(**data)
