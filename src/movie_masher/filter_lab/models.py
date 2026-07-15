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
    SCENE_ORDER = "scene_order"
    EMOTION = "emotion"
    MUSIC = "music"
    SOUNDSCAPE = "soundscape"
    SHOT_SELECTION = "shot_selection"
    NARRATION = "narration"
    GENRE = "genre"


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
    minimum_films: int = 1
    maximum_films: int | None = 1
    anchor_behavior: str = "anchor_timeline"
    cinematic_law: str = "Internal Transformation"
    affected_elements: tuple[str, ...] = ()
    quality_requirements: tuple[str, ...] = ()
    deterministic_seed_support: bool = True
    output_artifacts: tuple[str, ...] = ()
    affected_artifacts: tuple[str, ...] = ()
    intermediate_products: tuple[str, ...] = ()

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

    def validate_film_count(self, count: int) -> None:
        if count < self.minimum_films:
            raise ValueError(f"{self.name} requires at least {self.minimum_films} films; received {count}.")
        if self.maximum_films is not None and count > self.maximum_films:
            raise ValueError(f"{self.name} accepts at most {self.maximum_films} films; received {count}.")

    @property
    def is_multiworld(self) -> bool:
        return self.family_id == "multiworld"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reads_dimensions"] = [item.value for item in self.reads_dimensions]
        data["changes_dimensions"] = [item.value for item in self.changes_dimensions]
        return data


@dataclass
class FilterExecutionContext:
    films: list["FilmInput"] = field(default_factory=list)
    anchor_film_id: str | None = None
    film_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    scenes: dict[str, Any] | None = None
    shots: dict[str, Any] | None = None
    semantic_features: dict[str, Any] | None = None
    emotional_features: dict[str, Any] | None = None
    target_duration: float | None = None
    output_form: str = "best_short"
    random_seed: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)
    cache_references: dict[str, str] = field(default_factory=dict)

    @property
    def anchor_film(self) -> "FilmInput | None":
        if not self.films:
            return None
        anchor_id = self.anchor_film_id or self.films[0].id
        return next((film for film in self.films if film.id == anchor_id), None)


@dataclass(frozen=True)
class FilmInput:
    id: str
    media_path: Path
    label: str
    is_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "media_path": str(self.media_path),
            "label": self.label,
            "is_anchor": self.is_anchor,
        }


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
    cinematic_law: str = "Internal Transformation"
    anchor_behavior: str = "anchor_timeline"
    film_count: dict[str, int | None] = field(default_factory=lambda: {"minimum": 1, "maximum": 1})
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    affected_artifacts: list[str] = field(default_factory=list)
    intermediate_products: list[str] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformationPlan":
        return cls(**data)
