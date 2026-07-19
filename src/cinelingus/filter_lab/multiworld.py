from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import FilmInput, FilterDefinition


MULTIWORLD_STAGES = (
    "load_films",
    "inspect_films",
    "create_shared_timeline",
    "construct_world_model",
    "apply_cinematic_law",
    "generate_replacement_decisions",
    "review",
    "render",
)


@dataclass
class MultiworldRunState:
    definition: FilterDefinition
    films: tuple[FilmInput, ...]
    seed: int
    film_inspections: dict[str, dict[str, Any]] = field(default_factory=dict)
    shared_timeline: dict[str, Any] = field(default_factory=dict)
    world_model: dict[str, Any] = field(default_factory=dict)
    law_result: dict[str, Any] = field(default_factory=dict)
    replacement_decisions: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)

    @property
    def anchor(self) -> FilmInput:
        return next(film for film in self.films if film.is_anchor)


class MultiworldPipeline:
    """Contract-driven orchestration shared by every multi-film cinematic law."""

    def __init__(
        self,
        definition: FilterDefinition,
        films: Iterable[FilmInput | Path],
        *,
        anchor_index: int = 0,
        seed: int = 1,
        stage_callback: Callable[[str], None] | None = None,
    ) -> None:
        normalized = normalize_films(films, anchor_index=anchor_index)
        definition.validate_film_count(len(normalized))
        if not definition.is_multiworld:
            raise ValueError(f"{definition.name} is not a Multiworld filter.")
        self.state = MultiworldRunState(definition=definition, films=normalized, seed=int(seed))
        self.stage_callback = stage_callback
        self._complete("load_films")

    def inspect_films(self, inspector: Callable[[FilmInput], dict[str, Any]]) -> dict[str, dict[str, Any]]:
        self.state.film_inspections = {film.id: dict(inspector(film)) for film in self.state.films}
        self._complete("inspect_films")
        return self.state.film_inspections

    def create_shared_timeline(self, builder: Callable[[MultiworldRunState], dict[str, Any]] | None = None) -> dict[str, Any]:
        if builder is None:
            anchor = self.state.anchor
            anchor_data = self.state.film_inspections.get(anchor.id, {})
            self.state.shared_timeline = {
                "anchor_film_id": anchor.id,
                "behavior": self.state.definition.anchor_behavior,
                "duration": anchor_data.get("duration"),
                "film_ids": [film.id for film in self.state.films],
            }
        else:
            self.state.shared_timeline = dict(builder(self.state))
        self._complete("create_shared_timeline")
        return self.state.shared_timeline

    def construct_world_model(self, builder: Callable[[MultiworldRunState], dict[str, Any]] | None = None) -> dict[str, Any]:
        if builder is None:
            self.state.world_model = {
                "cinematic_law": self.state.definition.cinematic_law,
                "anchor_film_id": self.state.anchor.id,
                "films": [film.to_dict() for film in self.state.films],
                "film_inspections": self.state.film_inspections,
                "shared_timeline": self.state.shared_timeline,
                "affected_elements": list(self.state.definition.affected_elements),
                "deterministic_seed": self.state.seed,
            }
        else:
            self.state.world_model = dict(builder(self.state))
        self._complete("construct_world_model")
        return self.state.world_model

    def apply_cinematic_law(self, applicator: Callable[[MultiworldRunState], dict[str, Any]]) -> dict[str, Any]:
        if not self.state.definition.implemented:
            raise NotImplementedError("This filter is not yet implemented.")
        self.state.law_result = dict(applicator(self.state))
        self._complete("apply_cinematic_law")
        return self.state.law_result

    def generate_replacement_decisions(
        self, builder: Callable[[MultiworldRunState], dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        self.state.replacement_decisions = dict(builder(self.state) if builder else self.state.law_result)
        self._complete("generate_replacement_decisions")
        return self.state.replacement_decisions

    def review(self, reviewer: Callable[[MultiworldRunState], dict[str, Any]]) -> dict[str, Any]:
        self.state.review = dict(reviewer(self.state))
        self._complete("review")
        return self.state.review

    def render(self, renderer: Callable[[MultiworldRunState], dict[str, Any]]) -> dict[str, Any]:
        self.state.outputs = dict(renderer(self.state))
        self._complete("render")
        return self.state.outputs

    def _complete(self, stage: str) -> None:
        expected = MULTIWORLD_STAGES[len(self.state.completed_stages)]
        if stage != expected:
            raise RuntimeError(f"Multiworld stage '{stage}' cannot run before '{expected}'.")
        self.state.completed_stages.append(stage)
        if self.stage_callback:
            self.stage_callback(stage)


def normalize_films(films: Iterable[FilmInput | Path], *, anchor_index: int = 0) -> tuple[FilmInput, ...]:
    rows = list(films)
    if not rows:
        raise ValueError("At least one film is required.")
    if anchor_index < 0 or anchor_index >= len(rows):
        raise ValueError(f"Anchor index {anchor_index} is outside the {len(rows)} selected films.")
    resolved = []
    for value in rows:
        path = value.media_path if isinstance(value, FilmInput) else Path(value)
        try:
            resolved.append(str(Path(path).resolve()).casefold())
        except OSError:
            resolved.append(str(Path(path)).casefold())
    if len(set(resolved)) != len(resolved):
        raise ValueError("A Multiworld run requires distinct film paths; the same film was selected more than once.")
    normalized: list[FilmInput] = []
    for index, value in enumerate(rows):
        path = value.media_path if isinstance(value, FilmInput) else Path(value)
        normalized.append(FilmInput(id=f"film_{index + 1}", media_path=Path(path), label=f"Film {film_label(index)}", is_anchor=index == anchor_index))
    return tuple(normalized)


def film_label(index: int) -> str:
    if index < 0:
        raise ValueError("Film index must be non-negative.")
    value = index + 1
    label = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label
