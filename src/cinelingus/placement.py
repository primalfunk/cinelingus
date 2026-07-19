from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlacementResult:
    placer: str
    source_count: int
    destination_count: int
    placement_count: int
    strategy: str
    warnings: list[str] = field(default_factory=list)

    def to_plan_entry(self) -> dict[str, Any]:
        return {
            "placer": self.placer,
            "source_count": self.source_count,
            "destination_count": self.destination_count,
            "placement_count": self.placement_count,
            "strategy": self.strategy,
            "warnings": self.warnings,
        }


class PlaceIntoPerformances:
    placer = "PlaceIntoPerformances"

    def __init__(self, *, strategy: str = "chronological_best_fit") -> None:
        self.strategy = strategy

    def plan(self, source_items: list[dict[str, Any]], destination_performances: list[dict[str, Any]]) -> PlacementResult:
        warnings: list[str] = []
        if not source_items:
            warnings.append("no source dialogue selected")
        if not destination_performances:
            warnings.append("no destination performances selected")
        return PlacementResult(
            placer=self.placer,
            source_count=len(source_items),
            destination_count=len(destination_performances),
            placement_count=min(len(source_items), len(destination_performances)),
            strategy=self.strategy,
            warnings=warnings,
        )
