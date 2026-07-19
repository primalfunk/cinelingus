from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SelectionResult:
    selector: str
    object_type: str
    role: str
    source_artifact: str
    objects: list[dict[str, Any]]
    criteria: dict[str, Any] = field(default_factory=dict)

    def to_plan_entry(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "object_type": self.object_type,
            "role": self.role,
            "source_artifact": self.source_artifact,
            "count": len(self.objects),
            "criteria": self.criteria,
            "object_ids": [str(item.get("id", "")) for item in self.objects if item.get("id")],
        }


class BaseSelection:
    selector = "SelectObjects"
    object_type = "object"

    def __init__(self, *, role: str, source_artifact: str, criteria: dict[str, Any] | None = None) -> None:
        self.role = role
        self.source_artifact = source_artifact
        self.criteria = criteria or {}

    def select(self, artifact: dict[str, Any]) -> SelectionResult:
        return SelectionResult(
            selector=self.selector,
            object_type=self.object_type,
            role=self.role,
            source_artifact=self.source_artifact,
            objects=self._objects(artifact),
            criteria=self.criteria,
        )

    def _objects(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError


class SelectDialogue(BaseSelection):
    selector = "SelectDialogue"
    object_type = "dialogue_clip"

    def _objects(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            deepcopy(clip)
            for clip in artifact.get("clips", artifact.get("events", []))
            if float(clip.get("duration", 0.0) or 0.0) > 0.0 and clip.get("usable", True)
        ]


class SelectPerformances(BaseSelection):
    selector = "SelectPerformances"
    object_type = "performance"

    def _objects(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            deepcopy(performance)
            for performance in artifact.get("performances", [])
            if float(performance.get("duration", 0.0) or 0.0) > 0.0 and performance.get("usable", True)
        ]


class SelectPauses(BaseSelection):
    selector = "SelectPauses"
    object_type = "pause"

    def _objects(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        pauses: list[dict[str, Any]] = []
        for performance in artifact.get("performances", []):
            stats = performance.get("pause_statistics", {})
            if float(stats.get("max_pause", 0.0) or 0.0) <= 0.0:
                continue
            pauses.append(
                {
                    "id": f"{performance.get('id', 'performance')}:pause",
                    "performance_id": performance.get("id"),
                    "max_pause": stats.get("max_pause", 0.0),
                    "average_pause": stats.get("average_pause", 0.0),
                }
            )
        return pauses


class SelectShots(BaseSelection):
    selector = "SelectShots"
    object_type = "shot"

    def _objects(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        return [deepcopy(shot) for shot in artifact.get("shots", [])]


class SelectTimeline(BaseSelection):
    selector = "SelectTimeline"
    object_type = "speaking_window"

    def _objects(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            deepcopy(window)
            for window in artifact.get("windows", [])
            if float(window.get("duration", 0.0) or 0.0) > 0.0 and window.get("usable", True)
        ]
