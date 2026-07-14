from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import read_json


@dataclass(frozen=True)
class RemixMode:
    mode_id: str
    display_name: str
    short_description: str
    source_requirements: dict[str, Any]
    default_settings: dict[str, Any]
    candidate_generation_strategy: str
    scoring_profile: str
    assembly_strategy: str
    output_report_fields: tuple[str, ...]
    ui_visibility: str


@dataclass(frozen=True)
class RemixModeRegistry:
    default_mode_id: str
    modes: dict[str, RemixMode]

    @property
    def default_mode(self) -> RemixMode:
        return self.get(self.default_mode_id)

    def get(self, mode_id: str) -> RemixMode:
        try:
            return self.modes[mode_id]
        except KeyError as exc:
            raise KeyError(f"Unknown remix mode: {mode_id}") from exc

    def visible_modes(self) -> list[RemixMode]:
        order = {"default": 0, "advanced": 1, "planned": 2, "hidden": 3}
        return sorted(
            self.modes.values(),
            key=lambda mode: (order.get(mode.ui_visibility, 9), mode.display_name),
        )


def load_remix_mode_registry(root: Path) -> RemixModeRegistry:
    remix_dir = root / "remix_modes"
    registry = read_json(remix_dir / "registry.json")
    modes: dict[str, RemixMode] = {}
    for filename in registry.get("mode_files", []):
        data = read_json(remix_dir / str(filename))
        mode = RemixMode(
            mode_id=str(data["mode_id"]),
            display_name=str(data["display_name"]),
            short_description=str(data["short_description"]),
            source_requirements=dict(data.get("source_requirements", {})),
            default_settings=dict(data.get("default_settings", {})),
            candidate_generation_strategy=str(data["candidate_generation_strategy"]),
            scoring_profile=str(data["scoring_profile"]),
            assembly_strategy=str(data["assembly_strategy"]),
            output_report_fields=tuple(str(field) for field in data.get("output_report_fields", [])),
            ui_visibility=str(data.get("ui_visibility", "planned")),
        )
        modes[mode.mode_id] = mode
    return RemixModeRegistry(default_mode_id=str(registry["default_mode_id"]), modes=modes)
