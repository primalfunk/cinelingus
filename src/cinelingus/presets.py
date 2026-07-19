from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import read_json


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    description: str
    transformation_strategy: str
    input_requirements: list[str]
    selection_strategy: dict[str, Any]
    scheduling: dict[str, Any]
    render_outputs: dict[str, str]
    parameters: dict[str, Any]
    path: Path


def presets_dir(root: Path) -> Path:
    return root / "presets"


def list_presets(root: Path) -> list[Preset]:
    directory = presets_dir(root)
    if not directory.exists():
        return []
    return sorted((load_preset_file(path) for path in directory.glob("*.json")), key=lambda preset: preset.id)


def load_preset(root: Path, preset_id: str) -> Preset:
    candidates = [preset_id, f"{preset_id}.json"]
    for candidate in candidates:
        path = presets_dir(root) / candidate
        if path.exists():
            return load_preset_file(path)
    choices = ", ".join(preset.id for preset in list_presets(root)) or "none"
    raise ValueError(f"Unknown preset '{preset_id}'. Available presets: {choices}")


def load_preset_file(path: Path) -> Preset:
    raw = read_json(path)
    required = [
        "id",
        "name",
        "description",
        "input_requirements",
        "transformation_strategy",
        "selection_strategy",
        "scheduling",
        "render_outputs",
        "parameters",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Preset {path} is missing required fields: {', '.join(missing)}")
    return Preset(
        id=str(raw["id"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        transformation_strategy=str(raw["transformation_strategy"]),
        input_requirements=list(raw["input_requirements"]),
        selection_strategy=dict(raw["selection_strategy"]),
        scheduling=dict(raw["scheduling"]),
        render_outputs=dict(raw["render_outputs"]),
        parameters=dict(raw["parameters"]),
        path=path,
    )
