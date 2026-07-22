from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..util import write_json


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [canonicalize(item) for item in value]
        if normalized and all(isinstance(item, dict) for item in normalized):
            id_keys = (
                "shot_id", "transition_id", "speech_passage_id", "speaker_cluster_id",
                "dialogue_turn_id", "performance_id", "cinematic_moment_id",
                "editorial_observation_id", "provenance_id", "source_artifact_id",
            )
            for id_key in id_keys:
                if all(id_key in item for item in normalized):
                    return sorted(normalized, key=lambda item: str(item[id_key]))
        return normalized
    return value


def canonical_json(model: dict[str, Any]) -> str:
    return json.dumps(canonicalize(model), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_film_model(path: Path, model: dict[str, Any]) -> None:
    write_json(path, canonicalize(model))


def read_film_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

