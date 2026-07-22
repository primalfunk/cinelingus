from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..util import read_json
from .schema import FILM_MODEL_BUILDER_VERSION, FILM_MODEL_SCHEMA_VERSION


@dataclass(frozen=True)
class CacheDecision:
    reuse: bool
    status: str
    reasons: tuple[str, ...]


def evaluate_model_cache(model_path: Path, expected_signature: str, *, force: bool = False) -> CacheDecision:
    if force:
        return CacheDecision(False, "FORCE_REBUILD", ("Force rebuild was requested.",))
    if not model_path.exists():
        return CacheDecision(False, "CACHE_MISS", ("No cached FilmModel exists.",))
    try:
        model = read_json(model_path)
    except (OSError, ValueError) as exc:
        return CacheDecision(False, "INCOMPATIBLE_CACHE", (f"Cached FilmModel could not be read: {exc}",))
    reasons: list[str] = []
    if model.get("schema_version") != FILM_MODEL_SCHEMA_VERSION:
        reasons.append("FilmModel schema version changed.")
    if model.get("builder_version") != FILM_MODEL_BUILDER_VERSION:
        reasons.append("FilmModel builder version changed.")
    if model.get("created_from_signature") != expected_signature:
        reasons.append("Relevant source artifact or construction signature changed.")
    if model.get("validation_state", {}).get("status") not in {"VALID", "VALID_WITH_WARNINGS"}:
        reasons.append("Cached FilmModel is not valid.")
    if reasons:
        return CacheDecision(False, "REBUILD_REQUIRED", tuple(reasons))
    return CacheDecision(True, "CACHE_HIT", ("Schema, builder, source signature, and validation state are compatible.",))


def storage_footprint(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

