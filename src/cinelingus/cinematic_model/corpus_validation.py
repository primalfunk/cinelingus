from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..util import read_json
from .builder import SUPPORTED_ARTIFACTS, build_film_model
from .cache import evaluate_model_cache
from .reports import write_model_bundle
from .serialization import canonical_json
from .validation import validate_film_model


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    category: str
    artifact_dir: Path
    evidence_profile: str
    tier: str = "smoke"


def validate_corpus_cases(
    cases: Iterable[CorpusCase], *, schemas_dir: Path, output_root: Path,
) -> dict[str, Any]:
    """Build a bounded corpus without invoking analysis or modifying source media."""
    rows = [_measure_case(case, schemas_dir=schemas_dir, output_root=output_root) for case in cases]
    return {
        "schema_version": "1.0",
        "evaluation_type": "phase1_film_model_bounded_corpus",
        "case_count": len(rows),
        "all_valid": all(row["validation_status"] in {"VALID", "VALID_WITH_WARNINGS"} for row in rows),
        "all_deterministic": all(row["deterministic"] for row in rows),
        "all_cache_hits": all(row["cache_reuse_status"] == "CACHE_HIT" for row in rows),
        "source_media_unchanged": all(row["source_media_unchanged"] for row in rows),
        "cases": rows,
    }


def _measure_case(case: CorpusCase, *, schemas_dir: Path, output_root: Path) -> dict[str, Any]:
    artifacts = {
        name: case.artifact_dir / f"{name}.json"
        for name in SUPPORTED_ARTIFACTS
        if (case.artifact_dir / f"{name}.json").is_file()
    }
    if "movie" not in artifacts:
        raise FileNotFoundError(f"{case.case_id}: movie.json is missing from {case.artifact_dir}")
    movie = read_json(artifacts["movie"])
    media_path = Path(str(movie.get("path") or ""))
    before = _source_stat(media_path)

    tracemalloc.start()
    started = time.perf_counter()
    first = build_film_model(artifacts, schemas_dir=schemas_dir)
    first_build_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    started = time.perf_counter()
    second = build_film_model(dict(reversed(tuple(artifacts.items()))), schemas_dir=schemas_dir)
    second_build_seconds = time.perf_counter() - started
    first_json = canonical_json(first.model)
    second_json = canonical_json(second.model)

    output_dir = output_root / case.case_id
    started = time.perf_counter()
    paths = write_model_bundle(output_dir, first)
    serialization_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reloaded = read_json(paths["model"])
    reload_seconds = time.perf_counter() - started
    started = time.perf_counter()
    validation = validate_film_model(reloaded, schemas_dir)
    validation_seconds = time.perf_counter() - started
    cache = evaluate_model_cache(paths["model"], first.model["created_from_signature"])

    source_artifact_bytes = sum(path.stat().st_size for path in artifacts.values())
    model_bytes = paths["model"].stat().st_size
    provenance_bytes = len(canonical_json({
        "provenance": first.model["provenance"],
        "source_artifacts": first.model["source_artifacts"],
    }).encode("utf-8"))
    transcript = _transcript_duplication(first.model)
    after = _source_stat(media_path)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "tier": case.tier,
        "evidence_profile": case.evidence_profile,
        "artifact_dir": case.artifact_dir.resolve().as_posix(),
        "film_id": first.model["film_id"],
        "duration_seconds": first.model["timeline"]["duration"],
        "artifact_types": first.build_report["artifact_types_used"],
        "object_counts": first.build_report["object_counts"],
        "capability_status_counts": first.build_report["capability_status_counts"],
        "validation_status": validation["status"],
        "validation_error_count": validation["error_count"],
        "validation_warning_count": validation["warning_count"],
        "deterministic": first_json == second_json,
        "first_build_seconds": first_build_seconds,
        "second_build_seconds": second_build_seconds,
        "serialization_seconds": serialization_seconds,
        "reload_seconds": reload_seconds,
        "validation_seconds": validation_seconds,
        "peak_traced_memory_bytes": peak_bytes,
        "model_bytes": model_bytes,
        "source_artifact_bytes": source_artifact_bytes,
        "model_to_source_artifact_ratio": (model_bytes / source_artifact_bytes) if source_artifact_bytes else None,
        "provenance_registry_bytes": provenance_bytes,
        "provenance_share_of_model": (provenance_bytes / model_bytes) if model_bytes else None,
        "transcript_duplication": transcript,
        "cache_reuse_status": cache.status,
        "source_media_stat_available": before is not None,
        "source_media_unchanged": before == after,
    }


def _source_stat(path: Path) -> tuple[int, int] | None:
    if not str(path) or not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _transcript_duplication(model: dict[str, Any]) -> dict[str, int]:
    values: list[str] = []
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"transcript", "transcript_original", "normalized_transcript"} and isinstance(child, str) and child:
                    values.append(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for collection in ("speech_passages", "dialogue_turns", "performances"):
        visit(model.get(collection) or [])
    total = sum(len(value.encode("utf-8")) for value in values)
    unique = sum(len(value.encode("utf-8")) for value in set(values))
    return {"occurrence_bytes": total, "unique_value_bytes": unique, "estimated_duplicate_bytes": total - unique}
