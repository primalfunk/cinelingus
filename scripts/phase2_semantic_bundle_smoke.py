from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinelingus.semantic import (
    DeterministicFakeProvider, SemanticConfig, SemanticEntity, build_semantic_bundle,
    load_vector, top_k,
)
from cinelingus.util import read_json, write_json


def main() -> int:
    model_paths = sorted((ROOT / "temp" / "phase1_corpus_models").glob("*/film_model.json"))
    if not model_paths:
        raise FileNotFoundError("Run scripts/phase1_corpus_validation.py first.")
    output_root = ROOT / "temp" / "phase2_semantic_fake"
    rows = [_measure(path, output_root / path.parent.name) for path in model_paths]
    report = {
        "schema_version": "1.0", "evaluation_type": "phase2_semantic_bundle_mechanics",
        "provider_scope": "deterministic fake provider; no retrieval-quality claim",
        "model_count": len(rows), "all_ready": all(row["construction_state"] == "READY" for row in rows),
        "all_valid": all(row["validation_status"] == "VALID" for row in rows),
        "all_deterministic": all(row["deterministic_bundle"] for row in rows),
        "all_second_build_cache_hits": all(row["second_build_cache_hits"] == row["passage_count"] for row in rows),
        "source_media_unchanged": all(row["source_media_unchanged"] for row in rows),
        "models": rows,
    }
    output = ROOT / "evaluation" / "phase2_semantic_bundle_mechanics_20260721.json"
    write_json(output, report)
    print(output)
    print(
        f"models={len(rows)} ready={report['all_ready']} valid={report['all_valid']} "
        f"deterministic={report['all_deterministic']} cache_hits={report['all_second_build_cache_hits']} "
        f"source_unchanged={report['source_media_unchanged']}"
    )
    return 0 if all(report[key] for key in ("all_ready", "all_valid", "all_deterministic", "all_second_build_cache_hits", "source_media_unchanged")) else 1


def _measure(model_path: Path, output_dir: Path) -> dict:
    model = read_json(model_path)
    config = SemanticConfig()
    source_path = Path(str((model.get("media") or {}).get("source_path_reference") or ""))
    before = _stat(source_path)
    tracemalloc.start()
    started = time.perf_counter()
    first = build_semantic_bundle(model, output_dir, DeterministicFakeProvider(config), config, batch_size=32, resume=False)
    build_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    first_payload = (output_dir / "semantic_bundle.json").read_bytes()
    started = time.perf_counter()
    second = build_semantic_bundle(model, output_dir, DeterministicFakeProvider(config), config, batch_size=32)
    cached_build_seconds = time.perf_counter() - started
    second_payload = (output_dir / "semantic_bundle.json").read_bytes()
    entities = []
    for row in second.bundle["entities"]:
        if row["embedding_status"] in {"EMBEDDED", "TRUNCATED", "LOW_INFORMATION"}:
            entities.append(SemanticEntity(
                row["source_entity_id"], row["film_id"], "speech_passage", row.get("language_state"),
                load_vector(row, output_dir), {"source_provenance_id": row["source_provenance_id"]},
            ))
    search_started = time.perf_counter()
    matches = top_k(entities[0], entities, limit=5, source_film_id=model["film_id"], entity_type="speech_passage") if entities else ()
    search_seconds = time.perf_counter() - search_started
    return {
        "case_id": model_path.parent.name, "film_id": model["film_id"],
        "filename": (model.get("media") or {}).get("filename"),
        "passage_count": len(model.get("speech_passages") or []),
        "construction_state": second.bundle["construction_state"],
        "validation_status": second.validation_report["status"],
        "coverage": second.bundle["coverage"], "language_counts": _language_counts(second.bundle),
        "first_build_seconds": build_seconds, "cached_build_seconds": cached_build_seconds,
        "peak_traced_memory_bytes": peak_bytes, "bundle_storage_bytes": _size(output_dir),
        "second_build_cache_hits": second.cache_report["cache_hits"],
        "deterministic_bundle": first_payload == second_payload,
        "exact_top5_search_seconds": search_seconds,
        "sample_top5": [{"entity_id": row.candidate_entity_id, "raw_cosine": row.raw_cosine_similarity} for row in matches],
        "source_media_stat_available": before is not None,
        "source_media_unchanged": before == _stat(source_path),
    }


def _language_counts(bundle: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in bundle["entities"]:
        key = str(row.get("language_state") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _stat(path: Path) -> tuple[int, int] | None:
    if not str(path) or not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
