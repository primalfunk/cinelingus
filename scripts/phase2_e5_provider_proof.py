from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from cinelingus.semantic import (
    DEFAULT_E5_REVISION, LocalE5Provider, SemanticConfig, SemanticTextRole,
    build_semantic_bundle,
)
from cinelingus.util import read_json, write_json


def main() -> int:
    asset = ROOT / "models" / "semantic" / "intfloat-multilingual-e5-small" / DEFAULT_E5_REVISION
    model_path = ROOT / "temp" / "phase1_corpus_models" / "short_form_excerpt" / "film_model.json"
    if not asset.is_dir() or not model_path.is_file():
        raise FileNotFoundError("Prepare the pinned semantic model and Phase 1 corpus models first.")
    model = read_json(model_path)
    texts = (
        "The weather is lovely today.", "It is sunny outside.", "He drove to the stadium.",
        "El clima está precioso hoy.",
    )
    cpu_config = SemanticConfig(device="cpu")
    cpu_provider = LocalE5Provider(cpu_config, asset_dir=asset)
    started = time.perf_counter()
    first = cpu_provider.encode(texts, role=SemanticTextRole.PASSAGE)
    cpu_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    second = cpu_provider.encode(texts, role=SemanticTextRole.PASSAGE)
    cpu_repeat_seconds = time.perf_counter() - started
    output_dir = ROOT / "temp" / "phase2_e5_short_bundle"
    started = time.perf_counter()
    bundle_first = build_semantic_bundle(model, output_dir, cpu_provider, cpu_config, batch_size=4, resume=False)
    bundle_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    bundle_second = build_semantic_bundle(model, output_dir, cpu_provider, cpu_config, batch_size=4)
    bundle_cached_seconds = time.perf_counter() - started
    cuda_report = _cuda_check(asset, texts[:2])
    report = {
        "schema_version": "1.0", "evaluation_type": "phase2_pinned_e5_provider_proof",
        "provider_metadata": cpu_provider.describe(),
        "asset_size_bytes": sum(path.stat().st_size for path in asset.rglob("*") if path.is_file()),
        "cpu": {
            "first_load_and_four_entity_encode_seconds": cpu_first_seconds,
            "repeat_four_entity_encode_seconds": cpu_repeat_seconds,
            "repeat_exact": first.vectors == second.vectors,
            "dimensions": len(first.vectors[0]), "token_counts": list(first.token_counts),
            "truncated": list(first.truncated),
            "weather_sunny_cosine": _cosine(first.vectors[0], first.vectors[1]),
            "weather_stadium_cosine": _cosine(first.vectors[0], first.vectors[2]),
            "weather_spanish_cosine": _cosine(first.vectors[0], first.vectors[3]),
        },
        "cuda": cuda_report,
        "bundle": {
            "film_id": model["film_id"], "passage_count": len(model["speech_passages"]),
            "construction_state": bundle_second.bundle["construction_state"],
            "validation_status": bundle_second.validation_report["status"],
            "first_build_seconds": bundle_build_seconds, "cached_build_seconds": bundle_cached_seconds,
            "second_build_cache_hits": bundle_second.cache_report["cache_hits"],
            "deterministic_metadata": bundle_first.bundle == bundle_second.bundle,
            "coverage": bundle_second.bundle["coverage"],
        },
        "claims": [
            "Scores are raw transcript-vector cosine similarity, not probabilities.",
            "The weather examples are provider sanity checks, not Translation quality evidence.",
            "Cross-language quality is not inferred from one Spanish example.",
        ],
    }
    output = ROOT / "evaluation" / "phase2_e5_provider_proof_20260721.json"
    write_json(output, report)
    print(output)
    print(f"cpu_exact={report['cpu']['repeat_exact']} cuda={cuda_report['available']} bundle={report['bundle']['validation_status']} cache_hits={report['bundle']['second_build_cache_hits']}")
    return 0 if report["cpu"]["repeat_exact"] and report["bundle"]["validation_status"] == "VALID" else 1


def _cuda_check(asset: Path, texts: tuple[str, ...]) -> dict:
    if not torch.cuda.is_available():
        return {"available": False, "status": "NOT_AVAILABLE"}
    torch.cuda.reset_peak_memory_stats()
    provider = LocalE5Provider(SemanticConfig(device="cuda"), asset_dir=asset)
    started = time.perf_counter()
    first = provider.encode(texts, role=SemanticTextRole.PASSAGE)
    first_seconds = time.perf_counter() - started
    second = provider.encode(texts, role=SemanticTextRole.PASSAGE)
    return {
        "available": True, "status": "WORKING", "device_name": torch.cuda.get_device_name(0),
        "first_load_and_encode_seconds": first_seconds, "repeat_exact": first.vectors == second.vectors,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


if __name__ == "__main__":
    raise SystemExit(main())
