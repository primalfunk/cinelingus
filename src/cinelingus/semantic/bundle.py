from __future__ import annotations

import hashlib
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..util import read_json, stable_hash, write_json
from .config import SEMANTIC_BUILDER_VERSION, SEMANTIC_SCHEMA_VERSION, SemanticConfig, SemanticTextRole
from .providers import SemanticProvider, SemanticProviderUnavailable

USABLE_STATUSES = frozenset({"EMBEDDED", "TRUNCATED", "LOW_INFORMATION"})
ENTITY_STATUSES = USABLE_STATUSES | {"SKIPPED", "UNAVAILABLE", "FAILED"}


@dataclass(frozen=True)
class SemanticBuildResult:
    bundle: dict[str, Any]
    validation_report: dict[str, Any]
    cache_report: dict[str, Any]


def build_semantic_bundle(
    model: dict[str, Any], output_dir: Path, provider: SemanticProvider, config: SemanticConfig,
    *, batch_size: int = 16, resume: bool = True, entity_type: str = "speech_passage",
) -> SemanticBuildResult:
    if batch_size < 1:
        raise ValueError("Semantic batch size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "semantic_bundle.json"
    provider_metadata = provider.describe()
    effective_signature = _effective_embedding_signature(config, provider_metadata)
    source_entities, structural_exclusions = _semantic_source_entities(model, entity_type)
    prior = _compatible_prior(bundle_path, model, effective_signature, entity_type) if resume else None
    reusable = {
        row["source_entity_id"]: row for row in (prior or {}).get("entities", [])
        if _entity_vector_is_current(row, output_dir, config)
    }
    bundle = _new_bundle(model, provider_metadata, effective_signature, entity_type, len(source_entities), structural_exclusions)
    passages = sorted(source_entities, key=lambda row: str(row.get("speech_passage_id")))
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    hits = 0
    for passage in passages:
        inputs = _entity_inputs(model, passage, config, effective_signature)
        cached = reusable.get(inputs["source_entity_id"])
        if cached and cached.get("entity_cache_signature") == inputs["entity_cache_signature"]:
            bundle["entities"].append(cached)
            hits += 1
        elif not inputs["canonical_semantic_text"]:
            bundle["entities"].append(_non_vector_record(inputs, "SKIPPED", "Transcript is empty after deterministic whitespace normalization."))
        else:
            pending.append((passage, inputs))
    bundle["entities"].sort(key=lambda row: row["source_entity_id"])
    write_json(bundle_path, bundle)

    encoded = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset: offset + batch_size]
        texts = [inputs["canonical_semantic_text"] for _, inputs in batch]
        try:
            result = provider.encode(texts, role=SemanticTextRole.PASSAGE)
            if len(result.vectors) != len(batch):
                raise ValueError("Semantic provider returned the wrong batch size")
            for (_, inputs), vector, token_count, truncated in zip(batch, result.vectors, result.token_counts, result.truncated):
                record = _vector_record(inputs, vector, token_count, truncated, output_dir, config)
                bundle["entities"].append(record)
                encoded += 1
        except SemanticProviderUnavailable as exc:
            for _, inputs in batch:
                bundle["entities"].append(_non_vector_record(inputs, "UNAVAILABLE", f"{exc.state}: {exc}"))
        except Exception as exc:
            for _, inputs in batch:
                bundle["entities"].append(_non_vector_record(inputs, "FAILED", f"{type(exc).__name__}: {exc}"))
        bundle["entities"].sort(key=lambda row: row["source_entity_id"])
        bundle["coverage"] = _coverage(bundle["entities"], len(passages))
        write_json(bundle_path, bundle)

    bundle["coverage"] = _coverage(bundle["entities"], len(passages))
    bundle["capability_overlay"] = _capability_overlay(bundle)
    bundle["construction_state"] = "VALIDATED"
    preliminary = validate_semantic_bundle(bundle, output_dir, model, require_ready=False)
    if preliminary["status"] == "VALID":
        bundle["construction_state"] = "READY"
        bundle["validation_state"] = {"status": "VALID", "validator_version": preliminary["validator_version"], "errors": [], "warnings": preliminary["warnings"]}
    else:
        bundle["construction_state"] = "INVALID"
        bundle["validation_state"] = {"status": "INVALID", "validator_version": preliminary["validator_version"], "errors": preliminary["errors"], "warnings": preliminary["warnings"]}
    write_json(bundle_path, bundle)
    validation = validate_semantic_bundle(bundle, output_dir, model)
    cache_report = {
        "status": "COMPLETE", "entity_count": len(passages), "cache_hits": hits,
        "entities_encoded": encoded, "entities_accounted_without_encoding": len(passages) - encoded,
        "resume_used": prior is not None, "embedding_configuration_signature": effective_signature,
    }
    return SemanticBuildResult(bundle, validation, cache_report)


def validate_semantic_bundle(
    bundle: dict[str, Any], output_dir: Path, model: dict[str, Any], *, require_ready: bool = True,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    try:
        source_entities, expected_exclusions = _semantic_source_entities(model, str(bundle.get("entity_type")))
    except ValueError as exc:
        source_entities, expected_exclusions = [], []
        errors.append(_issue("STRUCTURAL", "entity_type", str(exc)))
    passages = {row.get("speech_passage_id") for row in source_entities}
    entities = bundle.get("entities") or []
    entity_ids = [row.get("source_entity_id") for row in entities]
    if bundle.get("film_id") != model.get("film_id") or bundle.get("film_model_signature") != model.get("created_from_signature"):
        errors.append(_issue("IDENTITY", "bundle", "FilmModel identity or construction signature does not match."))
    if len(entity_ids) != len(set(entity_ids)):
        errors.append(_issue("IDENTITY", "entities", "Duplicate semantic entity records are present."))
    if set(entity_ids) != passages:
        errors.append(_issue("COVERAGE", "entities", "Every SpeechPassage must have exactly one semantic accounting record."))
    if bundle.get("structural_exclusions", []) != expected_exclusions:
        errors.append(_issue("STRUCTURAL", "structural_exclusions", "Structural exclusions do not match deterministic FilmModel evidence."))
    for index, row in enumerate(entities):
        status = row.get("embedding_status")
        if status not in ENTITY_STATUSES:
            errors.append(_issue("STRUCTURAL", f"entities[{index}]", f"Unknown embedding status {status}."))
            continue
        if status in USABLE_STATUSES:
            path = output_dir / str(row.get("vector_locator"))
            if not path.is_file():
                errors.append(_issue("VECTOR", f"entities[{index}]", "Vector file is missing."))
                continue
            payload = path.read_bytes()
            expected_bytes = int(row.get("embedding_dimension", 0)) * 4
            if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != row.get("vector_digest"):
                errors.append(_issue("VECTOR", f"entities[{index}]", "Vector size or digest is invalid."))
                continue
            values = struct.unpack(f"<{row['embedding_dimension']}f", payload)
            norm = math.sqrt(sum(value * value for value in values))
            if not math.isfinite(norm) or not 0.999 <= norm <= 1.001:
                errors.append(_issue("VECTOR", f"entities[{index}]", "Vector is not finite and L2 normalized."))
        elif status in {"UNAVAILABLE", "FAILED"}:
            warnings.append(_issue("COVERAGE", f"entities[{index}]", row.get("status_reason") or status))
    if require_ready and bundle.get("construction_state") != "READY":
        errors.append(_issue("STATE", "construction_state", "Only a READY semantic bundle is scheduling-eligible."))
    status = "INVALID" if errors else "VALID"
    return {
        "validator_version": "semantic_bundle_validator_v1", "status": status,
        "error_count": len(errors), "warning_count": len(warnings), "errors": errors, "warnings": warnings,
    }


def load_vector(entity: dict[str, Any], output_dir: Path) -> tuple[float, ...]:
    if entity.get("embedding_status") not in USABLE_STATUSES:
        raise ValueError("Semantic entity has no usable vector")
    payload = (output_dir / str(entity["vector_locator"])).read_bytes()
    return tuple(struct.unpack(f"<{entity['embedding_dimension']}f", payload))


def _new_bundle(
    model: dict[str, Any], provider_metadata: dict[str, object], effective_signature: str,
    entity_type: str, source_count: int, structural_exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_SCHEMA_VERSION, "builder_version": SEMANTIC_BUILDER_VERSION,
        "construction_state": "BUILDING", "film_id": model["film_id"],
        "film_model_signature": model["created_from_signature"],
        "entity_type": entity_type, "embedding_configuration_signature": effective_signature,
        "provider_metadata": provider_metadata, "entities": [],
        "coverage": _coverage([], source_count), "structural_exclusions": structural_exclusions,
        "capability_overlay": {},
        "validation_state": {"status": "NOT_VALIDATED", "validator_version": None, "errors": [], "warnings": []},
    }


def _entity_inputs(model: dict[str, Any], passage: dict[str, Any], config: SemanticConfig, effective_signature: str) -> dict[str, Any]:
    text = " ".join(str(passage.get("original_transcript") or "").split())
    transcript_signature = stable_hash(str(passage.get("original_transcript") or ""))
    text_signature = stable_hash(text)
    entity_id = str(passage["speech_passage_id"])
    cache_signature = stable_hash({
        "film_model_signature": model["created_from_signature"], "source_entity_id": entity_id,
        "transcript_signature": transcript_signature, "canonical_semantic_text_signature": text_signature,
        "embedding_configuration_signature": effective_signature,
    })
    structural_metadata = {}
    if passage.get("source_entity_type", "speech_passage") != "speech_passage":
        structural_metadata = {
            "source_entity_type": passage["source_entity_type"],
            "structural_references": passage.get("structural_references", {}),
            "source_provenance_ids": passage.get("source_provenance_ids", [passage["provenance_id"]]),
        }
    return {
        "film_id": model["film_id"], "source_entity_id": entity_id,
        "source_time_range": {key: passage[key] for key in ("start", "end", "duration")},
        "transcript_signature": transcript_signature, "source_provenance_id": passage["provenance_id"],
        "canonical_semantic_text": text, "canonical_semantic_text_signature": text_signature,
        "language_state": passage.get("language"), "model_id": config.model_id,
        "model_revision": config.model_revision, "tokenizer_id": config.tokenizer_id,
        "semantic_configuration_signature": effective_signature,
        "embedding_dimension": config.dimensions, "entity_cache_signature": cache_signature,
        **structural_metadata,
    }


def _vector_record(
    inputs: dict[str, Any], vector: tuple[float, ...], token_count: int, truncated: bool,
    output_dir: Path, config: SemanticConfig,
) -> dict[str, Any]:
    if len(vector) != config.dimensions or any(not math.isfinite(value) for value in vector):
        raise ValueError("Semantic vector has invalid dimension or values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("Semantic vector cannot be zero")
    normalized = tuple(float(value / norm) for value in vector)
    payload = struct.pack(f"<{len(normalized)}f", *normalized)
    relative = Path("vectors") / f"{inputs['source_entity_id']}.f32"
    _write_bytes_atomic(output_dir / relative, payload)
    low_information = len(inputs["canonical_semantic_text"].split()) <= 3
    status = "TRUNCATED" if truncated else ("LOW_INFORMATION" if low_information else "EMBEDDED")
    return {
        **{key: value for key, value in inputs.items() if key != "canonical_semantic_text"},
        "embedding_status": status, "vector_locator": relative.as_posix(),
        "vector_digest": hashlib.sha256(payload).hexdigest(), "token_count": int(token_count),
        "truncation_state": "TRUNCATED" if truncated else "COMPLETE",
        "low_information": low_information, "status_reason": None,
        "validation_state": "VALID",
    }


def _non_vector_record(inputs: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        **{key: value for key, value in inputs.items() if key != "canonical_semantic_text"},
        "embedding_status": status, "vector_locator": None, "vector_digest": None,
        "token_count": None, "truncation_state": "NOT_APPLICABLE", "low_information": False,
        "status_reason": reason, "validation_state": "UNAVAILABLE" if status == "UNAVAILABLE" else status,
    }


def _coverage(entities: list[dict[str, Any]], source_count: int) -> dict[str, Any]:
    counts = {status.lower(): 0 for status in sorted(ENTITY_STATUSES)}
    for row in entities:
        counts[str(row["embedding_status"]).lower()] += 1
    return {"source_entity_count": source_count, "accounted_entity_count": len(entities), "status_counts": counts}


def _capability_overlay(bundle: dict[str, Any]) -> dict[str, Any]:
    usable = sum(row["embedding_status"] in USABLE_STATUSES for row in bundle["entities"])
    total = len(bundle["entities"])
    status = "AVAILABLE" if usable == total and total else ("PARTIAL" if usable else "UNAVAILABLE")
    record = {
        "status": status, "producing_artifact": "semantic_bundle.json",
        "configuration_signature": bundle["embedding_configuration_signature"],
        "implementation_version": bundle["builder_version"],
        "coverage": {"usable_entity_count": usable, "source_entity_count": total},
        "known_limitations": ["Transcript-vector similarity only; no dialogue function, intention, emotion, character, scene, or narrative understanding."],
    }
    return {"semantic_embeddings": record, "semantic_similarity": dict(record)}


def _compatible_prior(path: Path, model: dict[str, Any], effective_signature: str, entity_type: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        prior = read_json(path)
    except (OSError, ValueError):
        return None
    if (
        prior.get("schema_version") != SEMANTIC_SCHEMA_VERSION
        or prior.get("builder_version") != SEMANTIC_BUILDER_VERSION
        or prior.get("film_id") != model.get("film_id")
        or prior.get("film_model_signature") != model.get("created_from_signature")
        or prior.get("embedding_configuration_signature") != effective_signature
        or prior.get("entity_type") != entity_type
    ):
        return None
    return prior


def _semantic_source_entities(model: dict[str, Any], entity_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if entity_type == "speech_passage":
        return list(model.get("speech_passages") or []), []
    if entity_type not in {"dialogue_turn", "turn_sequence"}:
        raise ValueError(f"Unsupported semantic entity type: {entity_type}")
    passages = {str(row.get("speech_passage_id")): row for row in model.get("speech_passages") or []}
    turns: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    for turn in sorted(model.get("dialogue_turns") or [], key=lambda row: str(row.get("dialogue_turn_id"))):
        turn_id = str(turn.get("dialogue_turn_id") or "")
        references = [str(value) for value in turn.get("ordered_speech_passage_references") or []]
        reasons = []
        if not turn_id:
            reasons.append("missing_stable_turn_id")
        if not references or len(references) != len(set(references)):
            reasons.append("missing_or_non_unique_passage_order")
        if any(reference not in passages for reference in references):
            reasons.append("missing_passage_reference")
        if not turn.get("provenance_id"):
            reasons.append("missing_turn_provenance")
        if float(turn.get("end", 0.0) or 0.0) < float(turn.get("start", 0.0) or 0.0):
            reasons.append("invalid_time_range")
        if reasons:
            exclusions.append({"source_entity_type": "dialogue_turn", "source_entity_id": turn_id or None, "reasons": reasons})
            continue
        ordered_passages = [passages[reference] for reference in references]
        languages = {row.get("language") for row in ordered_passages if row.get("language")}
        turns[turn_id] = {
            "speech_passage_id": turn_id,
            "source_entity_type": "dialogue_turn",
            "start": float(turn["start"]), "end": float(turn["end"]),
            "duration": float(turn.get("duration", float(turn["end"]) - float(turn["start"]))),
            "original_transcript": " ".join(str(row.get("original_transcript") or "").strip() for row in ordered_passages).strip(),
            "language": next(iter(languages)) if len(languages) == 1 else ("mixed" if languages else None),
            "provenance_id": turn["provenance_id"],
            "source_provenance_ids": [turn["provenance_id"], *[row["provenance_id"] for row in ordered_passages]],
            "structural_references": {"ordered_speech_passage_ids": references},
        }
    if entity_type == "dialogue_turn":
        return list(turns.values()), exclusions
    sequences: list[dict[str, Any]] = []
    for performance in sorted(model.get("performances") or [], key=lambda row: str(row.get("performance_id"))):
        performance_id = str(performance.get("performance_id") or "")
        references = [str(value) for value in performance.get("dialogue_turn_references") or []]
        reasons = []
        if not performance_id:
            reasons.append("missing_stable_performance_id")
        if not references or len(references) != len(set(references)):
            reasons.append("missing_or_non_unique_turn_order")
        if any(reference not in turns for reference in references):
            reasons.append("missing_or_invalid_turn_reference")
        if not performance.get("provenance_id"):
            reasons.append("missing_performance_provenance")
        if reasons:
            exclusions.append({"source_entity_type": "turn_sequence", "source_entity_id": performance_id or None, "reasons": reasons})
            continue
        ordered_turns = [turns[reference] for reference in references]
        languages = {row.get("language") for row in ordered_turns if row.get("language")}
        sequences.append({
            "speech_passage_id": performance_id,
            "source_entity_type": "turn_sequence",
            "start": float(performance["start"]), "end": float(performance["end"]),
            "duration": float(performance.get("duration", float(performance["end"]) - float(performance["start"]))),
            "original_transcript": " ".join(row["original_transcript"] for row in ordered_turns).strip(),
            "language": next(iter(languages)) if len(languages) == 1 else ("mixed" if languages else None),
            "provenance_id": performance["provenance_id"],
            "source_provenance_ids": [performance["provenance_id"], *[value for row in ordered_turns for value in row["source_provenance_ids"]]],
            "structural_references": {"ordered_dialogue_turn_ids": references},
        })
    return sequences, exclusions


def _effective_embedding_signature(config: SemanticConfig, metadata: dict[str, object]) -> str:
    identity_fields = (
        "provider", "model_id", "model_revision", "tokenizer_id", "dimensions", "prefix_policy",
        "token_limit", "truncation_policy", "pooling_policy", "normalization", "precision",
        "execution_device", "runtime", "torch_version", "transformers_version",
        "asset_digest",
    )
    return stable_hash({
        "semantic_configuration_signature": config.configuration_signature,
        "provider_identity": {key: metadata.get(key) for key in identity_fields if key in metadata},
    })


def _entity_vector_is_current(row: dict[str, Any], output_dir: Path, config: SemanticConfig) -> bool:
    if row.get("embedding_status") not in USABLE_STATUSES:
        return row.get("embedding_status") == "SKIPPED"
    path = output_dir / str(row.get("vector_locator"))
    if not path.is_file() or int(row.get("embedding_dimension", 0)) != config.dimensions:
        return False
    payload = path.read_bytes()
    return len(payload) == config.dimensions * 4 and hashlib.sha256(payload).hexdigest() == row.get("vector_digest")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _issue(category: str, location: str, message: str) -> dict[str, str]:
    return {"category": category, "location": location, "message": message}
