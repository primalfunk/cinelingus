from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bundle import USABLE_STATUSES, load_vector, validate_semantic_bundle
from .config import SemanticMode
from .similarity import SemanticEntity
from ..util import stable_hash


@dataclass(frozen=True)
class SemanticScheduleContext:
    mode: SemanticMode
    weight: float
    source_by_reference: dict[str, SemanticEntity]
    destination_by_reference: dict[str, SemanticEntity]
    model_identity: dict[str, Any]
    source_by_start: dict[str, tuple[SemanticEntity, ...]] = field(default_factory=dict)
    destination_by_start: dict[str, tuple[SemanticEntity, ...]] = field(default_factory=dict)
    source_by_text: dict[str, tuple[SemanticEntity, ...]] = field(default_factory=dict)
    destination_by_text: dict[str, tuple[SemanticEntity, ...]] = field(default_factory=dict)
    source_by_performance: dict[str, SemanticEntity] = field(default_factory=dict)
    destination_by_performance: dict[str, SemanticEntity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", SemanticMode(self.mode))
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Semantic scheduling weight must be between 0 and 1")
        if self.mode is not SemanticMode.ASSISTED and self.weight != 0.0:
            raise ValueError("Only assisted semantic mode may influence ranking")

    @property
    def active(self) -> bool:
        return self.mode is SemanticMode.REPORT_ONLY or (self.mode is SemanticMode.ASSISTED and self.weight > 0.0)

    @classmethod
    def from_bundles(
        cls, *, mode: SemanticMode, weight: float,
        source_model: dict[str, Any], source_bundle: dict[str, Any], source_dir: Path,
        destination_model: dict[str, Any], destination_bundle: dict[str, Any], destination_dir: Path,
    ) -> "SemanticScheduleContext":
        for label, model, bundle in (("source", source_model, source_bundle), ("destination", destination_model, destination_bundle)):
            if bundle.get("entity_type") != "speech_passage":
                raise ValueError(f"{label} scheduling bundle is not passage-level")
            if bundle.get("construction_state") != "READY":
                raise ValueError(f"{label} semantic bundle is not READY")
            if bundle.get("film_id") != model.get("film_id") or bundle.get("film_model_signature") != model.get("created_from_signature"):
                raise ValueError(f"{label} semantic bundle does not match its FilmModel")
            directory = source_dir if label == "source" else destination_dir
            validation = validate_semantic_bundle(bundle, directory, model)
            if validation["status"] != "VALID":
                raise ValueError(f"{label} semantic bundle failed admission validation: {validation['errors']}")
        provider_identity = _provider_identity(source_bundle)
        if provider_identity != _provider_identity(destination_bundle):
            raise ValueError("Source and destination semantic bundles use incompatible providers")
        identity = {
            **provider_identity,
            "source_bundle_signature": stable_hash(source_bundle),
            "destination_bundle_signature": stable_hash(destination_bundle),
        }
        return cls(
            mode, weight,
            _index_by_source_reference(source_model, source_bundle, source_dir),
            _index_by_source_reference(destination_model, destination_bundle, destination_dir),
            identity,
            _index_by_start(source_model, source_bundle, source_dir),
            _index_by_start(destination_model, destination_bundle, destination_dir),
            _index_by_text(source_model, source_bundle, source_dir),
            _index_by_text(destination_model, destination_bundle, destination_dir),
            _index_by_performance_reference(source_model, source_bundle, source_dir),
            _index_by_performance_reference(destination_model, destination_bundle, destination_dir),
        )

    def annotate_clips(self, clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._annotate(row, self.source_by_reference, _clip_references(row), self.source_by_start, self.source_by_text) for row in clips]

    def annotate_windows(self, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        annotated = []
        for row in windows:
            item = dict(row)
            children = [self._annotate(child, self.destination_by_reference, [str(child.get("id"))], self.destination_by_start, self.destination_by_text) for child in row.get("speech_windows") or []]
            if children:
                item["speech_windows"] = children
                item = inherit_aggregate_semantics(item, children)
            else:
                item = self._annotate(item, self.destination_by_reference, [str(item.get("id"))], self.destination_by_start, self.destination_by_text)
            if not item.get("_semantic_vector"):
                item = self._annotate_performance_fallback(item, self.destination_by_performance)
            annotated.append(item)
        return annotated

    def annotate_source_performance_groups(self, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        annotated = []
        for group in groups:
            entity = self.source_by_performance.get(str(group.get("id")))
            clips = []
            for clip in group.get("clips") or []:
                item = dict(clip)
                if entity is not None and not item.get("_semantic_vector"):
                    item = self._apply_performance_entity(item, entity)
                clips.append(item)
            enriched = dict(group)
            enriched["clips"] = clips
            enriched = inherit_aggregate_semantics(enriched, clips)
            if not enriched.get("_semantic_vector"):
                enriched = self._annotate_performance_fallback(enriched, self.source_by_performance)
            annotated.append(enriched)
        return annotated

    def _annotate_performance_fallback(self, row: dict[str, Any], index: dict[str, SemanticEntity]) -> dict[str, Any]:
        reference = str(row.get("performance_id") or row.get("id") or "")
        entity = index.get(reference)
        return self._apply_performance_entity(row, entity) if entity is not None else row

    def _apply_performance_entity(self, row: dict[str, Any], entity: SemanticEntity) -> dict[str, Any]:
        item = dict(row)
        item["_semantic_vector"] = entity.vector
        item["_semantic_entity_ids"] = list(entity.provenance.get("speech_passage_ids") or [entity.entity_id])
        item["_semantic_language"] = entity.language
        item["_semantic_mode"] = self.mode.value
        item["_semantic_weight"] = self.weight
        item["_semantic_model_identity"] = self.model_identity
        item["_semantic_evidence_scope"] = "performance_passage_aggregate"
        return item

    def _annotate(
        self, row: dict[str, Any], index: dict[str, SemanticEntity], references: list[str],
        by_start: dict[str, tuple[SemanticEntity, ...]],
        by_text: dict[str, tuple[SemanticEntity, ...]],
    ) -> dict[str, Any]:
        item = dict(row)
        entities = [index[reference] for reference in references if reference in index]
        evidence_scope = "direct_passage" if entities else None
        # Clip libraries may retain raw event IDs while FilmModel passages point
        # at filtered-window IDs. Both artifacts preserve the canonical media
        # start, so use that deterministic bridge only when direct IDs miss.
        if not entities:
            entities = list(by_start.get(_start_key(row), ()))
            if entities:
                evidence_scope = "direct_passage"
        if not entities:
            entities = _boundary_start_matches(row, by_start)
            if entities:
                evidence_scope = "direct_passage_boundary_bridge"
        if not entities:
            entities = _text_matches(row, by_text)
            if entities:
                evidence_scope = "direct_passage_text_bridge"
        if entities:
            item["_semantic_vector"] = _mean_vector([entity.vector for entity in entities])
            item["_semantic_entity_ids"] = [entity.entity_id for entity in entities]
            languages = {entity.language for entity in entities if entity.language}
            item["_semantic_language"] = next(iter(languages)) if len(languages) == 1 else ("mixed" if languages else None)
            item["_semantic_evidence_scope"] = evidence_scope
        item["_semantic_mode"] = self.mode.value
        item["_semantic_weight"] = self.weight
        item["_semantic_model_identity"] = self.model_identity
        return item


def inherit_aggregate_semantics(target: dict[str, Any], children: list[dict[str, Any]]) -> dict[str, Any]:
    item = dict(target)
    vectors = [child["_semantic_vector"] for child in children if child.get("_semantic_vector")]
    if not vectors:
        return item
    item["_semantic_vector"] = _mean_vector(vectors)
    item["_semantic_entity_ids"] = [entity_id for child in children for entity_id in child.get("_semantic_entity_ids", [])]
    languages = {child.get("_semantic_language") for child in children if child.get("_semantic_language")}
    item["_semantic_language"] = next(iter(languages)) if len(languages) == 1 else ("mixed" if languages else None)
    for key in ("_semantic_mode", "_semantic_weight", "_semantic_model_identity"):
        if key in children[0]:
            item[key] = children[0][key]
    return item


def semantic_compatibility(source: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any] | None:
    mode_value = source.get("_semantic_mode") or destination.get("_semantic_mode")
    if not mode_value:
        return None
    mode = SemanticMode(mode_value)
    weight = float(source.get("_semantic_weight", destination.get("_semantic_weight", 0.0)) or 0.0)
    source_vector, destination_vector = source.get("_semantic_vector"), destination.get("_semantic_vector")
    available = bool(source_vector and destination_vector)
    raw, normalized = None, 0.0
    warnings: list[str] = []
    if available:
        raw = _cosine(source_vector, destination_vector)
        normalized = (raw + 1.0) / 2.0
    else:
        warnings.append("Source or destination passage semantic evidence is unavailable; legacy score preserved.")
    source_language, destination_language = source.get("_semantic_language"), destination.get("_semantic_language")
    relationship = "unknown"
    if source_language and destination_language:
        relationship = "same_language" if source_language == destination_language else "cross_language"
    return {
        "source_semantic_entity_ids": list(source.get("_semantic_entity_ids", [])),
        "destination_semantic_entity_ids": list(destination.get("_semantic_entity_ids", [])),
        "raw_cosine_similarity": raw, "normalized_semantic_contribution": normalized,
        "configured_weight": weight,
        "effective_weighted_contribution": weight * normalized if available and mode is SemanticMode.ASSISTED else 0.0,
        "language_relationship": relationship,
        "model_identity": source.get("_semantic_model_identity") or destination.get("_semantic_model_identity") or {},
        "source_evidence_scope": source.get("_semantic_evidence_scope", "direct_passage" if source_vector else "unavailable"),
        "destination_evidence_scope": destination.get("_semantic_evidence_scope", "direct_passage" if destination_vector else "unavailable"),
        "warnings": warnings, "fallback_state": "NONE" if available else "NEUTRAL_LEGACY_SCORE",
        "mode": mode.value, "available": available,
    }


def apply_semantic_contribution(base_score: float, semantic: dict[str, Any] | None) -> float:
    if not semantic or not semantic["available"] or semantic["mode"] != SemanticMode.ASSISTED.value:
        return base_score
    weight = float(semantic["configured_weight"])
    return base_score if weight <= 0.0 else base_score * (1.0 - weight) + float(semantic["normalized_semantic_contribution"]) * weight


def _index_by_source_reference(model: dict[str, Any], bundle: dict[str, Any], directory: Path) -> dict[str, SemanticEntity]:
    metadata = {row["source_entity_id"]: row for row in bundle.get("entities") or [] if row.get("embedding_status") in USABLE_STATUSES}
    index = {}
    for passage in model.get("speech_passages") or []:
        entity_id, reference = passage["speech_passage_id"], passage.get("source_transcript_reference")
        row = metadata.get(entity_id)
        if row and reference:
            index[str(reference)] = SemanticEntity(entity_id, model["film_id"], "speech_passage", passage.get("language"), load_vector(row, directory), {"source_provenance_id": row["source_provenance_id"]})
    return index


def _index_by_start(
    model: dict[str, Any], bundle: dict[str, Any], directory: Path,
) -> dict[str, tuple[SemanticEntity, ...]]:
    metadata = {row["source_entity_id"]: row for row in bundle.get("entities") or [] if row.get("embedding_status") in USABLE_STATUSES}
    grouped: dict[str, list[SemanticEntity]] = {}
    for passage in model.get("speech_passages") or []:
        entity_id = passage["speech_passage_id"]
        row = metadata.get(entity_id)
        if row is None:
            continue
        entity = SemanticEntity(
            entity_id, model["film_id"], "speech_passage", passage.get("language"),
            load_vector(row, directory), {
                "source_provenance_id": row["source_provenance_id"],
                "start": float(passage.get("start", 0.0) or 0.0),
                "normalized_text": _normalized_text(passage.get("original_transcript")),
            },
        )
        grouped.setdefault(_start_key(passage), []).append(entity)
    return {key: tuple(values) for key, values in grouped.items()}


def _index_by_text(
    model: dict[str, Any], bundle: dict[str, Any], directory: Path,
) -> dict[str, tuple[SemanticEntity, ...]]:
    metadata = {row["source_entity_id"]: row for row in bundle.get("entities") or [] if row.get("embedding_status") in USABLE_STATUSES}
    grouped: dict[str, list[SemanticEntity]] = {}
    for passage in model.get("speech_passages") or []:
        row = metadata.get(passage["speech_passage_id"])
        text = _normalized_text(passage.get("original_transcript"))
        if row is None or not text:
            continue
        grouped.setdefault(text, []).append(SemanticEntity(
            passage["speech_passage_id"], model["film_id"], "speech_passage", passage.get("language"),
            load_vector(row, directory), {
                "source_provenance_id": row["source_provenance_id"],
                "start": float(passage.get("start", 0.0) or 0.0),
            },
        ))
    return {key: tuple(values) for key, values in grouped.items()}


def _index_by_performance_reference(
    model: dict[str, Any], bundle: dict[str, Any], directory: Path,
) -> dict[str, SemanticEntity]:
    metadata = {row["source_entity_id"]: row for row in bundle.get("entities") or [] if row.get("embedding_status") in USABLE_STATUSES}
    passages = {str(row.get("speech_passage_id")): row for row in model.get("speech_passages") or []}
    vectors = {
        passage_id: SemanticEntity(
            passage_id, model["film_id"], "speech_passage", passage.get("language"),
            load_vector(metadata[passage_id], directory), {"source_provenance_id": metadata[passage_id]["source_provenance_id"]},
        )
        for passage_id, passage in passages.items() if passage_id in metadata
    }
    result = {}
    for performance in model.get("performances") or []:
        reference = str(performance.get("source_performance_reference") or "")
        if not reference:
            continue
        passage_ids = [str(value) for value in performance.get("speech_passage_references") or [] if str(value) in vectors]
        if not passage_ids:
            start, end = float(performance.get("start", 0.0) or 0.0), float(performance.get("end", 0.0) or 0.0)
            passage_ids = [
                passage_id for passage_id, passage in passages.items()
                if passage_id in vectors and min(end, float(passage.get("end", 0.0) or 0.0)) > max(start, float(passage.get("start", 0.0) or 0.0))
            ]
        entities = [vectors[value] for value in passage_ids]
        if not entities:
            continue
        languages = {entity.language for entity in entities if entity.language}
        result[reference] = SemanticEntity(
            str(performance.get("performance_id") or reference), model["film_id"], "performance_passage_aggregate",
            next(iter(languages)) if len(languages) == 1 else ("mixed" if languages else None),
            _mean_vector([entity.vector for entity in entities]),
            {"speech_passage_ids": passage_ids, "source_performance_reference": reference},
        )
    return result


def _start_key(row: dict[str, Any]) -> str:
    value = row.get("movie_timestamp", row.get("start", 0.0))
    return f"{float(value or 0.0):.3f}"


def _text_matches(
    row: dict[str, Any], by_text: dict[str, tuple[SemanticEntity, ...]],
) -> list[SemanticEntity]:
    text = _normalized_text(row.get("transcript") or row.get("original_transcript"))
    if not text:
        return []
    exact = by_text.get(text)
    if exact:
        if len(text.split()) >= 2:
            return list(exact)
        return _nearest_unambiguous(row, list(exact), maximum_delta=3.0, minimum_margin=0.5)
    if len(text.split()) < 2:
        return []
    candidates = [
        entity
        for passage_text, entities in by_text.items()
        if len(passage_text.split()) >= 2 and (f" {text} " in f" {passage_text} " or f" {passage_text} " in f" {text} ")
        for entity in entities
    ]
    if len(candidates) <= 1:
        return candidates
    start = float(row.get("movie_timestamp", row.get("start", 0.0)) or 0.0)
    return [min(candidates, key=lambda entity: abs(float(entity.provenance.get("start", 0.0)) - start))]


def _boundary_start_matches(
    row: dict[str, Any], by_start: dict[str, tuple[SemanticEntity, ...]],
) -> list[SemanticEntity]:
    start = float(row.get("movie_timestamp", row.get("start", 0.0)) or 0.0)
    row_text = _normalized_text(row.get("transcript") or row.get("original_transcript"))
    candidates = []
    for key, entities in by_start.items():
        delta = abs(float(key) - start)
        if delta > 1.0:
            continue
        for entity in entities:
            passage_text = str(entity.provenance.get("normalized_text") or "")
            if delta > 0.5 and (
                not row_text or not passage_text
                or SequenceMatcher(None, row_text, passage_text).ratio() < 0.5
            ):
                continue
            candidates.append(entity)
    return _nearest_unambiguous(row, candidates, maximum_delta=1.0, minimum_margin=0.05)


def _nearest_unambiguous(
    row: dict[str, Any], candidates: list[SemanticEntity], *, maximum_delta: float, minimum_margin: float,
) -> list[SemanticEntity]:
    if not candidates:
        return []
    start = float(row.get("movie_timestamp", row.get("start", 0.0)) or 0.0)
    ranked = sorted(
        ((abs(float(entity.provenance.get("start", 0.0)) - start), entity) for entity in candidates),
        key=lambda value: (value[0], value[1].entity_id),
    )
    if ranked[0][0] > maximum_delta:
        return []
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < minimum_margin:
        return []
    return [ranked[0][1]]


def _normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _provider_identity(bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle.get("provider_metadata") or {}
    keys = ("provider", "model_id", "model_revision", "tokenizer_id", "dimensions", "token_limit", "pooling_policy", "normalization", "precision", "asset_digest")
    return {key: metadata.get(key) for key in keys}


def _clip_references(clip: dict[str, Any]) -> list[str]:
    values = list(clip.get("event_ids") or [])
    if clip.get("event_id") and clip.get("event_id") not in values:
        values.append(clip["event_id"])
    return [str(value) for value in values]


def _mean_vector(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Cannot aggregate semantic vectors with different dimensions")
    return _normalize(tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)))


def _normalize(vector: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero semantic vector")
    return tuple(value / norm for value in vector)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("Semantic vector dimensions do not match")
    denominator = math.sqrt(sum(a*a for a in left) * sum(b*b for b in right))
    if denominator == 0.0:
        raise ValueError("Semantic vector cannot be zero")
    return max(-1.0, min(1.0, sum(a*b for a, b in zip(left, right)) / denominator))
