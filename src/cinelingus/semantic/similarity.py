from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SemanticEntity:
    entity_id: str
    film_id: str
    entity_type: str
    language: str | None
    vector: tuple[float, ...]
    provenance: dict[str, object]


@dataclass(frozen=True)
class SemanticMatch:
    query_entity_id: str
    candidate_entity_id: str
    raw_cosine_similarity: float
    normalized_scheduling_contribution: float
    language_relationship: str
    provenance: dict[str, object]


def compare_entities(query: SemanticEntity, candidate: SemanticEntity) -> SemanticMatch:
    if len(query.vector) != len(candidate.vector) or not query.vector:
        raise ValueError("Semantic vectors must have equal non-zero dimensions")
    query_norm = math.sqrt(sum(value * value for value in query.vector))
    candidate_norm = math.sqrt(sum(value * value for value in candidate.vector))
    if query_norm == 0.0 or candidate_norm == 0.0:
        raise ValueError("Semantic vectors cannot be zero")
    cosine = sum(left * right for left, right in zip(query.vector, candidate.vector)) / (query_norm * candidate_norm)
    cosine = max(-1.0, min(1.0, cosine))
    relationship = "unknown"
    if query.language and candidate.language:
        relationship = "same_language" if query.language == candidate.language else "cross_language"
    return SemanticMatch(
        query.entity_id, candidate.entity_id, cosine, (cosine + 1.0) / 2.0, relationship,
        {"query": query.provenance, "candidate": candidate.provenance},
    )


def top_k(
    query: SemanticEntity,
    candidates: Iterable[SemanticEntity],
    *,
    limit: int,
    source_film_id: str | None = None,
    entity_type: str | None = None,
    language: str | None = None,
    prohibited_entity_ids: frozenset[str] = frozenset(),
) -> tuple[SemanticMatch, ...]:
    if limit < 0:
        raise ValueError("Top-k limit cannot be negative")
    eligible = (
        candidate for candidate in candidates
        if candidate.entity_id != query.entity_id
        and candidate.entity_id not in prohibited_entity_ids
        and (source_film_id is None or candidate.film_id == source_film_id)
        and (entity_type is None or candidate.entity_type == entity_type)
        and (language is None or candidate.language == language)
    )
    matches = [compare_entities(query, candidate) for candidate in eligible]
    matches.sort(key=lambda row: (-row.raw_cosine_similarity, row.candidate_entity_id))
    return tuple(matches[:limit])
