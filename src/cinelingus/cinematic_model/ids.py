from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..util import stable_hash
from .schema import ID_GENERATION_VERSION

ENTITY_NAMESPACES = frozenset({
    "film", "shot", "transition", "speech", "speaker", "turn",
    "performance", "moment", "editorial", "artifact", "placement", "provenance",
})


def _identity_digest(namespace: str, film_id: str | None, evidence: Any) -> str:
    return stable_hash({
        "id_generation_version": ID_GENERATION_VERSION,
        "namespace": namespace,
        "film_id": film_id,
        "evidence": evidence,
    })


def stable_film_id(media_hash: str, inspection_signature: str | None = None) -> str:
    if not media_hash:
        raise ValueError("A stable media hash is required")
    return f"film_{_identity_digest('film', None, {'media_hash': media_hash, 'inspection_signature': inspection_signature})[:20]}"


def stable_entity_id(namespace: str, film_id: str, evidence: Any) -> str:
    if namespace not in ENTITY_NAMESPACES - {"film"}:
        raise ValueError(f"Unsupported FilmModel ID namespace: {namespace}")
    if not film_id.startswith("film_"):
        raise ValueError("A FilmModel film_id is required for film-local entities")
    return f"{namespace}_{_identity_digest(namespace, film_id, evidence)[:20]}"


@dataclass
class StableIdRegistry:
    """Generate IDs and fail if a truncated-token collision is observed."""

    film_id: str
    _signatures: dict[str, str] = field(default_factory=dict)

    def issue(self, namespace: str, evidence: Any) -> str:
        entity_id = stable_entity_id(namespace, self.film_id, evidence)
        signature = _identity_digest(namespace, self.film_id, evidence)
        prior = self._signatures.get(entity_id)
        if prior is not None and prior != signature:
            raise ValueError(f"Stable FilmModel ID collision: {entity_id}")
        self._signatures[entity_id] = signature
        return entity_id

