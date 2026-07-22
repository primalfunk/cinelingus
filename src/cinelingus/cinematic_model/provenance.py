from __future__ import annotations

from typing import Any

from .ids import StableIdRegistry
from .schema import FILM_MODEL_BUILDER_VERSION


def provenance_record(
    registry: StableIdRegistry,
    *,
    source_media_id: str,
    source_artifact_type: str,
    source_artifact_id: str,
    source_artifact_locator: str | None,
    source_artifact_schema_version: str | None,
    source_object_id: str | None,
    source_object_index: int | None,
    source_time_range: dict[str, float] | None,
    analysis_configuration_signature: str | None,
    producing_module: str,
    producing_model_or_heuristic: str | None,
    producer_version: str | None,
    migration_history: list[dict[str, Any]] | None = None,
    parent_provenance_ids: list[str] | None = None,
    transformed_fields: list[str] | None = None,
) -> dict[str, Any]:
    identity = {
        "source_artifact_id": source_artifact_id,
        "source_object_id": source_object_id,
        "source_object_index": source_object_index,
        "source_time_range": source_time_range,
        "analysis_configuration_signature": analysis_configuration_signature,
        "producing_module": producing_module,
    }
    return {
        "provenance_id": registry.issue("provenance", identity),
        "source_media_id": source_media_id,
        "source_artifact_type": source_artifact_type,
        "source_artifact_id": source_artifact_id,
        "source_artifact_locator": source_artifact_locator,
        "source_artifact_schema_version": source_artifact_schema_version,
        "source_object_id": source_object_id,
        "source_object_index": source_object_index,
        "source_time_range": source_time_range,
        "analysis_configuration_signature": analysis_configuration_signature,
        "producing_module": producing_module,
        "producing_model_or_heuristic": producing_model_or_heuristic,
        "producer_version": producer_version,
        "film_model_builder_version": FILM_MODEL_BUILDER_VERSION,
        "migration_history": migration_history or [],
        "parent_provenance_ids": sorted(parent_provenance_ids or []),
        "transformed_fields": sorted(transformed_fields or []),
    }

