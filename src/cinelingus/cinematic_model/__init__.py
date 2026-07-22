"""Versioned contracts for Cinelingus's normalized film representation."""

from .capabilities import (
    CAPABILITY_STATUSES,
    PHASE1_CAPABILITIES,
    PHASE1_UNSUPPORTED_CAPABILITIES,
    capability_record,
    initial_capability_manifest,
)
from .builder import BuildResult, FilmModelBuildError, build_film_model
from .confidence import confidence_record
from .ids import StableIdRegistry, stable_entity_id, stable_film_id
from .lookup import FilmModelView
from .schedule_bridge import (
    ScheduleBridgeError,
    compare_schedule_equivalence,
    ingest_schedule,
    reconstruct_schedule,
)
from .model import new_film_model
from .schema import (
    FILM_MODEL_BUILDER_VERSION,
    FILM_MODEL_SCHEMA_VERSION,
    ID_GENERATION_VERSION,
    TIMING_POLICY,
    canonical_interval,
    canonical_time,
)

__all__ = [
    "CAPABILITY_STATUSES",
    "BuildResult",
    "FILM_MODEL_BUILDER_VERSION",
    "FILM_MODEL_SCHEMA_VERSION",
    "FilmModelBuildError",
    "FilmModelView",
    "ID_GENERATION_VERSION",
    "PHASE1_CAPABILITIES",
    "PHASE1_UNSUPPORTED_CAPABILITIES",
    "StableIdRegistry",
    "ScheduleBridgeError",
    "TIMING_POLICY",
    "canonical_interval",
    "canonical_time",
    "capability_record",
    "build_film_model",
    "confidence_record",
    "compare_schedule_equivalence",
    "initial_capability_manifest",
    "ingest_schedule",
    "new_film_model",
    "stable_entity_id",
    "stable_film_id",
    "reconstruct_schedule",
]
