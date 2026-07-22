from __future__ import annotations

from typing import Any

from .capabilities import initial_capability_manifest
from .schema import FILM_MODEL_BUILDER_VERSION, FILM_MODEL_SCHEMA_VERSION, TIMING_POLICY


def new_film_model(
    *,
    film_id: str,
    media: dict[str, Any],
    created_from_signature: str,
    duration: int | float,
    frame_rate: int | float | None = None,
    audio_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Create an explicit empty FilmModel ready for adapter population."""
    if not film_id.startswith("film_"):
        raise ValueError("film_id must use the FilmModel film namespace")
    if not created_from_signature:
        raise ValueError("created_from_signature is required")
    duration_value = float(duration)
    if duration_value < 0:
        raise ValueError("FilmModel duration cannot be negative")
    return {
        "schema_version": FILM_MODEL_SCHEMA_VERSION,
        "builder_version": FILM_MODEL_BUILDER_VERSION,
        "film_id": film_id,
        "media": media,
        "timeline": {
            "start": 0.0,
            "end": duration_value,
            "duration": duration_value,
            "canonical_time_base": dict(TIMING_POLICY),
            "frame_rate": frame_rate,
            "audio_sample_rate": audio_sample_rate,
            "timing_tolerance_seconds": TIMING_POLICY["comparison_tolerance_seconds"],
            "ordered_temporal_indexes": {},
            "known_discontinuities": [],
        },
        "capabilities": initial_capability_manifest(),
        "shots": [],
        "transitions": [],
        "speech_passages": [],
        "speaker_clusters": [],
        "dialogue_turns": [],
        "performances": [],
        "cinematic_moments": [],
        "editorial_observations": [],
        "provenance": [],
        "confidence_summary": {"state": "unknown", "reason": "No adapters have populated this model.", "records_by_state": {}},
        "source_artifacts": [],
        "validation_state": {"status": "NOT_VALIDATED", "errors": [], "warnings": [], "validator_version": None},
        "created_from_signature": created_from_signature,
    }
