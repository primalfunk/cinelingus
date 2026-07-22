from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

FILM_MODEL_SCHEMA_VERSION = "1.0.0"
FILM_MODEL_BUILDER_VERSION = "minimal_cinematic_model_core_v1"
ID_GENERATION_VERSION = "film_local_evidence_hash_v1"

TIMING_POLICY: dict[str, Any] = {
    "unit": "seconds",
    "interval_convention": "half_open",
    "precision_decimal_places": 3,
    "rounding": "half_up",
    "comparison_tolerance_seconds": 0.001,
    "zero_length_policy": "preserve_with_validation_notice",
    "overlap_policy": "preserve_and_validate",
}


def canonical_time(value: int | float | str) -> float:
    """Normalize a source timestamp using the FilmModel timing policy."""
    quantum = Decimal("0.001")
    normalized = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    result = float(normalized)
    return 0.0 if result == -0.0 else result


def canonical_interval(start: int | float | str, end: int | float | str) -> dict[str, float]:
    normalized_start = canonical_time(start)
    normalized_end = canonical_time(end)
    if normalized_start < 0.0 or normalized_end < 0.0:
        raise ValueError("FilmModel time ranges cannot be negative")
    if normalized_end < normalized_start:
        raise ValueError("FilmModel interval end must not precede start")
    return {
        "start": normalized_start,
        "end": normalized_end,
        "duration": canonical_time(normalized_end - normalized_start),
    }

