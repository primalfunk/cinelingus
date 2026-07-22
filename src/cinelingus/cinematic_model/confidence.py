from __future__ import annotations

from typing import Any

CONFIDENCE_STATES = frozenset({
    "numeric", "categorical", "unavailable", "unknown", "not_applicable",
    "conflicting_evidence",
})
CALIBRATION_STATES = frozenset({"calibrated_probability", "uncalibrated", "not_applicable", "unknown"})
FALLBACK_STATES = frozenset({"direct", "fallback_derived", "mixed", "not_applicable", "unknown"})


def confidence_record(
    *,
    state: str,
    value: Any = None,
    scale: str | None,
    interpretation: str,
    evidence_source: str,
    calibration_state: str,
    fallback_state: str,
    contradiction_references: list[str] | None = None,
    provenance_id: str | None = None,
) -> dict[str, Any]:
    if state not in CONFIDENCE_STATES:
        raise ValueError(f"Unsupported confidence state: {state}")
    if calibration_state not in CALIBRATION_STATES:
        raise ValueError(f"Unsupported calibration state: {calibration_state}")
    if fallback_state not in FALLBACK_STATES:
        raise ValueError(f"Unsupported fallback state: {fallback_state}")
    if state == "numeric" and not isinstance(value, (int, float)):
        raise ValueError("Numeric confidence requires a numeric value")
    if state in {"unavailable", "unknown", "not_applicable"} and value is not None:
        raise ValueError(f"{state} confidence cannot carry a value")
    if calibration_state == "calibrated_probability" and (
        state != "numeric" or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("A calibrated probability must be numeric in [0, 1]")
    return {
        "state": state,
        "value": value,
        "scale": scale,
        "interpretation": interpretation,
        "evidence_source": evidence_source,
        "calibration_state": calibration_state,
        "fallback_state": fallback_state,
        "contradiction_references": sorted(contradiction_references or []),
        "provenance_id": provenance_id,
    }

