from __future__ import annotations

from pathlib import Path
from typing import Any

from ..util import read_json
from ..validation import ValidationError, _validate_object
from .capabilities import CAPABILITY_STATUSES
from .schema import TIMING_POLICY

MODEL_VALIDATOR_VERSION = "film_model_validator_v1"

ENTITY_COLLECTIONS = {
    "shots": "shot_id", "transitions": "transition_id", "speech_passages": "speech_passage_id",
    "speaker_clusters": "speaker_cluster_id", "dialogue_turns": "dialogue_turn_id",
    "performances": "performance_id", "cinematic_moments": "cinematic_moment_id",
    "editorial_observations": "editorial_observation_id", "provenance": "provenance_id",
    "source_artifacts": "source_artifact_id",
}


def validate_film_model(model: dict[str, Any], schemas_dir: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    try:
        _validate_object(model, read_json(schemas_dir / "film_model.schema.json"), "FilmModel")
    except ValidationError as exc:
        errors.append(_issue("STRUCTURAL", "FilmModel", str(exc)))

    indexes: dict[str, set[str]] = {}
    all_ids: dict[str, str] = {}
    for collection, id_key in ENTITY_COLLECTIONS.items():
        values: set[str] = set()
        for index, row in enumerate(model.get(collection) or []):
            entity_id = row.get(id_key)
            if not isinstance(entity_id, str) or not entity_id:
                errors.append(_issue("IDENTITY", f"{collection}[{index}]", f"Missing {id_key}."))
                continue
            if entity_id in values:
                errors.append(_issue("IDENTITY", f"{collection}[{index}]", f"Duplicate ID {entity_id}."))
            values.add(entity_id)
            prior = all_ids.get(entity_id)
            if prior and prior != collection:
                errors.append(_issue("IDENTITY", f"{collection}[{index}]", f"ID {entity_id} also occurs in {prior}."))
            all_ids[entity_id] = collection
        indexes[collection] = values

    duration = float(model.get("timeline", {}).get("duration", 0.0) or 0.0)
    tolerance = float(TIMING_POLICY["comparison_tolerance_seconds"])
    for collection in ("shots", "transitions", "speech_passages", "dialogue_turns", "performances", "cinematic_moments"):
        for index, row in enumerate(model.get(collection) or []):
            start, end = row.get("start"), row.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                errors.append(_issue("TEMPORAL", f"{collection}[{index}]", "Missing numeric time range."))
            elif start < -tolerance or end + tolerance < start or end > duration + tolerance:
                errors.append(_issue("TEMPORAL", f"{collection}[{index}]", f"Invalid time range [{start}, {end}) for duration {duration}."))
            elif abs(end - start) <= tolerance:
                warnings.append(_issue("TEMPORAL", f"{collection}[{index}]", "Zero-length or tolerance-length interval preserved."))

    reference_rules = {
        "shots": {"linked_transition_ids": "transitions", "linked_performance_ids": "performances", "linked_moment_ids": "cinematic_moments"},
        "transitions": {"preceding_shot_id": "shots", "following_shot_id": "shots"},
        "speech_passages": {"speaker_cluster_candidates": "speaker_clusters", "linked_dialogue_turn_id": "dialogue_turns", "linked_performance_ids": "performances"},
        "speaker_clusters": {"passage_references": "speech_passages", "turn_references": "dialogue_turns", "performance_references": "performances"},
        "dialogue_turns": {"ordered_speech_passage_references": "speech_passages", "speaker_cluster_reference": "speaker_clusters", "speaker_cluster_candidates": "speaker_clusters", "preceding_turn_reference": "dialogue_turns", "following_turn_reference": "dialogue_turns", "containing_performance_references": "performances"},
        "performances": {"speech_passage_references": "speech_passages", "dialogue_turn_references": "dialogue_turns", "speaker_cluster_references": "speaker_clusters", "shot_references": "shots", "transition_references": "transitions"},
        "cinematic_moments": {"shot_references": "shots", "transition_references": "transitions", "speech_passage_references": "speech_passages", "performance_references": "performances"},
        "editorial_observations": {"referenced_performance_ids": "performances", "referenced_speech_passage_ids": "speech_passages", "referenced_moment_ids": "cinematic_moments"},
    }
    for collection, fields in reference_rules.items():
        for index, row in enumerate(model.get(collection) or []):
            for field, target in fields.items():
                value = row.get(field)
                references = value if isinstance(value, list) else ([] if value is None else [value])
                for reference in references:
                    if reference not in indexes[target]:
                        errors.append(_issue("REFERENTIAL", f"{collection}[{index}].{field}", f"Unknown {target} reference {reference}."))

    provenance_ids = indexes["provenance"]
    for collection, id_key in ENTITY_COLLECTIONS.items():
        if collection == "provenance":
            continue
        for index, row in enumerate(model.get(collection) or []):
            if row.get("provenance_id") not in provenance_ids:
                errors.append(_issue("PROVENANCE", f"{collection}[{index}].provenance_id", "Missing or unknown provenance reference."))

    for name, capability in (model.get("capabilities") or {}).items():
        if capability.get("status") not in CAPABILITY_STATUSES:
            errors.append(_issue("CAPABILITY", f"capabilities.{name}", f"Unknown status {capability.get('status')}."))

    for collection in ENTITY_COLLECTIONS:
        if collection in {"provenance", "source_artifacts"}:
            continue
        for index, row in enumerate(model.get(collection) or []):
            for key, value in row.items():
                if (key == "confidence" or key.endswith("_confidence")) and isinstance(value, dict):
                    required = {"state", "scale", "interpretation", "evidence_source", "calibration_state", "fallback_state"}
                    if not required.issubset(value):
                        errors.append(_issue("CONFIDENCE", f"{collection}[{index}].{key}", "Incomplete confidence record."))

    status = "INVALID" if errors else ("VALID_WITH_WARNINGS" if warnings else "VALID")
    report = {
        "validator_version": MODEL_VALIDATOR_VERSION, "status": status,
        "error_count": len(errors), "warning_count": len(warnings), "errors": errors, "warnings": warnings,
        "checks": {
            "structural": "PASS" if not any(row["category"] == "STRUCTURAL" for row in errors) else "FAIL",
            "identity": "PASS" if not any(row["category"] == "IDENTITY" for row in errors) else "FAIL",
            "temporal": "PASS" if not any(row["category"] == "TEMPORAL" for row in errors) else "FAIL",
            "referential": "PASS" if not any(row["category"] == "REFERENTIAL" for row in errors) else "FAIL",
            "provenance": "PASS" if not any(row["category"] == "PROVENANCE" for row in errors) else "FAIL",
            "confidence": "PASS" if not any(row["category"] == "CONFIDENCE" for row in errors) else "FAIL",
            "capability": "PASS" if not any(row["category"] == "CAPABILITY" for row in errors) else "FAIL",
        },
    }
    model["validation_state"] = {
        "status": status, "errors": errors, "warnings": warnings, "validator_version": MODEL_VALIDATOR_VERSION,
    }
    return report


def _issue(category: str, location: str, message: str) -> dict[str, str]:
    return {"category": category, "location": location, "message": message}
