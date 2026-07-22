from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..util import stable_hash
from .ids import stable_entity_id
from .lookup import FilmModelView

SCHEDULE_BRIDGE_SCHEMA_VERSION = "1.0.0"
SCHEDULE_BRIDGE_VERSION = "translation_schedule_bridge_v1"


class ScheduleBridgeError(ValueError):
    pass


def ingest_schedule(
    schedule: dict[str, Any], *, source_model: dict[str, Any], destination_model: dict[str, Any],
    rendered_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _require_valid_model(source_model, "source")
    _require_valid_model(destination_model, "destination")
    source_hash = source_model.get("media", {}).get("media_hash")
    destination_hash = destination_model.get("media", {}).get("media_hash")
    if schedule.get("source_media_hash") not in {None, source_hash}:
        raise ScheduleBridgeError("Schedule source_media_hash does not match the source FilmModel")
    if schedule.get("destination_media_hash", schedule.get("media_hash")) not in {None, destination_hash}:
        raise ScheduleBridgeError("Schedule destination_media_hash does not match the destination FilmModel")

    source_view = FilmModelView(source_model)
    destination_view = FilmModelView(destination_model)
    schedule_signature = stable_hash(schedule)
    verification_by_placement = {
        str(row.get("editorial_placement_id")): row
        for row in (rendered_verification or {}).get("mappings") or []
        if row.get("editorial_placement_id")
    }
    traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_placement_ids: set[str] = set()
    for index, mapping in enumerate(schedule.get("mappings") or []):
        source_placement_id = str(mapping.get("editorial_placement_id") or mapping.get("mapping_id") or f"mapping_index:{index}")
        model_placement_id = stable_entity_id("placement", destination_model["film_id"], {
            "schedule_signature": schedule_signature, "source_placement_id": source_placement_id,
            "mapping_signature": stable_hash(mapping),
        })
        if model_placement_id in seen_placement_ids:
            errors.append(_issue("IDENTITY", index, "Duplicate model placement ID."))
        seen_placement_ids.add(model_placement_id)
        trace = _trace_mapping(
            index, mapping, model_placement_id=model_placement_id,
            source_model=source_model, destination_model=destination_model,
            source_view=source_view, destination_view=destination_view,
            verification=verification_by_placement.get(source_placement_id),
        )
        if trace["unresolved_references"]:
            warnings.append(_issue("TRACEABILITY", index, "; ".join(trace["unresolved_references"])))
        if trace["contradictions"]:
            warnings.append(_issue("CONTRADICTORY_EVIDENCE", index, "; ".join(trace["contradictions"])))
        traces.append(trace)

    status = "INVALID" if errors else ("VALID_WITH_WARNINGS" if warnings else "VALID")
    return {
        "schema_version": SCHEDULE_BRIDGE_SCHEMA_VERSION,
        "bridge_version": SCHEDULE_BRIDGE_VERSION,
        "schedule_signature": schedule_signature,
        "schedule_config_signature": schedule.get("config_signature"),
        "source_film_id": source_model["film_id"],
        "source_model_signature": source_model["created_from_signature"],
        "destination_film_id": destination_model["film_id"],
        "destination_model_signature": destination_model["created_from_signature"],
        "source_media_hash": source_hash,
        "destination_media_hash": destination_hash,
        "placement_count": len(schedule.get("mappings") or []),
        "canonical_schedule_payload": deepcopy(schedule),
        "placements": traces,
        "validation_state": {
            "status": status, "errors": errors, "warnings": warnings,
            "schedule_trace_readiness": "READY" if status == "VALID" else "DEGRADED" if not errors else "NOT_READY",
        },
    }


def reconstruct_schedule(bridge: dict[str, Any]) -> dict[str, Any]:
    if bridge.get("schema_version") != SCHEDULE_BRIDGE_SCHEMA_VERSION:
        raise ScheduleBridgeError(f"Unsupported schedule bridge schema: {bridge.get('schema_version')}")
    if bridge.get("validation_state", {}).get("status") == "INVALID":
        raise ScheduleBridgeError("Invalid schedule bridge cannot reconstruct a schedule")
    payload = bridge.get("canonical_schedule_payload")
    if not isinstance(payload, dict):
        raise ScheduleBridgeError("Schedule bridge has no canonical payload")
    if stable_hash(payload) != bridge.get("schedule_signature"):
        raise ScheduleBridgeError("Canonical schedule payload signature does not match the bridge")
    if len(payload.get("mappings") or []) != bridge.get("placement_count"):
        raise ScheduleBridgeError("Canonical schedule placement count does not match the bridge")
    return deepcopy(payload)


def compare_schedule_equivalence(
    original: dict[str, Any], reconstructed: dict[str, Any], *, bridge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    differences = _differences(original, reconstructed)
    classified = [{**row, "classification": _classify_difference(row)} for row in differences]
    original_mappings = original.get("mappings") or []
    reconstructed_mappings = reconstructed.get("mappings") or []
    checks = {
        "placement_count": len(original_mappings) == len(reconstructed_mappings),
        "placement_ids": _mapping_values(original_mappings, _placement_key) == _mapping_values(reconstructed_mappings, _placement_key),
        "placement_ordering": [stable_hash(row) for row in original_mappings] == [stable_hash(row) for row in reconstructed_mappings],
        "destination_timings": _mapping_values(original_mappings, _destination_timing) == _mapping_values(reconstructed_mappings, _destination_timing),
        "donor_timings": _mapping_values(original_mappings, _donor_timing) == _mapping_values(reconstructed_mappings, _donor_timing),
        "donor_media_identity": original.get("source_media_hash") == reconstructed.get("source_media_hash"),
        "speaker_references": _mapping_values(original_mappings, _speaker_refs) == _mapping_values(reconstructed_mappings, _speaker_refs),
        "performance_references": _mapping_values(original_mappings, _performance_refs) == _mapping_values(reconstructed_mappings, _performance_refs),
        "adaptation_parameters": _mapping_values(original_mappings, _adaptation) == _mapping_values(reconstructed_mappings, _adaptation),
        "suppression_parameters": _mapping_values(original_mappings, _suppression) == _mapping_values(reconstructed_mappings, _suppression),
        "fade_parameters": _mapping_values(original_mappings, _fades) == _mapping_values(reconstructed_mappings, _fades),
        "render_command_plan": _mapping_values(original_mappings, lambda row: row.get("render_operations")) == _mapping_values(reconstructed_mappings, lambda row: row.get("render_operations")),
        "schedule_config_signature": original.get("config_signature") == reconstructed.get("config_signature"),
        "canonical_payload": stable_hash(original) == stable_hash(reconstructed),
    }
    if bridge is not None:
        checks["trace_placement_count"] = len(bridge.get("placements") or []) == len(original_mappings)
        checks["trace_readiness"] = bridge.get("validation_state", {}).get("schedule_trace_readiness") == "READY"
        checks["donor_passage_trace"] = all(row.get("donor", {}).get("speech_passage_ids") for row in bridge.get("placements") or [])
        checks["destination_trace"] = all(
            row.get("destination", {}).get("speech_passage_ids") or row.get("destination", {}).get("performance_ids")
            for row in bridge.get("placements") or []
        )
    unacceptable = [row for row in classified if row["classification"] in {"behavioral difference", "invalid difference"}]
    equivalent = not differences and all(checks.values())
    return {
        "schema_version": "1.0", "comparison_version": "schedule_equivalence_v1",
        "equivalent": equivalent, "original_signature": stable_hash(original),
        "reconstructed_signature": stable_hash(reconstructed), "checks": checks,
        "difference_count": len(classified), "unacceptable_difference_count": len(unacceptable),
        "differences": classified,
    }


def _trace_mapping(
    index: int, mapping: dict[str, Any], *, model_placement_id: str,
    source_model: dict[str, Any], destination_model: dict[str, Any],
    source_view: FilmModelView, destination_view: FilmModelView,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    destination_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
    destination_end = destination_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
    donor_start = float(mapping.get("source_movie_timestamp", mapping.get("clip_movie_timestamp", 0.0)) or 0.0) + float(mapping.get("clip_trim_start", 0.0) or 0.0)
    donor_duration = float(mapping.get("clip_trim_duration", mapping.get("planned_render_duration", 0.0)) or 0.0)
    donor_end = donor_start + donor_duration

    destination_performances = _by_source_reference(destination_model, "performances", "source_performance_reference", mapping.get("destination_performance_id") or mapping.get("performance_id"))
    destination_passages = _passages_for_window(destination_model, mapping.get("window_id"))
    if not destination_passages:
        destination_passages = list(destination_view.overlapping("speech_passages", destination_start, destination_end))
    destination_turns = _turns_for(destination_model, destination_passages, destination_performances)
    destination_shots = list(destination_view.overlapping("shots", destination_start, destination_end))
    destination_moments = list(destination_view.overlapping("cinematic_moments", destination_start, destination_end))

    clip = _clip_record(source_model, mapping.get("clip_id"))
    donor_passage_ids = list(clip.get("speech_passage_ids") or []) if clip else []
    donor_passages = [source_view.get(item) for item in donor_passage_ids]
    if not donor_passages:
        donor_passages = list(source_view.overlapping("speech_passages", donor_start, donor_end))
        donor_passage_ids = [row["speech_passage_id"] for row in donor_passages]
    donor_performances = _by_source_reference(source_model, "performances", "source_performance_reference", mapping.get("source_performance_id"))
    donor_turns = _turns_for(source_model, donor_passages, donor_performances)
    donor_speakers = _by_source_reference(source_model, "speaker_clusters", "source_speaker_label", mapping.get("source_speaker_id"))

    source_placement_id = str(mapping.get("editorial_placement_id") or mapping.get("mapping_id") or f"mapping_index:{index}")
    editorial = list(destination_view.editorial_observations_for_placement(source_placement_id))
    unresolved: list[str] = []
    contradictions: list[str] = []
    if not destination_passages and not destination_performances:
        unresolved.append("No destination speech passage or performance reference resolved.")
    if not donor_passages:
        unresolved.append("No donor speech passage reference resolved.")
    if mapping.get("source_performance_id") and not donor_performances:
        unresolved.append("Source performance reference did not resolve.")
    if mapping.get("destination_performance_id") and not destination_performances:
        unresolved.append("Destination performance reference did not resolve.")
    editorial_clip_ids = {
        str(row.get("source_detail", {}).get("clip_id"))
        for row in editorial if row.get("source_detail", {}).get("clip_id") is not None
    }
    if editorial_clip_ids and str(mapping.get("clip_id")) not in editorial_clip_ids:
        contradictions.append(
            f"Schedule donor {mapping.get('clip_id')} conflicts with editorial donor(s) {', '.join(sorted(editorial_clip_ids))}."
        )
    return {
        "model_placement_id": model_placement_id,
        "source_placement_id": source_placement_id,
        "original_mapping_index": index,
        "mapping_signature": stable_hash(mapping),
        "destination": {
            "film_id": destination_model["film_id"], "media_hash": destination_model["media"]["media_hash"],
            "start": destination_start, "end": destination_end,
            "speech_passage_ids": [row["speech_passage_id"] for row in destination_passages],
            "dialogue_turn_ids": [row["dialogue_turn_id"] for row in destination_turns],
            "performance_ids": [row["performance_id"] for row in destination_performances],
            "shot_ids": [row["shot_id"] for row in destination_shots],
            "cinematic_moment_ids": [row["cinematic_moment_id"] for row in destination_moments],
        },
        "donor": {
            "film_id": source_model["film_id"], "media_hash": source_model["media"]["media_hash"],
            "source_clip_id": mapping.get("clip_id"), "start": donor_start, "end": donor_end,
            "speech_passage_ids": donor_passage_ids,
            "dialogue_turn_ids": [row["dialogue_turn_id"] for row in donor_turns],
            "performance_ids": [row["performance_id"] for row in donor_performances],
            "speaker_cluster_ids": [row["speaker_cluster_id"] for row in donor_speakers],
        },
        "score_evidence": {key: deepcopy(mapping.get(key)) for key in ("score", "score_components", "performance_similarity_score", "performance_similarity_components", "cinematic_compatibility_score", "cinematic_compatibility_components") if key in mapping},
        "adaptation_parameters": _adaptation(mapping),
        "suppression_parameters": _suppression(mapping),
        "verification_evidence": deepcopy(verification),
        "editorial_observation_ids": [row["editorial_observation_id"] for row in editorial],
        "repair_history": [deepcopy(row.get("source_detail")) for row in editorial if row.get("repair_strategy") or row.get("final_placement_state")],
        "unresolved_references": unresolved,
        "contradictions": contradictions,
        "trace_status": "COMPLETE" if not unresolved and not contradictions else "PARTIAL",
    }


def _require_valid_model(model: dict[str, Any], label: str) -> None:
    if model.get("validation_state", {}).get("status") not in {"VALID", "VALID_WITH_WARNINGS"}:
        raise ScheduleBridgeError(f"The {label} FilmModel is not valid")


def _by_source_reference(model: dict[str, Any], collection: str, field: str, source_id: Any) -> list[dict[str, Any]]:
    if source_id is None:
        return []
    return [row for row in model.get(collection) or [] if str(row.get(field)) == str(source_id)]


def _passages_for_window(model: dict[str, Any], source_id: Any) -> list[dict[str, Any]]:
    if source_id is None:
        return []
    base = str(source_id).split("@", 1)[0]
    return [row for row in model.get("speech_passages") or [] if str(row.get("source_transcript_reference")) in {str(source_id), base}]


def _turns_for(model: dict[str, Any], passages: list[dict[str, Any]], performances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = {row.get("linked_dialogue_turn_id") for row in passages if row.get("linked_dialogue_turn_id")}
    for performance in performances:
        ids.update(performance.get("dialogue_turn_references") or [])
    return [row for row in model.get("dialogue_turns") or [] if row.get("dialogue_turn_id") in ids]


def _clip_record(model: dict[str, Any], clip_id: Any) -> dict[str, Any] | None:
    for artifact in model.get("source_artifacts") or []:
        if artifact.get("logical_artifact_type") == "clip_library":
            return next((row for row in artifact.get("object_index") or [] if str(row.get("source_clip_id")) == str(clip_id)), None)
    return None


def _mapping_values(rows: list[dict[str, Any]], getter) -> list[Any]:
    return [getter(row) for row in rows]


def _placement_key(row: dict[str, Any]) -> Any:
    return row.get("editorial_placement_id") or row.get("mapping_id")


def _destination_timing(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("destination_timestamp"), row.get("planned_render_duration"), row.get("alignment_slot_start"), row.get("alignment_slot_end"))


def _donor_timing(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("source_movie_timestamp"), row.get("clip_movie_timestamp"), row.get("clip_trim_start"), row.get("clip_trim_duration"))


def _speaker_refs(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("source_speaker_id"), row.get("destination_speaker_id"), row.get("mapped_destination_speaker_id"), row.get("local_speaker_mapping"))


def _performance_refs(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("performance_id"), row.get("source_performance_id"), row.get("destination_performance_id"))


def _adaptation(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("stretch_factor", "clip_trim_start", "clip_trim_duration", "leading_silence", "trailing_silence", "planned_render_duration", "timing_strategy", "timing_adjustments", "render_operations")
    return {key: deepcopy(row.get(key)) for key in keys if key in row}


def _suppression(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("suppression_mode", "background_reconstruction_strategy", "fallback_reason", "enabled")
    return {key: deepcopy(row.get(key)) for key in keys if key in row}


def _fades(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [deepcopy(item) for item in row.get("render_operations") or [] if item.get("operation") == "fade_in_out"]


def _differences(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "left": left, "right": right, "kind": "type"}]
    if isinstance(left, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append({"path": f"{path}.{key}", "left": left.get(key), "right": right.get(key), "kind": "missing"})
            else:
                differences.extend(_differences(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path, "left": len(left), "right": len(right), "kind": "length"}]
        differences: list[dict[str, Any]] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_differences(left_item, right_item, f"{path}[{index}]"))
        return differences
    return [] if left == right else [{"path": path, "left": left, "right": right, "kind": "value"}]


def _classify_difference(row: dict[str, Any]) -> str:
    path = row["path"]
    if path.endswith("creation_timestamp"):
        return "serialization-only"
    if isinstance(row.get("left"), (int, float)) and isinstance(row.get("right"), (int, float)) and abs(float(row["left"]) - float(row["right"])) <= 0.001:
        return "harmless normalization"
    if path.startswith("$.film_model_"):
        return "expected migration difference"
    if path == "$.mappings" and row.get("kind") == "length":
        return "invalid difference"
    if path.startswith("$.mappings") or path in {"$.source_media_hash", "$.destination_media_hash", "$.media_hash"}:
        return "behavioral difference"
    return "expected migration difference"


def _issue(category: str, mapping_index: int, message: str) -> dict[str, Any]:
    return {"category": category, "mapping_index": mapping_index, "message": message}
