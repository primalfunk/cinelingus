from __future__ import annotations

from typing import Any

CAPABILITY_STATUSES = frozenset({"AVAILABLE", "PARTIAL", "FALLBACK", "UNAVAILABLE", "NOT_APPLICABLE"})

PHASE1_CAPABILITIES = (
    "media_inspection", "transcription", "word_timing", "diarization",
    "speaker_stitching", "shot_detection", "transition_evidence",
    "visual_performance_evidence", "performance_objects", "cinematic_moments",
    "editorial_verification", "editorial_repair_evidence", "schedule_provenance",
)

PHASE1_UNSUPPORTED_CAPABILITIES = (
    "semantic_embeddings", "semantic_similarity", "dialogue_function_classification",
    "character_identity", "active_speaker_attribution", "relationship_inference",
    "semantic_scene_understanding", "narrative_event_understanding",
)


def capability_record(
    *,
    status: str,
    producing_artifact_id: str | None = None,
    configuration_signature: str | None = None,
    implementation_version: str | None = None,
    coverage: dict[str, Any] | None = None,
    confidence_summary: dict[str, Any] | None = None,
    known_limitations: list[str] | None = None,
) -> dict[str, Any]:
    if status not in CAPABILITY_STATUSES:
        raise ValueError(f"Unsupported capability status: {status}")
    return {
        "status": status,
        "producing_artifact_id": producing_artifact_id,
        "configuration_signature": configuration_signature,
        "implementation_version": implementation_version,
        "coverage": coverage,
        "confidence_summary": confidence_summary,
        "known_limitations": sorted(known_limitations or []),
    }


def initial_capability_manifest() -> dict[str, dict[str, Any]]:
    manifest = {
        name: capability_record(status="UNAVAILABLE", known_limitations=["No compatible source artifact was supplied."])
        for name in PHASE1_CAPABILITIES
    }
    manifest.update({
        name: capability_record(status="UNAVAILABLE", known_limitations=["Explicitly out of scope for Phase 1."])
        for name in PHASE1_UNSUPPORTED_CAPABILITIES
    })
    return manifest

