from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..util import read_json, stable_hash
from .adapters import (
    ADAPTER_VERSION, AdapterContext, adapt_clip_library, adapt_editorial_observations, adapt_performances,
    adapt_schedule_registry, adapt_shots, adapt_speakers, adapt_cinematic_moments, adapt_speech,
    artifact_content_signature, media_identity,
)
from .capabilities import capability_record
from .ids import StableIdRegistry, stable_film_id
from .model import new_film_model
from .schema import FILM_MODEL_BUILDER_VERSION, FILM_MODEL_SCHEMA_VERSION, ID_GENERATION_VERSION, TIMING_POLICY
from .serialization import canonicalize
from .validation import validate_film_model

SUPPORTED_ARTIFACTS = (
    "movie", "dialogue_events", "timeline",
    "source_dialogue_dialogue_events", "source_dialogue_timeline",
    "destination_video_dialogue_events", "destination_video_timeline",
    "shots", "speaker_map", "performance",
    "cinematic_moments", "clip_library", "replacement_schedule", "editorial_decisions", "editorial_report",
)


class FilmModelBuildError(ValueError):
    pass


@dataclass(frozen=True)
class BuildResult:
    model: dict[str, Any]
    validation_report: dict[str, Any]
    migration_report: dict[str, Any]
    build_report: dict[str, Any]


def build_film_model(
    artifacts: dict[str, Path | dict[str, Any]], *, schemas_dir: Path,
) -> BuildResult:
    if "movie" not in artifacts:
        raise FilmModelBuildError("A canonical movie artifact is required")
    loaded: dict[str, dict[str, Any]] = {}
    locators: dict[str, str | None] = {}
    for logical_type, source in artifacts.items():
        if logical_type not in SUPPORTED_ARTIFACTS:
            continue
        if isinstance(source, Path):
            loaded[logical_type] = read_json(source)
            locators[logical_type] = str(source.resolve())
        else:
            loaded[logical_type] = source
            locators[logical_type] = None

    # Prefer the richer canonical artifact when callers supply redundant views.
    # Excluded views do not participate in the cache signature.
    if "dialogue_events" in loaded and "timeline" in loaded:
        loaded.pop("timeline")
        locators.pop("timeline", None)
    if "editorial_report" in loaded and "editorial_decisions" in loaded:
        loaded.pop("editorial_decisions")
        locators.pop("editorial_decisions", None)

    movie = loaded["movie"]
    media_hash = str(movie.get("media_hash") or "")
    if not media_hash:
        raise FilmModelBuildError("Movie artifact has no media_hash")
    incompatible = sorted(
        logical_type for logical_type, artifact in loaded.items()
        if artifact.get("media_hash") not in {None, media_hash}
    )
    if incompatible:
        raise FilmModelBuildError(f"Artifacts do not match movie media_hash: {', '.join(incompatible)}")

    artifact_signatures = {key: artifact_content_signature(value) for key, value in loaded.items()}
    film_id = stable_film_id(media_hash, artifact_signatures["movie"])
    created_from_signature = stable_hash({
        "media_hash": media_hash, "schema_version": FILM_MODEL_SCHEMA_VERSION,
        "builder_version": FILM_MODEL_BUILDER_VERSION, "id_generation_version": ID_GENERATION_VERSION,
        "timing_policy": TIMING_POLICY, "adapter_version": ADAPTER_VERSION,
        "source_artifacts": artifact_signatures,
    })
    media = media_identity(movie, film_id=film_id, source_signature=artifact_signatures["movie"])
    model = new_film_model(
        film_id=film_id, media=media, created_from_signature=created_from_signature,
        duration=media["duration"], frame_rate=media["frame_rate"],
        audio_sample_rate=movie.get("sample_rate"),
    )
    context = AdapterContext(model=model, registry=StableIdRegistry(film_id), media_hash=media_hash)
    movie_record = context.register_artifact("movie", movie, locators["movie"])
    model["capabilities"]["media_inspection"] = capability_record(
        status="AVAILABLE", producing_artifact_id=movie_record["source_artifact_id"],
        implementation_version=movie.get("tool_version"), coverage={"duration_seconds": media["duration"]},
    )

    speech_types = [
        logical_type for logical_type in (
            "dialogue_events", "timeline",
            "source_dialogue_dialogue_events", "source_dialogue_timeline",
            "destination_video_dialogue_events", "destination_video_timeline",
        )
        if logical_type in loaded
    ]
    for speech_type in speech_types:
        adapt_speech(context, speech_type, loaded[speech_type], locators[speech_type])
    if "shots" in loaded:
        adapt_shots(context, loaded["shots"], locators["shots"])
    if "speaker_map" in loaded:
        adapt_speakers(context, loaded["speaker_map"], locators["speaker_map"])
    if "performance" in loaded:
        adapt_performances(context, loaded["performance"], locators["performance"])
    if "clip_library" in loaded:
        adapt_clip_library(context, loaded["clip_library"], locators["clip_library"])
    if "cinematic_moments" in loaded:
        adapt_cinematic_moments(context, loaded["cinematic_moments"], locators["cinematic_moments"])
    editorial_type = "editorial_report" if "editorial_report" in loaded else ("editorial_decisions" if "editorial_decisions" in loaded else None)
    if editorial_type:
        adapt_editorial_observations(context, editorial_type, loaded[editorial_type], locators[editorial_type])
    if "replacement_schedule" in loaded:
        adapt_schedule_registry(context, loaded["replacement_schedule"], locators["replacement_schedule"])

    _build_temporal_indexes(model)
    _summarize_confidence(model)
    model = canonicalize(model)
    validation_report = validate_film_model(model, schemas_dir)
    migration_report = {
        "migration_version": ADAPTER_VERSION, "film_id": film_id,
        "source_artifact_count": len(model["source_artifacts"]), "artifacts": context.migration_rows,
        "id_mappings": canonicalize(context.id_maps), "source_artifacts_modified": False,
        "validation_status": validation_report["status"],
    }
    build_report = {
        "status": "COMPLETE" if validation_report["status"] in {"VALID", "VALID_WITH_WARNINGS"} else "INVALID",
        "film_id": film_id, "created_from_signature": created_from_signature,
        "schema_version": FILM_MODEL_SCHEMA_VERSION, "builder_version": FILM_MODEL_BUILDER_VERSION,
        "artifact_types_used": sorted(context.artifact_records),
        "object_counts": {key: len(model[key]) for key in ("shots", "transitions", "speech_passages", "speaker_clusters", "dialogue_turns", "performances", "cinematic_moments", "editorial_observations")},
        "capability_status_counts": _count_capabilities(model),
        "validation": {"status": validation_report["status"], "error_count": validation_report["error_count"], "warning_count": validation_report["warning_count"]},
    }
    return BuildResult(model=model, validation_report=validation_report, migration_report=migration_report, build_report=build_report)


def _build_temporal_indexes(model: dict[str, Any]) -> None:
    indexes: dict[str, list[str]] = {}
    for collection, id_key in (
        ("shots", "shot_id"), ("transitions", "transition_id"), ("speech_passages", "speech_passage_id"),
        ("dialogue_turns", "dialogue_turn_id"), ("performances", "performance_id"),
    ):
        indexes[collection] = [row[id_key] for row in sorted(model[collection], key=lambda item: (item["start"], item["end"], item[id_key]))]
    model["timeline"]["ordered_temporal_indexes"] = indexes


def _summarize_confidence(model: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for collection in ("shots", "transitions", "speech_passages", "speaker_clusters", "dialogue_turns", "performances"):
        for row in model[collection]:
            for key, value in row.items():
                if (key == "confidence" or key.endswith("_confidence")) and isinstance(value, dict):
                    state = str(value.get("state", "unknown"))
                    counts[state] = counts.get(state, 0) + 1
    model["confidence_summary"] = {
        "state": "available" if counts else "unavailable", "records_by_state": dict(sorted(counts.items())),
        "calibration_notice": "Source-defined scores remain uncalibrated unless explicitly declared otherwise.",
    }


def _count_capabilities(model: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for capability in model["capabilities"].values():
        status = capability["status"]
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
