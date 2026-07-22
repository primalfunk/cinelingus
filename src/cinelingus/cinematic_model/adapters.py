from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util import stable_hash
from .capabilities import capability_record
from .confidence import confidence_record
from .ids import StableIdRegistry
from .provenance import provenance_record
from .schema import canonical_interval

ADAPTER_VERSION = "phase1_read_adapters_v2_multirole_speech"
VOLATILE_SIGNATURE_FIELDS = frozenset({"creation_timestamp", "updated_timestamp", "path"})


def artifact_content_signature(data: dict[str, Any]) -> str:
    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items() if key not in VOLATILE_SIGNATURE_FIELDS}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    return stable_hash(stable(data))


def _confidence(value: Any, *, source: str, provenance_id: str | None, fallback: bool = False) -> dict[str, Any]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return confidence_record(
            state="numeric", value=float(value), scale="source_defined_unit_interval",
            interpretation="Source artifact score; calibration is not established.",
            evidence_source=source, calibration_state="uncalibrated",
            fallback_state="fallback_derived" if fallback else "direct", provenance_id=provenance_id,
        )
    return confidence_record(
        state="unknown", value=None, scale=None, interpretation="Source artifact did not report confidence.",
        evidence_source=source, calibration_state="unknown",
        fallback_state="fallback_derived" if fallback else "unknown", provenance_id=provenance_id,
    )


def _artifact_interval(ctx: "AdapterContext", row: dict[str, Any]) -> dict[str, float]:
    interval = canonical_interval(row.get("start", 0.0), row.get("end", row.get("start", 0.0)))
    media_duration = float(ctx.model["timeline"]["duration"])
    frame_rate = ctx.model["timeline"].get("frame_rate")
    frame_tolerance = (1.0 / float(frame_rate)) if isinstance(frame_rate, (int, float)) and frame_rate > 0 else 0.0
    boundary_tolerance = max(0.05, frame_tolerance)
    if media_duration < interval["end"] <= media_duration + boundary_tolerance:
        return canonical_interval(interval["start"], media_duration)
    return interval


@dataclass
class AdapterContext:
    model: dict[str, Any]
    registry: StableIdRegistry
    media_hash: str
    artifact_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    id_maps: dict[str, dict[str, str]] = field(default_factory=dict)
    migration_rows: list[dict[str, Any]] = field(default_factory=list)

    def register_artifact(self, logical_type: str, data: dict[str, Any], locator: str | None) -> dict[str, Any]:
        content_signature = artifact_content_signature(data)
        evidence = {
            "logical_type": logical_type,
            "media_hash": data.get("media_hash", self.media_hash),
            "schema_version": data.get("schema_version"),
            "config_signature": data.get("config_signature"),
            "content_signature": content_signature,
        }
        artifact_id = self.registry.issue("artifact", evidence)
        provenance = provenance_record(
            self.registry, source_media_id=self.model["film_id"], source_artifact_type=logical_type,
            source_artifact_id=artifact_id, source_artifact_locator=locator,
            source_artifact_schema_version=data.get("schema_version"), source_object_id=None,
            source_object_index=None, source_time_range=None,
            analysis_configuration_signature=data.get("config_signature"),
            producing_module="cinelingus.cinematic_model.adapters",
            producing_model_or_heuristic=data.get("detector") or data.get("diarization_tool"),
            producer_version=data.get("tool_version"),
            migration_history=[{"action": "registered_source_artifact", "adapter_version": ADAPTER_VERSION}],
        )
        record = {
            "source_artifact_id": artifact_id,
            "logical_artifact_type": logical_type,
            "logical_locator": locator,
            "locator_kind": "local_reference" if locator else "in_memory",
            "schema_version": data.get("schema_version"),
            "tool_version": data.get("tool_version"),
            "media_hash": data.get("media_hash"),
            "configuration_signature": data.get("config_signature"),
            "content_signature": content_signature,
            "adapter_version": ADAPTER_VERSION,
            "provenance_id": provenance["provenance_id"],
        }
        self.artifact_records[logical_type] = record
        self.model["source_artifacts"].append(record)
        self.model["provenance"].append(provenance)
        return record

    def object_provenance(
        self, logical_type: str, row: dict[str, Any], source_index: int, interval: dict[str, float] | None,
    ) -> str:
        artifact = self.artifact_records[logical_type]
        source_id = row.get("id") or row.get("speaker_id")
        source_interval = None
        migration_history: list[dict[str, Any]] = [{"action": "normalized", "adapter_version": ADAPTER_VERSION}]
        if "start" in row and "end" in row:
            source_interval = canonical_interval(row["start"], row["end"])
            if interval is not None and source_interval != interval:
                migration_history.append({
                    "action": "normalized_media_boundary_overshoot",
                    "source_time_range": source_interval,
                    "normalized_time_range": interval,
                    "policy": "clamp_within_50ms_or_one_frame",
                })
        provenance = provenance_record(
            self.registry, source_media_id=self.model["film_id"], source_artifact_type=logical_type,
            source_artifact_id=artifact["source_artifact_id"], source_artifact_locator=artifact["logical_locator"],
            source_artifact_schema_version=artifact["schema_version"],
            source_object_id=str(source_id) if source_id is not None else None,
            source_object_index=source_index if source_id is None else None, source_time_range=source_interval,
            analysis_configuration_signature=artifact["configuration_signature"],
            producing_module="cinelingus.cinematic_model.adapters",
            producing_model_or_heuristic=None, producer_version=ADAPTER_VERSION,
            migration_history=migration_history,
            transformed_fields=["id", "time_range", "confidence"],
        )
        self.model["provenance"].append(provenance)
        return provenance["provenance_id"]

    def map_id(self, namespace: str, source_id: Any, model_id: str) -> None:
        if source_id is not None:
            # Primary-role artifacts are adapted first. Preserve their reference
            # binding if another role happens to reuse the same local object ID.
            self.id_maps.setdefault(namespace, {}).setdefault(str(source_id), model_id)


def media_identity(movie: dict[str, Any], *, film_id: str, source_signature: str) -> dict[str, Any]:
    streams = list(movie.get("streams") or [])
    video = next((row for row in streams if row.get("codec_type") == "video"), None)
    audio = next((row for row in streams if row.get("codec_type") == "audio"), None)
    source_path = movie.get("path")
    return {
        "film_id": film_id,
        "media_hash": movie["media_hash"],
        "source_path_reference": source_path,
        "source_path_kind": "local_reference" if source_path else "unavailable",
        "normalized_source_path": Path(source_path).as_posix() if source_path else None,
        "filename": Path(source_path).name if source_path else "",
        "duration": float(movie.get("duration") or 0.0),
        "container": movie.get("codec"),
        "video_stream_summary": deepcopy(video),
        "audio_stream_summary": deepcopy(audio),
        "frame_rate": movie.get("frame_rate"),
        "resolution": movie.get("resolution"),
        "channel_layout": audio.get("channel_layout") if audio else None,
        "media_inspection_version": movie.get("tool_version"),
        "corpus_media_id": movie.get("corpus_media_id"),
        "source_artifact_signature": source_signature,
    }


def adapt_speech(ctx: AdapterContext, logical_type: str, artifact: dict[str, Any], locator: str | None) -> None:
    ctx.register_artifact(logical_type, artifact, locator)
    rows = artifact.get("events") if logical_type.endswith("dialogue_events") else artifact.get("windows")
    language = artifact.get("detected_language") or artifact.get("configured_language")
    for index, row in enumerate(rows or []):
        interval = _artifact_interval(ctx, row)
        evidence = {"artifact": ctx.artifact_records[logical_type]["source_artifact_id"], "source_id": row.get("id"), **interval}
        model_id = ctx.registry.issue("speech", evidence)
        provenance_id = ctx.object_provenance(logical_type, row, index, interval)
        transcript = str(row.get("transcript") or "")
        ctx.model["speech_passages"].append({
            "speech_passage_id": model_id, **interval, "original_transcript": transcript,
            "normalized_comparison_text": " ".join(transcript.split()).casefold(), "language": language,
            "transcription_confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id),
            "word_timing_references": deepcopy(row.get("words") or []),
            "punctuation_or_sentence_boundary_evidence": None,
            "source_transcript_reference": row.get("id"),
            "source_speaker_label": row.get("speaker"), "speaker_cluster_candidates": [],
            "linked_dialogue_turn_id": None, "linked_performance_ids": [],
            "provenance_id": provenance_id, "verification_eligibility": bool(transcript.strip()),
        })
        ctx.map_id("speech", row.get("id"), model_id)
    previous_transcription = ctx.model["capabilities"]["transcription"]
    previous_passage_count = int((previous_transcription.get("coverage") or {}).get("passage_count") or 0)
    previous_view_count = int((previous_transcription.get("coverage") or {}).get("speech_view_count") or 0)
    total_passage_count = previous_passage_count + len(rows or [])
    ctx.model["capabilities"]["transcription"] = capability_record(
        status="AVAILABLE" if rows else "PARTIAL", producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        configuration_signature=artifact.get("config_signature"), implementation_version=artifact.get("tool_version"),
        coverage={"passage_count": total_passage_count, "speech_view_count": previous_view_count + 1},
        known_limitations=[] if total_passage_count else ["No speech passages were detected."],
    )
    previous_word_timing = ctx.model["capabilities"]["word_timing"]
    previous_word_views = int((previous_word_timing.get("coverage") or {}).get("speech_view_count") or 0)
    previous_word_passages = int((previous_word_timing.get("coverage") or {}).get("passage_count") or 0)
    word_passage_count = previous_word_passages + sum(1 for row in rows or [] if row.get("words"))
    ctx.model["capabilities"]["word_timing"] = capability_record(
        status="AVAILABLE" if word_passage_count else "UNAVAILABLE",
        producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        implementation_version=artifact.get("tool_version"),
        coverage={"passage_count": word_passage_count, "speech_view_count": previous_word_views + 1},
        known_limitations=[] if word_passage_count else ["No word-level timing is present in the supplied speech artifacts."],
    )
    ctx.migration_rows.append(_migration_row(logical_type, artifact, len(rows or []), ["time range", "ID", "confidence"], ["raw transcript preserved"]))


def adapt_shots(ctx: AdapterContext, artifact: dict[str, Any], locator: str | None) -> None:
    logical_type = "shots"
    ctx.register_artifact(logical_type, artifact, locator)
    shot_rows = artifact.get("shots") or []
    for index, row in enumerate(shot_rows):
        interval = _artifact_interval(ctx, row)
        model_id = ctx.registry.issue("shot", {"artifact": ctx.artifact_records[logical_type]["source_artifact_id"], "source_id": row.get("id"), **interval})
        provenance_id = ctx.object_provenance(logical_type, row, index, interval)
        ctx.model["shots"].append({
            "shot_id": model_id, **interval, "source_shot_reference": row.get("id"),
            "boundary_confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id),
            "start_boundary_type": "source_shot_boundary", "end_boundary_type": "source_shot_boundary",
            "frame_or_timestamp_evidence": deepcopy(row.get("evidence")),
            "visual_analysis_version": artifact.get("core_evidence_version") or artifact.get("tool_version"),
            "provenance_id": provenance_id, "confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id),
            "linked_transition_ids": [], "linked_performance_ids": [], "linked_moment_ids": [],
        })
        ctx.map_id("shot", row.get("id"), model_id)
    for index, row in enumerate(artifact.get("transitions") or []):
        interval = _artifact_interval(ctx, row)
        transition_id = ctx.registry.issue("transition", {"artifact": ctx.artifact_records[logical_type]["source_artifact_id"], "source_id": row.get("id"), "kind": row.get("kind"), **interval})
        provenance_id = ctx.object_provenance(logical_type, row, len(shot_rows) + index, interval)
        preceding = _nearest_shot(ctx.model["shots"], interval["start"], edge="end")
        following = _nearest_shot(ctx.model["shots"], interval["end"], edge="start")
        ctx.model["transitions"].append({
            "transition_id": transition_id, **interval, "transition_evidence_type": row.get("kind") or "unknown",
            "current_classification": row.get("kind") or "unclassified",
            "classification_confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id),
            "source_artifact_reference": ctx.artifact_records[logical_type]["source_artifact_id"],
            "preceding_shot_id": preceding, "following_shot_id": following,
            "gradual_transition_candidate": "gradual" in str(row.get("kind", "")).lower(),
            "fade_or_black_guard_evidence": {key: row.get(key) for key in ("detected_black_start", "detected_black_end") if key in row},
            "provenance_id": provenance_id,
        })
        for shot in ctx.model["shots"]:
            if shot["shot_id"] in {preceding, following}:
                shot["linked_transition_ids"].append(transition_id)
        ctx.map_id("transition", row.get("id"), transition_id)
    ctx.model["capabilities"]["shot_detection"] = capability_record(
        status="AVAILABLE" if shot_rows else "UNAVAILABLE", producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        configuration_signature=artifact.get("config_signature"), implementation_version=artifact.get("core_evidence_version") or artifact.get("tool_version"),
        coverage={"shot_count": len(shot_rows)},
    )
    transitions = artifact.get("transitions") or []
    ctx.model["capabilities"]["transition_evidence"] = capability_record(
        status="AVAILABLE" if transitions else "PARTIAL", producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        coverage={"transition_count": len(transitions)}, known_limitations=[] if transitions else ["Shot artifact contains no explicit transition candidates."],
    )
    ctx.migration_rows.append(_migration_row(logical_type, artifact, len(shot_rows) + len(transitions), ["time range", "ID", "confidence", "transition links"], []))


def adapt_speakers(ctx: AdapterContext, artifact: dict[str, Any], locator: str | None) -> None:
    logical_type = "speaker_map"
    ctx.register_artifact(logical_type, artifact, locator)
    fallback = artifact.get("fallback_status") not in {None, "NONE"} or artifact.get("diarization_status") == "FALLBACK"
    speakers = artifact.get("speakers") or []
    segments = artifact.get("speaker_segments") or []
    for index, row in enumerate(speakers):
        source_label = row.get("speaker_id")
        appearances = [_artifact_interval(ctx, item) for item in segments if item.get("speaker_id") == source_label]
        evidence = {"artifact": ctx.artifact_records[logical_type]["source_artifact_id"], "source_label": source_label, "appearances": appearances}
        model_id = ctx.registry.issue("speaker", evidence)
        provenance_id = ctx.object_provenance(logical_type, row, index, None)
        ctx.model["speaker_clusters"].append({
            "speaker_cluster_id": model_id, "source_speaker_label": source_label,
            "appearance_intervals": appearances, "total_speaking_duration": float(row.get("total_duration") or 0.0),
            "passage_references": [], "turn_references": [], "performance_references": [],
            "diarization_backend": artifact.get("actual_backend"),
            "diarization_confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id, fallback=fallback),
            "fallback_state": artifact.get("fallback_status") or "UNKNOWN",
            "chunk_stitching_evidence": deepcopy(artifact.get("diagnostics", {}).get("chunk_stitching")),
            "source_diarization_reference": ctx.artifact_records[logical_type]["source_artifact_id"],
            "provenance_id": provenance_id,
        })
        ctx.map_id("speaker", source_label, model_id)
    for passage in ctx.model["speech_passages"]:
        speaker_id = ctx.id_maps.get("speaker", {}).get(str(passage.pop("source_speaker_label", None)))
        if speaker_id:
            passage["speaker_cluster_candidates"] = [speaker_id]
            next(row for row in ctx.model["speaker_clusters"] if row["speaker_cluster_id"] == speaker_id)["passage_references"].append(passage["speech_passage_id"])
    status = "FALLBACK" if fallback else ("AVAILABLE" if speakers else "UNAVAILABLE")
    ctx.model["capabilities"]["diarization"] = capability_record(
        status=status, producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        configuration_signature=artifact.get("config_signature"), implementation_version=artifact.get("model_name"),
        coverage={"speaker_cluster_count": len(speakers), "segment_count": len(segments)},
        known_limitations=list(artifact.get("warnings") or []),
    )
    ctx.model["capabilities"]["speaker_stitching"] = capability_record(
        status="PARTIAL" if speakers else "UNAVAILABLE", producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        known_limitations=["Film-local clusters do not assert character or cross-film identity."],
    )
    ctx.migration_rows.append(_migration_row(logical_type, artifact, len(speakers), ["cluster ID", "appearance intervals", "fallback state"], ["real-world identity unavailable"]))


def adapt_performances(ctx: AdapterContext, artifact: dict[str, Any], locator: str | None) -> None:
    logical_type = "performance"
    ctx.register_artifact(logical_type, artifact, locator)
    source_turn_map: dict[str, str] = {}
    performance_rows = artifact.get("performances") or []
    for performance_index, performance in enumerate(performance_rows):
        for turn_index, turn in enumerate(performance.get("ordered_turns") or []):
            source_id = str(turn.get("id") or f"{performance.get('id')}:turn:{turn_index}")
            if source_id in source_turn_map:
                continue
            interval = _artifact_interval(ctx, turn)
            provenance_id = ctx.object_provenance(logical_type, turn, performance_index * 100000 + turn_index, interval)
            turn_id = ctx.registry.issue("turn", {"artifact": ctx.artifact_records[logical_type]["source_artifact_id"], "source_id": source_id, **interval})
            speech_refs = [ctx.id_maps.get("speech", {}).get(source_id)]
            speech_refs = [item for item in speech_refs if item]
            speaker_id = ctx.id_maps.get("speaker", {}).get(str(turn.get("speaker_id")))
            ctx.model["dialogue_turns"].append({
                "dialogue_turn_id": turn_id, **interval, "ordered_speech_passage_references": speech_refs,
                "speaker_cluster_reference": speaker_id, "speaker_cluster_candidates": [speaker_id] if speaker_id else [],
                "transcript": str(turn.get("transcript") or ""), "preceding_turn_reference": None,
                "following_turn_reference": None, "containing_performance_references": [],
                "response_delay": turn.get("response_delay"), "overlap_or_interruption_evidence": deepcopy(turn.get("interruption")),
                "provenance_id": provenance_id, "confidence": _confidence(turn.get("confidence"), source=logical_type, provenance_id=provenance_id),
            })
            source_turn_map[source_id] = turn_id
            ctx.map_id("turn", source_id, turn_id)
            for passage_id in speech_refs:
                passage = next((item for item in ctx.model["speech_passages"] if item["speech_passage_id"] == passage_id), None)
                if passage is not None:
                    passage["linked_dialogue_turn_id"] = turn_id
    ordered_turns = sorted(ctx.model["dialogue_turns"], key=lambda row: (row["start"], row["end"], row["dialogue_turn_id"]))
    for index, turn in enumerate(ordered_turns):
        turn["preceding_turn_reference"] = ordered_turns[index - 1]["dialogue_turn_id"] if index else None
        turn["following_turn_reference"] = ordered_turns[index + 1]["dialogue_turn_id"] if index + 1 < len(ordered_turns) else None
    for index, row in enumerate(performance_rows):
        interval = _artifact_interval(ctx, row)
        provenance_id = ctx.object_provenance(logical_type, row, index, interval)
        performance_id = ctx.registry.issue("performance", {"artifact": ctx.artifact_records[logical_type]["source_artifact_id"], "source_id": row.get("id"), "signature": row.get("signature"), **interval})
        speech_refs = [ctx.id_maps.get("speech", {}).get(str(item)) for item in (row.get("dialogue_event_ids") or row.get("speaking_window_ids") or [])]
        turn_refs = [source_turn_map.get(str(item.get("id") or f"{row.get('id')}:turn:{turn_index}")) for turn_index, item in enumerate(row.get("ordered_turns") or [])]
        speaker_refs = [ctx.id_maps.get("speaker", {}).get(str(item)) for item in row.get("speaker_ids") or []]
        shot_refs = [ctx.id_maps.get("shot", {}).get(str(item)) for item in row.get("shot_ids") or []]
        normalized = {
            "performance_id": performance_id, **interval, "source_performance_reference": row.get("id"),
            "transcript": row.get("audio", {}).get("transcript") or " ".join(str(item.get("transcript") or "") for item in row.get("ordered_turns") or []),
            "speech_passage_references": [item for item in speech_refs if item], "dialogue_turn_references": [item for item in turn_refs if item],
            "speaker_cluster_references": [item for item in speaker_refs if item], "speaker_sequence": deepcopy(row.get("speaker_sequence") or []),
            "shot_references": [item for item in shot_refs if item], "transition_references": [],
            "render_history_references": deepcopy(row.get("render_history") or []), "review_history_references": deepcopy(row.get("review_history") or []),
            "provenance_id": provenance_id, "confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id),
            "source_detail": deepcopy(row),
        }
        ctx.model["performances"].append(normalized)
        ctx.map_id("performance", row.get("id"), performance_id)
        for ref in normalized["dialogue_turn_references"]:
            next(item for item in ctx.model["dialogue_turns"] if item["dialogue_turn_id"] == ref)["containing_performance_references"].append(performance_id)
        for ref in normalized["speech_passage_references"]:
            passage = next((item for item in ctx.model["speech_passages"] if item["speech_passage_id"] == ref), None)
            if passage:
                passage["linked_performance_ids"].append(performance_id)
        for ref in normalized["shot_references"]:
            next(item for item in ctx.model["shots"] if item["shot_id"] == ref)["linked_performance_ids"].append(performance_id)
        for ref in normalized["speaker_cluster_references"]:
            next(item for item in ctx.model["speaker_clusters"] if item["speaker_cluster_id"] == ref)["performance_references"].append(performance_id)
    ctx.model["capabilities"]["performance_objects"] = capability_record(
        status="AVAILABLE" if performance_rows else "UNAVAILABLE", producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        configuration_signature=artifact.get("config_signature"), implementation_version=artifact.get("tool_version"),
        coverage={"performance_count": len(performance_rows), "dialogue_turn_count": len(ctx.model["dialogue_turns"])},
        known_limitations=["Dialogue turns are a structural normalized view; no semantic function is inferred."],
    )
    visual_rows = [row for row in performance_rows if row.get("visual")]
    ctx.model["capabilities"]["visual_performance_evidence"] = capability_record(
        status="PARTIAL" if visual_rows else "UNAVAILABLE",
        producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        configuration_signature=artifact.get("config_signature"), implementation_version=artifact.get("tool_version"),
        coverage={"performances_with_embedded_visual_summary": len(visual_rows)},
        known_limitations=["This capability reflects existing embedded measurements and does not add visual interpretation."],
    )
    ctx.migration_rows.append(_migration_row(logical_type, artifact, len(performance_rows), ["all source performance fields retained in source_detail", "stable references"], ["semantic turn function unavailable"]))


def adapt_clip_library(ctx: AdapterContext, artifact: dict[str, Any], locator: str | None) -> None:
    logical_type = "clip_library"
    record = ctx.register_artifact(logical_type, artifact, locator)
    clip_index: list[dict[str, Any]] = []
    for clip in artifact.get("clips") or []:
        source_event_ids = [str(item) for item in (clip.get("event_ids") or ([clip.get("event_id")] if clip.get("event_id") else []))]
        passage_ids = [ctx.id_maps.get("speech", {}).get(item) for item in source_event_ids]
        passage_ids = [item for item in passage_ids if item]
        clip_row = {
            "source_clip_id": clip.get("id"), "source_event_ids": source_event_ids,
            "speech_passage_ids": passage_ids, "movie_timestamp": clip.get("movie_timestamp"),
            "duration": clip.get("duration"), "speaker_label": clip.get("speaker_id") or clip.get("speaker"),
            "transcript": clip.get("transcript"), "local_clip_reference": clip.get("path"),
        }
        clip_index.append(clip_row)
        for passage_id in passage_ids:
            passage = next(row for row in ctx.model["speech_passages"] if row["speech_passage_id"] == passage_id)
            passage.setdefault("source_clip_references", []).append(str(clip.get("id")))
    record["object_index"] = sorted(clip_index, key=lambda row: str(row.get("source_clip_id")))
    ctx.migration_rows.append(_migration_row(
        logical_type, artifact, len(clip_index), ["clip-to-passage references", "local clip references classified as local"], [],
    ))


def adapt_schedule_registry(ctx: AdapterContext, artifact: dict[str, Any], locator: str | None) -> None:
    logical_type = "replacement_schedule"
    record = ctx.register_artifact(logical_type, artifact, locator)
    record["placement_count"] = len(artifact.get("mappings") or [])
    record["schedule_config_signature"] = artifact.get("config_signature")
    ctx.model["capabilities"]["schedule_provenance"] = capability_record(
        status="AVAILABLE", producing_artifact_id=record["source_artifact_id"],
        configuration_signature=artifact.get("config_signature"), implementation_version=artifact.get("tool_version"),
        coverage={"placement_count": record["placement_count"]},
        known_limitations=["Schedule evidence describes one explicit transformation run."],
    )
    ctx.migration_rows.append(_migration_row(
        logical_type, artifact, record["placement_count"], ["schedule registered for explicit bridge ingestion"], [],
    ))


def adapt_cinematic_moments(ctx: AdapterContext, artifact: dict[str, Any], locator: str | None) -> None:
    logical_type = "cinematic_moments"
    ctx.register_artifact(logical_type, artifact, locator)
    moment_rows = artifact.get("moments") or []
    for index, row in enumerate(moment_rows):
        interval = _artifact_interval(ctx, row)
        provenance_id = ctx.object_provenance(logical_type, row, index, interval)
        moment_id = ctx.registry.issue("moment", {
            "artifact": ctx.artifact_records[logical_type]["source_artifact_id"],
            "source_id": row.get("id"), **interval,
        })
        shot_refs = [ctx.id_maps.get("shot", {}).get(str(item)) for item in row.get("shot_ids") or []]
        transition_refs = [item["transition_id"] for item in ctx.model["transitions"] if _overlaps(interval, item)]
        speech_refs = [item["speech_passage_id"] for item in ctx.model["speech_passages"] if _overlaps(interval, item)]
        performance_refs = [item["performance_id"] for item in ctx.model["performances"] if _overlaps(interval, item)]
        virtual = bool(row.get("virtual_boundary") or row.get("virtual_boundary_state"))
        ctx.model["cinematic_moments"].append({
            "cinematic_moment_id": moment_id, **interval, "source_moment_reference": row.get("id"),
            "shot_references": [item for item in shot_refs if item], "transition_references": transition_refs,
            "speech_passage_references": speech_refs, "performance_references": performance_refs,
            "boundary_type": "virtual" if virtual else "source_analysis_boundary",
            "boundary_confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id),
            "virtual_boundary_state": virtual, "stillness_evidence": deepcopy(row.get("stillness_evidence")),
            "speech_boundary_evidence": deepcopy(row.get("speech_boundary_evidence")),
            "transition_guard_evidence": deepcopy(row.get("transition_guard_evidence")),
            "source_analysis_version": artifact.get("backend_version") or artifact.get("tool_version"),
            "provenance_id": provenance_id,
            "confidence": _confidence(row.get("confidence"), source=logical_type, provenance_id=provenance_id),
        })
        ctx.map_id("moment", row.get("id"), moment_id)
        for shot_id in [item for item in shot_refs if item]:
            next(item for item in ctx.model["shots"] if item["shot_id"] == shot_id)["linked_moment_ids"].append(moment_id)
    ctx.model["capabilities"]["cinematic_moments"] = capability_record(
        status="AVAILABLE" if moment_rows else "UNAVAILABLE",
        producing_artifact_id=ctx.artifact_records[logical_type]["source_artifact_id"],
        configuration_signature=artifact.get("config_signature"), implementation_version=artifact.get("backend_version") or artifact.get("tool_version"),
        coverage={"moment_count": len(moment_rows)},
        known_limitations=["Cinematic moments are structural boundaries, not semantic scenes."],
    )
    ctx.migration_rows.append(_migration_row(logical_type, artifact, len(moment_rows), ["stable references", "boundary state", "time range"], ["semantic scene meaning unavailable"]))


def adapt_editorial_observations(
    ctx: AdapterContext, logical_type: str, artifact: dict[str, Any], locator: str | None,
) -> None:
    ctx.register_artifact(logical_type, artifact, locator)
    decisions = artifact.get("decisions") or []
    imported = 0
    for decision_index, decision in enumerate(decisions):
        start = decision.get("destination_start")
        end = decision.get("destination_end")
        interval = _artifact_interval(ctx, {"start": start, "end": end}) if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None
        failures = decision.get("failures") or [None]
        for failure_index, failure in enumerate(failures):
            failure = failure or {}
            source_row = dict(decision)
            source_row["id"] = f"{decision.get('placement_key', decision_index)}:{failure_index}"
            provenance_id = ctx.object_provenance(logical_type, source_row, decision_index * 1000 + failure_index, interval)
            placement_id = str(decision.get("placement_key") or f"mapping_index:{decision.get('mapping_index', decision_index)}")
            observation_id = ctx.registry.issue("editorial", {
                "artifact": ctx.artifact_records[logical_type]["source_artifact_id"],
                "placement_id": placement_id, "failure_index": failure_index,
                "failure_category": failure.get("category"), "final_state": decision.get("final_state"),
            })
            performance_refs = []
            source_performance = decision.get("window_id")
            if source_performance is not None:
                mapped = ctx.id_maps.get("performance", {}).get(str(source_performance))
                if mapped:
                    performance_refs.append(mapped)
            if not performance_refs and interval:
                performance_refs = [row["performance_id"] for row in ctx.model["performances"] if _overlaps(interval, row)]
            speech_refs = [row["speech_passage_id"] for row in ctx.model["speech_passages"] if interval and _overlaps(interval, row)]
            moment_refs = [row["cinematic_moment_id"] for row in ctx.model["cinematic_moments"] if interval and _overlaps(interval, row)]
            confidence_value = failure.get("confidence")
            ctx.model["editorial_observations"].append({
                "editorial_observation_id": observation_id, "observation_scope": "schedule_placement",
                "referenced_placement_id": placement_id, "referenced_performance_ids": performance_refs,
                "referenced_speech_passage_ids": speech_refs, "referenced_moment_ids": moment_refs,
                "failure_category": failure.get("category"), "severity": failure.get("severity"),
                "confidence": _confidence(confidence_value, source=logical_type, provenance_id=provenance_id),
                "evidence": deepcopy(failure.get("evidence") or decision.get("problem_evidence") or []),
                "recommendation": decision.get("recommendation"),
                "repairability_estimate": deepcopy(decision.get("repairability")),
                "repair_strategy": decision.get("repair_strategy") or failure.get("recommended_repair"),
                "final_placement_state": decision.get("final_state"),
                "verification_references": deepcopy(decision.get("verification_references") or []),
                "pass_number": decision.get("pass_number"), "candidate_reference": decision.get("candidate_reference"),
                "rollback_state": decision.get("rollback_state"), "provenance_id": provenance_id,
                "source_detail": deepcopy(decision),
            })
            imported += 1
    artifact_id = ctx.artifact_records[logical_type]["source_artifact_id"]
    ctx.model["capabilities"]["editorial_verification"] = capability_record(
        status="AVAILABLE" if decisions else "PARTIAL", producing_artifact_id=artifact_id,
        implementation_version=artifact.get("decision_engine_version") or artifact.get("editorial_system_version"),
        coverage={"decision_count": len(decisions), "observation_count": imported},
        known_limitations=["Editorial evidence is scoped to its originating run and schedule."],
    )
    repair_count = sum(1 for row in decisions if row.get("repair_strategy") or row.get("repairability"))
    ctx.model["capabilities"]["editorial_repair_evidence"] = capability_record(
        status="AVAILABLE" if repair_count else "UNAVAILABLE", producing_artifact_id=artifact_id,
        coverage={"decisions_with_repair_evidence": repair_count},
        known_limitations=["Repair evidence is historical; it is not an instruction to rerun repair."],
    )
    ctx.migration_rows.append(_migration_row(
        logical_type, artifact, imported,
        ["placement-scoped observations", "stable model references", "confidence metadata"],
        ["unresolved references remain absent rather than inferred"],
    ))


def _nearest_shot(shots: list[dict[str, Any]], timestamp: float, *, edge: str) -> str | None:
    if not shots:
        return None
    return min(shots, key=lambda row: (abs(float(row[edge]) - timestamp), row["shot_id"]))["shot_id"]


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return float(left["start"]) < float(right["end"]) and float(right["start"]) < float(left["end"])


def _migration_row(logical_type: str, artifact: dict[str, Any], count: int, normalized: list[str], unavailable: list[str]) -> dict[str, Any]:
    return {
        "source_artifact": logical_type, "source_schema": artifact.get("schema_version"),
        "objects_imported": count, "fields_preserved_directly": ["source artifact remains read-only"],
        "fields_normalized": normalized, "fields_omitted": [], "fields_unavailable": unavailable,
        "warnings": [], "validation_issues": [],
    }
