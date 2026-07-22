from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from cinelingus.cinematic_model.builder import FilmModelBuildError, build_film_model
from cinelingus.cinematic_model.cache import evaluate_model_cache
from cinelingus.cinematic_model.lookup import FilmModelView
from cinelingus.cinematic_model.reports import SEMANTIC_LIMITATION, compare_models, write_model_bundle
from cinelingus.cinematic_model.schedule_bridge import (
    ScheduleBridgeError, compare_schedule_equivalence, ingest_schedule, reconstruct_schedule,
)
from cinelingus.cinematic_model.serialization import canonical_json
from cinelingus.cinematic_model.validation import validate_film_model
from cinelingus.cli import main
from cinelingus.util import read_json, write_json
from cinelingus.validation import validate_artifact


def _artifacts() -> dict[str, dict]:
    media_hash = "d" * 64
    return {
        "movie": {
            "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": media_hash,
            "creation_timestamp": "volatile", "path": "C:/movies/example.mp4", "duration": 10.0,
            "resolution": "1920x1080", "frame_rate": 24.0, "sample_rate": 48000,
            "channels": 2, "codec": "h264",
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "aac", "channel_layout": "stereo"},
            ],
        },
        "dialogue_events": {
            "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": media_hash,
            "creation_timestamp": "volatile", "config_signature": "speech-config", "detected_language": "en",
            "events": [{"id": "e1", "start": 1.0, "end": 3.0, "duration": 2.0, "transcript": "Hello there.", "confidence": 0.8, "speaker": "speaker_001"}],
        },
        "shots": {
            "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": media_hash,
            "creation_timestamp": "volatile", "config_signature": "shot-config", "core_evidence_version": "core-v1",
            "shots": [
                {"id": "s1", "start": 0.0, "end": 5.0, "duration": 5.0, "confidence": 0.9},
                {"id": "s2", "start": 5.0, "end": 10.0, "duration": 5.0, "confidence": 0.85},
            ],
            "transitions": [{"id": "tr1", "kind": "gradual_candidate", "start": 4.9, "end": 5.1, "confidence": 0.6}],
        },
        "speaker_map": {
            "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": media_hash,
            "creation_timestamp": "volatile", "config_signature": "speaker-config", "actual_backend": "pyannote",
            "model_name": "speaker-model", "diarization_status": "SUCCESS", "fallback_status": "NONE",
            "speakers": [{"speaker_id": "speaker_001", "total_duration": 2.0, "event_count": 1, "first_seen": 1.0, "last_seen": 3.0, "confidence": 0.7}],
            "speaker_segments": [{"id": "seg1", "speaker_id": "speaker_001", "start": 1.0, "end": 3.0, "duration": 2.0}],
            "warnings": [], "diagnostics": {},
        },
        "performance": {
            "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": media_hash,
            "creation_timestamp": "volatile", "config_signature": "performance-config",
            "performances": [{
                "id": "p000001", "start": 1.0, "end": 3.0, "duration": 2.0,
                "dialogue_event_ids": ["e1"], "shot_ids": ["s1"], "speaker_ids": ["speaker_001"],
                "speaker_sequence": ["speaker_001"], "ordered_turns": [{
                    "id": "e1", "start": 1.0, "end": 3.0, "duration": 2.0,
                    "speaker_id": "speaker_001", "transcript": "Hello there.", "confidence": 0.8,
                }],
                "audio": {"transcript": "Hello there.", "confidence": 0.8},
                "render_history": [], "review_history": [], "confidence": 0.75, "signature": "performance-1",
            }],
        },
    }


def test_builder_normalizes_and_links_existing_artifacts() -> None:
    result = build_film_model(_artifacts(), schemas_dir=Path("schemas"))
    assert result.validation_report["status"] == "VALID"
    assert result.build_report["object_counts"] == {
        "shots": 2, "transitions": 1, "speech_passages": 1, "speaker_clusters": 1,
        "dialogue_turns": 1, "performances": 1, "cinematic_moments": 0, "editorial_observations": 0,
    }
    performance = result.model["performances"][0]
    assert performance["speech_passage_references"] == [result.model["speech_passages"][0]["speech_passage_id"]]
    assert performance["speaker_cluster_references"] == [result.model["speaker_clusters"][0]["speaker_cluster_id"]]
    assert result.model["capabilities"]["semantic_scene_understanding"]["status"] == "UNAVAILABLE"
    assert result.migration_report["source_artifacts_modified"] is False


def test_builder_is_deterministic_and_ignores_volatile_timestamps() -> None:
    first = _artifacts()
    second = deepcopy(first)
    second["movie"]["creation_timestamp"] = "different"
    second["dialogue_events"]["creation_timestamp"] = "different"
    a = build_film_model(first, schemas_dir=Path("schemas"))
    b = build_film_model(dict(reversed(list(second.items()))), schemas_dir=Path("schemas"))
    assert a.model["film_id"] == b.model["film_id"]
    assert a.model["created_from_signature"] == b.model["created_from_signature"]
    assert canonical_json(a.model) == canonical_json(b.model)


def test_redundant_optional_view_does_not_change_builder_signature() -> None:
    artifacts = _artifacts()
    artifacts["editorial_report"] = {"schema_version": "1.0", "editorial_system_version": "v1", "decisions": []}
    artifacts["editorial_decisions"] = {"schema_version": "1.0", "decision_engine_version": "v1", "decisions": [{"volatile_redundant_view": 1}]}
    first = build_film_model(artifacts, schemas_dir=Path("schemas"))
    artifacts["editorial_decisions"]["decisions"][0]["volatile_redundant_view"] = 2
    second = build_film_model(artifacts, schemas_dir=Path("schemas"))
    assert first.model["created_from_signature"] == second.model["created_from_signature"]


def test_builder_merges_primary_and_additional_role_speech_with_provenance() -> None:
    artifacts = _artifacts()
    artifacts.pop("dialogue_events")
    artifacts["timeline"] = {
        "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": "d" * 64,
        "configured_language": "en", "windows": [{
            "id": "w1", "start": 1.1, "end": 3.1, "transcript": "Hello there.", "confidence": 0.7,
        }],
    }
    artifacts["source_dialogue_dialogue_events"] = {
        "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": "d" * 64,
        "detected_language": "en", "events": [{
            "id": "e1", "start": 1.0, "end": 3.0, "transcript": "Hello there.", "confidence": 0.8,
        }],
    }
    result = build_film_model(artifacts, schemas_dir=Path("schemas"))
    assert result.validation_report["status"] == "VALID"
    assert len(result.model["speech_passages"]) == 2
    assert {row["source_transcript_reference"] for row in result.model["speech_passages"]} == {"e1", "w1"}
    assert {row["source_artifact_type"] for row in result.model["provenance"]} >= {
        "timeline", "source_dialogue_dialogue_events",
    }
    assert result.model["capabilities"]["transcription"]["coverage"] == {
        "passage_count": 2, "speech_view_count": 2,
    }


def test_builder_rejects_cross_film_artifacts() -> None:
    artifacts = _artifacts()
    artifacts["shots"]["media_hash"] = "e" * 64
    with pytest.raises(FilmModelBuildError, match="shots"):
        build_film_model(artifacts, schemas_dir=Path("schemas"))


def test_validator_reports_dangling_references_by_category() -> None:
    result = build_film_model(_artifacts(), schemas_dir=Path("schemas"))
    model = deepcopy(result.model)
    model["performances"][0]["shot_references"] = ["shot_missing"]
    report = validate_film_model(model, Path("schemas"))
    assert report["status"] == "INVALID"
    assert report["checks"]["referential"] == "FAIL"
    assert any(issue["category"] == "REFERENTIAL" for issue in report["errors"])


def test_moment_adapter_keeps_structural_boundary_language() -> None:
    artifacts = _artifacts()
    artifacts["cinematic_moments"] = {
        "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": "d" * 64,
        "config_signature": "moment-config", "backend_version": "core-v1",
        "moments": [{"id": "moment_source", "start": 0.5, "end": 4.0, "duration": 3.5, "shot_ids": ["s1"], "virtual_boundary": True}],
    }
    result = build_film_model(artifacts, schemas_dir=Path("schemas"))
    moment = result.model["cinematic_moments"][0]
    assert result.validation_report["status"] == "VALID"
    assert moment["boundary_type"] == "virtual"
    assert moment["virtual_boundary_state"] is True
    assert result.model["capabilities"]["cinematic_moments"]["status"] == "AVAILABLE"


def test_near_media_boundary_overshoot_is_clamped_and_provenanced() -> None:
    artifacts = _artifacts()
    artifacts["dialogue_events"]["events"][0]["end"] = 10.02
    artifacts["dialogue_events"]["events"][0]["duration"] = 9.02
    artifacts["performance"]["performances"] = []
    result = build_film_model(artifacts, schemas_dir=Path("schemas"))
    passage = result.model["speech_passages"][0]
    provenance = next(row for row in result.model["provenance"] if row["provenance_id"] == passage["provenance_id"])
    assert passage["end"] == 10.0
    assert provenance["source_time_range"]["end"] == 10.02
    assert any(row["action"] == "normalized_media_boundary_overshoot" for row in provenance["migration_history"])


def test_editorial_adapter_preserves_run_scope_and_final_state() -> None:
    artifacts = _artifacts()
    artifacts["editorial_decisions"] = {
        "schema_version": "1.0", "decision_engine_version": "editorial-v1",
        "decisions": [{
            "placement_key": "editorial_placement_000001", "mapping_index": 0,
            "window_id": "p000001", "destination_start": 1.0, "destination_end": 3.0,
            "recommendation": "repair", "repair_strategy": "repair_performance_structure",
            "repairability": {"score": 0.8, "class": "high"},
            "final_state": "BEST_KNOWN_UNRESOLVED",
            "failures": [{"category": "performance_mismatch", "severity": "medium", "confidence": 0.8, "evidence": {"reason": "fixture"}}],
        }],
    }
    result = build_film_model(artifacts, schemas_dir=Path("schemas"))
    observation = result.model["editorial_observations"][0]
    assert result.validation_report["status"] == "VALID"
    assert observation["observation_scope"] == "schedule_placement"
    assert observation["referenced_placement_id"] == "editorial_placement_000001"
    assert observation["final_placement_state"] == "BEST_KNOWN_UNRESOLVED"
    assert observation["referenced_performance_ids"] == [result.model["performances"][0]["performance_id"]]
    assert result.model["capabilities"]["editorial_repair_evidence"]["status"] == "AVAILABLE"


def test_lookup_api_traces_objects_without_semantic_querying() -> None:
    artifacts = _artifacts()
    artifacts["editorial_decisions"] = {
        "schema_version": "1.0", "decision_engine_version": "editorial-v1",
        "decisions": [{"placement_key": "placement_1", "window_id": "p000001", "destination_start": 1.0, "destination_end": 3.0, "failures": []}],
    }
    result = build_film_model(artifacts, schemas_dir=Path("schemas"))
    view = FilmModelView(result.model)
    performance = result.model["performances"][0]
    turn_id = performance["dialogue_turn_references"][0]
    assert view.get(performance["performance_id"]) == performance
    assert view.performances_containing_turn(turn_id) == (performance,)
    assert len(view.shots_intersecting_performance(performance["performance_id"])) == 1
    assert view.source_artifact_for(performance["performance_id"])["logical_artifact_type"] == "performance"
    assert view.provenance_chain(performance["performance_id"])
    assert view.editorial_observations_for_placement("placement_1")
    assert view.capability_status("semantic_similarity")["status"] == "UNAVAILABLE"


def test_model_bundle_report_and_cache_decisions_are_deterministic(tmp_path) -> None:
    result = build_film_model(_artifacts(), schemas_dir=Path("schemas"))
    paths = write_model_bundle(tmp_path, result)
    report = paths["report"].read_text(encoding="utf-8")
    assert SEMANTIC_LIMITATION in report
    assert "Schedule-trace readiness: NOT READY" in report
    hit = evaluate_model_cache(paths["model"], result.model["created_from_signature"])
    assert hit.reuse is True
    changed = evaluate_model_cache(paths["model"], "different-signature")
    assert changed.reuse is False
    assert changed.status == "REBUILD_REQUIRED"
    forced = evaluate_model_cache(paths["model"], result.model["created_from_signature"], force=True)
    assert forced.status == "FORCE_REBUILD"
    assert compare_models(result.model, result.model)["equivalent"] is True


def test_developer_cli_builds_and_reuses_opt_in_bundle(tmp_path, capsys) -> None:
    artifact_dir = tmp_path / "cache-role"
    output_dir = tmp_path / "model-output"
    for name, artifact in _artifacts().items():
        write_json(artifact_dir / f"{name}.json", artifact)
    argv = ["build-film-model", "--artifact-dir", str(artifact_dir), "--output", str(output_dir)]
    assert main(argv) == 0
    assert (output_dir / "film_model.json").is_file()
    capsys.readouterr()
    assert main(argv) == 0
    assert "CACHE_HIT" in capsys.readouterr().out
    assert main(["validate-film-model", str(output_dir / "film_model.json")]) == 0


def test_developer_cli_merges_an_additional_speech_role(tmp_path) -> None:
    artifact_dir = tmp_path / "destination_video"
    speech_dir = tmp_path / "source_dialogue"
    output_dir = tmp_path / "model-output"
    for name, artifact in _artifacts().items():
        write_json(artifact_dir / f"{name}.json", artifact)
    supplemental = deepcopy(_artifacts()["dialogue_events"])
    supplemental["events"][0].update({"id": "source-e1", "transcript": "Supplemental role."})
    write_json(speech_dir / "dialogue_events.json", supplemental)
    argv = [
        "build-film-model", "--artifact-dir", str(artifact_dir),
        "--include-speech-role", str(speech_dir), "--output", str(output_dir),
    ]
    assert main(argv) == 0
    model = read_json(output_dir / "film_model.json")
    assert len(model["speech_passages"]) == 2
    assert {row["source_transcript_reference"] for row in model["speech_passages"]} == {"e1", "source-e1"}


def _schedule_fixture() -> dict:
    return {
        "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": "d" * 64,
        "source_media_hash": "d" * 64, "destination_media_hash": "d" * 64,
        "config_signature": "schedule-config", "dialogue_suppression": "replace",
        "mappings": [{
            "editorial_placement_id": "editorial_placement_000001", "window_id": "e1",
            "performance_id": "p000001", "destination_performance_id": "p000001",
            "source_performance_id": "p000001", "clip_id": "c000001",
            "source_speaker_id": "speaker_001", "destination_speaker_id": "speaker_001",
            "destination_timestamp": 1.0, "planned_render_duration": 2.0,
            "source_movie_timestamp": 1.0, "clip_movie_timestamp": 1.0,
            "clip_trim_start": 0.0, "clip_trim_duration": 2.0, "stretch_factor": 1.0,
            "leading_silence": 0.0, "trailing_silence": 0.0, "suppression_mode": "replace",
            "enabled": True, "score": 0.8, "score_components": {"fit": 0.8},
            "render_operations": [{"operation": "fade_in_out", "duration": None}],
        }],
    }


def test_schedule_bridge_round_trip_is_lossless_and_traced(tmp_path) -> None:
    source_artifacts = _artifacts()
    source_artifacts["clip_library"] = {
        "schema_version": "1.0", "tool_version": "0.3.0", "media_hash": "d" * 64,
        "config_signature": "clips", "clips": [{
            "id": "c000001", "event_id": "e1", "event_ids": ["e1"],
            "movie_timestamp": 1.0, "duration": 2.0, "speaker_id": "speaker_001",
            "transcript": "Hello there.", "path": "C:/cache/clip.wav",
        }],
    }
    destination_artifacts = _artifacts()
    destination_artifacts["replacement_schedule"] = _schedule_fixture()
    source = build_film_model(source_artifacts, schemas_dir=Path("schemas")).model
    destination = build_film_model(destination_artifacts, schemas_dir=Path("schemas")).model
    schedule = _schedule_fixture()
    bridge = ingest_schedule(schedule, source_model=source, destination_model=destination)
    reconstructed = reconstruct_schedule(bridge)
    comparison = compare_schedule_equivalence(schedule, reconstructed, bridge=bridge)
    assert bridge["validation_state"]["schedule_trace_readiness"] == "READY"
    assert bridge["placements"][0]["donor"]["speech_passage_ids"]
    assert bridge["placements"][0]["destination"]["performance_ids"]
    assert reconstructed == schedule
    assert comparison["equivalent"] is True
    assert all(comparison["checks"].values())
    bridge_path = tmp_path / "schedule_bridge.json"
    write_json(bridge_path, bridge)
    assert validate_artifact("schedule_bridge", bridge_path, Path("schemas"))["placement_count"] == 1


def test_schedule_equivalence_classifies_behavioral_changes() -> None:
    original = _schedule_fixture()
    changed = deepcopy(original)
    changed["mappings"][0]["destination_timestamp"] = 2.0
    comparison = compare_schedule_equivalence(original, changed)
    assert comparison["equivalent"] is False
    assert comparison["unacceptable_difference_count"] == 1
    assert comparison["differences"][0]["classification"] == "behavioral difference"


def test_schedule_bridge_rejects_tampered_canonical_payload() -> None:
    artifacts = _artifacts()
    model = build_film_model(artifacts, schemas_dir=Path("schemas")).model
    bridge = ingest_schedule(_schedule_fixture(), source_model=model, destination_model=model)
    bridge["canonical_schedule_payload"]["mappings"][0]["score"] = 0.1
    with pytest.raises(ScheduleBridgeError, match="signature"):
        reconstruct_schedule(bridge)


def test_schedule_bridge_discloses_editorial_donor_contradiction() -> None:
    artifacts = _artifacts()
    artifacts["editorial_report"] = {
        "schema_version": "1.0", "editorial_system_version": "v1",
        "decisions": [{
            "placement_key": "editorial_placement_000001", "window_id": "p000001",
            "destination_start": 1.0, "destination_end": 3.0, "clip_id": "different_clip",
            "failures": [], "final_state": "IMPROVED_ACCEPTED",
        }],
    }
    source_artifacts = _artifacts()
    source_artifacts["clip_library"] = {
        "schema_version": "1.0", "media_hash": "d" * 64,
        "clips": [{"id": "c000001", "event_ids": ["e1"], "movie_timestamp": 1.0, "duration": 2.0}],
    }
    source = build_film_model(source_artifacts, schemas_dir=Path("schemas")).model
    destination = build_film_model(artifacts, schemas_dir=Path("schemas")).model
    bridge = ingest_schedule(_schedule_fixture(), source_model=source, destination_model=destination)
    assert bridge["validation_state"]["status"] == "VALID_WITH_WARNINGS"
    assert bridge["validation_state"]["schedule_trace_readiness"] == "DEGRADED"
    assert bridge["placements"][0]["contradictions"]
