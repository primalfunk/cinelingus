from __future__ import annotations

import json
from pathlib import Path

import pytest

from cinelingus.cinematic_model import (
    StableIdRegistry,
    canonical_interval,
    canonical_time,
    confidence_record,
    initial_capability_manifest,
    new_film_model,
    stable_entity_id,
    stable_film_id,
)
from cinelingus.cinematic_model.serialization import canonical_json, read_film_model, write_film_model
from cinelingus.validation import validate_artifact


def test_published_json_schemas_declare_their_dialect() -> None:
    for name in ("film_model.schema.json", "schedule_bridge.schema.json"):
        schema = json.loads((Path("schemas") / name).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_film_and_entity_ids_are_evidence_stable() -> None:
    film_id = stable_film_id("a" * 64, "inspection-v1")
    evidence = {"start": 1.25, "end": 2.5, "source_object_id": "shot_7"}
    assert film_id == stable_film_id("a" * 64, "inspection-v1")
    assert stable_entity_id("shot", film_id, evidence) == stable_entity_id("shot", film_id, dict(reversed(list(evidence.items()))))
    assert stable_entity_id("shot", film_id, evidence).startswith("shot_")


def test_registry_reuses_identical_evidence() -> None:
    registry = StableIdRegistry(stable_film_id("b" * 64))
    assert registry.issue("speech", {"source": "event_1"}) == registry.issue("speech", {"source": "event_1"})


def test_canonical_timing_is_half_up_milliseconds() -> None:
    assert canonical_time("1.2345") == 1.235
    assert canonical_interval(1, 1.2345) == {"start": 1.0, "end": 1.235, "duration": 0.235}
    with pytest.raises(ValueError):
        canonical_interval(2, 1)


def test_confidence_does_not_conflate_missing_states_or_probabilities() -> None:
    unknown = confidence_record(
        state="unknown", value=None, scale=None, interpretation="Not reported",
        evidence_source="legacy", calibration_state="unknown", fallback_state="unknown",
    )
    unavailable = confidence_record(
        state="unavailable", value=None, scale=None, interpretation="No detector",
        evidence_source="builder", calibration_state="not_applicable", fallback_state="not_applicable",
    )
    assert unknown != unavailable
    with pytest.raises(ValueError):
        confidence_record(
            state="numeric", value=1.2, scale="unit_interval", interpretation="probability",
            evidence_source="test", calibration_state="calibrated_probability", fallback_state="direct",
        )


def test_phase1_unsupported_capabilities_are_explicit() -> None:
    manifest = initial_capability_manifest()
    assert manifest["active_speaker_attribution"]["status"] == "UNAVAILABLE"
    assert "out of scope" in manifest["active_speaker_attribution"]["known_limitations"][0].lower()


def test_canonical_serialization_round_trips_deterministically(tmp_path) -> None:
    model = {"shots": [{"shot_id": "shot_b"}, {"shot_id": "shot_a"}], "schema_version": "1.0.0"}
    path = tmp_path / "film_model.json"
    write_film_model(path, model)
    loaded = read_film_model(path)
    assert [row["shot_id"] for row in loaded["shots"]] == ["shot_a", "shot_b"]
    assert canonical_json(loaded) == canonical_json(json.loads(canonical_json(model)))


def test_empty_film_model_is_explicit_and_schema_valid(tmp_path) -> None:
    film_id = stable_film_id("c" * 64)
    model = new_film_model(
        film_id=film_id,
        media={
            "film_id": film_id,
            "media_hash": "c" * 64,
            "source_path_reference": "C:/local/movie.mp4",
            "source_path_kind": "local_reference",
            "normalized_source_path": None,
            "filename": "movie.mp4",
            "duration": 12.0,
            "container": None,
            "video_stream_summary": None,
            "audio_stream_summary": None,
            "frame_rate": None,
            "resolution": None,
            "channel_layout": None,
            "media_inspection_version": None,
            "corpus_media_id": None,
            "source_artifact_signature": "inspection-signature",
        },
        created_from_signature="builder-signature",
        duration=12.0,
    )
    assert model["shots"] == []
    assert model["capabilities"]["semantic_similarity"]["status"] == "UNAVAILABLE"
    path = tmp_path / "film_model.json"
    write_film_model(path, model)
    assert validate_artifact("film_model", path, Path("schemas")) == read_film_model(path)
