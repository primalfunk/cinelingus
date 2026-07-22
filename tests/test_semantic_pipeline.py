from pathlib import Path
from types import SimpleNamespace

from cinelingus.pipeline import Pipeline
from cinelingus.semantic import DeterministicFakeProvider, SemanticConfig, build_semantic_bundle
from cinelingus.util import write_json


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def _pipeline(tmp_path: Path, mode: str, weight: float) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = SimpleNamespace(semantic_mode=mode, semantic_weight=weight)
    pipeline.source = SimpleNamespace(cache_dir=tmp_path / "source")
    pipeline.destination = SimpleNamespace(cache_dir=tmp_path / "destination")
    pipeline.logger = _Logger()
    return pipeline


def _write_semantics(root: Path, *, film_id: str, passage_id: str, reference: str, text: str) -> None:
    model_dir = root / "cinematic_model"
    semantic_dir = model_dir / "semantic"
    model = {
        "film_id": film_id,
        "created_from_signature": f"{film_id}-signature",
        "speech_passages": [{
            "speech_passage_id": passage_id,
            "source_transcript_reference": reference,
            "original_transcript": text,
            "language": "en",
            "start": 0.0,
            "end": 1.0,
            "duration": 1.0,
            "provenance_id": f"{passage_id}-provenance",
        }],
    }
    write_json(model_dir / "film_model.json", model)
    config = SemanticConfig(dimensions=8)
    build_semantic_bundle(model, semantic_dir, DeterministicFakeProvider(config), config)


def test_disabled_pipeline_does_not_require_semantic_artifacts(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "SEMANTIC_DISABLED", 0.0)
    assert pipeline._semantic_schedule_context() is None


def test_active_pipeline_uses_neutral_fallback_when_artifacts_are_missing(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "SEMANTIC_ASSISTED", 0.2)
    context = pipeline._semantic_schedule_context()
    assert context is not None and context.active
    assert context.source_by_reference == {}
    assert context.model_identity["fallback"] == "NEUTRAL_LEGACY_SCORE"
    assert "preserving legacy scores" in pipeline.logger.messages[-1]


def test_pipeline_loads_compatible_ready_bundles(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "SEMANTIC_REPORT_ONLY", 0.0)
    _write_semantics(pipeline.source.cache_dir, film_id="source-film", passage_id="source-passage", reference="e1", text="weather")
    _write_semantics(pipeline.destination.cache_dir, film_id="destination-film", passage_id="destination-passage", reference="w1", text="sunny")

    context = pipeline._semantic_schedule_context()

    assert context is not None
    assert set(context.source_by_reference) == {"e1"}
    assert set(context.destination_by_reference) == {"w1"}
    assert context.model_identity["source_bundle_signature"]
    assert context.model_identity["destination_bundle_signature"]


def test_context_bridges_raw_clip_ids_to_filtered_passages_by_canonical_start(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "SEMANTIC_REPORT_ONLY", 0.0)
    _write_semantics(pipeline.source.cache_dir, film_id="source-film", passage_id="source-passage", reference="w1", text="stormy weather")
    _write_semantics(pipeline.destination.cache_dir, film_id="destination-film", passage_id="destination-passage", reference="w2", text="sunny")

    context = pipeline._semantic_schedule_context()
    annotated = context.annotate_clips([{
        "id": "c1", "event_id": "e1", "event_ids": ["e1"],
        "movie_timestamp": 0.0, "duration": 1.0,
    }])

    assert annotated[0]["_semantic_entity_ids"] == ["source-passage"]

    text_linked = context.annotate_clips([{
        "id": "c2", "event_id": "e2", "event_ids": ["e2"],
        "movie_timestamp": 9.0, "duration": 1.0, "transcript": "Stormy weather",
    }])
    assert text_linked[0]["_semantic_entity_ids"] == ["source-passage"]


def test_pipeline_rejects_ready_bundle_when_vector_is_corrupted(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "SEMANTIC_ASSISTED", 0.2)
    _write_semantics(pipeline.source.cache_dir, film_id="source-film", passage_id="source-passage", reference="e1", text="weather")
    _write_semantics(pipeline.destination.cache_dir, film_id="destination-film", passage_id="destination-passage", reference="w1", text="sunny")
    source_vector = next((pipeline.source.cache_dir / "cinematic_model" / "semantic" / "vectors").glob("*.f32"))
    source_vector.write_bytes(b"corrupted")

    context = pipeline._semantic_schedule_context()

    assert context is not None
    assert context.source_by_reference == {}
    assert context.model_identity["fallback"] == "NEUTRAL_LEGACY_SCORE"
