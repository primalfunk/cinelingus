from pathlib import Path

from cinelingus.schedule import build_schedule
from cinelingus.semantic import SemanticEntity, SemanticMode, SemanticScheduleContext


def _context(mode: SemanticMode, weight: float) -> SemanticScheduleContext:
    source = {
        "e1": SemanticEntity("speech_source_1", "film_source", "speech_passage", "en", (1.0, 0.0), {}),
        "e2": SemanticEntity("speech_source_2", "film_source", "speech_passage", "en", (0.0, 1.0), {}),
    }
    destination = {"w1": SemanticEntity("speech_destination", "film_destination", "speech_passage", "en", (0.0, 1.0), {})}
    return SemanticScheduleContext(mode, weight, source, destination, {"model_id": "fake"})


def _schedule(tmp_path: Path, *, context=None):
    clips = [
        {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": "1.wav", "movie_timestamp": 0.0, "duration": 2.0, "transcript": "weather", "confidence": 0.8, "usable": True},
        {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": "2.wav", "movie_timestamp": 2.0, "duration": 2.0, "transcript": "sunny", "confidence": 0.8, "usable": True},
    ]
    windows = [{"id": "w1", "start": 0.0, "end": 2.0, "duration": 2.0, "transcript": "sunny", "confidence": 0.8, "usable": True}]
    return build_schedule(clips=clips, windows=windows, source_hash="s", destination_hash="d", max_time_stretch=0.1, output_path=tmp_path / "schedule.json", scheduling_mode="best_fit", best_fit_lookahead=2, semantic_context=context)


def test_disabled_and_assisted_zero_weight_use_exact_legacy_mapping(tmp_path: Path) -> None:
    baseline = _schedule(tmp_path / "base")
    disabled = _schedule(tmp_path / "disabled", context=_context(SemanticMode.DISABLED, 0.0))
    zero = _schedule(tmp_path / "zero", context=_context(SemanticMode.ASSISTED, 0.0))
    assert baseline["mappings"] == disabled["mappings"] == zero["mappings"]
    assert "semantic_scoring" not in disabled and "semantic_scoring" not in zero


def test_report_only_reports_scores_without_changing_selection(tmp_path: Path) -> None:
    baseline = _schedule(tmp_path / "base")
    report = _schedule(tmp_path / "report", context=_context(SemanticMode.REPORT_ONLY, 0.0))
    assert baseline["mappings"][0]["clip_id"] == report["mappings"][0]["clip_id"] == "c1"
    assert baseline["mappings"][0]["score"] == report["mappings"][0]["score"]
    assert report["mappings"][0]["semantic_compatibility"]["mode"] == "SEMANTIC_REPORT_ONLY"


def test_assisted_semantics_reranks_only_already_legal_candidates(tmp_path: Path) -> None:
    assisted = _schedule(tmp_path, context=_context(SemanticMode.ASSISTED, 0.2))
    assert assisted["mappings"][0]["clip_id"] == "c2"
    semantic = assisted["mappings"][0]["semantic_compatibility"]
    assert semantic["raw_cosine_similarity"] == 1.0
    assert semantic["configured_weight"] == 0.2
    assert semantic["fallback_state"] == "NONE"


def test_assisted_semantics_cannot_reduce_transcript_completeness(tmp_path: Path) -> None:
    clips = [
        {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": "1.wav", "movie_timestamp": 0.0, "duration": 2.0, "transcript": "This is a complete sentence.", "confidence": 0.8, "usable": True},
        {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": "2.wav", "movie_timestamp": 2.0, "duration": 2.0, "transcript": "unfinished fragment", "confidence": 0.8, "usable": True},
    ]
    windows = [{"id": "w1", "start": 0.0, "end": 2.0, "duration": 2.0, "transcript": "sunny", "confidence": 0.8, "usable": True}]

    schedule = build_schedule(
        clips=clips, windows=windows, source_hash="s", destination_hash="d",
        max_time_stretch=0.1, output_path=tmp_path / "schedule.json",
        scheduling_mode="best_fit", best_fit_lookahead=2,
        semantic_context=_context(SemanticMode.ASSISTED, 0.2),
    )

    assert schedule["mappings"][0]["clip_id"] == "c1"


def test_performance_aggregate_fills_only_missing_clip_semantics() -> None:
    context = _context(SemanticMode.REPORT_ONLY, 0.0)
    aggregate = SemanticEntity(
        "performance-source", "film_source", "performance_passage_aggregate", "en",
        (0.0, 1.0), {"speech_passage_ids": ["speech-a", "speech-b"]},
    )
    context = SemanticScheduleContext(
        context.mode, context.weight, context.source_by_reference,
        context.destination_by_reference, context.model_identity,
        source_by_performance={"p1": aggregate},
    )
    groups = context.annotate_source_performance_groups([{
        "id": "p1", "clips": [
            {"id": "direct", "_semantic_vector": (1.0, 0.0), "_semantic_entity_ids": ["direct"]},
            {"id": "missing"},
        ],
    }])
    assert groups[0]["clips"][0]["_semantic_entity_ids"] == ["direct"]
    assert groups[0]["clips"][1]["_semantic_entity_ids"] == ["speech-a", "speech-b"]
    assert groups[0]["clips"][1]["_semantic_evidence_scope"] == "performance_passage_aggregate"


def test_boundary_start_bridge_accepts_small_unique_analysis_jitter() -> None:
    entity = SemanticEntity(
        "passage", "film_source", "speech_passage", "en", (1.0, 0.0),
        {"start": 10.0, "normalized_text": "when"},
    )
    context = SemanticScheduleContext(
        SemanticMode.REPORT_ONLY, 0.0, {}, {}, {"model_id": "fake"},
        source_by_start={"10.000": (entity,)},
    )

    clip = context.annotate_clips([{"id": "clip", "movie_timestamp": 10.2, "transcript": "mistranscribed"}])[0]

    assert clip["_semantic_entity_ids"] == ["passage"]
    assert clip["_semantic_evidence_scope"] == "direct_passage_boundary_bridge"

    corroborated = context.annotate_clips([{"id": "clip", "movie_timestamp": 10.9, "transcript": "Went"}])[0]
    assert corroborated["_semantic_entity_ids"] == ["passage"]

    rejected = context.annotate_clips([{"id": "clip", "movie_timestamp": 10.9, "transcript": "unrelated"}])[0]
    assert "_semantic_vector" not in rejected


def test_one_token_text_bridge_requires_nearest_temporal_match_to_be_unambiguous() -> None:
    near = SemanticEntity("near", "film_source", "speech_passage", "en", (1.0, 0.0), {"start": 11.8})
    far = SemanticEntity("far", "film_source", "speech_passage", "en", (0.0, 1.0), {"start": 7.5})
    context = SemanticScheduleContext(
        SemanticMode.REPORT_ONLY, 0.0, {}, {}, {"model_id": "fake"},
        source_by_text={"oh": (far, near)},
    )

    matched = context.annotate_clips([{"id": "clip", "movie_timestamp": 10.0, "transcript": "Oh"}])[0]
    assert matched["_semantic_entity_ids"] == ["near"]
    assert matched["_semantic_evidence_scope"] == "direct_passage_text_bridge"

    ambiguous_context = SemanticScheduleContext(
        SemanticMode.REPORT_ONLY, 0.0, {}, {}, {"model_id": "fake"},
        source_by_text={"oh": (
            SemanticEntity("left", "film_source", "speech_passage", "en", (1.0, 0.0), {"start": 9.0}),
            SemanticEntity("right", "film_source", "speech_passage", "en", (0.0, 1.0), {"start": 11.0}),
        )},
    )
    ambiguous = ambiguous_context.annotate_clips([{"id": "clip", "movie_timestamp": 10.0, "transcript": "Oh"}])[0]
    assert "_semantic_vector" not in ambiguous
