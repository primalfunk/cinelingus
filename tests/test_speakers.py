from pathlib import Path
import math
import struct
import wave

from cinelingus.mutations import speaker_aware_shuffle_selection
from cinelingus.speakers import (
    annotate_artifact_speakers,
    apply_speaker_mapping_to_schedule,
    build_speaker_map,
    build_speaker_mapping,
    build_item_speaker_assignments,
    diarization_backend_status,
    diarization_setup_status,
    enrich_performances_with_speakers,
    speaker_map_diagnostics,
    speaker_mapping_summary,
    speaker_map_has_real_diarization,
    speaker_map_identity_ready,
    speaker_preservation_summary,
    _hf_token,
    _load_pyannote_audio_input,
)
from cinelingus.validation import validate_artifact


def _items():
    return [
        {"id": "e1", "start": 0.0, "end": 1.0, "duration": 1.0},
        {"id": "e2", "start": 1.2, "end": 2.0, "duration": 0.8},
        {"id": "e3", "start": 5.0, "end": 6.0, "duration": 1.0},
    ]


def test_speaker_map_generation_exports_valid_artifact(tmp_path: Path) -> None:
    output = tmp_path / "speaker_map.json"

    speaker_map = build_speaker_map(media_hash="hash", speech_items=_items(), output_path=output)

    assert speaker_map["speaker_count"] == 2
    assert speaker_map["speaker_segments"][0]["speaker_id"] == "speaker_001"
    assert speaker_map["speaker_segments"][1]["speaker_id"] == "speaker_002"
    validate_artifact("speaker_map", output, Path.cwd() / "schemas")


def test_pyannote_backend_falls_back_explicitly_when_unavailable(tmp_path: Path) -> None:
    output = tmp_path / "speaker_map.json"

    speaker_map = build_speaker_map(
        media_hash="hash",
        speech_items=_items(),
        output_path=output,
        backend="pyannote",
        audio_path=tmp_path / "missing.wav",
    )

    assert speaker_map["requested_backend"] == "pyannote"
    assert speaker_map["diarization_tool"] == "heuristic_timing_v1"
    assert "falling back to heuristic speaker labels" in speaker_map["warnings"][0]
    validate_artifact("speaker_map", output, Path.cwd() / "schemas")


def test_diarization_backend_status_reports_heuristic_and_unknown() -> None:
    assert diarization_backend_status(backend="heuristic") == {"backend": "heuristic", "available": True, "reason": None}

    status = diarization_backend_status(backend="bogus")

    assert status["available"] is False
    assert status["reason"] == "unknown diarization backend: bogus"


def test_diarization_setup_status_reports_token_requirement() -> None:
    status = diarization_setup_status(backend="pyannote", hf_token="")

    if status["available"] is False:
        assert "pyannote.audio is not installed" in status["reason"] or "HUGGINGFACE_TOKEN" in status["reason"]


def test_hf_token_reads_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test_token")

    assert _hf_token() == "hf_test_token"


def test_pyannote_audio_input_loader_reads_pcm_wav_without_torchcodec(tmp_path: Path) -> None:
    audio = tmp_path / "probe.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        for index in range(1600):
            sample = int(1000 * math.sin(index / 10))
            handle.writeframes(struct.pack("<h", sample))

    loaded = _load_pyannote_audio_input(audio)

    assert loaded["sample_rate"] == 16000
    assert tuple(loaded["waveform"].shape) == (1, 1600)



def test_speaker_ids_attach_to_dialogue_events_and_clips(tmp_path: Path) -> None:
    speaker_map = build_speaker_map(media_hash="hash", speech_items=_items(), output_path=tmp_path / "speaker_map.json")
    events = {"media_hash": "hash", "events": _items()}
    clips = {
        "media_hash": "hash",
        "clips": [
            {"id": "c1", "movie_timestamp": 0.0, "duration": 1.0},
            {"id": "c2", "movie_timestamp": 1.2, "duration": 0.8},
        ],
    }

    annotated_events = annotate_artifact_speakers(events, speaker_map, collection_key="events")
    annotated_clips = annotate_artifact_speakers(clips, speaker_map, collection_key="clips")

    assert annotated_events["events"][0]["speaker_id"] == "speaker_001"
    assert annotated_events["events"][1]["speaker_id"] == "speaker_002"
    assert annotated_clips["clips"][0]["speaker_id"] == "speaker_001"
    assert annotated_clips["clips"][1]["speaker_id"] == "speaker_002"


def test_performance_speaker_pattern_generation(tmp_path: Path) -> None:
    speaker_map = build_speaker_map(media_hash="hash", speech_items=_items(), output_path=tmp_path / "speaker_map.json")
    performances = {
        "media_hash": "hash",
        "performances": [
            {
                "id": "p1",
                "start": 0.0,
                "end": 2.2,
                "estimated_speaker_count": 1,
                "signature": {},
            }
        ],
    }

    enriched = enrich_performances_with_speakers(performances, speaker_map)
    row = enriched["performances"][0]

    assert row["speaker_ids"] == ["speaker_001", "speaker_002"]
    assert row["dominant_speaker_id"] in {"speaker_001", "speaker_002"}
    assert row["speaker_pattern"] == "speaker_001 speaker_002"
    assert row["signature"]["speaker_count"] == 2


def test_self_shuffle_prefers_same_speaker_when_available() -> None:
    clips = [
        {"id": "c1", "speaker_id": "speaker_001"},
        {"id": "c2", "speaker_id": "speaker_002"},
        {"id": "c3", "speaker_id": "speaker_001"},
    ]
    windows = [
        {"id": "w1", "speaker_id": "speaker_002"},
        {"id": "w2", "speaker_id": "speaker_001"},
    ]

    selected = speaker_aware_shuffle_selection(clips, windows, seed=1)

    assert selected[0]["speaker_id"] == "speaker_002"
    assert selected[1]["speaker_id"] == "speaker_001"


def test_speaker_mapping_pairs_ranked_source_and_destination_speakers(tmp_path: Path) -> None:
    source = {
        "media_hash": "source",
        "speakers": [
            {"speaker_id": "speaker_002", "total_duration": 4.0, "event_count": 2, "first_seen": 0.0, "last_seen": 4.0, "confidence": 0.5},
            {"speaker_id": "speaker_001", "total_duration": 9.0, "event_count": 4, "first_seen": 0.0, "last_seen": 9.0, "confidence": 0.6},
        ],
    }
    destination = {
        "media_hash": "dest",
        "speakers": [
            {"speaker_id": "speaker_001", "total_duration": 3.0, "event_count": 1, "first_seen": 0.0, "last_seen": 3.0, "confidence": 0.5},
            {"speaker_id": "speaker_002", "total_duration": 8.0, "event_count": 3, "first_seen": 0.0, "last_seen": 8.0, "confidence": 0.7},
        ],
    }

    mapping = build_speaker_mapping(source_speaker_map=source, destination_speaker_map=destination, output_path=tmp_path / "speaker_mapping.json")

    assert mapping["mappings"][0]["source_speaker_id"] == "speaker_001"
    assert mapping["mappings"][0]["destination_speaker_id"] == "speaker_002"
    validate_artifact("speaker_mapping", tmp_path / "speaker_mapping.json", Path.cwd() / "schemas")


def test_apply_speaker_mapping_to_schedule_marks_followed_and_fallback() -> None:
    schedule = {
        "mappings": [
            {"source_speaker_id": "speaker_001", "destination_speaker_id": "speaker_002", "enabled": True},
            {"source_speaker_id": "speaker_002", "destination_speaker_id": "speaker_002", "enabled": True},
        ]
    }
    speaker_mapping = {
        "mapping_strategy": "rank_by_speaker_presence_v1",
        "source_media_hash": "source",
        "destination_media_hash": "dest",
        "mappings": [
            {"source_speaker_id": "speaker_001", "destination_speaker_id": "speaker_002"},
            {"source_speaker_id": "speaker_002", "destination_speaker_id": "speaker_001"},
        ],
    }

    annotated = apply_speaker_mapping_to_schedule(schedule, speaker_mapping)

    assert annotated["mappings"][0]["speaker_mapping_followed"] is True
    assert annotated["mappings"][1]["speaker_mapping_followed"] is False
    assert annotated["mappings"][1]["speaker_mapping_fallback_reason"] == "performance_fit_overrode_speaker_mapping"
    assert speaker_mapping_summary(annotated)["speaker_mapping_followed_rate"] == 0.5


def test_speaker_preservation_summary_counts_fallbacks() -> None:
    schedule = {
        "mappings": [
            {"enabled": True, "source_speaker_id": "speaker_001", "destination_speaker_id": "speaker_001", "speaker_match_preserved": True},
            {"enabled": True, "source_speaker_id": "speaker_001", "destination_speaker_id": "speaker_002", "speaker_match_preserved": False, "speaker_fallback_reason": "no_same_speaker_fit"},
        ]
    }

    summary = speaker_preservation_summary(schedule)

    assert summary["speaker_aware_mapping_count"] == 2
    assert summary["same_speaker_count"] == 1
    assert summary["same_speaker_rate"] == 0.5
    assert summary["fallback_reasons"] == {"no_same_speaker_fit": 1}

def test_pyannote_zero_segments_falls_back_to_heuristic(monkeypatch, tmp_path: Path) -> None:
    class EmptyDiarization:
        def itertracks(self, yield_label: bool = False):
            return iter(())

    import cinelingus.speakers as speakers

    monkeypatch.setattr(speakers, "_pyannote_unavailable_reason", lambda audio_path, hf_token: None)
    monkeypatch.setattr(speakers, "_resolve_pyannote_device", lambda device: None)
    monkeypatch.setattr(speakers, "_load_pyannote_audio_input", lambda audio_path: {"waveform": None, "sample_rate": 16000})

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_name, token=None):
            return cls()

        def __call__(self, audio):
            return EmptyDiarization()

    import sys
    import types

    module = types.ModuleType("pyannote.audio")
    module.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)

    output = tmp_path / "speaker_map.json"
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"placeholder")

    speaker_map = build_speaker_map(
        media_hash="hash",
        speech_items=_items(),
        output_path=output,
        backend="pyannote",
        audio_path=audio,
        hf_token="hf_test",
    )

    assert speaker_map["requested_backend"] == "pyannote"
    assert speaker_map["diarization_tool"] == "heuristic_timing_v1"
    assert speaker_map["speaker_count"] == 2
    assert speaker_map["diagnostics"]["fallback_used"] is True
    assert speaker_map["diagnostics"]["labeled_item_count"] == 3
    assert "produced no usable speaker segments" in speaker_map["warnings"][0]


def test_speaker_map_diagnostics_reports_unavailable_coverage() -> None:
    diagnostics = speaker_map_diagnostics(
        {
            "requested_backend": "pyannote",
            "diarization_tool": "pyannote.audio",
            "speaker_count": 0,
            "speaker_segments": [],
            "warnings": [],
        },
        _items(),
    )

    assert diagnostics["status"] == "unavailable"
    assert diagnostics["labeled_item_count"] == 0
    assert diagnostics["labeled_item_rate"] == 0.0


def test_annotated_artifact_identity_changes_with_speaker_map_content() -> None:
    artifact = {"events": [{"id": "e1", "start": 0.0, "end": 1.0}]}
    first = {
        "media_hash": "hash",
        "schema_version": "2.0",
        "config_signature": "same-config",
        "diarization_tool": "pyannote",
        "speaker_segments": [{"source_id": "e1", "speaker_id": "speaker_001", "confidence": 0.8}],
    }
    second = {
        **first,
        "speaker_segments": [{"source_id": "e1", "speaker_id": "speaker_002", "confidence": 0.8}],
    }

    first_annotated = annotate_artifact_speakers(artifact, first, collection_key="events")
    second_annotated = annotate_artifact_speakers(artifact, second, collection_key="events")

    assert first_annotated["speaker_map_content_signature"] != second_annotated["speaker_map_content_signature"]


def test_partial_pyannote_map_counts_as_real_diarization() -> None:
    speaker_map = {
        "actual_backend": "pyannote_partial",
        "speaker_segments": [
            {"speaker_id": "speaker_001", "source_id": "e1", "confidence": 0.8},
            {"speaker_id": "unknown_speaker_001", "source_id": "e2", "confidence": 0.45, "fallback_label": True},
        ],
    }

    assert speaker_map_has_real_diarization(speaker_map) is True


def test_heuristic_map_does_not_count_as_real_diarization() -> None:
    speaker_map = {
        "actual_backend": "heuristic_timing_v1",
        "speaker_segments": [{"speaker_id": "speaker_001", "source_id": "e1", "confidence": 0.45}],
    }

    assert speaker_map_has_real_diarization(speaker_map) is False


def test_speaker_mapping_excludes_unknown_fallback_speakers(tmp_path: Path) -> None:
    source = {
        "media_hash": "source",
        "speakers": [
            {"speaker_id": "unknown_speaker_001", "total_duration": 100.0, "event_count": 20, "confidence": 0.45},
            {"speaker_id": "speaker_001", "total_duration": 10.0, "event_count": 3, "confidence": 0.8},
        ],
    }
    destination = {
        "media_hash": "destination",
        "speakers": [
            {"speaker_id": "unknown_speaker_001", "total_duration": 90.0, "event_count": 18, "confidence": 0.45},
            {"speaker_id": "speaker_002", "total_duration": 9.0, "event_count": 3, "confidence": 0.8},
        ],
    }

    mapping = build_speaker_mapping(
        source_speaker_map=source,
        destination_speaker_map=destination,
        output_path=tmp_path / "speaker_mapping.json",
    )

    assert mapping["source_speaker_count"] == 1
    assert mapping["destination_speaker_count"] == 1
    assert mapping["mappings"][0]["source_speaker_id"] == "speaker_001"
    assert mapping["mappings"][0]["destination_speaker_id"] == "speaker_002"


def test_one_diarization_turn_can_directly_support_multiple_transcript_items() -> None:
    items = [
        {'id': 'e1', 'start': 0.0, 'end': 1.0},
        {'id': 'e2', 'start': 1.0, 'end': 2.0},
        {'id': 'e3', 'start': 2.0, 'end': 3.0},
    ]
    turns = [
        {'id': 'segment_1', 'start': 0.0, 'end': 3.0, 'speaker_id': 'speaker_001', 'confidence': 0.8},
    ]

    assignments = build_item_speaker_assignments(items, turns)

    assert [row['source_id'] for row in assignments] == ['e1', 'e2', 'e3']
    assert {row['speaker_id'] for row in assignments} == {'speaker_001'}
    assert {row['assignment_method'] for row in assignments} == {'DIRECT_OVERLAP'}
    diagnostics = speaker_map_diagnostics(
        {
            'requested_backend': 'pyannote',
            'actual_backend': 'pyannote',
            'diarization_tool': 'pyannote',
            'diarization_status': 'SUCCESS',
            'alignment_status': 'COMPLETE',
            'fallback_status': 'NONE',
            'speaker_count': 1,
            'speaker_segments': turns,
            'speaker_assignments': assignments,
            'warnings': [],
        },
        items,
    )
    assert diagnostics['direct_item_count'] == 3
    assert diagnostics['direct_item_rate'] == 1.0
    assert diagnostics['fallback_used'] is False


def test_nonoverlapping_item_uses_explicit_continuity_provenance() -> None:
    items = [
        {'id': 'e1', 'start': 0.0, 'end': 1.0},
        {'id': 'e2', 'start': 1.05, 'end': 1.15},
        {'id': 'e3', 'start': 1.2, 'end': 2.0},
    ]
    turns = [
        {'id': 's1', 'start': 0.0, 'end': 1.0, 'speaker_id': 'speaker_001', 'confidence': 0.8},
        {'id': 's2', 'start': 1.2, 'end': 2.0, 'speaker_id': 'speaker_001', 'confidence': 0.8},
    ]

    assignments = build_item_speaker_assignments(items, turns)
    inferred = next(row for row in assignments if row['source_id'] == 'e2')

    assert inferred['speaker_id'] == 'speaker_001'
    assert inferred['assignment_method'] == 'CONTINUITY_INFERENCE'
    assert inferred['fallback_label'] is True
    assert len(inferred['supporting_segment_ids']) == 2


def test_distant_unmapped_item_is_heuristic_and_identity_gate_uses_direct_rate() -> None:
    items = [
        {'id': 'e1', 'start': 0.0, 'end': 1.0},
        {'id': 'e2', 'start': 5.0, 'end': 6.0},
    ]
    turns = [{'id': 's1', 'start': 0.0, 'end': 1.0, 'speaker_id': 'speaker_001', 'confidence': 0.8}]
    assignments = build_item_speaker_assignments(items, turns)
    heuristic = next(row for row in assignments if row['source_id'] == 'e2')
    speaker_map = {
        'actual_backend': 'pyannote',
        'diarization_status': 'SUCCESS',
        'speaker_segments': turns,
        'diagnostics': {'diarization_status': 'SUCCESS', 'direct_item_rate': 0.5},
    }

    assert heuristic['assignment_method'] == 'HEURISTIC'
    assert heuristic['speaker_id'].startswith('unknown_')
    assert speaker_map_identity_ready(speaker_map) is False
    assert speaker_map_identity_ready(speaker_map, minimum_direct_rate=0.5) is True


def test_overlapping_speakers_record_ambiguity_without_fabricating_certainty() -> None:
    assignments = build_item_speaker_assignments(
        [{'id': 'e1', 'start': 0.0, 'end': 1.0}],
        [
            {'id': 's1', 'start': 0.0, 'end': 0.9, 'speaker_id': 'speaker_001', 'confidence': 0.8},
            {'id': 's2', 'start': 0.1, 'end': 0.9, 'speaker_id': 'speaker_002', 'confidence': 0.8},
        ],
    )

    assert assignments[0]['assignment_method'] == 'AMBIGUOUS_OVERLAP'
    assert assignments[0]['evidence_status'] == 'DIRECT_AMBIGUOUS'
    assert assignments[0]['competing_speakers'][0]['speaker_id'] == 'speaker_002'


def test_successful_pyannote_run_stays_successful_when_one_turn_spans_many_items(monkeypatch, tmp_path: Path) -> None:
    import cinelingus.speakers as speakers

    items = [
        {'id': 'e1', 'start': 0.0, 'end': 1.0},
        {'id': 'e2', 'start': 1.0, 'end': 2.0},
        {'id': 'e3', 'start': 2.0, 'end': 3.0},
    ]
    usable_turns = [
        {
            'id': 'segment_000001',
            'start': 0.0,
            'end': 3.0,
            'duration': 3.0,
            'speaker_id': 'speaker_001',
            'speaker': 'speaker_001',
            'confidence': 0.8,
            'speaker_confidence': 0.8,
            'source_id': 'e1',
            'valid': True,
        }
    ]
    monkeypatch.setattr(speakers, '_pyannote_unavailable_reason', lambda **kwargs: None)
    monkeypatch.setattr(
        speakers,
        'run_pyannote_diagnostic',
        lambda **kwargs: {
            'usable_turns': usable_turns,
            'actual_backend': 'pyannote',
            'status': 'success',
            'fallback_reason': None,
        },
    )
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'placeholder')
    output = tmp_path / 'speaker_map.json'

    speaker_map = speakers._build_pyannote_speaker_map(
        media_hash='hash',
        speech_items=items,
        output_path=output,
        audio_path=audio,
        model_name='test-model',
        config_signature='signature',
        device='cpu',
        hf_token='token',
    )

    assert speaker_map is not None
    assert speaker_map['actual_backend'] == 'pyannote'
    assert speaker_map['diarization_status'] == 'SUCCESS'
    assert speaker_map['alignment_status'] == 'COMPLETE'
    assert speaker_map['fallback_status'] == 'NONE'
    assert speaker_map['diagnostics']['direct_item_count'] == 3
    validate_artifact('speaker_map', output, Path.cwd() / 'schemas')

