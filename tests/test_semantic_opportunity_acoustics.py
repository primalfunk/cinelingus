import wave
from pathlib import Path

from cinelingus.semantic.opportunity_acoustics import audit_semantic_opportunity_audio
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 32000)


def test_opportunity_audio_audit_caches_clips_and_includes_two_cycle_partner(tmp_path: Path) -> None:
    first, partner = tmp_path / "first.wav", tmp_path / "partner.wav"
    _wav(first); _wav(partner)
    screen = tmp_path / "semantic_schedule_screen.json"
    write_json(screen, {
        "experiment_signature": "screen-one",
        "acoustic_repair": {"quarantined_source_performance_ids": ["prior"]},
        "semantic_opportunity_audit": {"opportunities": [{
            "globally_admissible": True, "global_admission_mode": "TWO_CYCLE",
            "source_performance_id": "primary", "clip_ids": ["c1"], "semantic_delta": 0.2,
            "two_cycle_swap": {
                "state": "ADMISSIBLE_TWO_CYCLE", "replacement_source_performance_id": "partner",
                "net_semantic_delta": 0.1,
            },
        }]},
    })
    clips = [
        {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": str(first), "duration": 2.0, "transcript": "hello"},
        {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": str(partner), "duration": 2.0, "transcript": "goodbye"},
    ]
    performances = {"performances": [
        {"id": "primary", "dialogue_event_ids": ["e1"]},
        {"id": "partner", "dialogue_event_ids": ["e2"]},
    ]}
    calls = []

    def transcriber(**kwargs):
        calls.append(kwargs)
        return {"media_hash": kwargs["media_hash"], "windows": [
            {"id": "w1", "start": 1.0, "end": 3.0, "transcript": "hello", "confidence": 0.9},
            {"id": "w2", "start": 4.0, "end": 6.0, "transcript": "noise", "confidence": 0.9},
        ]}

    output = tmp_path / "audit"
    report = audit_semantic_opportunity_audio(
        screen_path=screen, clips=clips, source_performances=performances,
        output_dir=output, transcriber=transcriber,
    )
    assert len(calls) == 2
    assert report["audited_source_performance_count"] == 2
    assert report["rejected_source_performance_ids"] == ["partner"]
    assert report["repair_lineage"]["quarantined_source_performance_ids"] == ["prior"]
    validate_artifact(
        "semantic_opportunity_acoustic_audit",
        output / "semantic_opportunity_acoustic_audit.json", Path("schemas"),
    )

    reused = audit_semantic_opportunity_audio(
        screen_path=screen, clips=clips, source_performances=performances,
        output_dir=output,
        transcriber=lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert reused["transcription_cache_state"] == "REUSED"
    assert len(calls) == 2
    assert reused["rejected_source_performance_ids"] == ["partner"]


def test_opportunity_audio_audit_rejects_intended_words_with_adjacent_dialogue(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.wav"; _wav(clip_path)
    screen = tmp_path / "screen.json"
    write_json(screen, {"experiment_signature": "screen", "semantic_opportunity_audit": {"opportunities": [{
        "globally_admissible": True, "source_performance_id": "p1", "clip_ids": ["c1"],
        "semantic_delta": 0.2,
    }]}})

    def transcriber(**kwargs):
        return {"media_hash": kwargs["media_hash"], "windows": [{
            "start": 1.0, "end": 3.0, "transcript": "noise hello", "confidence": 0.9,
        }]}

    report = audit_semantic_opportunity_audio(
        screen_path=screen,
        clips=[{"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": str(clip_path), "duration": 2.0, "transcript": "hello"}],
        source_performances={"performances": [{"id": "p1", "dialogue_event_ids": ["e1"]}]},
        output_dir=tmp_path / "audit", transcriber=transcriber,
    )
    assert report["rejected_source_performance_ids"] == ["p1"]
    assert report["clips"][0]["health_state"] == "REJECTED"
