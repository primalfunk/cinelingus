import wave
from pathlib import Path

from cinelingus.semantic.clip_boundary_repair import repair_semantic_clip_boundaries
from cinelingus.util import read_json
from cinelingus.validation import validate_artifact
from cinelingus.render_verification import lexically_equivalent_transcript


def _wav(path: Path, seconds: float = 20.0, rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * int(seconds * rate) * channels)


def test_boundary_repair_requires_repeatable_exact_candidate_and_writes_overlay(tmp_path: Path) -> None:
    audio = tmp_path / "analysis.wav"
    _wav(audio)
    original_one, original_two = tmp_path / "c1.wav", tmp_path / "c2.wav"
    _wav(original_one, 2.0, 48000, 2); _wav(original_two, 2.0, 48000, 2)
    clips = {
        "schema_version": "1.0", "tool_version": "test", "media_hash": "film",
        "creation_timestamp": "then", "clips": [
            {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": str(original_one),
             "movie_timestamp": 5.0, "duration": 2.0, "transcript": "hello", "speech_rate": 0.5,
             "average_loudness": None},
            {"id": "c2", "event_id": "e2", "event_ids": ["e2"], "path": str(original_two),
             "movie_timestamp": 12.0, "duration": 2.0, "transcript": "goodbye", "speech_rate": 0.5,
             "average_loudness": None},
        ],
    }
    events = {"events": [
        {"id": "prior", "start": 1.0, "end": 2.0},
        {"id": "e1", "start": 5.0, "end": 7.0},
        {"id": "e2", "start": 12.0, "end": 14.0},
        {"id": "next", "start": 18.0, "end": 19.0},
    ]}
    audit = {"audit_signature": "audit", "clips": [
        {"clip_id": "c1", "health_state": "REJECTED"},
        {"clip_id": "c2", "health_state": "REJECTED"},
    ]}

    def extractor(source: Path, start: float, duration: float, target: Path) -> None:
        _wav(target, duration, 48000, 2)

    calls = []

    def transcriber(**kwargs):
        calls.append(kwargs["output_path"].name)
        if "discovery" in kwargs["output_path"].name:
            # Six 2-second candidates with one-second reel gaps. Only c1 at -0.25
            # is exact; c2's apparent match contains an adjacent word.
            windows = [
                {"start": 4.0, "end": 6.0, "transcript": "hello", "confidence": 0.9},
                {"start": 13.0, "end": 15.0, "transcript": "noise goodbye", "confidence": 0.9},
            ]
        else:
            windows = [{"start": 1.0, "end": 3.0, "transcript": "hello", "confidence": 0.9}]
        return {"media_hash": kwargs["media_hash"], "windows": windows}

    output = tmp_path / "repair"
    report = repair_semantic_clip_boundaries(
        clip_library=clips, dialogue_events=events, acoustic_audit=audit,
        analysis_audio=audio, output_dir=output, offsets=(0.0, -0.25, 0.25),
        transcriber=transcriber, extractor=extractor,
    )
    assert report["repair_state"] == "PARTIAL_REPAIR"
    assert report["repaired_clip_count"] == 1
    assert report["unrecovered_clip_count"] == 1
    assert len(calls) == 2
    first = report["clips"][0]
    assert first["repair_state"] == "REPAIRED"
    assert first["selected_candidate"]["candidate_start"] == 4.75
    assert report["clips"][1]["repair_state"] == "UNRECOVERED"
    overlay = read_json(output / "repaired_clip_library.json")
    assert overlay["clips"][0]["movie_timestamp"] == 4.75
    assert overlay["clips"][0]["boundary_repair"]["evidence_state"] == "REPEATABLE_LEXICAL_TRANSCRIPT"
    assert overlay["clips"][1]["path"] == str(original_two)
    validate_artifact(
        "semantic_clip_boundary_repair", output / "semantic_clip_boundary_repair.json", Path("schemas"),
    )


def test_boundary_repair_does_not_search_across_neighbor_event(tmp_path: Path) -> None:
    audio = tmp_path / "analysis.wav"; _wav(audio, 10.0)
    original = tmp_path / "clip.wav"; _wav(original, 2.0, 48000, 2)
    starts = []

    def extractor(source: Path, start: float, duration: float, target: Path) -> None:
        starts.append(start); _wav(target, duration, 48000, 2)

    report = repair_semantic_clip_boundaries(
        clip_library={"schema_version": "1.0", "tool_version": "test", "media_hash": "film", "creation_timestamp": "then", "clips": [
            {"id": "c1", "event_id": "e1", "event_ids": ["e1"], "path": str(original),
             "movie_timestamp": 5.0, "duration": 2.0, "transcript": "hello", "speech_rate": 0.5, "average_loudness": None},
        ]},
        dialogue_events={"events": [
            {"id": "prior", "start": 3.0, "end": 4.8}, {"id": "e1", "start": 5.0, "end": 7.0},
            {"id": "next", "start": 7.2, "end": 9.0},
        ]},
        acoustic_audit={"clips": [{"clip_id": "c1", "health_state": "REJECTED"}]},
        analysis_audio=audio, output_dir=tmp_path / "out", offsets=(-1.0, 0.0, 1.0),
        extractor=extractor,
        transcriber=lambda **kwargs: {"media_hash": kwargs["media_hash"], "windows": []},
    )
    assert starts == [4.8, 5.0, 5.2]
    assert report["repair_state"] == "NO_REPAIR_CANDIDATE"


def test_bounded_breathy_transcript_equivalence_rejects_extra_words() -> None:
    assert lexically_equivalent_transcript("Sorry", "Sorryhh!")
    assert not lexically_equivalent_transcript("Sorry", "Whoa, sorry!")
    assert not lexically_equivalent_transcript("Gromit", "Grommet")
    assert lexically_equivalent_transcript(
        "Get off my building, Sima Punk!", "Get off my building, cyma punk!",
    )
    assert not lexically_equivalent_transcript(
        "And now to finish you off", "Now to finish you off",
    )
