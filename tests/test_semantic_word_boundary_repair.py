import wave
from pathlib import Path

from cinelingus.semantic.word_boundary_repair import repair_semantic_word_boundaries
from cinelingus.util import read_json
from cinelingus.validation import validate_artifact


def _wav(path: Path, seconds: float = 20.0, rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes(b"\0\0" * int(seconds * rate) * channels)


def test_word_boundary_repair_aligns_and_independently_verifies_derived_clip(tmp_path: Path) -> None:
    audio = tmp_path / "analysis.wav"; _wav(audio)
    original = tmp_path / "original.wav"; _wav(original, 3.0, 48000, 2)
    library = {
        "schema_version": "1.0", "tool_version": "test", "media_hash": "film", "creation_timestamp": "then",
        "clips": [{
            "id": "c1", "path": str(original), "movie_timestamp": 10.0, "duration": 3.0,
            "transcript": "Hello there", "speech_rate": 0.667, "average_loudness": None,
        }],
    }
    extractions = []

    def extractor(source, start, duration, target, rate, channels):
        extractions.append((start, duration, rate, channels)); _wav(target, duration, rate, channels)

    def words(**kwargs):
        return {"media_hash": kwargs["media_hash"], "text": "noise hello there tail", "words": [
            {"start": 0.5, "end": 0.8, "text": "noise"},
            {"start": 4.4, "end": 4.8, "text": "Hello"},
            {"start": 4.8, "end": 5.2, "text": "there"},
            {"start": 6.0, "end": 6.3, "text": "tail"},
        ]}

    def verifier(**kwargs):
        return {"media_hash": kwargs["media_hash"], "windows": [
            {"start": 0.0, "end": 1.0, "transcript": "Hello there", "confidence": 0.9},
        ]}

    output = tmp_path / "repair"
    report = repair_semantic_word_boundaries(
        clip_library=library,
        rejection_evidence={"preflight_signature": "proof", "mapping_decisions": [{"clip_id": "c1", "state": "REJECTED"}]},
        analysis_audio=audio, output_dir=output, word_transcriber=words,
        verifier=verifier, extractor=extractor,
    )
    assert report["repair_state"] == "ALL_REPAIRED"
    assert report["repaired_clip_ids"] == ["c1"]
    assert extractions[0] == (6.0, 11.0, 16000, 1)
    assert extractions[1][0] == 10.32
    overlay = read_json(output / "repaired_clip_library.json")
    assert overlay["clips"][0]["word_boundary_repair"]["aligned_words"] == ["Hello", "there"]
    validate_artifact("semantic_word_boundary_repair", output / "semantic_word_boundary_repair.json", Path("schemas"))


def test_word_boundary_repair_refuses_unaligned_metadata(tmp_path: Path) -> None:
    audio = tmp_path / "analysis.wav"; _wav(audio)
    original = tmp_path / "original.wav"; _wav(original, 2.0, 48000, 2)
    report = repair_semantic_word_boundaries(
        clip_library={"clips": [{"id": "c1", "path": str(original), "movie_timestamp": 5.0, "duration": 2.0, "transcript": "missing words", "speech_rate": 1.0, "average_loudness": None}]},
        rejection_evidence={"mapping_decisions": [{"clip_id": "c1", "state": "REJECTED"}]},
        analysis_audio=audio, output_dir=tmp_path / "out",
        extractor=lambda source, start, duration, target, rate, channels: _wav(target, duration, rate, channels),
        word_transcriber=lambda **kwargs: {"media_hash": kwargs["media_hash"], "text": "other speech", "words": [{"start": 1.0, "end": 1.5, "text": "other"}]},
        verifier=lambda **kwargs: (_ for _ in ()).throw(AssertionError("verification should not run")),
    )
    assert report["repair_state"] == "NO_REPAIR"
    assert report["clips"][0]["reason"] == "intended_words_not_found_in_context"
