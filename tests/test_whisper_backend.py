from pathlib import Path

from cinelingus.whisper_backend import _fallback_candidates, _transcribe_with_model_fallback


class FakeModel:
    def __init__(self, name):
        self.name = name

    def transcribe(self, _path, **_options):
        return {"language": "en", "segments": []}


class FakeWhisper:
    def __init__(self, failures):
        self.failures = set(failures)
        self.loaded = []

    def load_model(self, name, device):
        self.loaded.append((name, device))
        if name in self.failures:
            raise RuntimeError(f"cannot load {name}")
        return FakeModel(name)


def test_fallback_candidates_keep_tiny_as_fast_preview_floor():
    assert _fallback_candidates("medium") == ["medium", "small", "base", "tiny"]
    assert _fallback_candidates("small") == ["small", "base", "tiny"]
    assert _fallback_candidates("tiny") == ["tiny"]


def test_transcribe_with_model_fallback_reports_actual_model(tmp_path: Path):
    fake = FakeWhisper(failures={"medium"})

    model_name, result, warning = _transcribe_with_model_fallback(
        whisper=fake,
        transcription_audio=tmp_path / "audio.wav",
        requested_model="medium",
        device="cpu",
        language=None,
    )

    assert model_name == "small"
    assert result["segments"] == []
    assert "fell back to 'small'" in warning
    assert fake.loaded[:2] == [("medium", "cpu"), ("small", "cpu")]


def test_model_fallback_forwards_word_timestamp_options(tmp_path: Path):
    fake = FakeWhisper(failures=set())
    model_name, _, _ = _transcribe_with_model_fallback(
        whisper=fake, transcription_audio=tmp_path / "audio.wav",
        requested_model="small", device="cpu", language="en",
        transcribe_options={"word_timestamps": True, "condition_on_previous_text": False},
    )
    assert model_name == "small"
