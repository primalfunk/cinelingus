from pathlib import Path

import pytest

from cinelingus.contract_kernel import (
    MediaDescriptor,
    OutputExtentResolver,
    compile_run_contract,
    write_run_contract,
)
from cinelingus.filter_lab.registry import default_filter_registry


SCHEMAS = Path.cwd() / "schemas"


def _media(path: str, *, video: float, audio: float, media_hash: str) -> MediaDescriptor:
    return MediaDescriptor.from_probe(
        path=Path(path),
        media_hash=media_hash,
        probe={
            "format": {"duration": str(max(video, audio))},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264", "duration": str(video), "start_time": "0"},
                {"index": 1, "codec_type": "audio", "codec_name": "aac", "duration": str(audio), "start_time": "0", "sample_rate": "48000", "channels": 2},
            ],
        },
    )


def test_extent_resolver_uses_video_stream_instead_of_container_duration() -> None:
    pokemon = _media("pokemon.mp4", video=1341.826522, audio=1341.887, media_hash="pokemon")

    extent = OutputExtentResolver().resolve([pokemon])

    assert extent.duration == 1341.827
    assert extent.anchor_video_duration == 1341.827
    assert extent.required_audio_durations == (1341.887,)
    assert extent.curtailed is False


def test_extent_resolver_curtails_anchor_to_supporting_audio() -> None:
    anchor = _media("anchor.mp4", video=5400.0, audio=5399.0, media_hash="anchor")
    support = _media("support.mp4", video=7200.0, audio=3600.0, media_hash="support")

    extent = OutputExtentResolver().resolve([anchor, support])

    assert extent.duration == 3600.0
    assert extent.curtailed is True


def test_run_contract_is_stable_for_same_media_and_filter(tmp_path: Path) -> None:
    definition = default_filter_registry().get("time.foreshadow")
    media = _media("pokemon.mp4", video=1341.826522, audio=1341.887, media_hash="pokemon")

    first = compile_run_contract(definition=definition, media=[media])
    second = compile_run_contract(definition=definition, media=[media])
    path = tmp_path / "run_contract.json"
    write_run_contract(first, path, SCHEMAS)

    assert first.contract_id == second.contract_id
    assert first.timeline.duration == 1341.827
    assert first.repetition_policy["policy"] == "forbidden"
    assert path.exists()


def test_contract_requires_speaker_identity_only_when_filter_declares_it() -> None:
    media = _media("film.mp4", video=60.0, audio=60.0, media_hash="film")
    ordinary = compile_run_contract(definition=default_filter_registry().get("time.foreshadow"), media=[media])
    identity = compile_run_contract(definition=default_filter_registry().get("identity.possession"), media=[media])

    assert ordinary.analysis_requirements["speaker_identity"] == "optional"
    assert "weak" in ordinary.analysis_requirements["accepted_speaker_quality"]
    assert identity.analysis_requirements["speaker_identity"] == "required"
    assert identity.analysis_requirements["accepted_speaker_quality"] == ["direct"]


def test_contract_rejects_wrong_film_count() -> None:
    definition = default_filter_registry().get("multiworld.translation")
    media = _media("film.mp4", video=60.0, audio=60.0, media_hash="film")

    with pytest.raises(ValueError, match="at least 2 films"):
        compile_run_contract(definition=definition, media=[media])
