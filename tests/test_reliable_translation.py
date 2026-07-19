from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from cinelingus.montage import build_full_timeline_plan, build_montage_render_acceptance
from cinelingus.reliable_inputs import (
    MediaPreflightError,
    chooser_initial_directory,
    default_input_directory,
    preflight_media_inputs,
)
from cinelingus.render import mux_video
from cinelingus.tools import ffprobe_json, run


def test_default_input_directory_prefers_downloads_music(tmp_path: Path) -> None:
    preferred = tmp_path / "Downloads" / "Music"
    preferred.mkdir(parents=True)

    assert default_input_directory(tmp_path) == preferred


def test_default_input_directory_falls_back_to_downloads(tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    downloads.mkdir()

    assert default_input_directory(tmp_path) == downloads


def test_chooser_uses_selected_file_then_session_directory(tmp_path: Path) -> None:
    default = tmp_path / "default"
    session = tmp_path / "session"
    selected_dir = tmp_path / "selected"
    default.mkdir()
    session.mkdir()
    selected_dir.mkdir()
    selected = selected_dir / "film.mp4"
    selected.write_bytes(b"film")

    assert chooser_initial_directory(selected, last_directory=session, default_directory=default) == selected_dir
    assert chooser_initial_directory(None, last_directory=session, default_directory=default) == session


def test_preflight_predicts_shortest_audio_supported_duration(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.mp4"
    support = tmp_path / "support.mp4"
    anchor.write_bytes(b"anchor")
    support.write_bytes(b"support")

    def fake_probe(path: Path) -> dict:
        duration = "12.0" if path == anchor else "8.0"
        return {
            "format": {"duration": duration},
            "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
        }

    report = preflight_media_inputs([anchor, support], output_dir=tmp_path / "output", probe=fake_probe)

    assert report["predicted_output_duration"] == 8.0
    assert report["anchor_curtailed"] is True
    assert report["input_scope"] == "complete_media_files"


def test_preflight_rejects_supporting_film_without_audio(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.mp4"
    support = tmp_path / "silent.mp4"
    anchor.write_bytes(b"anchor")
    support.write_bytes(b"support")

    def fake_probe(path: Path) -> dict:
        streams = [{"codec_type": "video"}]
        if path == anchor:
            streams.append({"codec_type": "audio"})
        return {"format": {"duration": "8.0"}, "streams": streams}

    with pytest.raises(MediaPreflightError, match="no usable audio stream"):
        preflight_media_inputs([anchor, support], output_dir=tmp_path / "output", probe=fake_probe)


def test_render_acceptance_rejects_an_extra_audio_stream(tmp_path: Path) -> None:
    plan = _full_timeline_plan(anchor_duration=12.0, support_duration=8.0)
    acceptance = build_montage_render_acceptance(
        plan=plan,
        encoded_probe={
            "format": {"duration": "8.0"},
            "streams": [
                {"codec_type": "video", "duration": "8.0"},
                {"codec_type": "audio", "duration": "8.0"},
                {"codec_type": "audio", "duration": "8.0"},
            ],
        },
        output_path=tmp_path / "acceptance.json",
    )

    assert acceptance["acceptance_status"] == "FAIL"
    assert acceptance["checks"]["exactly_one_replacement_audio_stream"] is False


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is unavailable")
def test_real_mux_curtails_anchor_and_keeps_one_replacement_audio_stream(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.mp4"
    support = tmp_path / "support.mp4"
    replacement = tmp_path / "replacement.wav"
    output = tmp_path / "translation.mp4"
    _make_test_film(anchor, duration=3.0, frequency=330)
    _make_test_film(support, duration=2.0, frequency=660)
    _write_sine_wav(replacement, duration=2.0, frequency=880)

    preflight = preflight_media_inputs([anchor, support], output_dir=tmp_path / "output")
    mux_video(destination_video=anchor, dialogue_wav=replacement, output_path=output)
    probe = ffprobe_json(output)
    plan = _full_timeline_plan(
        anchor_duration=3.0,
        support_duration=float(preflight["predicted_output_duration"]),
    )
    acceptance = build_montage_render_acceptance(
        plan=plan,
        encoded_probe=probe,
        output_path=tmp_path / "montage_render_acceptance.json",
        timing_tolerance_seconds=0.08,
    )

    assert preflight["predicted_output_duration"] == pytest.approx(2.0, abs=0.03)
    assert acceptance["acceptance_status"] == "PASS"
    contract = acceptance["provenance"]["encoded_stream_contract"]
    assert contract["video_stream_count"] == 1
    assert contract["audio_stream_count"] == 1
    assert contract["audio_duration"] == pytest.approx(2.0, abs=0.08)


def _full_timeline_plan(*, anchor_duration: float, support_duration: float) -> dict:
    return build_full_timeline_plan(
        filter_id="multiworld.translation",
        filter_contract_version="2.0",
        anchor_source_id="anchor",
        anchor_media_hash="anchor-hash",
        anchor_duration=anchor_duration,
        supporting_audio_durations=[support_duration],
        random_seed=1,
        governing_relationship="continuous supporting soundtrack",
        laws={
            "visual": "COMPLETE_ANCHOR_TIMELINE_FROM_ZERO",
            "temporal": "ANCHOR_CHRONOLOGY_PRESERVED",
            "dialogue": "SUPPORTING_AUDIO_REPLACES_ANCHOR_AUDIO",
            "requested_audio": "TRANSLATION_LAW",
            "actual_audio_method": "CONTINUOUS_SOURCE_SOUNDTRACK_BED",
        },
    )


def _make_test_film(path: Path, *, duration: float, frequency: int) -> None:
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=160x90:r=24:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    ])


def _write_sine_wav(path: Path, *, duration: float, frequency: int) -> None:
    sample_rate = 48_000
    frame_count = int(duration * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = int(12_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wav.writeframes(frames)
