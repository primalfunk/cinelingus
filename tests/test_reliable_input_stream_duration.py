from pathlib import Path

from cinelingus.reliable_inputs import preflight_media_inputs


def test_preflight_uses_supporting_audio_stream_duration_not_container_duration(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.mp4"
    support = tmp_path / "support.mp4"
    anchor.write_bytes(b"anchor")
    support.write_bytes(b"support")

    def fake_probe(path: Path) -> dict:
        audio_duration = "12.0" if path == anchor else "8.0"
        return {
            "format": {"duration": "12.0"},
            "streams": [
                {"codec_type": "video", "duration": "12.0"},
                {"codec_type": "audio", "duration": audio_duration},
            ],
        }

    report = preflight_media_inputs([anchor, support], output_dir=tmp_path / "output", probe=fake_probe)

    assert report["films"][1]["duration"] == 12.0
    assert report["films"][1]["audio_duration"] == 8.0
    assert report["predicted_output_duration"] == 8.0
    assert report["anchor_curtailed"] is True
