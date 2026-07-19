from pathlib import Path

from cinelingus.publish import publish_single_video


def test_publish_keeps_video_and_lightweight_audit_sidecars(tmp_path: Path) -> None:
    output = tmp_path / "output"
    work = output / "best_short" / "runs" / "one"
    work.mkdir(parents=True)
    video = work / "final.mp4"
    video.write_bytes(b"video")
    (output / "run_report.json").write_text("{}")
    existing = output / "cinelingus_cinelingus_2026-01-01_00-00-00.mp4"
    audio = work / "intermediate.wav"
    report = work / "mutation_report.json"
    log = work / "run.log"
    sibling = output / "multiworld" / "other-run"
    sibling.mkdir(parents=True)
    sibling_video = sibling / "still-running.mp4"
    sibling_audio = sibling / "still-running.wav"
    existing.write_bytes(b"old")
    audio.write_bytes(b"audio")
    report.write_text("{}")
    log.write_text("diagnostics")
    sibling_video.write_bytes(b"other video")
    sibling_audio.write_bytes(b"other audio")

    published = publish_single_video(video=video, output_dir=output, process="Best Short")

    assert published.parent == output
    assert published.name.startswith("cinelingus_best-short_")
    assert existing.exists()
    assert report.exists()
    assert log.exists()
    assert (output / "run_report.json").exists()
    assert sibling_video.exists()
    assert sibling_audio.exists()
    assert not video.exists()
    assert not audio.exists()
