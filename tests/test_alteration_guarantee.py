from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from cinelingus import alteration
from cinelingus.alteration import (
    evaluate_requested_alteration,
    evaluate_universal_alteration,
    measure_sampled_audio_difference,
    render_universal_alteration,
)
from cinelingus.transformations.base import TransformationResult
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def _filter_acceptance(path: Path, *, mapped: float = 8.0, render: float = 100.0) -> Path:
    write_json(path, {
        "status": "pass",
        "measurements": {
            "dialogue_coverage": mapped / render,
            "mapped_dialogue_duration": mapped,
            "render_duration": render,
            "source_placement_count": 6,
            "occupied_timeline_buckets": 3,
            "timeline_bucket_count": 4,
        },
    })
    return path


def test_requested_filter_requires_material_authored_extent(tmp_path: Path) -> None:
    video = tmp_path / "result.mp4"
    video.write_bytes(b"video")
    acceptance_path = _filter_acceptance(tmp_path / "filter_acceptance.json")
    result = TransformationResult(
        transformation_id="memory.dream",
        outputs={"video": video},
        artifacts={"filter_acceptance": acceptance_path},
    )

    report = evaluate_requested_alteration(
        result=result,
        anchor=tmp_path / "anchor.mp4",
        output=video,
        expected_duration=100.0,
        output_dir=tmp_path / "output",
    )

    assert report["status"] == "PASS"
    assert report["measurements"]["effective_alteration_ratio"] == 0.08
    validate_artifact("alteration_acceptance", Path(report["artifact_path"]), Path.cwd() / "schemas")


def test_requested_filter_with_negligible_extent_fails_alteration_gate(tmp_path: Path) -> None:
    video = tmp_path / "result.mp4"
    video.write_bytes(b"video")
    result = TransformationResult(
        transformation_id="memory.dream",
        outputs={"video": video},
        artifacts={"filter_acceptance": _filter_acceptance(
            tmp_path / "filter_acceptance.json", mapped=1.0, render=100.0
        )},
    )

    report = evaluate_requested_alteration(
        result=result,
        anchor=tmp_path / "anchor.mp4",
        output=video,
        expected_duration=100.0,
        output_dir=tmp_path / "output",
    )

    assert report["status"] == "FAIL"
    assert report["checks"]["minimum_authored_timeline_ratio"] is False


def test_multi_input_universal_renderer_maps_supporting_audio(monkeypatch, tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.mp4"
    donor = tmp_path / "donor.mp4"
    anchor.write_bytes(b"anchor")
    donor.write_bytes(b"donor")
    commands = []

    def fake_run(args):
        commands.append(args)
        Path(args[-1]).write_bytes(b"altered")

    monkeypatch.setattr(alteration, "run", fake_run)
    output, method, manifest = render_universal_alteration(
        (anchor, donor), tmp_path / "output", 42.0
    )

    assert output.exists()
    assert method.endswith("VIDEO_COPY")
    command = commands[0]
    audio_filter = command[command.index("-filter_complex") + 1]
    assert "[1:a:0]" in audio_filter
    assert "[0:a:0]" not in audio_filter
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "[aout]"
    assert manifest["supporting_contributor_count"] == 1
    assert manifest["contributors"][1]["media_hash"] != manifest["anchor"]["media_hash"]


def test_single_input_universal_renderer_transforms_complete_audio(monkeypatch, tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.mp4"
    anchor.write_bytes(b"anchor")
    commands = []

    def fake_run(args):
        commands.append(args)
        Path(args[-1]).write_bytes(b"altered")

    monkeypatch.setattr(alteration, "run", fake_run)
    _output, _method, manifest = render_universal_alteration(
        (anchor,), tmp_path / "output", 42.0
    )

    audio_filter = commands[0][commands[0].index("-filter_complex") + 1]
    assert "[0:a:0]" in audio_filter
    assert "equalizer=" in audio_filter
    assert "aecho=" in audio_filter
    assert manifest["audio_coverage_ratio"] == 1.0


def test_sampled_audio_difference_detects_full_track_change(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.mp4"
    output = tmp_path / "output.mp4"
    anchor.write_bytes(b"anchor")
    output.write_bytes(b"output")

    def fake_runner(args):
        source = Path(args[args.index("-i") + 1])
        destination = Path(args[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            frequency = 220.0 if source == anchor else 660.0
            frames = b"".join(
                struct.pack("<h", int(9000 * math.sin(2 * math.pi * frequency * index / 8000)))
                for index in range(8000)
            )
            handle.writeframes(frames)

    report = measure_sampled_audio_difference(
        anchor=anchor,
        output=output,
        duration=30.0,
        working_dir=tmp_path / "samples",
        runner=fake_runner,
    )

    assert report["comparable_sample_count"] == 5
    assert report["changed_sample_ratio"] == 1.0


def test_universal_acceptance_requires_duration_provenance_and_sample_difference(
    monkeypatch, tmp_path: Path
) -> None:
    anchor = tmp_path / "anchor.mp4"
    output = tmp_path / "altered.mp4"
    anchor.write_bytes(b"anchor")
    output.write_bytes(b"altered")
    monkeypatch.setattr(
        alteration,
        "measure_sampled_audio_difference",
        lambda **_kwargs: {
            "sample_count": 5,
            "comparable_sample_count": 5,
            "changed_sample_count": 5,
            "changed_sample_ratio": 1.0,
            "samples": [],
        },
    )
    manifest = {
        "strategy": "MULTI_INPUT_COMPLETE_SUPPORTING_AUDIO",
        "anchor": {"path": str(anchor), "media_hash": "anchor-hash"},
        "contributors": [
            {"path": str(anchor), "media_hash": "anchor-hash", "role": "transformed_anchor"},
            {"path": "donor.mp4", "media_hash": "donor-hash", "role": "supporting_audio"},
        ],
        "audio_coverage_ratio": 1.0,
        "audio_filter": "[1:a:0]anull[aout]",
        "expected_duration": 42.0,
        "output": str(output),
    }
    duration_acceptance = {
        "status": "PASS",
        "duration": 42.0,
        "duration_contract": {"checks": {"container": True, "video": True, "audio": True}},
    }

    report = evaluate_universal_alteration(
        manifest=manifest,
        duration_acceptance=duration_acceptance,
        output_dir=tmp_path / "acceptance",
    )

    assert report["status"] == "PASS"
    assert all(report["checks"].values())
