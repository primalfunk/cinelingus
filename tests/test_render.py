from pathlib import Path
import wave

from cinelingus import render
from cinelingus.render import build_preview_schedule, preview_bounds, scheduled_audio_duration


def test_mux_video_curtails_picture_to_audio(monkeypatch, tmp_path: Path) -> None:
    commands = []
    monkeypatch.setattr(render, "run", lambda args: commands.append(args))

    render.mux_video(
        destination_video=tmp_path / "longer_video.mp4",
        dialogue_wav=tmp_path / "complete_supporting_audio.wav",
        output_path=tmp_path / "final.mp4",
        duration=42.25,
    )

    command = commands[0]
    assert "-shortest" in command
    assert command[command.index("-t") + 1] == "42.250"
    assert command.index("-t") < command.index("-shortest")
    assert command.index("-shortest") < command.index(str(tmp_path / "final.mp4"))


def test_render_montage_visual_uses_exact_plan_boundaries_and_source_audio(monkeypatch, tmp_path: Path) -> None:
    commands = []
    monkeypatch.setattr(render, "run", lambda args: commands.append(args))

    render.render_montage_visual(
        input_video=tmp_path / "film.mp4",
        selected_moments=[
            {"id": "m1", "visual_boundary": {"start": 2.0, "end": 5.5}},
            {"id": "m2", "visual_boundary": {"start": 11.0, "end": 15.0}},
        ],
        output_path=tmp_path / "montage.mp4",
    )

    command = commands[0]
    filters = command[command.index("-filter_complex") + 1]
    assert "trim=start=2.000:end=5.500" in filters
    assert "trim=start=11.000:end=15.000" in filters
    assert "atrim=start=2.000:end=5.500" in filters
    assert "atrim=start=11.000:end=15.000" in filters
    assert "concat=n=2:v=1:a=1" in filters
    assert "[outa]" in command


def test_render_multi_source_montage_uses_each_declared_film_and_soundtrack(monkeypatch, tmp_path: Path) -> None:
    sources = [tmp_path / f"film_{index}.mp4" for index in range(3)]
    for source in sources:
        source.write_bytes(b"media")
    commands = []
    monkeypatch.setattr(render, "ffprobe_json", lambda _path: {"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]})
    monkeypatch.setattr(render, "run", lambda args: commands.append(args))

    render.render_multi_source_montage(
        selected_moments=[
            {"id": f"phase_{index}", "source_path": str(source), "visual_boundary": {"start": index * 10.0, "end": (index + 1) * 10.0}}
            for index, source in enumerate(sources)
        ],
        output_path=tmp_path / "triangle_base.mp4",
    )

    command = commands[0]
    filters = command[command.index("-filter_complex") + 1]
    assert all(str(source) in command for source in sources)
    assert "concat=n=3:v=1:a=1" in filters
    assert filters.count("atrim=start=") == 3
    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in filters


def test_render_dialogue_wav_skips_disabled_mappings(monkeypatch, tmp_path: Path) -> None:
    enabled_clip = tmp_path / "enabled.wav"
    disabled_clip = tmp_path / "disabled.wav"
    enabled_clip.write_text("enabled")
    disabled_clip.write_text("disabled")
    commands = []

    def fake_run(args):
        commands.append(args)
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("wav")

    monkeypatch.setattr(render, "run", fake_run)

    render.render_dialogue_wav(
        schedule={
            "mappings": [
                {
                    "enabled": True,
                    "clip_path": str(enabled_clip),
                    "destination_timestamp": 0.0,
                    "clip_trim_start": 0.0,
                    "clip_trim_duration": 1.0,
                    "stretch_factor": 1.0,
                    "highpass_hz": 180,
                    "lowpass_hz": 4200,
                    "gain_db": -14.0,
                },
                {
                    "enabled": False,
                    "clip_path": str(disabled_clip),
                    "destination_timestamp": 1.0,
                    "clip_trim_start": 0.0,
                    "clip_trim_duration": 1.0,
                    "stretch_factor": 1.0,
                },
            ]
        },
        duration=2.0,
        output_path=tmp_path / "out.wav",
        sample_rate=48000,
        channels=2,
        target_lufs=-18.0,
        fade_duration=0.015,
    )

    command_text = " ".join(" ".join(command) for command in commands)
    assert str(enabled_clip) in command_text
    assert str(disabled_clip) not in command_text
    assert "highpass=f=180.0" in command_text
    assert "lowpass=f=4200.0" in command_text
    assert "volume=-14.00dB" in command_text


def test_render_schedule_over_original_audio_uses_original_track_and_mutes_windows(monkeypatch, tmp_path: Path) -> None:
    original = tmp_path / "movie.mp4"
    clip = tmp_path / "clip.wav"
    original.write_text("movie")
    clip.write_text("clip")
    commands = []

    def fake_run(args):
        commands.append(args)
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("wav")

    monkeypatch.setattr(render, "run", fake_run)

    render.render_schedule_over_original_audio(
        original_media=original,
        schedule={
            "mappings": [
                {
                    "enabled": True,
                    "clip_path": str(clip),
                    "destination_timestamp": 3.0,
                    "planned_render_duration": 2.0,
                    "clip_trim_start": 0.0,
                    "clip_trim_duration": 1.0,
                    "stretch_factor": 1.0,
                }
            ]
        },
        duration=10.0,
        output_path=tmp_path / "self_shuffle.wav",
        sample_rate=48000,
        channels=2,
        target_lufs=-18.0,
        fade_duration=0.015,
    )

    command_text = " ".join(" ".join(command) for command in commands)
    assert str(original) in command_text
    assert "volume=enable='between(t,3.000,4.000)':volume=-28.0dB" in command_text
    assert str(clip) in command_text


def test_render_schedule_ducks_only_literal_clip_activity(monkeypatch, tmp_path: Path) -> None:
    original = tmp_path / "movie.mp4"
    clip = tmp_path / "clip.wav"
    original.write_text("movie")
    with wave.open(str(clip), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        silent = (0).to_bytes(2, "little", signed=True)
        active = (3000).to_bytes(2, "little", signed=True)
        handle.writeframes(silent * 9600 + active * 19200 + silent * 19200)
    commands = []

    def fake_run(args):
        commands.append(args)
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("wav")

    monkeypatch.setattr(render, "run", fake_run)
    schedule = {"mappings": [{
        "enabled": True, "clip_path": str(clip), "destination_timestamp": 3.0,
        "planned_render_duration": 1.0, "clip_trim_start": 0.0,
        "clip_trim_duration": 1.0, "stretch_factor": 1.0,
    }]}

    render.render_schedule_over_original_audio(
        original_media=original, schedule=schedule, duration=10.0,
        output_path=tmp_path / "mix.wav", sample_rate=48000, channels=2,
        target_lufs=-18.0, fade_duration=0.015,
    )

    command_text = " ".join(" ".join(command) for command in commands)
    assert "volume=enable='between(t,3.200,3.600)':volume=-28.0dB" in command_text
    assert schedule["audio_ducking"]["strategy"] == "clip_activity_exact_v1"
    assert schedule["audio_ducking"]["rendered_region_count"] == 1


def test_render_schedule_over_original_audio_accepts_explicit_mute_regions(monkeypatch, tmp_path: Path) -> None:
    original = tmp_path / "movie.mp4"
    clip = tmp_path / "clip.wav"
    original.write_text("movie")
    clip.write_text("clip")
    commands = []

    def fake_run(args):
        commands.append(args)
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("wav")

    monkeypatch.setattr(render, "run", fake_run)

    render.render_schedule_over_original_audio(
        original_media=original,
        schedule={
            "mappings": [
                {
                    "enabled": True,
                    "clip_path": str(clip),
                    "destination_timestamp": 3.0,
                    "planned_render_duration": 2.0,
                    "clip_trim_start": 0.0,
                    "clip_trim_duration": 1.0,
                    "stretch_factor": 1.0,
                }
            ]
        },
        duration=10.0,
        output_path=tmp_path / "cinelingus.wav",
        sample_rate=48000,
        channels=2,
        target_lufs=-18.0,
        fade_duration=0.015,
        mute_regions=[{"start": 2.0, "duration": 5.0}],
        duck_db=-24.0,
    )

    command_text = " ".join(" ".join(command) for command in commands)
    assert "volume=enable='between(t,2.000,7.000)':volume=-24.0dB" in command_text


def test_render_schedule_can_hard_suppress_carrier_speech_without_audible_bed(monkeypatch, tmp_path: Path) -> None:
    original = tmp_path / "movie.mp4"
    clip = tmp_path / "clip.wav"
    original.write_text("movie")
    clip.write_text("clip")
    commands = []

    def fake_run(args):
        commands.append(args)
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("wav")

    monkeypatch.setattr(render, "run", fake_run)
    schedule = {"mappings": [{
        "enabled": True, "clip_path": str(clip), "destination_timestamp": 2.0,
        "clip_trim_start": 0.0, "clip_trim_duration": 1.0, "stretch_factor": 1.0,
    }]}
    regions = [{"id": "speech_1", "start": 1.0, "duration": 2.0}]

    render.render_schedule_over_original_audio(
        original_media=original, schedule=schedule, duration=5.0,
        output_path=tmp_path / "mix.wav", sample_rate=48000, channels=2,
        target_lufs=-18.0, fade_duration=0.015, mute_regions=regions,
        suppression_mode="hard_mute", suppression_fade_duration=0.05,
    )

    command_text = " ".join(" ".join(command) for command in commands)
    assert "volume='" in command_text
    assert "lte(t,3.000),0" in command_text
    assert "-28.0dB" not in command_text
    assert schedule["audio_ducking"] == {
        "strategy": "hard_carrier_speech_suppression_v1",
        "requested_region_count": 1,
        "rendered_region_count": 1,
        "duck_db": None,
        "suppression_mode": "hard_mute",
        "suppression_floor": "DIGITAL_SILENCE",
        "edge_fade_seconds": 0.05,
        "residual_speech_test": "NOT_ACOUSTICALLY_MEASURED",
    }
    assert schedule["audio_suppression"] == schedule["audio_ducking"]
    assert "volume=enable='between(t,3.000,5.000)':volume=-28.0dB" not in command_text
    assert str(clip) in command_text


def test_hard_suppression_reconstructs_only_from_detected_non_speech_neighbors(monkeypatch, tmp_path: Path) -> None:
    original = tmp_path / "movie.mp4"
    clip = tmp_path / "clip.wav"
    original.write_text("movie")
    clip.write_text("clip")
    commands = []

    def fake_run(args):
        commands.append(args)
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("wav")

    monkeypatch.setattr(render, "run", fake_run)
    schedule = {
        "destination_speech_regions": [{"id": "speech_1", "start": 1.0, "end": 3.0, "duration": 2.0}],
        "mappings": [{
            "enabled": True, "clip_path": str(clip), "destination_timestamp": 1.0,
            "clip_trim_start": 0.0, "clip_trim_duration": 2.0, "stretch_factor": 1.0,
        }],
    }

    render.render_schedule_over_original_audio(
        original_media=original, schedule=schedule, duration=5.0,
        output_path=tmp_path / "mix.wav", sample_rate=48000, channels=2,
        target_lufs=-18.0, fade_duration=0.015,
        mute_regions=[{"start": 1.0, "duration": 2.0}],
        suppression_mode="hard_mute", suppression_fade_duration=0.05,
        background_reconstruction="neighboring_non_speech_with_adaptive_crossfades",
    )

    report = schedule["background_reconstruction_report"]
    assert report["reconstructed_region_count"] == 1
    assert report["silence_fallback_region_count"] == 0
    source = report["sources"][0]
    assert source["source_start"] >= 3.08
    assert source["target_start"] == 1.0
    assert source["selection_score"] > 0.0
    assert source["score_components"]["duration_fit"] > 0.9
    assert source["candidate_count"] == 1
    command_text = " ".join(" ".join(command) for command in commands)
    assert "aloop=loop=-1" not in command_text
    assert source["loop_required"] is False
    assert source["coverage_ratio"] > 0.9
    assert "afade=t=in" in command_text
    assert schedule["performance_summary"]["voice_residue"] == "NOT_ACOUSTICALLY_MEASURED"


def test_ambience_reconstruction_refuses_short_or_reused_beds() -> None:
    plan = render._neighboring_ambience_plan(
        target_regions=[
            {"start": 1.0, "duration": 3.0},
            {"start": 5.0, "duration": 3.0},
        ],
        protected_speech_regions=[
            {"start": 1.0, "end": 4.0},
            {"start": 5.0, "end": 8.0},
        ],
        duration=9.0,
    )

    assert len(plan) <= 1
    assert all(row["loop_required"] is False for row in plan)
    assert len({(row["source_start"], row["source_end"]) for row in plan}) == len(plan)


def test_scheduled_audio_duration_trims_to_available_scheduled_audio() -> None:
    schedule = {
        "mappings": [
            {"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.5},
            {"enabled": False, "destination_timestamp": 9.0, "planned_render_duration": 10.0},
        ]
    }

    assert scheduled_audio_duration(schedule, 20.0) == 3.5


def test_scheduled_audio_duration_truncates_to_destination_duration() -> None:
    schedule = {"mappings": [{"enabled": True, "destination_timestamp": 8.0, "planned_render_duration": 5.0}]}

    assert scheduled_audio_duration(schedule, 10.0) == 10.0


def test_preview_bounds_adds_padding_and_caps_to_destination() -> None:
    mappings = [
        {"destination_timestamp": 5.0, "planned_render_duration": 2.0},
        {"destination_timestamp": 9.0, "planned_render_duration": 5.0},
    ]

    assert preview_bounds(mappings, destination_duration=12.0, padding=1.0) == (4.0, 12.0)


def test_build_preview_schedule_rebases_selected_mappings() -> None:
    schedule = {
        "mappings": [
            {
                "destination_timestamp": 10.0,
                "enabled": False,
                "alignment_slot_start": 10.0,
                "alignment_slot_end": 12.0,
                "shot_start": 9.5,
                "shot_end": 12.5,
            }
        ]
    }

    preview = build_preview_schedule(schedule, schedule["mappings"], start_time=8.5)

    assert preview["mappings"][0]["destination_timestamp"] == 1.5
    assert preview["mappings"][0]["alignment_slot_start"] == 1.5
    assert preview["mappings"][0]["alignment_slot_end"] == 3.5
    assert preview["mappings"][0]["shot_start"] == 1.0
    assert preview["mappings"][0]["shot_end"] == 4.0
    assert preview["mappings"][0]["enabled"] is True
    assert schedule["mappings"][0]["destination_timestamp"] == 10.0
