from pathlib import Path

from movie_masher import render
from movie_masher.render import build_preview_schedule, preview_bounds, scheduled_audio_duration


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
    assert "volume=enable='between(t,3.000,5.000)':volume=-28.0dB" in command_text
    assert str(clip) in command_text


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
        output_path=tmp_path / "movie_masher.wav",
        sample_rate=48000,
        channels=2,
        target_lufs=-18.0,
        fade_duration=0.015,
        mute_regions=[{"start": 2.0, "duration": 5.0}],
        duck_db=-24.0,
    )

    command_text = " ".join(" ".join(command) for command in commands)
    assert "volume=enable='between(t,2.000,7.000)':volume=-24.0dB" in command_text
    assert "volume=enable='between(t,3.000,5.000)':volume=-28.0dB" not in command_text
    assert str(clip) in command_text


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
