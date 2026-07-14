from pathlib import Path

from movie_masher.cli import build_parser
from movie_masher.config import load_config


def test_config_overrides_paths_mode_and_quick_seconds(tmp_path: Path) -> None:
    config = load_config(Path.cwd())
    updated = config.with_overrides(
        mode="fast_preview",
        quick_seconds=12.5,
        destination_video=tmp_path / "dest.mp4",
        source_dialogue=tmp_path / "source.mp4",
        output_dir=tmp_path / "out",
    )

    assert updated.transcription_mode == "fast_preview"
    assert updated.quick_test_seconds == 12.5
    assert updated.destination_video == tmp_path / "dest.mp4"
    assert updated.source_dialogue == tmp_path / "source.mp4"
    assert updated.output_dir == tmp_path / "out"
    assert config.destination_video != updated.destination_video
    assert config.visual_scene_threshold == 0.35
    assert config.visual_min_shot_duration == 0.5
    assert config.shot_boundary_mode == "soft"
    assert config.cinematic_filter == "balanced"
    assert config.transcription_mode == "balanced"
    assert config.whisper_model == "small"
    assert config.whisper_language == "en"
    assert config.speaker_diarization_backend == "pyannote"
    assert config.speaker_diarization_model == "pyannote/speaker-diarization-community-1"
    assert config.speaker_diarization_device == "auto"
    assert config.target_duration_seconds == 180.0
    assert config.minimum_duration_seconds == 120.0
    assert config.maximum_duration_seconds == 300.0
    assert config.allow_full_movie_mode is False


def test_config_advanced_overrides() -> None:
    config = load_config(Path.cwd())

    updated = config.with_overrides(
        shot_boundary_mode="strict",
        scheduling_mode="best_fit",
        best_fit_lookahead=12,
        max_time_stretch=0.2,
        visual_scene_threshold=0.45,
        visual_min_shot_duration=0.75,
        target_lufs=-20.0,
        audio_fade_duration=0.03,
        original_duck_db=-24.0,
        cinematic_filter="rhythm",
        speaker_diarization_backend="pyannote",
        speaker_diarization_device="cpu",
    )

    assert updated.shot_boundary_mode == "strict"
    assert updated.scheduling_mode == "best_fit"
    assert updated.best_fit_lookahead == 12
    assert updated.max_time_stretch == 0.2
    assert updated.visual_scene_threshold == 0.45
    assert updated.visual_min_shot_duration == 0.75
    assert updated.target_lufs == -20.0
    assert updated.audio_fade_duration == 0.03
    assert updated.original_duck_db == -24.0
    assert updated.cinematic_filter == "rhythm"
    assert updated.speaker_diarization_backend == "pyannote"
    assert updated.speaker_diarization_device == "cpu"


def test_cli_accepts_output_dir_override(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--output-dir", str(tmp_path / "chosen"), "inspect"])

    assert args.output_dir == tmp_path / "chosen"


def test_quality_modes_keep_tiny_only_for_fast_preview() -> None:
    config = load_config(Path.cwd())

    assert config.with_overrides(mode="fast_preview").whisper_model == "tiny"
    assert config.with_overrides(mode="balanced").whisper_model == "small"
    assert config.with_overrides(mode="quality").whisper_model == "medium"


def test_missing_whisper_language_defaults_to_english(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """{
  "destination_video": "source/movie_1.mpg",
  "source_dialogue": "source/movie_2.mp4",
  "cache_dir": "cache",
  "output_dir": "output",
  "temp_dir": "temp",
  "speech_backend": "whisper",
  "transcription_mode": "balanced",
  "whisper_model": "small",
  "quality_modes": {
    "balanced": {"whisper_model": "small"}
  },
  "silence_noise_db": -35,
  "silence_min_duration": 0.35,
  "min_speech_duration": 0.25,
  "merge_gap": 0.25,
  "max_time_stretch": 0.1,
  "render_sample_rate": 48000,
  "render_channels": 2,
  "target_lufs": -18.0
}
""",
        encoding="utf-8",
    )

    config = load_config(Path.cwd(), config_path)

    assert config.whisper_language == "en"
