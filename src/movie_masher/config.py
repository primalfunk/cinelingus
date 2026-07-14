from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .cinematic_filters import FILTER_CHOICES
from .util import read_json


@dataclass(frozen=True)
class AppConfig:
    root: Path
    destination_video: Path
    source_dialogue: Path
    cache_dir: Path
    output_dir: Path
    temp_dir: Path
    speech_backend: str
    transcription_mode: str
    whisper_model: str
    whisper_language: str | None
    quick_test_seconds: float | None
    quality_modes: dict[str, dict[str, Any]]
    filter_min_duration: float
    filter_max_duration: float
    filter_min_confidence: float
    filter_min_chars_per_second: float
    filter_max_chars_per_second: float
    filter_repeated_text_window: int
    silence_noise_db: int
    silence_min_duration: float
    min_speech_duration: float
    merge_gap: float
    max_time_stretch: float
    scheduling_mode: str
    best_fit_lookahead: int
    render_sample_rate: int
    render_channels: int
    target_lufs: float
    audio_fade_duration: float
    original_duck_db: float
    visual_scene_threshold: float
    visual_min_shot_duration: float
    shot_boundary_mode: str
    cinematic_filter: str
    enable_speaker_awareness: bool
    speaker_diarization_backend: str
    speaker_diarization_model: str
    speaker_diarization_device: str
    target_duration_seconds: float
    minimum_duration_seconds: float
    maximum_duration_seconds: float
    prefer_high_speaker_confidence: bool
    prefer_clean_dialogue_timing: bool
    prefer_funny_or_surprising_matches: bool
    allow_full_movie_mode: bool

    def with_overrides(
        self,
        *,
        mode: str | None = None,
        quick_seconds: float | None = None,
        destination_video: Path | None = None,
        source_dialogue: Path | None = None,
        output_dir: Path | None = None,
        shot_boundary_mode: str | None = None,
        scheduling_mode: str | None = None,
        best_fit_lookahead: int | None = None,
        max_time_stretch: float | None = None,
        visual_scene_threshold: float | None = None,
        visual_min_shot_duration: float | None = None,
        target_lufs: float | None = None,
        audio_fade_duration: float | None = None,
        original_duck_db: float | None = None,
        cinematic_filter: str | None = None,
        enable_speaker_awareness: bool | None = None,
        speaker_diarization_backend: str | None = None,
        speaker_diarization_model: str | None = None,
        speaker_diarization_device: str | None = None,
        target_duration_seconds: float | None = None,
        minimum_duration_seconds: float | None = None,
        maximum_duration_seconds: float | None = None,
        allow_full_movie_mode: bool | None = None,
    ) -> "AppConfig":
        config = self
        if mode:
            if mode not in self.quality_modes:
                choices = ", ".join(sorted(self.quality_modes))
                raise ValueError(f"Unknown transcription mode '{mode}'. Available modes: {choices}")
            mode_config = self.quality_modes[mode]
            config = replace(
                config,
                transcription_mode=mode,
                whisper_model=str(mode_config.get("whisper_model", self.whisper_model)),
            )
        if quick_seconds is not None:
            config = replace(config, quick_test_seconds=quick_seconds)
        path_updates = {}
        if destination_video is not None:
            path_updates["destination_video"] = destination_video
        if source_dialogue is not None:
            path_updates["source_dialogue"] = source_dialogue
        if output_dir is not None:
            path_updates["output_dir"] = output_dir
        if path_updates:
            config = replace(config, **path_updates)
        advanced_updates: dict[str, Any] = {}
        if shot_boundary_mode is not None:
            if shot_boundary_mode not in {"off", "soft", "strict"}:
                raise ValueError("shot_boundary_mode must be off, soft, or strict")
            advanced_updates["shot_boundary_mode"] = shot_boundary_mode
        if scheduling_mode is not None:
            if scheduling_mode not in {"strict_order", "best_fit", "window_fill", "whole_line_fill", "performance_fill"}:
                raise ValueError("scheduling_mode must be strict_order, best_fit, window_fill, whole_line_fill, or performance_fill")
            advanced_updates["scheduling_mode"] = scheduling_mode
        if best_fit_lookahead is not None:
            if best_fit_lookahead < 1:
                raise ValueError("best_fit_lookahead must be at least 1")
            advanced_updates["best_fit_lookahead"] = best_fit_lookahead
        if max_time_stretch is not None:
            if max_time_stretch < 0:
                raise ValueError("max_time_stretch must be non-negative")
            advanced_updates["max_time_stretch"] = max_time_stretch
        if visual_scene_threshold is not None:
            if not 0 <= visual_scene_threshold <= 1:
                raise ValueError("visual_scene_threshold must be between 0 and 1")
            advanced_updates["visual_scene_threshold"] = visual_scene_threshold
        if visual_min_shot_duration is not None:
            if visual_min_shot_duration <= 0:
                raise ValueError("visual_min_shot_duration must be greater than zero")
            advanced_updates["visual_min_shot_duration"] = visual_min_shot_duration
        if target_lufs is not None:
            advanced_updates["target_lufs"] = target_lufs
        if audio_fade_duration is not None:
            if audio_fade_duration < 0:
                raise ValueError("audio_fade_duration must be non-negative")
            advanced_updates["audio_fade_duration"] = audio_fade_duration
        if original_duck_db is not None:
            if original_duck_db > 0:
                raise ValueError("original_duck_db must be zero or negative")
            advanced_updates["original_duck_db"] = original_duck_db
        if cinematic_filter is not None:
            if cinematic_filter not in FILTER_CHOICES:
                choices = ", ".join(FILTER_CHOICES)
                raise ValueError(f"cinematic_filter must be one of: {choices}")
            advanced_updates["cinematic_filter"] = cinematic_filter
        if enable_speaker_awareness is not None:
            advanced_updates["enable_speaker_awareness"] = bool(enable_speaker_awareness)
        if speaker_diarization_backend is not None:
            if speaker_diarization_backend not in {"heuristic", "pyannote"}:
                raise ValueError("speaker_diarization_backend must be heuristic or pyannote")
            advanced_updates["speaker_diarization_backend"] = speaker_diarization_backend
        if speaker_diarization_model is not None:
            advanced_updates["speaker_diarization_model"] = speaker_diarization_model
        if speaker_diarization_device is not None:
            if speaker_diarization_device not in {"auto", "cpu", "cuda"}:
                raise ValueError("speaker_diarization_device must be auto, cpu, or cuda")
            advanced_updates["speaker_diarization_device"] = speaker_diarization_device
        if target_duration_seconds is not None:
            if target_duration_seconds <= 0:
                raise ValueError("target_duration_seconds must be greater than zero")
            advanced_updates["target_duration_seconds"] = float(target_duration_seconds)
        if minimum_duration_seconds is not None:
            if minimum_duration_seconds <= 0:
                raise ValueError("minimum_duration_seconds must be greater than zero")
            advanced_updates["minimum_duration_seconds"] = float(minimum_duration_seconds)
        if maximum_duration_seconds is not None:
            if maximum_duration_seconds <= 0:
                raise ValueError("maximum_duration_seconds must be greater than zero")
            advanced_updates["maximum_duration_seconds"] = float(maximum_duration_seconds)
        if allow_full_movie_mode is not None:
            advanced_updates["allow_full_movie_mode"] = bool(allow_full_movie_mode)
        if advanced_updates:
            config = replace(config, **advanced_updates)
        return config


def load_config(root: Path, config_path: Path | None = None) -> AppConfig:
    path = config_path or root / "config" / "default.json"
    raw = read_json(path)

    def p(name: str) -> Path:
        value = Path(raw[name])
        return value if value.is_absolute() else root / value

    mode = str(raw.get("transcription_mode", "fast_preview"))
    quality_modes = dict(raw.get("quality_modes", {}))
    model = str(raw.get("whisper_model", quality_modes.get(mode, {}).get("whisper_model", "tiny")))
    if mode in quality_modes:
        model = str(quality_modes[mode].get("whisper_model", model))

    quick_value = raw.get("quick_test_seconds")
    return AppConfig(
        root=root,
        destination_video=p("destination_video"),
        source_dialogue=p("source_dialogue"),
        cache_dir=p("cache_dir"),
        output_dir=p("output_dir"),
        temp_dir=p("temp_dir"),
        speech_backend=str(raw.get("speech_backend", "fallback")),
        transcription_mode=mode,
        whisper_model=model,
        whisper_language=str(raw.get("whisper_language") or "en"),
        quick_test_seconds=float(quick_value) if quick_value is not None else None,
        quality_modes=quality_modes,
        filter_min_duration=float(raw.get("filter_min_duration", 0.35)),
        filter_max_duration=float(raw.get("filter_max_duration", 12.0)),
        filter_min_confidence=float(raw.get("filter_min_confidence", 0.2)),
        filter_min_chars_per_second=float(raw.get("filter_min_chars_per_second", 0.5)),
        filter_max_chars_per_second=float(raw.get("filter_max_chars_per_second", 28.0)),
        filter_repeated_text_window=int(raw.get("filter_repeated_text_window", 4)),
        silence_noise_db=int(raw["silence_noise_db"]),
        silence_min_duration=float(raw["silence_min_duration"]),
        min_speech_duration=float(raw["min_speech_duration"]),
        merge_gap=float(raw["merge_gap"]),
        max_time_stretch=float(raw["max_time_stretch"]),
        scheduling_mode=str(raw.get("scheduling_mode", "strict_order")),
        best_fit_lookahead=int(raw.get("best_fit_lookahead", 8)),
        render_sample_rate=int(raw["render_sample_rate"]),
        render_channels=int(raw["render_channels"]),
        target_lufs=float(raw["target_lufs"]),
        audio_fade_duration=float(raw.get("audio_fade_duration", 0.015)),
        original_duck_db=float(raw.get("original_duck_db", -28.0)),
        visual_scene_threshold=float(raw.get("visual_scene_threshold", 0.35)),
        visual_min_shot_duration=float(raw.get("visual_min_shot_duration", 0.5)),
        shot_boundary_mode=str(raw.get("shot_boundary_mode", "soft")),
        cinematic_filter=str(raw.get("cinematic_filter", "balanced")),
        enable_speaker_awareness=bool(raw.get("enable_speaker_awareness", True)),
        speaker_diarization_backend=str(raw.get("speaker_diarization_backend", "heuristic")),
        speaker_diarization_model=str(raw.get("speaker_diarization_model", "pyannote/speaker-diarization-community-1")),
        speaker_diarization_device=str(raw.get("speaker_diarization_device", "auto")),
        target_duration_seconds=float(raw.get("target_duration_seconds", 180.0)),
        minimum_duration_seconds=float(raw.get("minimum_duration_seconds", 120.0)),
        maximum_duration_seconds=float(raw.get("maximum_duration_seconds", 300.0)),
        prefer_high_speaker_confidence=bool(raw.get("prefer_high_speaker_confidence", True)),
        prefer_clean_dialogue_timing=bool(raw.get("prefer_clean_dialogue_timing", True)),
        prefer_funny_or_surprising_matches=bool(raw.get("prefer_funny_or_surprising_matches", True)),
        allow_full_movie_mode=bool(raw.get("allow_full_movie_mode", False)),
    )
