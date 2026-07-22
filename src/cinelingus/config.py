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
    prefer_high_speaker_confidence: bool
    prefer_clean_dialogue_timing: bool
    prefer_funny_or_surprising_matches: bool
    semantic_mode: str = "SEMANTIC_DISABLED"
    semantic_weight: float = 0.0
    dialogue_suppression: str = "hard_mute"
    suppression_padding: float = 0.04
    background_reconstruction: str = "neighboring_non_speech_with_adaptive_crossfades"
    verify_voice_residue: bool = True
    residue_correction_passes: int = 1
    residue_correction_padding: float = 0.12
    editorial_refinement_enabled: bool = True
    editorial_max_passes: int = 2
    editorial_acceptance_threshold: float = 0.72
    editorial_min_word_coverage: float = 0.72
    editorial_max_repairs_per_pass: int = 24
    editorial_incremental_render: bool = True
    editorial_suppress_unresolved: bool = False
    editorial_benchmark_failure_category: str | None = None
    film_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.semantic_mode not in {"SEMANTIC_DISABLED", "SEMANTIC_REPORT_ONLY", "SEMANTIC_ASSISTED"}:
            raise ValueError("semantic_mode must be SEMANTIC_DISABLED, SEMANTIC_REPORT_ONLY, or SEMANTIC_ASSISTED")
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1")
        if self.semantic_mode != "SEMANTIC_ASSISTED" and self.semantic_weight != 0.0:
            raise ValueError("Only SEMANTIC_ASSISTED may use a non-zero semantic_weight")

    @property
    def films(self) -> tuple[Path, ...]:
        """Canonical ordered film inputs; the first film is the anchor."""
        if self.film_paths:
            return self.film_paths
        if self.source_dialogue == self.destination_video:
            return (self.destination_video,)
        return (self.destination_video, self.source_dialogue)

    @property
    def anchor_film(self) -> Path:
        return self.films[0]

    def with_films(self, films: list[Path] | tuple[Path, ...], *, anchor_index: int = 0) -> "AppConfig":
        rows = tuple(Path(path) for path in films)
        if not rows:
            raise ValueError("At least one film is required.")
        if anchor_index < 0 or anchor_index >= len(rows):
            raise ValueError(f"Anchor index {anchor_index} is outside the {len(rows)} selected films.")
        anchor = rows[anchor_index]
        donors = rows[:anchor_index] + rows[anchor_index + 1:]
        ordered = (anchor, *donors)
        return replace(
            self,
            destination_video=anchor,
            source_dialogue=donors[0] if donors else anchor,
            film_paths=ordered,
        )

    def with_overrides(
        self,
        *,
        mode: str | None = None,
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
        dialogue_suppression: str | None = None,
        cinematic_filter: str | None = None,
        enable_speaker_awareness: bool | None = None,
        speaker_diarization_backend: str | None = None,
        speaker_diarization_model: str | None = None,
        speaker_diarization_device: str | None = None,
        semantic_mode: str | None = None,
        semantic_weight: float | None = None,
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
        path_updates = {}
        if destination_video is not None:
            path_updates["destination_video"] = destination_video
        if source_dialogue is not None:
            path_updates["source_dialogue"] = source_dialogue
        if output_dir is not None:
            path_updates["output_dir"] = output_dir
        if path_updates:
            if "destination_video" in path_updates or "source_dialogue" in path_updates:
                path_updates["film_paths"] = ()
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
        if dialogue_suppression is not None:
            if dialogue_suppression not in {"hard_mute", "duck"}:
                raise ValueError("dialogue_suppression must be hard_mute or duck")
            advanced_updates["dialogue_suppression"] = dialogue_suppression
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
        next_semantic_mode = semantic_mode or config.semantic_mode
        next_semantic_weight = config.semantic_weight if semantic_weight is None else semantic_weight
        if semantic_mode is not None:
            if semantic_mode not in {"SEMANTIC_DISABLED", "SEMANTIC_REPORT_ONLY", "SEMANTIC_ASSISTED"}:
                raise ValueError("semantic_mode must be SEMANTIC_DISABLED, SEMANTIC_REPORT_ONLY, or SEMANTIC_ASSISTED")
            advanced_updates["semantic_mode"] = semantic_mode
        if semantic_weight is not None:
            if not 0.0 <= semantic_weight <= 1.0:
                raise ValueError("semantic_weight must be between 0 and 1")
            advanced_updates["semantic_weight"] = semantic_weight
        if next_semantic_mode != "SEMANTIC_ASSISTED" and next_semantic_weight != 0.0:
            raise ValueError("Only SEMANTIC_ASSISTED may use a non-zero semantic_weight")
        if advanced_updates:
            config = replace(config, **advanced_updates)
        return config


def load_config(root: Path, config_path: Path | None = None) -> AppConfig:
    path = config_path or root / "config" / "default.json"
    raw = read_json(path)

    def p(name: str) -> Path:
        value = Path(raw[name])
        return value if value.is_absolute() else root / value

    mode = str(raw.get("transcription_mode", "quality"))
    quality_modes = dict(raw.get("quality_modes", {}))
    model = str(raw.get("whisper_model", quality_modes.get(mode, {}).get("whisper_model", "medium")))
    if mode in quality_modes:
        model = str(quality_modes[mode].get("whisper_model", model))

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
        prefer_high_speaker_confidence=bool(raw.get("prefer_high_speaker_confidence", True)),
        prefer_clean_dialogue_timing=bool(raw.get("prefer_clean_dialogue_timing", True)),
        prefer_funny_or_surprising_matches=bool(raw.get("prefer_funny_or_surprising_matches", True)),
        semantic_mode=str(raw.get("semantic_mode", "SEMANTIC_DISABLED")),
        semantic_weight=float(raw.get("semantic_weight", 0.0)),
        dialogue_suppression=str(raw.get("dialogue_suppression", "hard_mute")),
        suppression_padding=float(raw.get("suppression_padding", 0.04)),
        background_reconstruction=str(raw.get("background_reconstruction", "neighboring_non_speech_with_adaptive_crossfades")),
        verify_voice_residue=bool(raw.get("verify_voice_residue", True)),
        residue_correction_passes=max(0, int(raw.get("residue_correction_passes", 1))),
        residue_correction_padding=max(0.0, float(raw.get("residue_correction_padding", 0.12))),
        editorial_refinement_enabled=bool(raw.get("editorial_refinement_enabled", True)),
        editorial_max_passes=max(0, int(raw.get("editorial_max_passes", 2))),
        editorial_acceptance_threshold=max(0.0, min(1.0, float(raw.get("editorial_acceptance_threshold", 0.72)))),
        editorial_min_word_coverage=max(0.0, min(1.0, float(raw.get("editorial_min_word_coverage", 0.72)))),
        editorial_max_repairs_per_pass=max(1, int(raw.get("editorial_max_repairs_per_pass", 24))),
        editorial_incremental_render=bool(raw.get("editorial_incremental_render", True)),
        editorial_suppress_unresolved=bool(raw.get("editorial_suppress_unresolved", False)),
        editorial_benchmark_failure_category=(
            str(raw["editorial_benchmark_failure_category"])
            if raw.get("editorial_benchmark_failure_category") else None
        ),
    )
