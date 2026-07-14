from __future__ import annotations

from pathlib import Path
from typing import Any

from movie_masher.filters import usable_rows
from movie_masher.render import mux_video, render_schedule_over_original_audio
from movie_masher.schedule import build_schedule
from movie_masher.mutations import enforce_self_shuffle_changed_lines, speaker_aware_shuffle_selection, _mark_speaker_shuffle_fallbacks
from movie_masher.transformation_verbs import self_shuffle_transformation_plan

from .base import Transformation, TransformationMetadata, TransformationResult


class SelfShuffleTransformation(Transformation):
    metadata = TransformationMetadata(
        id="self_shuffle",
        display_name="Self Shuffle",
        description="Shuffle one film's own dialogue back into its speaking windows.",
        required_inputs=("destination_video",),
        generated_outputs=("self_shuffle_schedule.json", "self_shuffle_dialogue.wav", "self_shuffle_output.mp4", "transformation_report.json"),
        supported_modes=("fast_preview", "balanced", "quality"),
        version="1.0",
    )

    def __init__(self, context):
        super().__init__(context)
        self._working_pipeline: Any | None = None
        self._schedule_path: Path | None = None
        self._schedule: dict[str, Any] | None = None

    def validate_inputs(self) -> None:
        config = self.context.pipeline.config
        if not config.destination_video.exists():
            raise FileNotFoundError(f"Destination video does not exist: {config.destination_video}")

    def select(self) -> dict[str, Any]:
        from movie_masher.pipeline import Pipeline

        base = self.context.pipeline
        force = self.context.force
        single_config = base.config.with_overrides(source_dialogue=base.config.destination_video)
        single = Pipeline(single_config, cancel_check=base.cancel_check, stage_callback=base.stage_callback)
        self._working_pipeline = single

        single.inspect(force=force)
        single.extract_source_dialogue(force=force)
        single.filter_source_dialogue(force=force)
        library = single.build_clip_library(force=force)
        single.detect_destination_timeline(force=force)
        timeline = single.filter_destination_timeline(force=force)
        return {"clip_library": library, "timeline": timeline}

    def transform(self, selections: dict[str, Any]) -> dict[str, Any]:
        single = self._require_working_pipeline()
        seed = int(self.context.parameters.get("seed", 1))
        library = selections["clip_library"]
        timeline = selections["timeline"]
        schedule_path = single.destination.cache_dir / "self_shuffle_schedule.json"
        self._schedule_path = schedule_path
        signature = single._signature(
            "self_shuffle_schedule",
            single.destination.media_hash,
            library.get("config_signature"),
            timeline.get("config_signature"),
            seed,
        )
        schedule = single._load_current("replacement_schedule", schedule_path, single.destination.media_hash, signature, self.context.force)
        schedule_regenerated = schedule is None
        if schedule is not None and _requires_self_shuffle_regeneration(schedule):
            single.logger.info(f"cached self-shuffle schedule is stale, regenerating: {schedule_path}")
            schedule = None
            schedule_regenerated = True
        if schedule is None:
            windows = usable_rows(timeline["windows"])
            clips = speaker_aware_shuffle_selection(library["clips"], windows, seed=seed)
            single.logger.info(f"building self-shuffle schedule for {len(windows)} usable windows")
            schedule = build_schedule(
                clips=clips,
                windows=windows,
                source_hash=single.destination.media_hash,
                destination_hash=single.destination.media_hash,
                max_time_stretch=single.config.max_time_stretch,
                output_path=schedule_path,
                scheduling_mode="whole_line_fill",
                best_fit_lookahead=single.config.best_fit_lookahead,
                transformation_name="self_shuffle",
                transformation_history=self_shuffle_transformation_plan(),
                cinematic_filter=single.config.cinematic_filter,
            )
            schedule["config_signature"] = signature
            schedule["mutation_id"] = "self_shuffle"
            schedule["self_shuffle_render_strategy"] = "dialogue_only_v1"
            for mapping in schedule.get("mappings", []):
                mapping["mutation_operation"] = "self_shuffle"
            _mark_speaker_shuffle_fallbacks(schedule)
            enforce_self_shuffle_changed_lines(schedule=schedule, clips=library["clips"])
            single._write_and_validate("replacement_schedule", schedule_path, schedule)
            single.logger.info(f"self-shuffle schedule mappings: {len(schedule['mappings'])}")
        else:
            single.logger.info(f"reused self-shuffle schedule: {schedule_path}")
        self._schedule = schedule
        return {"schedule": schedule, "schedule_path": schedule_path, "schedule_regenerated": schedule_regenerated}

    def validate(self, transformed: dict[str, Any]) -> None:
        schedule = transformed.get("schedule", {})
        if not schedule.get("mappings"):
            raise ValueError("Self Shuffle transformation produced no schedule mappings.")

    def render(self, transformed: dict[str, Any]) -> dict[str, Path]:
        single = self._require_working_pipeline()
        schedule = transformed["schedule"]
        schedule_path = transformed["schedule_path"]
        dest_movie = single._inspect_one(single.destination, force=False)
        audio_output = single.config.output_dir / "self_shuffle_dialogue.wav"
        video_output = single.config.output_dir / "self_shuffle_output.mp4"
        force = self.context.force
        schedule_changed = bool(transformed.get("schedule_regenerated")) or _is_stale(audio_output, schedule_path)

        if force or schedule_changed or not audio_output.exists():
            render_duration = float(dest_movie["duration"])
            single.logger.info(f"rendering self-shuffle soundtrack duration={render_duration:.3f}s")
            render_schedule_over_original_audio(
                original_media=single.destination.media_path,
                schedule=schedule,
                duration=render_duration,
                output_path=audio_output,
                sample_rate=single.config.render_sample_rate,
                channels=single.config.render_channels,
                target_lufs=single.config.target_lufs,
                fade_duration=single.config.audio_fade_duration,
            )
            single.logger.info(f"rendered self-shuffle soundtrack: {audio_output}")
        else:
            single.logger.info(f"reused self-shuffle audio: {audio_output}")

        if force or schedule_changed or _is_stale(video_output, audio_output) or not video_output.exists():
            single.logger.info("muxing self-shuffle video")
            mux_video(destination_video=single.destination.media_path, dialogue_wav=audio_output, output_path=video_output)
            single.logger.info(f"rendered self-shuffle video: {video_output}")
        else:
            single.logger.info(f"reused self-shuffle video: {video_output}")

        return {"schedule": schedule_path, "audio": audio_output, "video": video_output}

    def generate_report(self, result: TransformationResult) -> Path | None:
        from .report import write_transformation_report

        return write_transformation_report(
            metadata=self.metadata,
            pipeline=self._require_working_pipeline(),
            result=result,
        )

    def _require_working_pipeline(self):
        if self._working_pipeline is None:
            raise RuntimeError("Self Shuffle working pipeline has not been initialized.")
        return self._working_pipeline


def _is_stale(output_path: Path, input_path: Path) -> bool:
    if not output_path.exists() or not input_path.exists():
        return True
    return input_path.stat().st_mtime > output_path.stat().st_mtime


def _requires_self_shuffle_regeneration(schedule: dict[str, Any]) -> bool:
    if schedule.get("self_shuffle_render_strategy") != "dialogue_only_v1":
        return True
    for mapping in schedule.get("mappings", []):
        if not mapping.get("enabled", True):
            continue
        if mapping.get("clip_movie_timestamp") is None or mapping.get("source_movie_timestamp") is None:
            return True
    return False