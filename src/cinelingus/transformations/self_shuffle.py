from __future__ import annotations

from pathlib import Path
from typing import Any

from cinelingus.filters import usable_rows
from cinelingus.filter_lab.acceptance import validate_filter_output
from cinelingus.montage import FULL_TIMELINE_POLICY_VERSION, build_full_timeline_plan, build_montage_render_acceptance, rebase_schedule_to_montage
from cinelingus.render import mux_video, render_schedule_over_original_audio
from cinelingus.schedule import build_schedule
from cinelingus.mutations import enforce_self_shuffle_changed_lines, speaker_aware_shuffle_selection, _mark_speaker_shuffle_fallbacks
from cinelingus.transformation_verbs import self_shuffle_transformation_plan
from cinelingus.tools import ffprobe_json

from .base import Transformation, TransformationMetadata, TransformationResult


class SelfShuffleTransformation(Transformation):
    metadata = TransformationMetadata(
        id="self_shuffle",
        display_name="Self Shuffle",
        description="Shuffle one film's own dialogue back into its speaking windows.",
        required_inputs=("destination_video",),
        generated_outputs=("self_shuffle_schedule.json", "montage_plan.json", "self_shuffle_dialogue.wav", "self_shuffle_output.mp4", "transformation_report.json"),
        supported_modes=("fast_preview", "balanced", "quality"),
        version="2.0",
    )

    def __init__(self, context):
        super().__init__(context)
        self._working_pipeline: Any | None = None
        self._schedule_path: Path | None = None
        self._schedule: dict[str, Any] | None = None
        self._montage_plan_path: Path | None = None
        self._montage_acceptance_path: Path | None = None
        self._filter_acceptance_path: Path | None = None

    def validate_inputs(self) -> None:
        config = self.context.pipeline.config
        if not config.destination_video.exists():
            raise FileNotFoundError(f"Destination video does not exist: {config.destination_video}")

    def select(self) -> dict[str, Any]:
        from cinelingus.pipeline import Pipeline

        base = self.context.pipeline
        force = self.context.force
        single_config = base.config.with_overrides(source_dialogue=base.config.destination_video)
        single = Pipeline(single_config, cancel_check=base.cancel_check, stage_callback=base.stage_callback)
        self._working_pipeline = single

        destination_movie, _source_movie = single.inspect(force=force)
        single.extract_source_dialogue(force=force)
        single.filter_source_dialogue(force=force)
        library = single.build_clip_library(force=force)
        timeline_raw = single.detect_destination_timeline(force=force)
        timeline = single.filter_destination_timeline(force=force)
        visual = single.analyze_visual(force=force)
        return {"clip_library": library, "timeline": timeline, "visual": visual, "destination_movie": destination_movie}

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
            selections["visual"].get("shots", {}).get("config_signature"),
            FULL_TIMELINE_POLICY_VERSION,
            seed,
        )
        schedule = single._load_current("replacement_schedule", schedule_path, single.destination.media_hash, signature, self.context.force)
        schedule_regenerated = schedule is None
        expected_plan_path = single.config.output_dir / "self_shuffle" / "montage_plan.json"
        if schedule is not None and not expected_plan_path.exists():
            single.logger.info(f"cached self-shuffle montage plan is missing, regenerating: {expected_plan_path}")
            schedule = None
            schedule_regenerated = True
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
            schedule["self_shuffle_render_strategy"] = "full_source_timeline_v1"
            for mapping in schedule.get("mappings", []):
                mapping["mutation_operation"] = "self_shuffle"
            _mark_speaker_shuffle_fallbacks(schedule)
            enforce_self_shuffle_changed_lines(schedule=schedule, clips=library["clips"])
            duration = float(selections["destination_movie"]["duration"])
            plan = build_full_timeline_plan(
                filter_id="translation.self_shuffle",
                filter_contract_version=self.metadata.version,
                anchor_source_id="film_a",
                anchor_media_hash=single.destination.media_hash,
                anchor_duration=duration,
                supporting_audio_durations=[duration],
                shot_ids=[
                    str(row["id"])
                    for row in selections["visual"].get("shots", {}).get("shots", [])
                ],
                random_seed=seed,
                governing_relationship="self_recollection",
                laws={
                    "visual": "COMPLETE_SOURCE_TIMELINE_FROM_ZERO",
                    "temporal": "DESTINATION_CHRONOLOGY_PRESERVED",
                    "dialogue": "same-film dialogue reassigned across the complete timeline",
                    "requested_audio": "DIALOGUE_REPLACEMENT_PLUS_CONTINUOUS_AMBIENT_BED",
                    "actual_audio_method": "CONTINUOUS_SOURCE_SOUNDTRACK_BED",
                },
                schedule=schedule,
                repetition_authorized=bool(self.context.parameters.get("allow_line_reuse", False)),
                repetition_authorization_basis=(
                    "FILTER_PARAMETER:allow_line_reuse"
                    if bool(self.context.parameters.get("allow_line_reuse", False))
                    else None
                ),
            )
            schedule = rebase_schedule_to_montage(schedule, plan)
            schedule["montage_native"] = True
            schedule["full_timeline_native"] = True
            schedule["input_scope"] = "complete_media_files"
            schedule["duration_policy"] = dict(plan["duration_resolution"])
            schedule["dead_air_policy"] = "SOURCE_SOUNDTRACK_BED_WITH_SUSTAINED_SILENCE_REJECTION"
            schedule["config_signature"] = signature
            plan_path = expected_plan_path
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            single._write_and_validate("montage_plan", plan_path, plan)
            single._write_and_validate("replacement_schedule", schedule_path, schedule)
            self._montage_plan_path = plan_path
            single.logger.info(f"self-shuffle full timeline: {plan['actual_duration']:.3f}s, {len(schedule['mappings'])} dialogue mappings")
        else:
            single.logger.info(f"reused self-shuffle schedule: {schedule_path}")
            self._montage_plan_path = expected_plan_path
        self._schedule = schedule
        return {"schedule": schedule, "schedule_path": schedule_path, "montage_plan_path": self._montage_plan_path, "schedule_regenerated": schedule_regenerated}

    def validate(self, transformed: dict[str, Any]) -> None:
        schedule = transformed.get("schedule", {})
        if not schedule.get("mappings"):
            raise ValueError("Self Shuffle transformation produced no schedule mappings.")
        if schedule.get("self_shuffle_render_strategy") != "full_source_timeline_v1":
            raise ValueError("Self Shuffle did not produce a complete-source timeline schedule.")

    def create_filter_artifacts(self, transformed: dict[str, Any]) -> dict[str, Path]:
        artifacts = super().create_filter_artifacts(transformed)
        artifacts["montage_plan"] = transformed["montage_plan_path"]
        return artifacts

    def render(self, transformed: dict[str, Any]) -> dict[str, Path]:
        single = self._require_working_pipeline()
        schedule = transformed["schedule"]
        schedule_path = transformed["schedule_path"]
        plan_path = transformed["montage_plan_path"]
        from cinelingus.util import read_json
        plan = read_json(plan_path)
        audio_output = single.config.output_dir / "self_shuffle_dialogue.wav"
        video_output = single.config.output_dir / "self_shuffle_output.mp4"
        force = self.context.force
        schedule_changed = bool(transformed.get("schedule_regenerated")) or _is_stale(audio_output, schedule_path)

        if force or schedule_changed or not audio_output.exists():
            render_duration = float(schedule["render_duration"])
            single.logger.info(f"rendering self-shuffle over continuous source soundtrack duration={render_duration:.3f}s")
            render_schedule_over_original_audio(
                original_media=single.destination.media_path,
                schedule=schedule,
                duration=render_duration,
                output_path=audio_output,
                sample_rate=single.config.render_sample_rate,
                channels=single.config.render_channels,
                target_lufs=single.config.target_lufs,
                fade_duration=single.config.audio_fade_duration,
                mute_regions=schedule.get("mappings", []),
            )
            single.logger.info(f"rendered self-shuffle soundtrack: {audio_output}")
        else:
            single.logger.info(f"reused self-shuffle audio: {audio_output}")

        if force or schedule_changed or _is_stale(video_output, audio_output) or not video_output.exists():
            single.logger.info("muxing full-timeline self-shuffle")
            mux_video(destination_video=single.destination.media_path, dialogue_wav=audio_output, output_path=video_output)
            single.logger.info(f"rendered self-shuffle video: {video_output}")
        else:
            single.logger.info(f"reused self-shuffle video: {video_output}")

        self._filter_acceptance_path = single.config.output_dir / "self_shuffle" / "filter_acceptance.json"
        validate_filter_output(
            filter_id="translation.self_shuffle",
            schedule=schedule,
            final_video=video_output,
            replacement_audio=audio_output,
            output_path=self._filter_acceptance_path,
            schemas_dir=single.schemas_dir,
        )

        acceptance_path = single.config.output_dir / "self_shuffle" / "montage_render_acceptance.json"
        acceptance = build_montage_render_acceptance(plan=plan, encoded_probe=ffprobe_json(video_output), output_path=acceptance_path)
        single._write_and_validate("montage_render_acceptance", acceptance_path, acceptance)
        self._montage_acceptance_path = acceptance_path

        return {"schedule": schedule_path, "audio": audio_output, "video": video_output}

    def generate_report(self, result: TransformationResult) -> Path | None:
        from .report import write_transformation_report

        if self._montage_acceptance_path is not None:
            result.artifacts["montage_render_acceptance"] = self._montage_acceptance_path
        if self._filter_acceptance_path is not None:
            result.artifacts["filter_acceptance"] = self._filter_acceptance_path

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
    if schedule.get("self_shuffle_render_strategy") != "full_source_timeline_v1":
        return True
    if schedule.get("dead_air_policy") != "SOURCE_SOUNDTRACK_BED_WITH_SUSTAINED_SILENCE_REJECTION":
        return True
    for mapping in schedule.get("mappings", []):
        if not mapping.get("enabled", True):
            continue
        if mapping.get("clip_movie_timestamp") is None or mapping.get("source_movie_timestamp") is None:
            return True
    return False
