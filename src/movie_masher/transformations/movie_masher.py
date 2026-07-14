from __future__ import annotations

from pathlib import Path
from typing import Any

from ..transformation_plan import build_movie_masher_plan, write_transformation_plan
from .base import Transformation, TransformationMetadata, TransformationResult


class MovieMasherTransformation(Transformation):
    metadata = TransformationMetadata(
        id="movie_masher",
        display_name="Transposition",
        description="Replace destination dialogue using dialogue extracted from another film.",
        required_inputs=("destination_video", "source_dialogue"),
        generated_outputs=("replacement_dialogue.wav", "movie_masher_output.mp4", "transformation_report.json"),
        supported_modes=("fast_preview", "balanced", "quality"),
        version="1.0",
    )

    def __init__(self, context):
        super().__init__(context)
        self._selections: dict[str, Any] = {}
        self._transformed: dict[str, Any] = {}
        self._plan_path: Path | None = None

    def validate_inputs(self) -> None:
        config = self.context.pipeline.config
        if not config.destination_video.exists():
            raise FileNotFoundError(f"Destination video does not exist: {config.destination_video}")
        if not config.source_dialogue.exists():
            raise FileNotFoundError(f"Source dialogue does not exist: {config.source_dialogue}")

    def select(self) -> dict[str, Any]:
        pipeline = self.context.pipeline
        force = self.context.force
        destination_movie, source_movie = pipeline.inspect(force=force)
        source_events = pipeline.extract_source_dialogue(force=force, source_movie=source_movie)
        filtered_source_events = pipeline.filter_source_dialogue_from_events(source_events, force=force)
        clip_library = pipeline.build_clip_library_from_events(filtered_source_events, force=force)
        destination_timeline = pipeline.detect_destination_timeline(force=force, dest_movie=destination_movie)
        filtered_destination_timeline = pipeline.filter_destination_timeline_from_timeline(destination_timeline, force=force)
        visual = pipeline.analyze_visual(force=force, dest_movie=destination_movie)
        source_performances = pipeline.build_source_performances(source_events=filtered_source_events, force=force)
        destination_performances = pipeline.build_destination_performances(
            timeline=filtered_destination_timeline,
            visual=visual,
            force=force,
        )
        self._selections = {
            "destination_movie": destination_movie,
            "source_movie": source_movie,
            "source_events": source_events,
            "filtered_source_events": filtered_source_events,
            "clip_library": clip_library,
            "destination_timeline": destination_timeline,
            "filtered_destination_timeline": filtered_destination_timeline,
            "visual": visual,
            "source_performances": source_performances,
            "destination_performances": destination_performances,
        }
        plan = build_movie_masher_plan(
            root=pipeline.config.root,
            destination_movie=destination_movie,
            source_movie=source_movie,
            clip_library=clip_library,
            destination_timeline=filtered_destination_timeline,
            visual=visual,
            source_performances=source_performances,
            destination_performances=destination_performances,
            output_dir=pipeline.config.output_dir,
            max_time_stretch=pipeline.config.max_time_stretch,
        )
        self._plan_path = write_transformation_plan(
            plan=plan,
            output_path=pipeline.config.output_dir / self.metadata.id / "transformation_plan.json",
            latest_path=pipeline.config.output_dir / "transformation_plan.json",
            schemas_dir=pipeline.schemas_dir,
        )
        return self._selections

    def transform(self, selections: dict[str, Any]) -> dict[str, Any]:
        schedule = self.context.pipeline.schedule_from_artifacts(
            library=selections["clip_library"],
            timeline=selections["filtered_destination_timeline"],
            visual=selections["visual"],
            destination_performances=selections.get("destination_performances"),
            source_performances=selections.get("source_performances"),
            force=self.context.force,
        )
        self._transformed = {"schedule": schedule}
        return self._transformed

    def validate(self, transformed: dict[str, Any]) -> None:
        schedule = transformed.get("schedule", {})
        if not schedule.get("mappings"):
            raise ValueError("Transposition produced no schedule mappings.")

    def render(self, transformed: dict[str, Any]) -> dict[str, Path]:
        pipeline = self.context.pipeline
        force = self.context.force
        audio = pipeline.render_audio_from_schedule(
            schedule=transformed["schedule"],
            dest_movie=self._selections["destination_movie"],
            force=force,
        )
        video = pipeline.render_video_from_audio(audio=audio, force=force)
        return {
            "audio": audio,
            "video": video,
        }

    def generate_report(self, result: TransformationResult) -> Path | None:
        from .report import write_transformation_report

        if self._plan_path is not None:
            result.artifacts["transformation_plan"] = self._plan_path
        report = write_transformation_report(
            metadata=self.metadata,
            pipeline=self.context.pipeline,
            result=result,
        )
        self.context.pipeline.generate_reports(
            destination_movie=self._selections.get("destination_movie"),
            source_movie=self._selections.get("source_movie"),
            source_events=self._selections.get("source_events"),
            filtered_source_events=self._selections.get("filtered_source_events"),
            clip_library=self._selections.get("clip_library"),
            destination_timeline=self._selections.get("destination_timeline"),
            filtered_destination_timeline=self._selections.get("filtered_destination_timeline"),
            schedule=self._transformed.get("schedule"),
            visual=self._selections.get("visual"),
            source_performances=self._selections.get("source_performances"),
            destination_performances=self._selections.get("destination_performances"),
            transformation_plan=self._plan_path,
        )
        return report
