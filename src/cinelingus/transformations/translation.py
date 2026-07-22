from __future__ import annotations

from pathlib import Path
from typing import Any

from ..filter_lab.acceptance import validate_filter_output, validate_schedule_quality
from ..filter_lab.multiworld import MultiworldPipeline
from ..filter_lab.registry import default_filter_registry
from ..montage import build_full_timeline_plan, build_montage_render_acceptance, rebase_schedule_to_montage
from ..mutations import render_mutation_media  # compatibility export for older integrations
from ..tools import ffprobe_json
from ..transformation_plan import build_translation_plan, write_transformation_plan
from .base import Transformation, TransformationMetadata, TransformationResult


class TranslationTransformation(Transformation):
    metadata = TransformationMetadata(
        id="translation",
        display_name="Translation",
        description="Transfer spoken performances across two films under the Translation law.",
        required_inputs=("films",),
        generated_outputs=("replacement_dialogue.wav", "translation_output.mp4", "transformation_report.json"),
        supported_modes=("fast_preview", "balanced", "quality"),
        version="1.0",
    )

    def __init__(self, context):
        super().__init__(context)
        self._selections: dict[str, Any] = {}
        self._transformed: dict[str, Any] = {}
        self._plan_path: Path | None = None
        self._montage_plan_path: Path | None = None
        self._montage_acceptance_path: Path | None = None
        self._filter_acceptance_path: Path | None = None
        self._multiworld: MultiworldPipeline | None = None

    def validate_inputs(self) -> None:
        config = self.context.pipeline.config
        definition = default_filter_registry().get("translation")
        definition.validate_film_count(len(config.films))
        for film in config.films:
            if not film.exists():
                raise FileNotFoundError(f"Film does not exist: {film}")
        self._multiworld = MultiworldPipeline(
            definition,
            config.films,
            seed=int(self.context.parameters.get("seed", 1)),
            stage_callback=getattr(
                self.context.pipeline,
                "_publish_multiworld_stage",
                lambda stage: self.context.pipeline.logger.info(f"multiworld stage: {stage}"),
            ),
        )

    def select(self) -> dict[str, Any]:
        pipeline = self.context.pipeline
        force = self.context.force
        destination_movie, source_movie = pipeline.inspect(force=force)
        assert self._multiworld is not None
        inspections = {self._multiworld.state.films[0].id: destination_movie, self._multiworld.state.films[1].id: source_movie}
        self._multiworld.inspect_films(lambda film: inspections[film.id])
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
        self._multiworld.create_shared_timeline(
            lambda state: {
                "anchor_film_id": state.anchor.id,
                "behavior": state.definition.anchor_behavior,
                "duration": destination_movie.get("duration"),
                "speaking_windows": filtered_destination_timeline.get("windows", []),
            }
        )
        self._multiworld.construct_world_model(
            lambda state: {
                "cinematic_law": state.definition.cinematic_law,
                "anchor_film_id": state.anchor.id,
                "films": [film.to_dict() for film in state.films],
                "shared_timeline": state.shared_timeline,
                "dialogue_events": filtered_source_events,
                "performances": {"anchor": destination_performances, "donor": source_performances},
                "scenes": {"anchor": destination_performances, "donor": source_performances},
                "shots": visual,
            }
        )
        plan = build_translation_plan(
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
        assert self._multiworld is not None
        law_result = self._multiworld.apply_cinematic_law(
            lambda _state: {"schedule": self.context.pipeline.schedule_from_artifacts(
                library=selections["clip_library"],
                timeline=selections["filtered_destination_timeline"],
                visual=selections["visual"],
                destination_performances=selections.get("destination_performances"),
                source_performances=selections.get("source_performances"),
                force=self.context.force,
            )}
        )
        decisions = self._multiworld.generate_replacement_decisions(lambda _state: law_result)
        schedule = decisions["schedule"]
        schedule["multiworld"] = {
            "cinematic_law": self._multiworld.state.definition.cinematic_law,
            "anchor_film_id": self._multiworld.state.anchor.id,
            "film_ids": [film.id for film in self._multiworld.state.films],
            "completed_stages": list(self._multiworld.state.completed_stages),
        }
        pipeline = self.context.pipeline
        definition = default_filter_registry().get("multiworld.translation")
        destination_movie = selections["destination_movie"]
        source_movie = selections["source_movie"]
        # Container duration can include padding beyond the primary audio or
        # video stream. The rendered mix and mux are bounded by those streams.
        destination_duration = min(
            _primary_stream_duration(destination_movie, "video"),
            _primary_stream_duration(destination_movie, "audio"),
        )
        source_duration = _primary_stream_duration(source_movie, "audio")
        shot_ids = [
            str(row["id"])
            for row in selections["visual"].get("shots", {}).get("shots", [])
            if float(row.get("start", 0.0)) < min(destination_duration, source_duration)
        ]
        montage_plan = build_full_timeline_plan(
            filter_id=definition.id,
            filter_contract_version=definition.version,
            anchor_source_id=self._multiworld.state.anchor.id,
            anchor_media_hash=pipeline.destination.media_hash,
            anchor_duration=destination_duration,
            supporting_audio_durations=[source_duration],
            shot_ids=shot_ids,
            random_seed=int(self.context.parameters.get("seed", 1)),
            governing_relationship=definition.cinematic_law,
            laws={
                "visual": "COMPLETE_ANCHOR_TIMELINE_FROM_ZERO",
                "temporal": definition.preserves.get("time", "ANCHOR_CHRONOLOGY_PRESERVED"),
                "dialogue": definition.operational_description,
                "requested_audio": "TRANSLATION_LAW",
                "actual_audio_method": "CONTINUOUS_SOURCE_SOUNDTRACK_BED",
            },
            schedule=schedule,
        )
        schedule = rebase_schedule_to_montage(schedule, montage_plan)
        schedule["montage_native"] = True
        schedule["full_timeline_native"] = True
        schedule["input_scope"] = "complete_media_files"
        schedule["duration_policy"] = dict(montage_plan["duration_resolution"])
        output_dir = pipeline.config.output_dir / self.metadata.id
        self._montage_plan_path = output_dir / "montage_plan.json"
        pipeline._write_and_validate("montage_plan", self._montage_plan_path, montage_plan)
        self._transformed = {"schedule": schedule, "montage_plan": montage_plan}
        return self._transformed

    def validate(self, transformed: dict[str, Any]) -> None:
        schedule = transformed.get("schedule", {})
        if not schedule.get("mappings"):
            raise ValueError("Translation produced no schedule mappings.")
        validate_schedule_quality(schedule)
        assert self._multiworld is not None
        self._multiworld.review(lambda _state: {"status": "pass", "mapping_count": len(schedule["mappings"])})

    def render(self, transformed: dict[str, Any]) -> dict[str, Path]:
        pipeline = self.context.pipeline
        assert self._multiworld is not None
        publish_runtime_stage = getattr(pipeline, "_publish_runtime_stage", lambda _stage: None)
        def render_world(_state):
            output_dir = pipeline.config.output_dir / self.metadata.id
            audio = output_dir / "replacement_dialogue.wav"
            video = output_dir / "translation_output.mp4"
            publish_runtime_stage("render_audio")
            audio = pipeline.render_audio_from_schedule(
                schedule=transformed["schedule"],
                dest_movie={
                    **self._selections["destination_movie"],
                    "duration": float(transformed["montage_plan"]["actual_duration"]),
                },
                force=self.context.force,
                output_path=audio,
                persist_schedule=False,
            )
            publish_runtime_stage("render_video")
            video = pipeline.render_video_from_audio(
                audio=audio,
                force=self.context.force,
                output_path=video,
                duration=float(transformed["montage_plan"]["actual_duration"]),
            )
            publish_runtime_stage("finalize")
            self._filter_acceptance_path = output_dir / "filter_acceptance.json"
            validate_filter_output(
                filter_id="multiworld.translation",
                schedule=transformed["schedule"],
                final_video=video,
                replacement_audio=audio,
                output_path=self._filter_acceptance_path,
                schemas_dir=pipeline.schemas_dir,
            )
            self._montage_acceptance_path = output_dir / "montage_render_acceptance.json"
            acceptance = build_montage_render_acceptance(
                plan=transformed["montage_plan"],
                encoded_probe=ffprobe_json(video),
                output_path=self._montage_acceptance_path,
            )
            pipeline._write_and_validate("montage_render_acceptance", self._montage_acceptance_path, acceptance)
            return {"audio": audio, "video": video}

        return self._multiworld.render(render_world)

    def generate_report(self, result: TransformationResult) -> Path | None:
        from .report import write_transformation_report

        if self._plan_path is not None:
            result.artifacts["transformation_plan"] = self._plan_path
        if self._montage_plan_path is not None:
            result.artifacts["montage_plan"] = self._montage_plan_path
        if self._montage_acceptance_path is not None:
            result.artifacts["montage_render_acceptance"] = self._montage_acceptance_path
        if self._filter_acceptance_path is not None:
            result.artifacts["filter_acceptance"] = self._filter_acceptance_path
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


def _primary_stream_duration(movie: dict[str, Any], codec_type: str) -> float:
    fallback = float(movie["duration"])
    streams = [row for row in movie.get("streams", []) if row.get("codec_type") == codec_type]
    if not streams:
        return fallback
    primary = next((row for row in streams if (row.get("disposition") or {}).get("default") == 1), streams[0])
    try:
        duration = float(primary.get("duration") or 0.0)
    except (TypeError, ValueError):
        return fallback
    return duration if duration > 0.0 else fallback
