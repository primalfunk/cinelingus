from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any

from .audio_provenance import compare_wav_audio, extract_audio_for_provenance, verify_audio_provenance
from .cache import CacheEntry, ensure_cache, update_manifest
from .cir import build_cinematic_index
from .clips import slice_clips
from .config import AppConfig
from .detect import detect_voice_windows, extract_analysis_audio, write_dialogue_events, write_timeline
from .dialogue_reel import build_dialogue_scene_artifact, build_scene_pair_candidates, build_vignette_reel_report, build_vignette_schedule, offset_vignette_schedule, select_vignette_reel
from .filters import FilterConfig, filter_dialogue_events, filter_timeline, usable_rows
from .filter_lab.registry import default_filter_registry
from .filter_lab.integration import build_strategy_schedule, write_filter_artifacts
from .filter_lab.acceptance import validate_filter_output
from .logging import RunLogger
from .media import inspect_media
from .mutations import (
    MUTATIONS,
    build_drift_schedule,
    build_echo_schedule,
    build_mutation_plan,
    build_mutation_report,
    build_self_shuffle_schedule,
    get_mutation,
    render_mutation_media,
)
from .performance import build_performances, performance_windows
from .performance_library import build_performance_library
from .performance_diagnostics import build_performance_diagnostics
from .performance_report import build_performance_placement_report
from .problem_report import build_problem_region_report
from .presets import Preset, load_preset
from .progress import ProgressState, format_progress_status
from .publish import publish_single_video
from .render import build_preview_schedule, concat_media_files, concat_wav_files, extract_video_segment, mux_video, mux_video_segment, preview_bounds, render_dialogue_wav, render_schedule_over_original_audio, scheduled_audio_duration
from .reports import build_run_report, write_report_files
from .remix_modes import load_remix_mode_registry
from .short_form import build_short_remix_candidates, build_short_remix_report, build_short_remix_schedule, expanded_short_window, select_best_short_candidate
from .review_analysis import build_review_analysis
from .schedule import build_schedule
from .shot_context import annotate_windows_with_shots, build_visual_schedule_report
from .speakers import annotate_artifact_speakers, apply_speaker_mapping_to_schedule, build_speaker_map, build_speaker_mapping, enrich_performances_with_speakers, speaker_map_content_signature, speaker_map_has_real_diarization
from .taste import build_editorial_highlights, default_taste_profile
from .transformations import TransformationContext, TransformationResult, default_registry
from .util import read_json, stable_hash, write_json
from .validation import validate_artifact
from .visual import build_visual_report, detect_shots
from .whisper_backend import transcribe_with_whisper


class Pipeline:
    def __init__(self, config: AppConfig, cancel_check=None, stage_callback=None) -> None:
        self.config = config
        self.cancel_check = cancel_check
        self.stage_callback = stage_callback
        self.diarization_attempt_registry: set[str] = set()
        self.logger = RunLogger(config.output_dir / "run.log")
        self.destination = ensure_cache(config.cache_dir, config.destination_video, "destination_video")
        self.source = ensure_cache(config.cache_dir, config.source_dialogue, "source_dialogue")
        self.schemas_dir = config.root / "schemas"

    def _publish_diarization_stage(self, stage: str) -> None:
        self.logger.info(f"Diarization stage: {stage}")
        if self.stage_callback is not None:
            self.stage_callback(stage)

    def _clear_diarization_stage(self) -> None:
        if self.stage_callback is not None:
            self.stage_callback("")

    def _check_cancel(self) -> None:
        cancel_check = getattr(self, "cancel_check", None)
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Run cancelled by user.")

    def inspect(self, *, force: bool = False) -> tuple[dict, dict]:
        self._check_cancel()
        return self._inspect_one(self.destination, force=force), self._inspect_one(self.source, force=force)

    def extract_source_dialogue(self, *, force: bool = False, source_movie: dict | None = None) -> dict:
        self._check_cancel()
        if source_movie is None:
            _, source_movie = self.inspect()
        audio_path = self.source.cache_dir / "analysis_audio.wav"
        output_path = self.source.cache_dir / "dialogue_events.json"
        signature = self._signature("dialogue_events", self.source.media_hash)
        cached = self._load_current("dialogue_events", output_path, self.source.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused source dialogue events: {output_path}")
            return cached

        self.logger.info("extracting source analysis audio")
        extract_analysis_audio(self.source.media_path, audio_path)
        if self.config.speech_backend == "whisper":
            self.logger.info(
                f"transcribing source dialogue with Whisper mode={self.config.transcription_mode} model={self.config.whisper_model}"
            )
            events = transcribe_with_whisper(
                audio_path=audio_path,
                media_hash=self.source.media_hash,
                output_path=output_path,
                model_name=self.config.whisper_model,
                language=self.config.whisper_language,
                artifact_type="dialogue_events",
                transcription_mode=self.config.transcription_mode,
                quick_test_seconds=self.config.quick_test_seconds,
            )
        else:
            self.logger.info("detecting source dialogue with FFmpeg silence fallback")
            windows = detect_voice_windows(
                audio_path,
                source_movie["duration"],
                noise_db=self.config.silence_noise_db,
                min_silence=self.config.silence_min_duration,
                min_speech=self.config.min_speech_duration,
                merge_gap=self.config.merge_gap,
            )
            events = write_dialogue_events(output_path, self.source.media_hash, windows)
        events["config_signature"] = signature
        events["input_role"] = "source_dialogue"
        self._write_and_validate("dialogue_events", output_path, events)
        update_manifest(
            self.source,
            "dialogue_extracted",
            {
                "movie": str(self.source.cache_dir / "movie.json"),
                "analysis_audio": str(audio_path),
                "dialogue_events": str(output_path),
            },
        )
        self.logger.info(f"source dialogue events: {len(events['events'])}")
        return events

    def filter_source_dialogue(self, *, force: bool = False) -> dict:
        self._check_cancel()
        raw = self.extract_source_dialogue(force=False)
        if self.config.enable_speaker_awareness:
            speaker_map = self.build_source_speaker_map(source_events=raw, force=force)
            raw = annotate_artifact_speakers(raw, speaker_map, collection_key="events")
        return self.filter_source_dialogue_from_events(raw, force=force)

    def filter_source_dialogue_from_events(self, raw: dict, *, force: bool = False) -> dict:
        if self.config.enable_speaker_awareness and raw.get("events") and not any(event.get("speaker_id") for event in raw.get("events", [])):
            speaker_map = self.build_source_speaker_map(source_events=raw, force=force)
            raw = annotate_artifact_speakers(raw, speaker_map, collection_key="events")
        output_path = self.source.cache_dir / "filtered_dialogue_events.json"
        signature = self._signature(
            "filtered_dialogue_events",
            self.source.media_hash,
            raw.get("config_signature"),
            raw.get("speaker_map_content_signature"),
        )
        cached = self._load_current("filtered_dialogue_events", output_path, self.source.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused filtered source dialogue: {output_path}")
            return cached
        self.logger.info("filtering source dialogue events")
        filtered = filter_dialogue_events(raw, self._filter_config(), output_path)
        filtered["config_signature"] = signature
        self._write_and_validate("filtered_dialogue_events", output_path, filtered)
        update_manifest(self.source, "dialogue_filtered", {"filtered_dialogue_events": str(output_path)})
        stats = filtered["filter_stats"]
        self.logger.info(f"filtered source dialogue: {stats['usable_count']} usable / {stats['raw_count']} raw")
        return filtered

    def build_source_speaker_map(self, *, source_events: dict | None = None, force: bool = False) -> dict:
        source_events = source_events or self.extract_source_dialogue(force=False)
        output_path = self.source.cache_dir / "speaker_map.json"
        signature = self._signature("speaker_map", self.source.media_hash)
        cached = self._load_current("speaker_map", output_path, self.source.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused source speaker map: {output_path}")
            return cached
        self.logger.info(f"building source speaker map backend={self.config.speaker_diarization_backend}")
        audio_path = self.source.cache_dir / "analysis_audio.wav"
        if not audio_path.exists():
            extract_analysis_audio(self.source.media_path, audio_path)
        speaker_map = build_speaker_map(
            media_hash=self.source.media_hash,
            speech_items=source_events.get("events", []),
            output_path=output_path,
            config_signature=signature,
            audio_path=audio_path,
            backend=self.config.speaker_diarization_backend,
            model_name=self.config.speaker_diarization_model,
            device=self.config.speaker_diarization_device,
            role="source",
            stage_callback=self._publish_diarization_stage,
            log=self.logger.info,
            attempt_registry=self.diarization_attempt_registry,
        )
        validate_artifact("speaker_map", output_path, self.schemas_dir)
        update_manifest(self.source, "speakers_mapped", {"speaker_map": str(output_path)})
        diagnostics = speaker_map.get("diagnostics", {})
        self.logger.info(
            f"source speakers: {speaker_map.get('speaker_count', 0)} "
            f"backend={diagnostics.get('effective_backend', speaker_map.get('diarization_tool'))} "
            f"coverage={diagnostics.get('labeled_item_count', 0)}/{diagnostics.get('speech_item_count', len(source_events.get('events', [])))} "
            f"status={diagnostics.get('status', 'unknown')}"
        )
        self._clear_diarization_stage()
        return speaker_map

    def build_destination_speaker_map(self, *, timeline: dict | None = None, force: bool = False) -> dict:
        timeline = timeline or self.detect_destination_timeline(force=False)
        output_path = self.destination.cache_dir / "speaker_map.json"
        signature = self._signature("speaker_map", self.destination.media_hash)
        cached = self._load_current("speaker_map", output_path, self.destination.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused destination speaker map: {output_path}")
            return cached
        self.logger.info(f"building destination speaker map backend={self.config.speaker_diarization_backend}")
        audio_path = self.destination.cache_dir / "analysis_audio.wav"
        if not audio_path.exists():
            extract_analysis_audio(self.destination.media_path, audio_path)
        speaker_map = build_speaker_map(
            media_hash=self.destination.media_hash,
            speech_items=timeline.get("windows", []),
            output_path=output_path,
            config_signature=signature,
            audio_path=audio_path,
            backend=self.config.speaker_diarization_backend,
            model_name=self.config.speaker_diarization_model,
            device=self.config.speaker_diarization_device,
            role="destination",
            stage_callback=self._publish_diarization_stage,
            log=self.logger.info,
            attempt_registry=self.diarization_attempt_registry,
        )
        validate_artifact("speaker_map", output_path, self.schemas_dir)
        update_manifest(self.destination, "speakers_mapped", {"speaker_map": str(output_path)})
        diagnostics = speaker_map.get("diagnostics", {})
        self.logger.info(
            f"destination speakers: {speaker_map.get('speaker_count', 0)} "
            f"backend={diagnostics.get('effective_backend', speaker_map.get('diarization_tool'))} "
            f"coverage={diagnostics.get('labeled_item_count', 0)}/{diagnostics.get('speech_item_count', len(timeline.get('windows', [])))} "
            f"status={diagnostics.get('status', 'unknown')}"
        )
        self._clear_diarization_stage()
        return speaker_map

    def build_clip_library(self, *, force: bool = False) -> dict:
        self._check_cancel()
        events = self.filter_source_dialogue(force=False)
        return self.build_clip_library_from_events(events, force=force)

    def build_clip_library_from_events(self, events: dict, *, force: bool = False) -> dict:
        output_path = self.source.cache_dir / "clip_library.json"
        signature = self._signature("clip_library", self.source.media_hash, events.get("config_signature"))
        cached = self._load_current("clip_library", output_path, self.source.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused clip library: {output_path}")
            return cached

        source_events = usable_rows(events["events"])
        self.logger.info(f"slicing source dialogue clips from {len(source_events)} usable events")
        library = slice_clips(
            self.source.media_path,
            self.source.media_hash,
            source_events,
            self.source.cache_dir / "clips",
            output_path,
        )
        library["config_signature"] = signature
        self._write_and_validate("clip_library", output_path, library)
        update_manifest(
            self.source,
            "clips_built",
            {"clip_library": str(output_path), "clips_dir": str(self.source.cache_dir / "clips")},
        )
        self.logger.info(f"clips: {len(library['clips'])}")
        return library

    def detect_destination_timeline(self, *, force: bool = False, dest_movie: dict | None = None) -> dict:
        self._check_cancel()
        if dest_movie is None:
            dest_movie, _ = self.inspect()
        audio_path = self.destination.cache_dir / "analysis_audio.wav"
        output_path = self.destination.cache_dir / "timeline.json"
        signature = self._signature("timeline", self.destination.media_hash)
        cached = self._load_current("timeline", output_path, self.destination.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused destination timeline: {output_path}")
            return cached

        self.logger.info("extracting destination analysis audio")
        extract_analysis_audio(self.destination.media_path, audio_path)
        if self.config.speech_backend == "whisper":
            self.logger.info(
                f"transcribing destination timeline with Whisper mode={self.config.transcription_mode} model={self.config.whisper_model}"
            )
            timeline = transcribe_with_whisper(
                audio_path=audio_path,
                media_hash=self.destination.media_hash,
                output_path=output_path,
                model_name=self.config.whisper_model,
                language=self.config.whisper_language,
                artifact_type="timeline",
                transcription_mode=self.config.transcription_mode,
                quick_test_seconds=self.config.quick_test_seconds,
            )
        else:
            self.logger.info("detecting destination timeline with FFmpeg silence fallback")
            windows = detect_voice_windows(
                audio_path,
                dest_movie["duration"],
                noise_db=self.config.silence_noise_db,
                min_silence=self.config.silence_min_duration,
                min_speech=self.config.min_speech_duration,
                merge_gap=self.config.merge_gap,
            )
            timeline = write_timeline(output_path, self.destination.media_hash, windows)
        timeline["config_signature"] = signature
        timeline["input_role"] = "destination_video"
        self._write_and_validate("timeline", output_path, timeline)
        update_manifest(
            self.destination,
            "timeline_detected",
            {
                "movie": str(self.destination.cache_dir / "movie.json"),
                "analysis_audio": str(audio_path),
                "timeline": str(output_path),
            },
        )
        self.logger.info(f"destination windows: {len(timeline['windows'])}")
        return timeline

    def filter_destination_timeline(self, *, force: bool = False) -> dict:
        self._check_cancel()
        raw = self.detect_destination_timeline(force=False)
        if self.config.enable_speaker_awareness:
            speaker_map = self.build_destination_speaker_map(timeline=raw, force=force)
            raw = annotate_artifact_speakers(raw, speaker_map, collection_key="windows")
        return self.filter_destination_timeline_from_timeline(raw, force=force)

    def filter_destination_timeline_from_timeline(self, raw: dict, *, force: bool = False) -> dict:
        if self.config.enable_speaker_awareness and raw.get("windows") and not any(window.get("speaker_id") for window in raw.get("windows", [])):
            speaker_map = self.build_destination_speaker_map(timeline=raw, force=force)
            raw = annotate_artifact_speakers(raw, speaker_map, collection_key="windows")
        output_path = self.destination.cache_dir / "filtered_timeline.json"
        signature = self._signature(
            "filtered_timeline",
            self.destination.media_hash,
            raw.get("config_signature"),
            raw.get("speaker_map_content_signature"),
        )
        cached = self._load_current("filtered_timeline", output_path, self.destination.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused filtered destination timeline: {output_path}")
            return cached
        self.logger.info("filtering destination timeline")
        filtered = filter_timeline(raw, self._filter_config(), output_path)
        filtered["config_signature"] = signature
        self._write_and_validate("filtered_timeline", output_path, filtered)
        update_manifest(self.destination, "timeline_filtered", {"filtered_timeline": str(output_path)})
        stats = filtered["filter_stats"]
        self.logger.info(f"filtered destination timeline: {stats['usable_count']} usable / {stats['raw_count']} raw")
        return filtered

    def analyze_visual(self, *, force: bool = False, dest_movie: dict | None = None) -> dict[str, dict]:
        self._check_cancel()
        if dest_movie is None:
            dest_movie, _ = self.inspect(force=False)
        shots_path = self.destination.cache_dir / "shots.json"
        visual_report_path = self.destination.cache_dir / "visual_report.json"
        signature = self._signature("shots", self.destination.media_hash)
        shots = self._load_current("shots", shots_path, self.destination.media_hash, signature, force)
        if shots:
            self.logger.info(f"reused visual shot analysis: {shots_path}")
        else:
            self.logger.info("detecting destination visual shot boundaries")
            shots = detect_shots(
                media_path=self.destination.media_path,
                media_hash=self.destination.media_hash,
                duration=float(dest_movie["duration"]),
                output_path=shots_path,
                threshold=self.config.visual_scene_threshold,
                min_shot_duration=self.config.visual_min_shot_duration,
                config_signature=signature,
            )
            self._write_and_validate("shots", shots_path, shots)

        report = None
        if not force and visual_report_path.exists():
            try:
                cached_report = validate_artifact("visual_report", visual_report_path, self.schemas_dir)
                if cached_report.get("media_hash") == self.destination.media_hash and cached_report.get("config_signature") == signature:
                    report = cached_report
                    self.logger.info(f"reused visual report: {visual_report_path}")
            except ValueError as exc:
                self.logger.info(f"invalid cached visual report, regenerating: {exc}")
        if report is None:
            report = build_visual_report(shots_artifact=shots, movie=dest_movie, output_path=visual_report_path)
            self._write_and_validate("visual_report", visual_report_path, report)

        update_manifest(
            self.destination,
            "visual_analyzed",
            {"shots": str(shots_path), "visual_report": str(visual_report_path)},
        )
        self.logger.info(f"visual shots: {len(shots['shots'])}")
        return {"shots": shots, "visual_report": report}

    def build_source_performances(
        self,
        *,
        source_events: dict | None = None,
        force: bool = False,
    ) -> dict:
        self._check_cancel()
        source_events = source_events or self.filter_source_dialogue(force=False)
        output_path = self.source.cache_dir / "performance.json"
        signature = self._signature("performance", self.source.media_hash, source_events.get("config_signature"), "source_dialogue")
        cached = self._load_current("performance", output_path, self.source.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused source performances: {output_path}")
            return cached
        events = source_events.get("events", [])
        performances = build_performances(
            media_hash=self.source.media_hash,
            role="source_dialogue",
            output_path=output_path,
            speaking_windows=events,
            dialogue_events=events,
            max_pause=2.0,
            config_signature=signature,
        )
        if self.config.enable_speaker_awareness:
            speaker_map = self.build_source_speaker_map(source_events=source_events, force=False)
            performances = enrich_performances_with_speakers(performances, speaker_map)
        self._write_and_validate("performance", output_path, performances)
        update_manifest(self.source, "performances_built", {"performance": str(output_path)})
        self.logger.info(f"source performances: {len(performances['performances'])}")
        return performances


    def build_source_performance_library(
        self,
        *,
        source_performances: dict | None = None,
        clip_library: dict | None = None,
        force: bool = False,
    ) -> dict:
        clip_library = clip_library or self.build_clip_library(force=False)
        source_performances = source_performances or self.build_source_performances(force=False)
        output_path = self.source.cache_dir / "performance_library.json"
        signature = self._signature(
            "performance_library",
            self.source.media_hash,
            source_performances.get("config_signature"),
            clip_library.get("config_signature"),
        )
        cached = self._load_current("performance_library", output_path, self.source.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused source performance library: {output_path}")
            return cached
        library = build_performance_library(
            media_hash=self.source.media_hash,
            performances=source_performances,
            clips=clip_library.get("clips", []),
            output_path=output_path,
            config_signature=signature,
        )
        self._write_and_validate("performance_library", output_path, library)
        update_manifest(self.source, "performance_library_built", {"performance_library": str(output_path)})
        self.logger.info(f"source performance library: {len(library['performances'])}")
        return library

    def build_destination_performances(
        self,
        *,
        timeline: dict | None = None,
        visual: dict | None = None,
        force: bool = False,
    ) -> dict:
        self._check_cancel()
        timeline = timeline or self.filter_destination_timeline(force=False)
        visual = visual or self.analyze_visual(force=False)
        output_path = self.destination.cache_dir / "performance.json"
        signature = self._signature(
            "performance",
            self.destination.media_hash,
            timeline.get("config_signature"),
            visual.get("shots", {}).get("config_signature"),
            "destination_video",
        )
        cached = self._load_current("performance", output_path, self.destination.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused destination performances: {output_path}")
            return cached
        performances = build_performances(
            media_hash=self.destination.media_hash,
            role="destination_video",
            output_path=output_path,
            speaking_windows=timeline.get("windows", []),
            shots=visual.get("shots", {}).get("shots", []),
            max_pause=2.0,
            config_signature=signature,
        )
        if self.config.enable_speaker_awareness:
            speaker_map = self.build_destination_speaker_map(timeline=timeline, force=False)
            performances = enrich_performances_with_speakers(performances, speaker_map)
        self._write_and_validate("performance", output_path, performances)
        update_manifest(self.destination, "performances_built", {"performance": str(output_path)})
        self.logger.info(f"destination performances: {len(performances['performances'])}")
        return performances

    def build_speaker_mapping(self, *, force: bool = False) -> dict | None:
        if not self.config.enable_speaker_awareness:
            return None
        source_path = self.source.cache_dir / "speaker_map.json"
        destination_path = self.destination.cache_dir / "speaker_map.json"
        if not source_path.exists() or not destination_path.exists():
            return None
        source_speaker_map = validate_artifact("speaker_map", source_path, self.schemas_dir)
        destination_speaker_map = validate_artifact("speaker_map", destination_path, self.schemas_dir)
        if self.config.prefer_high_speaker_confidence:
            fallback_roles = [
                role for role, speaker_map in (("source", source_speaker_map), ("destination", destination_speaker_map))
                if not speaker_map_has_real_diarization(speaker_map)
            ]
            if fallback_roles:
                raise RuntimeError(
                    "Speaker-aware rendering requires real diarization; no validated Pyannote segments were available for "
                    + " and ".join(fallback_roles)
                    + ". Regenerate speaker maps after fixing Pyannote or disable high-confidence speaker preference."
                )
        output_path = self.config.output_dir / "speaker_mapping.json"
        signature = self._signature(
            "speaker_mapping",
            self.source.media_hash,
            self.destination.media_hash,
            speaker_map_content_signature(source_speaker_map),
            speaker_map_content_signature(destination_speaker_map),
        )
        if not force and output_path.exists():
            cached = validate_artifact("speaker_mapping", output_path, self.schemas_dir)
            if cached.get("config_signature") == signature:
                self.logger.info(f"reused speaker mapping: {output_path}")
                return cached
        self.logger.info("building experimental speaker mapping")
        mapping = build_speaker_mapping(
            source_speaker_map=source_speaker_map,
            destination_speaker_map=destination_speaker_map,
            output_path=output_path,
            config_signature=signature,
        )
        validate_artifact("speaker_mapping", output_path, self.schemas_dir)
        return mapping

    def schedule(self, *, force: bool = False) -> dict:
        self._check_cancel()
        source_events = self.filter_source_dialogue(force=False)
        library = self.build_clip_library_from_events(source_events, force=False)
        timeline = self.filter_destination_timeline(force=False)
        visual = self.analyze_visual(force=False)
        source_performances = self.build_source_performances(source_events=source_events, force=force)
        self.build_source_performance_library(source_performances=source_performances, clip_library=library, force=force)
        destination_performances = self.build_destination_performances(timeline=timeline, visual=visual, force=force)
        return self.schedule_from_artifacts(
            library=library,
            timeline=timeline,
            visual=visual,
            destination_performances=destination_performances,
            source_performances=source_performances,
            force=force,
        )

    def schedule_from_artifacts(
        self,
        *,
        library: dict,
        timeline: dict,
        visual: dict,
        destination_performances: dict | None = None,
        source_performances: dict | None = None,
        force: bool = False,
    ) -> dict:
        shots = visual["shots"]
        if source_performances is not None:
            self.build_source_performance_library(source_performances=source_performances, clip_library=library, force=force)
        output_path = self.destination.cache_dir / "replacement_schedule.json"
        visual_report_path = self.destination.cache_dir / "visual_schedule_report.json"
        signature = self._signature(
            "replacement_schedule",
            self.source.media_hash,
            self.destination.media_hash,
            library.get("config_signature"),
            timeline.get("config_signature"),
            shots.get("config_signature"),
            destination_performances.get("config_signature") if destination_performances else "",
            source_performances.get("config_signature") if source_performances else "",
        )
        cached = self._load_current("replacement_schedule", output_path, self.destination.media_hash, signature, force)
        if cached:
            self.logger.info(f"reused replacement schedule: {output_path}")
            if not visual_report_path.exists() or force:
                report_timeline = dict(timeline)
                report_timeline["windows"] = annotate_windows_with_shots(usable_rows(timeline["windows"]), shots.get("shots", []))
                report = build_visual_schedule_report(
                    shots_artifact=shots,
                    timeline=report_timeline,
                    schedule=cached,
                    output_path=visual_report_path,
                )
                self._write_and_validate("visual_schedule_report", visual_report_path, report)
            return cached

        if destination_performances is not None:
            windows = performance_windows(destination_performances)
            windows = _attach_performance_speech_windows(windows, timeline.get("windows", []))
        else:
            windows = annotate_windows_with_shots(usable_rows(timeline["windows"]), shots.get("shots", []))
        self.logger.info(f"building replacement schedule for {len(windows)} usable windows")
        schedule = build_schedule(
            clips=library["clips"],
            windows=windows,
            source_hash=self.source.media_hash,
            destination_hash=self.destination.media_hash,
            max_time_stretch=self.config.max_time_stretch,
            output_path=output_path,
            scheduling_mode=self.config.scheduling_mode,
            best_fit_lookahead=self.config.best_fit_lookahead,
            shot_boundary_mode=self.config.shot_boundary_mode,
            source_performances=source_performances,
            cinematic_filter=self.config.cinematic_filter,
        )
        speaker_mapping = self.build_speaker_mapping(force=force)
        if speaker_mapping is not None:
            schedule = apply_speaker_mapping_to_schedule(schedule, speaker_mapping)
        schedule["config_signature"] = signature
        self._write_and_validate("replacement_schedule", output_path, schedule)
        report_timeline = dict(timeline)
        report_timeline["windows"] = windows
        visual_schedule_report = build_visual_schedule_report(
            shots_artifact=shots,
            timeline=report_timeline,
            schedule=schedule,
            output_path=visual_report_path,
        )
        self._write_and_validate("visual_schedule_report", visual_report_path, visual_schedule_report)
        update_manifest(
            self.destination,
            "schedule_built",
            {"replacement_schedule": str(output_path), "visual_schedule_report": str(visual_report_path)},
        )
        self.logger.info(f"schedule mappings: {len(schedule['mappings'])}")
        return schedule

    def render_audio(self, *, force: bool = False) -> Path:
        schedule = self.schedule(force=False)
        dest_movie = self._inspect_one(self.destination, force=False)
        return self.render_audio_from_schedule(schedule=schedule, dest_movie=dest_movie, force=force)

    def render_audio_from_schedule(self, *, schedule: dict, dest_movie: dict, force: bool = False) -> Path:
        self._check_cancel()
        output = self.config.output_dir / "replacement_dialogue.wav"
        schedule_path = self.destination.cache_dir / "replacement_schedule.json"
        if output.exists() and not force and not _is_stale(output, schedule_path):
            self.logger.info(f"reused rendered audio: {output}")
            return output
        render_duration = float(dest_movie["duration"])
        self.logger.info(
            f"rendering dialogue-only replacement soundtrack duration={render_duration:.3f}s "
            f"mappings={len([mapping for mapping in schedule.get('mappings', []) if mapping.get('enabled', True)])}"
        )
        render_with_bed = hasattr(self.config, "original_duck_db")
        renderer = render_schedule_over_original_audio if render_with_bed else render_dialogue_wav
        render_kwargs = {"original_media": self.destination.media_path} if render_with_bed else {}
        renderer(
            **render_kwargs,
            schedule=schedule,
            duration=render_duration,
            output_path=output,
            sample_rate=self.config.render_sample_rate,
            channels=self.config.render_channels,
            target_lufs=self.config.target_lufs,
            fade_duration=self.config.audio_fade_duration,
            **({"mute_regions": _speech_mute_regions(schedule, padding=0.04, merge_gap=0.08, duration=render_duration),
                "duck_db": self.config.original_duck_db} if render_with_bed else {}),
        )
        self.logger.info(f"rendered audio: {output}")
        return output

    def render_video(self, *, force: bool = False) -> Path:
        audio = self.render_audio(force=force)
        return self.render_video_from_audio(audio=audio, force=force)

    def render_video_from_audio(self, *, audio: Path, force: bool = False) -> Path:
        self._check_cancel()
        output = self.config.output_dir / "movie_masher_output.mp4"
        if output.exists() and not force and not _is_stale(output, audio):
            self.logger.info(f"reused rendered video: {output}")
            return output
        self.logger.info("muxing final video")
        mux_video(destination_video=self.destination.media_path, dialogue_wav=audio, output_path=output)
        self.logger.info(f"rendered video: {output}")
        return output

    def render_preview(self, mapping_indices: list[int], *, video: bool = True) -> dict[str, Path | float]:
        if not mapping_indices:
            raise ValueError("Select at least one mapping to preview.")
        schedule = self.schedule(force=False)
        dest_movie = self._inspect_one(self.destination, force=False)
        mappings = schedule.get("mappings", [])
        selected = []
        for index in mapping_indices:
            if index < 0 or index >= len(mappings):
                raise ValueError(f"Preview mapping index out of range: {index}")
            selected.append(mappings[index])
        start_time, end_time = preview_bounds(selected, float(dest_movie["duration"]))
        duration = round(end_time - start_time, 3)
        preview_schedule = build_preview_schedule(schedule, selected, start_time)
        preview_dir = self.config.output_dir / "previews"
        stem = f"preview_{int(start_time * 1000):09d}_{int(end_time * 1000):09d}"
        audio_output = preview_dir / f"{stem}.wav"
        video_output = preview_dir / f"{stem}.mp4"
        self.logger.info(f"rendering preview audio start={start_time:.3f}s duration={duration:.3f}s mappings={len(selected)}")
        render_dialogue_wav(
            schedule=preview_schedule,
            duration=duration,
            output_path=audio_output,
            sample_rate=self.config.render_sample_rate,
            channels=self.config.render_channels,
            target_lufs=self.config.target_lufs,
            fade_duration=self.config.audio_fade_duration,
            batch_size=min(40, max(1, len(selected))),
        )
        result: dict[str, Path | float] = {"audio": audio_output, "start": start_time, "end": end_time, "duration": duration}
        if video:
            self.logger.info("muxing preview video")
            mux_video_segment(
                destination_video=self.destination.media_path,
                dialogue_wav=audio_output,
                output_path=video_output,
                start_time=start_time,
                duration=duration,
            )
            result["video"] = video_output
        return result

    def render_problem_region_previews(self, *, padding: float = 1.0, max_regions: int | None = None) -> dict[str, Any]:
        problem_path = self.config.output_dir / "problem_regions.json"
        if not problem_path.exists():
            self.generate_reports()
        problem_report = read_json(problem_path)
        final_video = self.config.output_dir / "movie_masher_output.mp4"
        if not final_video.exists():
            raise FileNotFoundError(f"Final output video is missing: {final_video}")
        dest_movie = self._inspect_one(self.destination, force=False)
        destination_duration = float(dest_movie.get("duration", 0.0) or 0.0)
        preview_dir = self.config.output_dir / "previews" / "problem_regions"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for stale in preview_dir.glob("problem_*.mp4"):
            stale.unlink()
        previews = []
        problems = problem_report.get("problems", [])
        if max_regions is not None:
            problems = problems[:max(0, max_regions)]
        for index, problem in enumerate(problems, start=1):
            start = max(0.0, float(problem.get("start", 0.0) or 0.0) - max(0.0, padding))
            end = float(problem.get("end", start + float(problem.get("duration", 0.0) or 0.0)) or start) + max(0.0, padding)
            if destination_duration > 0:
                end = min(destination_duration, end)
            if end <= start:
                end = min(destination_duration or start + 0.5, start + 0.5)
            duration = round(max(0.001, end - start), 3)
            stem = f"problem_{index:03d}_{problem.get('problem_type', 'region')}_{int(start * 1000):09d}_{int(end * 1000):09d}"
            output_path = preview_dir / f"{_safe_filename(stem)}.mp4"
            self.logger.info(f"rendering problem preview {index}/{len(problems)} start={start:.3f}s duration={duration:.3f}s")
            extract_video_segment(
                input_video=final_video,
                output_path=output_path,
                start_time=start,
                duration=duration,
            )
            previews.append(
                {
                    "index": index,
                    "problem_type": problem.get("problem_type"),
                    "severity": problem.get("severity"),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": duration,
                    "performance_id": problem.get("performance_id"),
                    "mapping_indices": problem.get("mapping_indices", []),
                    "reason": problem.get("reason"),
                    "path": str(output_path),
                }
            )
        manifest = {
            "schema_version": "1.0",
            "source_problem_report": str(problem_path),
            "source_video": str(final_video),
            "preview_count": len(previews),
            "previews": previews,
        }
        manifest_path = preview_dir / "problem_region_previews.json"
        write_json(manifest_path, manifest)
        text_path = preview_dir / "problem_region_previews.txt"
        text_path.write_text(_format_problem_preview_manifest(manifest), encoding="utf-8")
        return {"manifest": manifest_path, "text": text_path, "directory": preview_dir, "previews": previews}

    def run_all(self, *, force: bool = False) -> Path:
        result = self.execute_transformation("movie_masher", force=force)
        video = result.outputs["video"]
        try:
            previews = self.render_problem_region_previews(max_regions=10)
            self.logger.info(f"problem preview clips: {len(previews.get('previews', []))}")
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            self.logger.info(f"problem preview generation skipped: {exc}")
        if not hasattr(self, "config"):
            return video
        return publish_single_video(video=video, output_dir=self.config.output_dir, process="movie-masher")


    def execute_transformation(
        self,
        transformation_id: str,
        *,
        force: bool = False,
        parameters: dict[str, Any] | None = None,
    ):
        registry = default_filter_registry()
        resolved, migration = registry.resolve_id(transformation_id)
        definition = registry.get(resolved)
        if migration:
            self.logger.info(migration)
        implementation_key = definition.implementation_key
        if not definition.implemented or not implementation_key:
            raise ValueError(f"{definition.name} is in development and cannot be executed.")
        if definition.execution_mode == "scheduling_strategy":
            paths = self.run_mutation(implementation_key, force=force, parameters=parameters)
            artifacts = {key: value for key, value in paths.items() if key not in {"video", "audio"}}
            return TransformationResult(transformation_id=definition.id, outputs={key: paths[key] for key in ("video", "audio") if key in paths}, artifacts=artifacts)
        context = TransformationContext(pipeline=self, force=force, parameters=parameters or {})
        transformation = default_registry().create(implementation_key, context)
        return transformation.execute()

    def generate_reports(
        self,
        *,
        destination_movie: dict | None = None,
        source_movie: dict | None = None,
        source_events: dict | None = None,
        filtered_source_events: dict | None = None,
        clip_library: dict | None = None,
        destination_timeline: dict | None = None,
        filtered_destination_timeline: dict | None = None,
        schedule: dict | None = None,
        visual: dict | None = None,
        source_performances: dict | None = None,
        destination_performances: dict | None = None,
        transformation_plan: Path | None = None,
    ) -> dict[str, Path]:
        if schedule is None:
            self.schedule(force=False)
        destination_movie = destination_movie or validate_artifact("movie", self.destination.cache_dir / "movie.json", self.schemas_dir)
        source_movie = source_movie or validate_artifact("movie", self.source.cache_dir / "movie.json", self.schemas_dir)
        source_events = source_events or validate_artifact("dialogue_events", self.source.cache_dir / "dialogue_events.json", self.schemas_dir)
        filtered_source_events = filtered_source_events or validate_artifact(
            "filtered_dialogue_events", self.source.cache_dir / "filtered_dialogue_events.json", self.schemas_dir
        )
        clip_library = clip_library or validate_artifact("clip_library", self.source.cache_dir / "clip_library.json", self.schemas_dir)
        destination_timeline = destination_timeline or validate_artifact("timeline", self.destination.cache_dir / "timeline.json", self.schemas_dir)
        filtered_destination_timeline = filtered_destination_timeline or validate_artifact(
            "filtered_timeline", self.destination.cache_dir / "filtered_timeline.json", self.schemas_dir
        )
        schedule = schedule or validate_artifact(
            "replacement_schedule", self.destination.cache_dir / "replacement_schedule.json", self.schemas_dir
        )
        source_performance_path = self.source.cache_dir / "performance.json"
        destination_performance_path = self.destination.cache_dir / "performance.json"
        if source_performances is None and source_performance_path.exists():
            source_performances = validate_artifact("performance", source_performance_path, self.schemas_dir)
        if destination_performances is None and destination_performance_path.exists():
            destination_performances = validate_artifact("performance", destination_performance_path, self.schemas_dir)
        source_speaker_map = None
        destination_speaker_map = None
        source_speaker_path = self.source.cache_dir / "speaker_map.json"
        destination_speaker_path = self.destination.cache_dir / "speaker_map.json"
        if source_speaker_path.exists():
            source_speaker_map = validate_artifact("speaker_map", source_speaker_path, self.schemas_dir)
        if destination_speaker_path.exists():
            destination_speaker_map = validate_artifact("speaker_map", destination_speaker_path, self.schemas_dir)
        shots = visual.get("shots") if visual else None
        visual_report = visual.get("visual_report") if visual else None
        shots_path = self.destination.cache_dir / "shots.json"
        visual_report_path = self.destination.cache_dir / "visual_report.json"
        visual_schedule_report = None
        visual_schedule_report_path = self.destination.cache_dir / "visual_schedule_report.json"
        if shots is None and shots_path.exists():
            shots = validate_artifact("shots", shots_path, self.schemas_dir)
        if visual_report is None and visual_report_path.exists():
            visual_report = validate_artifact("visual_report", visual_report_path, self.schemas_dir)
        if visual_schedule_report_path.exists():
            visual_schedule_report = validate_artifact("visual_schedule_report", visual_schedule_report_path, self.schemas_dir)
        review_notes = None
        review_notes_path = self.destination.cache_dir / "review_notes.json"
        if review_notes_path.exists():
            review_notes = validate_artifact("review_notes", review_notes_path, self.schemas_dir)
        review_analysis = None
        if review_notes is not None:
            review_analysis_path = self.destination.cache_dir / "review_analysis.json"
            review_analysis = build_review_analysis(
                review_notes=review_notes,
                schedule=schedule,
                output_path=review_analysis_path,
            )
            validate_artifact("review_analysis", review_analysis_path, self.schemas_dir)
        audio_output = self.config.output_dir / "replacement_dialogue.wav"
        video_output = self.config.output_dir / "movie_masher_output.mp4"
        transformation_report_path = self.config.output_dir / "movie_masher" / "transformation_report.json"
        latest_transformation_report_path = self.config.output_dir / "transformation_report.json"
        transformation_plan_path = transformation_plan or (self.config.output_dir / "movie_masher" / "transformation_plan.json")
        latest_transformation_plan_path = self.config.output_dir / "transformation_plan.json"
        if transformation_report_path.exists():
            validate_artifact("transformation_report", transformation_report_path, self.schemas_dir)
        if latest_transformation_report_path.exists():
            validate_artifact("transformation_report", latest_transformation_report_path, self.schemas_dir)
        if transformation_plan_path.exists():
            validate_artifact("transformation_plan", transformation_plan_path, self.schemas_dir)
        if latest_transformation_plan_path.exists():
            validate_artifact("transformation_plan", latest_transformation_plan_path, self.schemas_dir)
        performance_report_json = self.config.output_dir / "performance_placement_report.json"
        performance_report_csv = self.config.output_dir / "performance_placement_report.csv"
        performance_report_txt = self.config.output_dir / "performance_placement_report.txt"
        performance_placement_report = build_performance_placement_report(
            schedule=schedule,
            source_performances=source_performances,
            destination_performances=destination_performances,
            output_json=performance_report_json,
            output_csv=performance_report_csv,
            output_txt=performance_report_txt,
        )
        validate_artifact("performance_placement_report", performance_report_json, self.schemas_dir)
        performance_diagnostics_json = self.config.output_dir / "performance_diagnostics.json"
        performance_diagnostics = build_performance_diagnostics(
            schedule=schedule,
            output_path=performance_diagnostics_json,
        )
        validate_artifact("performance_diagnostics", performance_diagnostics_json, self.schemas_dir)
        taste_profile_json = self.config.output_dir / "taste_profile.json"
        taste_profile = default_taste_profile(output_path=taste_profile_json)
        validate_artifact("taste_profile", taste_profile_json, self.schemas_dir)
        editorial_highlights_json = self.config.output_dir / "editorial_highlights.json"
        editorial_highlights = build_editorial_highlights(
            schedule=schedule,
            performance_diagnostics=performance_diagnostics,
            taste_profile=taste_profile,
            output_path=editorial_highlights_json,
        )
        validate_artifact("editorial_highlights", editorial_highlights_json, self.schemas_dir)
        problem_report_json = self.config.output_dir / "problem_regions.json"
        problem_report_csv = self.config.output_dir / "problem_regions.csv"
        problem_report_txt = self.config.output_dir / "problem_regions.txt"
        problem_region_report = build_problem_region_report(
            schedule=schedule,
            output_json=problem_report_json,
            output_csv=problem_report_csv,
            output_txt=problem_report_txt,
        )
        report = build_run_report(
            config=self.config,
            source_hash=self.source.media_hash,
            destination_hash=self.destination.media_hash,
            destination_movie=destination_movie,
            source_movie=source_movie,
            source_events=source_events,
            filtered_source_events=filtered_source_events,
            clip_library=clip_library,
            destination_timeline=destination_timeline,
            filtered_destination_timeline=filtered_destination_timeline,
            schedule=schedule,
            visual_schedule_report=visual_schedule_report,
            review_notes=review_notes,
            review_analysis=review_analysis,
            source_performances=source_performances,
            destination_performances=destination_performances,
            performance_placement_report=performance_placement_report,
            problem_region_report=problem_region_report,
            editorial_highlights=editorial_highlights,
            source_speaker_map=source_speaker_map,
            destination_speaker_map=destination_speaker_map,
            audio_output=audio_output,
            video_output=video_output,
        )
        paths = write_report_files(report, schedule, self.config.output_dir)
        paths["performance_placement_report"] = performance_report_json
        paths["performance_placement_report_csv"] = performance_report_csv
        paths["performance_placement_report_txt"] = performance_report_txt
        paths["performance_diagnostics"] = performance_diagnostics_json
        paths["taste_profile"] = taste_profile_json
        paths["editorial_highlights"] = editorial_highlights_json
        paths["problem_regions"] = problem_report_json
        paths["problem_regions_csv"] = problem_report_csv
        paths["problem_regions_txt"] = problem_report_txt
        index_path = self.config.output_dir / "cinematic_index.json"
        build_cinematic_index(
            root=self.config.root,
            output_path=index_path,
            destination_movie=destination_movie,
            source_movie=source_movie,
            source_events=source_events,
            filtered_source_events=filtered_source_events,
            clip_library=clip_library,
            destination_timeline=destination_timeline,
            filtered_destination_timeline=filtered_destination_timeline,
            schedule=schedule,
            audio_output=audio_output,
            video_output=video_output,
            run_report_json=paths["json"],
            schedule_report_csv=paths["csv"],
            destination_cache=self.destination.cache_dir,
            source_cache=self.source.cache_dir,
            shots=shots,
            visual_report=visual_report,
            visual_schedule_report=visual_schedule_report,
            review_notes=review_notes,
            review_analysis=review_analysis,
            source_performances=source_performances,
            destination_performances=destination_performances,
            transformation_report=transformation_report_path if transformation_report_path.exists() else None,
            transformation_plan=transformation_plan_path if transformation_plan_path.exists() else None,
        )
        validate_artifact("cinematic_index", index_path, self.schemas_dir)
        paths["cir"] = index_path
        if transformation_report_path.exists():
            paths["transformation_report"] = transformation_report_path
        if latest_transformation_report_path.exists():
            paths["latest_transformation_report"] = latest_transformation_report_path
        if transformation_plan_path.exists():
            paths["transformation_plan"] = transformation_plan_path
        if latest_transformation_plan_path.exists():
            paths["latest_transformation_plan"] = latest_transformation_plan_path
        self.logger.info(f"wrote run reports: {paths['json']}, {paths['txt']}, {paths['csv']}, {paths['cir']}")
        return paths



    def run_best_short_remix(
        self,
        *,
        app_mode: str = "Movie Masher",
        mutation_id: str = "self_shuffle",
        preference: str = "balanced",
        filter_parameters: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Path]:
        started_at = time.time()
        progress = ProgressState.start("source_loading", "Source loading", total=6, status_message="Preparing Dialogue Reel")
        self._log_progress(progress.update(current=1))
        self.logger.info("dialogue reel: building schedule and scene artifacts")
        if app_mode == "Movie Masher":
            schedule = self.schedule(force=force)
            media_path = self.destination.media_path
            media_hash = self.destination.media_hash
            source_media_path = self.source.media_path
            source_media_hash = self.source.media_hash
            dest_movie = self._inspect_one(self.destination, force=False)
            source_performances = self.build_source_performances(force=False)
            destination_performances = self.build_destination_performances(force=False)
            output_root = self.config.output_dir / "best_short"
        else:
            single, dest_movie, schedule, _schedule_path, source_performances, destination_performances = self._build_single_film_mutation_schedule(mutation_id=mutation_id, force=force, parameters=filter_parameters)
            media_path = single.destination.media_path
            media_hash = single.destination.media_hash
            source_media_path = media_path
            source_media_hash = media_hash
            output_root = single.config.output_dir / "best_short" / mutation_id
        active_filter_id = "movie_masher" if app_mode == "Movie Masher" else mutation_id
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_mode = "movie_masher" if app_mode == "Movie Masher" else mutation_id
        source_hash_for_output = schedule.get("source_media_hash") or source_media_hash
        output_root = output_root / "runs" / f"{run_stamp}_{run_mode}_{media_hash[:8]}_{source_hash_for_output[:8]}"
        output_root.mkdir(parents=True, exist_ok=True)
        self._log_progress(progress.update(current=2, stage_id="dialogue_scene_generation", stage_label="Dialogue scenes", status_message="Converting performances to scenes"))

        for index, mapping in enumerate(schedule.get("mappings", [])):
            mapping["_schedule_index"] = index

        source_scenes_path = output_root / "source_dialogue_scenes.json"
        destination_scenes_path = output_root / "destination_dialogue_scenes.json"
        scene_candidates_path = output_root / "scene_pair_candidates.json"
        reel_selection_path = output_root / "vignette_reel_selection.json"
        source_scenes = build_dialogue_scene_artifact(
            media_hash=source_media_hash,
            role="source_dialogue",
            performances=source_performances,
            output_path=source_scenes_path,
        )
        destination_scenes = build_dialogue_scene_artifact(
            media_hash=media_hash,
            role="destination_video",
            performances=destination_performances,
            output_path=destination_scenes_path,
        )
        self._log_progress(progress.update(current=3, stage_id="scene_pair_scoring", stage_label="Scene pair scoring", status_message="Scoring scene pairs"))
        selected_definition = default_filter_registry().get(active_filter_id)
        sparse_filter_short = app_mode != "Movie Masher" and selected_definition.sparse_schedule
        minimum_dialogue_coverage = 0.0 if sparse_filter_short else 0.6
        scene_candidates = build_scene_pair_candidates(
            schedule=schedule,
            source_scenes=source_scenes,
            destination_scenes=destination_scenes,
            self_shuffle=(app_mode != "Movie Masher" and mutation_id == "self_shuffle"),
            minimum_temporal_separation=30.0,
            minimum_dialogue_coverage=minimum_dialogue_coverage,
            output_path=scene_candidates_path,
        )
        reel = select_vignette_reel(
            candidates=scene_candidates,
            target_duration_seconds=self.config.target_duration_seconds,
            minimum_duration_seconds=self.config.minimum_duration_seconds,
            maximum_duration_seconds=self.config.maximum_duration_seconds,
            minimum_dialogue_coverage=minimum_dialogue_coverage,
            output_path=reel_selection_path,
        )
        selected_vignettes = reel.get("selected_vignettes", [])
        if not selected_vignettes:
            raise ValueError("Dialogue reel selection produced no vignettes.")
        if len(selected_vignettes) < 2:
            self.logger.info("dialogue reel warning: only one vignette selected; output is a fallback, not a full multi-moment reel")
        self.logger.info(
            f"dialogue reel: selected {len(selected_vignettes)} vignettes "
            f"duration={reel.get('actual_scene_duration_seconds')}s candidates={scene_candidates.get('candidate_count')}"
        )

        date_stamp = datetime.now().strftime("%Y-%m-%d")
        stem = f"movie_masher_dialogue_reel_{date_stamp}"
        vignette_dir = output_root / "vignettes"
        vignette_dir.mkdir(parents=True, exist_ok=True)
        visual_segment = output_root / f"_{stem}_visual_reel_original_audio_DO_NOT_REVIEW.mp4"
        audio_output = output_root / f"{stem}.wav"
        video_output = output_root / f"FINAL_{stem}.mp4"
        report_output = output_root / f"{stem}_report.json"
        latest_report_output = output_root / "output_report.json"
        audio_provenance_output = output_root / "audio_provenance.json"
        filter_acceptance_output = output_root / "filter_acceptance.json"

        self._log_progress(progress.update(current=4, stage_id="vignette_render", stage_label="Vignette render", status_message="Rendering selected moments"))
        vignette_videos: list[Path] = []
        vignette_audios: list[Path] = []
        original_segments: list[Path] = []
        vignette_outputs: list[dict[str, Any]] = []
        combined_mappings: list[dict[str, Any]] = []
        reel_cursor = 0.0
        for vignette in selected_vignettes:
            vignette_index = int(vignette.get("vignette_index", len(vignette_videos) + 1) or len(vignette_videos) + 1)
            start_time = max(0.0, float(vignette.get("destination_start", 0.0) or 0.0) - 0.5)
            duration = max(0.001, float(vignette.get("destination_duration", 0.0) or 0.0) + 1.0)
            visual_path = vignette_dir / f"vignette_{vignette_index:03d}_visual_original_audio_DO_NOT_REVIEW.mp4"
            audio_path = vignette_dir / f"vignette_{vignette_index:03d}.wav"
            video_path = vignette_dir / f"vignette_{vignette_index:03d}.mp4"
            self.logger.info(
                f"vignette {vignette_index}: destination={vignette.get('destination_scene_id')} "
                f"source={vignette.get('source_scene_id')} start={start_time:.3f}s duration={duration:.3f}s"
            )
            extract_video_segment(input_video=media_path, output_path=visual_path, start_time=start_time, duration=duration)
            vignette_schedule = build_vignette_schedule(schedule, vignette, padding=0.5)
            if sparse_filter_short:
                mute_regions = _speech_mute_regions(vignette_schedule, padding=0.35, merge_gap=0.25, duration=duration)
                render_mutation_media(
                    original_media=visual_path,
                    schedule=vignette_schedule,
                    duration=duration,
                    audio_output=audio_path,
                    video_output=video_path,
                    sample_rate=self.config.render_sample_rate,
                    channels=self.config.render_channels,
                    target_lufs=self.config.target_lufs,
                    fade_duration=self.config.audio_fade_duration,
                    mute_regions=mute_regions,
                )
            else:
                render_dialogue_wav(
                    schedule=vignette_schedule,
                    duration=duration,
                    output_path=audio_path,
                    sample_rate=self.config.render_sample_rate,
                    channels=self.config.render_channels,
                    target_lufs=self.config.target_lufs,
                    fade_duration=self.config.audio_fade_duration,
                )
                mux_video(destination_video=visual_path, dialogue_wav=audio_path, output_path=video_path)
            mapping_start_index = len(combined_mappings)
            vignette_videos.append(video_path)
            vignette_audios.append(audio_path)
            original_segments.append(visual_path)
            combined_mappings.extend(vignette_schedule.get("mappings", []))
            reel_segment = {'mappings': combined_mappings[mapping_start_index:]}
            reel_segment = offset_vignette_schedule(reel_segment, offset_seconds=reel_cursor)
            combined_mappings[mapping_start_index:] = reel_segment['mappings']
            reel_cursor += duration
            vignette_outputs.append(
                {
                    "vignette_index": vignette_index,
                    "destination_scene_id": vignette.get("destination_scene_id"),
                    "source_scene_id": vignette.get("source_scene_id"),
                    "score": vignette.get("overall_score"),
                    "component_scores": vignette.get("component_scores", {}),
                    "reason_selected": vignette.get("reason_selected"),
                    "destination_timestamp": vignette.get("destination_start"),
                    "donor_timestamp": vignette.get("source_start"),
                    "video": str(video_path),
                    "audio": str(audio_path),
                    "visual_original_audio": str(visual_path),
                }
            )

        self._log_progress(progress.update(current=5, stage_id="render_export", stage_label="Render/export", status_message="Concatenating dialogue reel"))
        concat_wav_files(inputs=vignette_audios, output_path=audio_output, sample_rate=self.config.render_sample_rate, channels=self.config.render_channels)
        concat_media_files(inputs=original_segments, output_path=visual_segment, reencode=True)
        mux_video(destination_video=visual_segment, dialogue_wav=audio_output, output_path=video_output)

        final_audio_analysis_path = output_root / f"_{stem}_final_audio_for_verification.wav"
        original_audio_analysis_path = output_root / f"_{stem}_original_segment_audio_for_verification.wav"
        final_audio_stats = extract_audio_for_provenance(
            media_path=video_output,
            output_path=final_audio_analysis_path,
            sample_rate=self.config.render_sample_rate,
            channels=self.config.render_channels,
        )
        original_audio_stats = extract_audio_for_provenance(
            media_path=visual_segment,
            output_path=original_audio_analysis_path,
            sample_rate=self.config.render_sample_rate,
            channels=self.config.render_channels,
        )
        final_vs_replacement = compare_wav_audio(left_path=final_audio_analysis_path, right_path=audio_output)
        final_vs_original = compare_wav_audio(left_path=final_audio_analysis_path, right_path=original_audio_analysis_path)
        final_audio_stats["diff_from_replacement_rms"] = final_vs_replacement["diff_rms"]
        final_audio_stats["diff_from_original_segment_rms"] = final_vs_original["diff_rms"]
        reel_schedule = dict(schedule)
        reel_schedule["mappings"] = combined_mappings
        reel_schedule["selected_mode"] = "dialogue_reel"
        reel_schedule["vignette_count"] = len(selected_vignettes)
        reel_schedule['render_duration'] = round(reel_cursor, 3)
        audio_provenance = verify_audio_provenance(
            root=self.config.root,
            destination_video=self.destination.media_path if app_mode == "Movie Masher" else media_path,
            destination_hash=media_hash,
            source_dialogue=source_media_path,
            source_hash=source_media_hash,
            schedule=schedule,
            short_schedule=reel_schedule,
            replacement_audio=audio_output,
            final_video=video_output,
            visual_segment=visual_segment,
            output_path=audio_provenance_output,
            final_audio_analysis=final_audio_stats,
            original_segment_analysis=original_audio_stats,
        )
        validate_filter_output(
            filter_id=active_filter_id,
            schedule=reel_schedule,
            final_video=video_output,
            replacement_audio=audio_output,
            output_path=filter_acceptance_output,
            schemas_dir=self.schemas_dir,
            audio_provenance=audio_provenance,
        )
        report = build_vignette_reel_report(
            reel=reel,
            candidates=scene_candidates,
            destination_scenes=destination_scenes,
            source_scenes=source_scenes,
            output_video=video_output,
            output_audio=audio_output,
            output_path=report_output,
            vignette_outputs=vignette_outputs,
            audio_provenance=audio_provenance,
        )
        write_json(latest_report_output, report)
        self._log_progress(progress.update(current=6, stage_id="complete", stage_label="Finished", status_message="Dialogue reel complete"))
        self.logger.info(f"Dialogue reel rendered: {video_output}")
        rendered_video_output = video_output
        video_output = publish_single_video(video=rendered_video_output, output_dir=self.config.output_dir, process=f"{run_mode}-short")
        _rewrite_published_video_references(artifact_paths=[report_output, latest_report_output, audio_provenance_output, filter_acceptance_output], rendered_video=rendered_video_output, published_video=video_output, root=self.config.root)
        return {
            "video": video_output,
            "audio": audio_output,
            "report": report_output,
            "latest_report": latest_report_output,
            "candidates": scene_candidates_path,
            "reel_selection": reel_selection_path,
            "audio_provenance": audio_provenance_output,
            "filter_acceptance": filter_acceptance_output,
        }

    def _build_single_film_mutation_schedule(self, *, mutation_id: str, force: bool = False, parameters: dict[str, Any] | None = None):
        definition = get_mutation(mutation_id)
        params = {**definition.default_parameters, **(parameters or {})}
        single_config = self.config.with_overrides(source_dialogue=self.config.destination_video)
        single = Pipeline(single_config, cancel_check=self.cancel_check, stage_callback=self.stage_callback)
        destination_movie, _source_movie = single.inspect(force=force)
        source_events = single.extract_source_dialogue(force=force, source_movie=destination_movie)
        if single.config.enable_speaker_awareness:
            source_speaker_map = single.build_source_speaker_map(source_events=source_events, force=force)
            source_events = annotate_artifact_speakers(source_events, source_speaker_map, collection_key="events")
        filtered_events = single.filter_source_dialogue_from_events(source_events, force=force)
        clip_library = single.build_clip_library_from_events(filtered_events, force=force)
        timeline = single.detect_destination_timeline(force=force, dest_movie=destination_movie)
        visual = single.analyze_visual(force=force)
        if single.config.enable_speaker_awareness:
            destination_speaker_map = single.build_destination_speaker_map(timeline=timeline, force=force)
            timeline = annotate_artifact_speakers(timeline, destination_speaker_map, collection_key="windows")
        filtered_timeline = single.filter_destination_timeline_from_timeline(timeline, force=force)
        source_performances = single.build_source_performances(source_events=filtered_events, force=force)
        destination_performances = single.build_destination_performances(timeline=filtered_timeline, visual=visual, force=force)
        clips = clip_library.get("clips", [])
        _annotate_clips_with_dialogue_scene_ids(clips=clips, source_performances=source_performances)
        duration = float(destination_movie["duration"])
        mutation_dir = single.config.output_dir / "mutations" / mutation_id
        mutation_dir.mkdir(parents=True, exist_ok=True)
        schedule_path = mutation_dir / f"{mutation_id}_schedule.json"
        if mutation_id == "echo":
            schedule = build_echo_schedule(clips=clips, duration=duration, parameters=params)
        elif mutation_id == "drift":
            schedule = build_drift_schedule(clips=clips, duration=duration, parameters=params)
        elif mutation_id == "self_shuffle":
            schedule = build_self_shuffle_schedule(
                clips=clips,
                windows=performance_windows(destination_performances),
                media_hash=single.destination.media_hash,
                max_time_stretch=single.config.max_time_stretch,
                output_path=schedule_path,
                seed=int(params.get("seed", 1)),
                best_fit_lookahead=single.config.best_fit_lookahead,
                cinematic_filter=single.config.cinematic_filter,
                source_performances=source_performances,
            )
        elif default_filter_registry().get(mutation_id).execution_mode == "scheduling_strategy":
            schedule = build_strategy_schedule(
                mutation_id,
                clips=clips,
                windows=usable_rows(filtered_timeline.get("windows", [])),
                duration=duration,
                parameters=params,
                progress_callback=single.logger.info,
            )
            _annotate_schedule_with_destination_performance_ids(schedule=schedule, destination_performances=destination_performances)
        else:
            raise ValueError(f"Unsupported mutation: {mutation_id}")
        schedule["media_hash"] = single.destination.media_hash
        schedule["source_media_hash"] = single.destination.media_hash
        schedule["destination_media_hash"] = single.destination.media_hash
        schedule["config_signature"] = single._signature("mutation", mutation_id, single.destination.media_hash, clip_library.get("config_signature"), filtered_timeline.get("config_signature"), params)
        write_filter_artifacts(
            pipeline=single,
            filter_id=mutation_id,
            parameters=params,
            schedule=schedule,
            output_dir=mutation_dir,
            output_form="best_short",
            target_duration=single.config.target_duration_seconds,
        )
        write_json(schedule_path, schedule)
        return single, destination_movie, schedule, schedule_path, source_performances, destination_performances

    def run_self_shuffle(self, *, seed: int = 1, force: bool = False) -> dict[str, Path]:
        result = self.execute_transformation("self_shuffle", force=force, parameters={"seed": seed})
        return {
            "schedule": result.outputs["schedule"],
            "audio": result.outputs["audio"],
            "video": result.outputs["video"],
            "transformation_report": result.artifacts.get(
                "transformation_report",
                self.config.output_dir / "self_shuffle" / "transformation_report.json",
            ),
        }



    def run_mutation(
        self,
        mutation_id: str,
        *,
        force: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        definition = get_mutation(mutation_id)
        params = {**definition.default_parameters, **(parameters or {})}
        single_config = self.config.with_overrides(source_dialogue=self.config.destination_video)
        single = Pipeline(single_config, cancel_check=self.cancel_check, stage_callback=self.stage_callback)
        destination_movie, _source_movie = single.inspect(force=force)
        source_events = single.extract_source_dialogue(force=force, source_movie=destination_movie)
        if single.config.enable_speaker_awareness:
            source_speaker_map = single.build_source_speaker_map(source_events=source_events, force=force)
            source_events = annotate_artifact_speakers(source_events, source_speaker_map, collection_key="events")
        filtered_events = single.filter_source_dialogue_from_events(source_events, force=force)
        clip_library = single.build_clip_library_from_events(filtered_events, force=force)
        timeline = single.detect_destination_timeline(force=force, dest_movie=destination_movie)
        visual = single.analyze_visual(force=force)
        if single.config.enable_speaker_awareness:
            destination_speaker_map = single.build_destination_speaker_map(timeline=timeline, force=force)
            timeline = annotate_artifact_speakers(timeline, destination_speaker_map, collection_key="windows")
        filtered_timeline = single.filter_destination_timeline_from_timeline(timeline, force=force)
        source_performances = single.build_source_performances(source_events=filtered_events, force=force)
        destination_performances = single.build_destination_performances(timeline=filtered_timeline, visual=visual, force=force)
        clips = clip_library.get("clips", [])
        _annotate_clips_with_dialogue_scene_ids(clips=clips, source_performances=source_performances)
        duration = float(destination_movie["duration"])
        mutation_dir = single.config.output_dir / "mutations" / mutation_id
        mutation_dir.mkdir(parents=True, exist_ok=True)
        schedule_path = mutation_dir / f"{mutation_id}_schedule.json"
        audio_output = mutation_dir / f"{mutation_id}_audio.wav"
        video_output = mutation_dir / f"{mutation_id}_output.mp4"
        plan_path = mutation_dir / "mutation_plan.json"
        report_path = mutation_dir / "mutation_report.json"
        acceptance_path = mutation_dir / "filter_acceptance.json"
        warnings: list[str] = []

        if mutation_id == "echo":
            schedule = build_echo_schedule(clips=clips, duration=duration, parameters=params)
            mute_regions = schedule.get("mappings", []) if bool(params.get("duck_original_at_echoes", True)) else []
        elif mutation_id == "drift":
            schedule = build_drift_schedule(clips=clips, duration=duration, parameters=params)
            mute_regions = filtered_timeline.get("windows", []) if bool(params.get("preserve_original_soundtrack", True)) else None
        elif mutation_id == "self_shuffle":
            schedule = build_self_shuffle_schedule(
                clips=clips,
                windows=performance_windows(destination_performances),
                media_hash=single.destination.media_hash,
                max_time_stretch=single.config.max_time_stretch,
                output_path=schedule_path,
                seed=int(params.get("seed", 1)),
                best_fit_lookahead=single.config.best_fit_lookahead,
                cinematic_filter=single.config.cinematic_filter,
                source_performances=source_performances,
            )
            mute_regions = _speech_mute_regions(schedule, padding=0.35, merge_gap=0.25, duration=duration)
        elif default_filter_registry().get(mutation_id).execution_mode == "scheduling_strategy":
            schedule = build_strategy_schedule(
                mutation_id,
                clips=clips,
                windows=usable_rows(filtered_timeline.get("windows", [])),
                duration=duration,
                parameters=params,
                progress_callback=single.logger.info,
            )
            _annotate_schedule_with_destination_performance_ids(schedule=schedule, destination_performances=destination_performances)
            mute_regions = _speech_mute_regions(schedule, padding=0.35, merge_gap=0.25, duration=duration)
        else:
            raise ValueError(f"Unsupported mutation: {mutation_id}")

        schedule["media_hash"] = single.destination.media_hash
        schedule["source_media_hash"] = single.destination.media_hash
        schedule["destination_media_hash"] = single.destination.media_hash
        schedule["config_signature"] = single._signature("mutation", mutation_id, single.destination.media_hash, clip_library.get("config_signature"), filtered_timeline.get("config_signature"), params)
        filter_artifacts = write_filter_artifacts(
            pipeline=single,
            filter_id=mutation_id,
            parameters=params,
            schedule=schedule,
            output_dir=mutation_dir,
            output_form="full_length",
            target_duration=duration,
        )
        write_json(schedule_path, schedule)

        selected = [{"id": mapping.get("clip_id"), "type": "dialogue_clip"} for mapping in schedule.get("mappings", [])]
        operations = [{"operation": mapping.get("mutation_operation", mutation_id), "mapping": mapping.get("window_id")} for mapping in schedule.get("mappings", [])]
        placements = [
            {"clip_id": mapping.get("clip_id"), "destination_timestamp": mapping.get("destination_timestamp")}
            for mapping in schedule.get("mappings", [])
        ]
        plan = build_mutation_plan(
            mutation_id=mutation_id,
            source_media_hash=single.destination.media_hash,
            source_path=single.destination.media_path,
            selected_objects=selected,
            operations=operations,
            placements=placements,
            render_strategy={"video": "preserve_source_video", "audio": "mix_mutated_dialogue_over_source_audio"},
            expected_output_path=video_output,
            output_path=plan_path,
            parameters=params,
            warnings=warnings,
        )
        single._write_and_validate("mutation_plan", plan_path, plan)

        render_mutation_media(
            original_media=single.destination.media_path,
            schedule=schedule,
            duration=duration,
            audio_output=audio_output,
            video_output=video_output,
            sample_rate=single.config.render_sample_rate,
            channels=single.config.render_channels,
            target_lufs=single.config.target_lufs,
            fade_duration=single.config.audio_fade_duration,
            mute_regions=mute_regions,
        )
        validate_filter_output(
            filter_id=mutation_id,
            schedule=schedule,
            final_video=video_output,
            replacement_audio=audio_output,
            output_path=acceptance_path,
            schemas_dir=single.schemas_dir,
        )
        schedule["filter_acceptance_path"] = str(acceptance_path)
        report = build_mutation_report(
            mutation_id=mutation_id,
            source_path=single.destination.media_path,
            source_media_hash=single.destination.media_hash,
            parameters=params,
            plan_path=plan_path,
            output_video=video_output,
            output_audio=audio_output,
            schedule=schedule,
            output_path=report_path,
            warnings=warnings,
        )
        single._write_and_validate("mutation_report", report_path, report)
        self.logger.info(f"mutation {mutation_id} rendered: {video_output}")
        rendered_video_output = video_output
        video_output = publish_single_video(video=rendered_video_output, output_dir=self.config.output_dir, process=mutation_id)
        _rewrite_published_video_references(artifact_paths=[schedule_path, plan_path, report_path, acceptance_path, *filter_artifacts.values()], rendered_video=rendered_video_output, published_video=video_output, root=self.config.root)
        return {"video": video_output, "audio": audio_output, "schedule": schedule_path, "mutation_plan": plan_path, "mutation_report": report_path, "filter_acceptance": acceptance_path, **filter_artifacts}

    def run_preset(
        self,
        preset_id: str,
        *,
        force: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        preset = load_preset(self.config.root, preset_id)
        return self.run_loaded_preset(preset, force=force, parameters=parameters)

    def run_loaded_preset(
        self,
        preset: Preset,
        *,
        force: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        params = parameters or {}
        self.logger.info(f"running preset: {preset.id}")
        if preset.transformation_strategy == "movie_masher":
            result = self.execute_transformation("movie_masher", force=force, parameters=params)
            return {
                "video": result.outputs["video"],
                "audio": self.config.output_dir / preset.render_outputs.get("audio", "replacement_dialogue.wav"),
                "schedule": self.destination.cache_dir / "replacement_schedule.json",
                "transformation_report": result.artifacts.get(
                    "transformation_report",
                    self.config.output_dir / "movie_masher" / "transformation_report.json",
                ),
            }
        if preset.transformation_strategy == "self_shuffle":
            seed = int(params.get("seed", preset.parameters.get("seed", {}).get("default", preset.scheduling.get("seed", 1))))
            result = self.execute_transformation("self_shuffle", force=force, parameters={"seed": seed})
            return {
                "schedule": result.outputs["schedule"],
                "audio": result.outputs["audio"],
                "video": result.outputs["video"],
                "transformation_report": result.artifacts.get(
                    "transformation_report",
                    self.config.output_dir / "self_shuffle" / "transformation_report.json",
                ),
            }
        raise ValueError(f"Unsupported preset transformation strategy: {preset.transformation_strategy}")

    def validate_existing(self) -> dict[str, bool]:
        checks = {
            "destination_movie": self.destination.cache_dir / "movie.json",
            "source_movie": self.source.cache_dir / "movie.json",
            "dialogue_events": self.source.cache_dir / "dialogue_events.json",
            "filtered_dialogue_events": self.source.cache_dir / "filtered_dialogue_events.json",
            "clip_library": self.source.cache_dir / "clip_library.json",
            "source_speaker_map": self.source.cache_dir / "speaker_map.json",
            "timeline": self.destination.cache_dir / "timeline.json",
            "filtered_timeline": self.destination.cache_dir / "filtered_timeline.json",
            "destination_speaker_map": self.destination.cache_dir / "speaker_map.json",
            "speaker_mapping": self.config.output_dir / "speaker_mapping.json",
            "replacement_schedule": self.destination.cache_dir / "replacement_schedule.json",
            "source_performance": self.source.cache_dir / "performance.json",
            "source_performance_library": self.source.cache_dir / "performance_library.json",
            "destination_performance": self.destination.cache_dir / "performance.json",
            "shots": self.destination.cache_dir / "shots.json",
            "visual_report": self.destination.cache_dir / "visual_report.json",
            "visual_schedule_report": self.destination.cache_dir / "visual_schedule_report.json",
            "cinematic_index": self.config.output_dir / "cinematic_index.json",
            "latest_transformation_report": self.config.output_dir / "transformation_report.json",
            "movie_masher_transformation_report": self.config.output_dir / "movie_masher" / "transformation_report.json",
            "self_shuffle_transformation_report": self.config.output_dir / "self_shuffle" / "transformation_report.json",
            "performance_placement_report": self.config.output_dir / "performance_placement_report.json",
            "latest_transformation_plan": self.config.output_dir / "transformation_plan.json",
            "movie_masher_transformation_plan": self.config.output_dir / "movie_masher" / "transformation_plan.json",
        }
        artifact_types = {
            "destination_movie": "movie",
            "source_movie": "movie",
            "dialogue_events": "dialogue_events",
            "filtered_dialogue_events": "filtered_dialogue_events",
            "clip_library": "clip_library",
            "source_speaker_map": "speaker_map",
            "timeline": "timeline",
            "filtered_timeline": "filtered_timeline",
            "destination_speaker_map": "speaker_map",
            "speaker_mapping": "speaker_mapping",
            "replacement_schedule": "replacement_schedule",
            "source_performance": "performance",
            "source_performance_library": "performance_library",
            "destination_performance": "performance",
            "shots": "shots",
            "visual_report": "visual_report",
            "visual_schedule_report": "visual_schedule_report",
            "cinematic_index": "cinematic_index",
            "latest_transformation_report": "transformation_report",
            "movie_masher_transformation_report": "transformation_report",
            "self_shuffle_transformation_report": "transformation_report",
            "performance_placement_report": "performance_placement_report",
            "latest_transformation_plan": "transformation_plan",
            "movie_masher_transformation_plan": "transformation_plan",
        }
        result = {}
        for name, path in checks.items():
            result[name] = path.exists()
            if path.exists():
                validate_artifact(artifact_types[name], path, self.schemas_dir)
        return result

    def _log_progress(self, state: ProgressState) -> None:
        self.logger.info(format_progress_status(state))

    def _inspect_one(self, entry: CacheEntry, *, force: bool) -> dict:
        movie_path = entry.cache_dir / "movie.json"
        cached = self._load_current("movie", movie_path, entry.media_hash, None, force)
        if cached:
            self.logger.info(f"reused media inspection for {entry.role}: {movie_path}")
            return cached
        self.logger.info(f"inspecting {entry.role}: {entry.media_path}")
        data = inspect_media(entry.media_path, entry.media_hash, movie_path)
        self._write_and_validate("movie", movie_path, data)
        update_manifest(entry, "inspected", {"movie": str(movie_path)})
        return data

    def _signature(self, phase: str, *parts: Any) -> str:
        payload: dict[str, Any] = {"phase": phase, "parts": parts}

        if phase in {"dialogue_events", "timeline"}:
            payload.update(
                {
                    "speech_backend": self.config.speech_backend,
                    "transcription_mode": self.config.transcription_mode,
                    "whisper_model": self.config.whisper_model,
                    "whisper_language": self.config.whisper_language,
                    "quick_test_seconds": self.config.quick_test_seconds,
                    "silence_noise_db": self.config.silence_noise_db,
                    "silence_min_duration": self.config.silence_min_duration,
                    "min_speech_duration": self.config.min_speech_duration,
                    "merge_gap": self.config.merge_gap,
                }
            )
        elif phase in {"filtered_dialogue_events", "filtered_timeline"}:
            payload.update(
                {
                    "filter_min_duration": self.config.filter_min_duration,
                    "filter_max_duration": self.config.filter_max_duration,
                    "filter_min_confidence": self.config.filter_min_confidence,
                    "filter_min_chars_per_second": self.config.filter_min_chars_per_second,
                    "filter_max_chars_per_second": self.config.filter_max_chars_per_second,
                    "filter_repeated_text_window": self.config.filter_repeated_text_window,
                    "speaker_awareness": self.config.speaker_diarization_backend if self.config.enable_speaker_awareness else "off",
                    "speaker_mapping": "content_dependent_v4" if self.config.enable_speaker_awareness else "off",
                }
            )
        elif phase == "clip_library":
            payload.update({"clip_schema": "complete_utterances_v2", "maximum_utterance_duration": 12.0})
        elif phase == "shots":
            payload.update(
                {
                    "visual_scene_threshold": self.config.visual_scene_threshold,
                    "visual_min_shot_duration": self.config.visual_min_shot_duration,
                }
            )
        elif phase == "speaker_map":
            payload.update(
                {
                    "speaker_schema": "speaker_backend_v4_stable_input" if self.config.enable_speaker_awareness else "off",
                    "speaker_diarization_backend": self.config.speaker_diarization_backend,
                    "speaker_diarization_model": self.config.speaker_diarization_model,
                    "speaker_diarization_device": self.config.speaker_diarization_device,
                    "silence_noise_db": self.config.silence_noise_db,
                    "silence_min_duration": self.config.silence_min_duration,
                    "min_speech_duration": self.config.min_speech_duration,
                    "merge_gap": self.config.merge_gap,
                }
            )
        elif phase == "speaker_mapping":
            payload.update({"speaker_mapping_schema": "rank_real_pyannote_speakers_v3" if self.config.enable_speaker_awareness else "off"})
        elif phase == "performance":
            payload.update({"performance_schema": "signature_v3_speaker_content", "speaker_awareness": self.config.speaker_diarization_backend if self.config.enable_speaker_awareness else "off"})
        elif phase == "performance_library":
            payload.update({"performance_library_schema": "1.0", "speaker_awareness": self.config.speaker_diarization_backend if self.config.enable_speaker_awareness else "off"})
        elif phase == "replacement_schedule":
            payload.update(
                {
                    "max_time_stretch": self.config.max_time_stretch,
                    "scheduling_mode": self.config.scheduling_mode,
                    "best_fit_lookahead": self.config.best_fit_lookahead,
                    "shot_boundary_mode": self.config.shot_boundary_mode,
                    "cinematic_filter": self.config.cinematic_filter,
                    "source_reuse_policy": "forbidden_by_default",
                    "schedule_schema": "source_timestamp_v19_no_implicit_reuse",
                }
            )

        elif phase == "audio_render":
            payload.update(
                {
                    "render_sample_rate": self.config.render_sample_rate,
                    "render_channels": self.config.render_channels,
                    "target_lufs": self.config.target_lufs,
                    "audio_fade_duration": self.config.audio_fade_duration,
                    "cinematic_filter": self.config.cinematic_filter,
                }
            )
        elif phase == "self_shuffle_schedule":
            payload.update(
                {
                    "max_time_stretch": self.config.max_time_stretch,
                    "best_fit_lookahead": self.config.best_fit_lookahead,
                    "speaker_awareness": self.config.speaker_diarization_backend if self.config.enable_speaker_awareness else "off",
                    "changed_line_policy": "no_original_overlap_v1",
                    "schedule_schema": "source_timestamp_v2",
                    "render_strategy": "dialogue_only_v1",
                    "scheduling_mode": "whole_line_fill",
                    "cinematic_filter": self.config.cinematic_filter,
                }
            )

        return stable_hash(payload)

    def _filter_config(self) -> FilterConfig:
        return FilterConfig(
            min_duration=self.config.filter_min_duration,
            max_duration=self.config.filter_max_duration,
            min_confidence=self.config.filter_min_confidence,
            min_chars_per_second=self.config.filter_min_chars_per_second,
            max_chars_per_second=self.config.filter_max_chars_per_second,
            repeated_text_window=self.config.filter_repeated_text_window,
        )

    def _load_current(
        self,
        artifact_type: str,
        path: Path,
        media_hash: str,
        signature: str | None,
        force: bool,
    ) -> dict | None:
        if force or not path.exists():
            return None
        try:
            data = validate_artifact(artifact_type, path, self.schemas_dir)
        except ValueError as exc:
            self.logger.info(f"invalid cached {artifact_type}, regenerating: {exc}")
            return None
        if data.get("media_hash") != media_hash:
            self.logger.info(f"cached {artifact_type} media hash mismatch, regenerating: {path}")
            return None
        if signature is not None and data.get("config_signature") != signature:
            self.logger.info(f"cached {artifact_type} config changed, regenerating: {path}")
            return None
        return data

    def _write_and_validate(self, artifact_type: str, path: Path, data: dict) -> None:
        from .util import write_json

        write_json(path, data)
        validate_artifact(artifact_type, path, self.schemas_dir)


def _annotate_clips_with_dialogue_scene_ids(*, clips: list[dict[str, Any]], source_performances: dict[str, Any]) -> None:
    performances = list(source_performances.get("performances", []))
    for clip in clips:
        timestamp = _float_for_pipeline(clip.get("movie_timestamp", clip.get("start")), None)
        if timestamp is None:
            continue
        for performance in performances:
            start = _float_for_pipeline(performance.get("start"), 0.0) or 0.0
            end = _float_for_pipeline(performance.get("end"), start + (_float_for_pipeline(performance.get("duration"), 0.0) or 0.0)) or start
            if start - 0.001 <= timestamp <= end + 0.001:
                clip["source_performance_id"] = performance.get("id")
                clip["source_performance_type"] = performance.get("conversation_type")
                clip["source_performance_duration"] = performance.get("duration")
                clip["source_speaker_sequence"] = performance.get("speaker_sequence", [])
                clip["source_turn_pattern"] = performance.get("turn_pattern", "")
                break


def _annotate_schedule_with_destination_performance_ids(*, schedule: dict[str, Any], destination_performances: dict[str, Any]) -> None:
    performances = list(destination_performances.get("performances", []))
    for mapping in schedule.get("mappings", []):
        timestamp = _float_for_pipeline(mapping.get("destination_timestamp"), None)
        if timestamp is None:
            continue
        for performance in performances:
            start = _float_for_pipeline(performance.get("start"), 0.0) or 0.0
            end = _float_for_pipeline(performance.get("end"), start + (_float_for_pipeline(performance.get("duration"), 0.0) or 0.0)) or start
            if start - 0.001 <= timestamp <= end + 0.001:
                mapping["destination_performance_id"] = performance.get("id")
                break


def _float_for_pipeline(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _rewrite_published_video_references(*, artifact_paths: list[Path], rendered_video: Path, published_video: Path, root: Path) -> None:
    old_absolute = str(rendered_video.resolve())
    new_absolute = str(published_video.resolve())
    old_relative = str(rendered_video)
    new_relative = str(published_video)
    try:
        old_relative = rendered_video.resolve().relative_to(root.resolve()).as_posix()
        new_relative = published_video.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            if value in {str(rendered_video), old_absolute}:
                return new_absolute
            if value in {old_relative, old_relative.replace("/", "\\") }:
                return new_relative
        return value

    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            continue
        data = read_json(artifact_path)
        updated = replace(data)
        if updated != data:
            write_json(artifact_path, updated)


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


def _format_problem_preview_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        "Cinelingus Problem Region Previews",
        "===================================",
        f"Preview count: {manifest.get('preview_count')}",
        f"Source video: {manifest.get('source_video')}",
        "",
    ]
    for preview in manifest.get("previews", []):
        lines.append(
            f"{preview.get('index')}. {preview.get('problem_type')} {preview.get('start')}s-{preview.get('end')}s: {preview.get('path')}"
        )
        if preview.get("reason"):
            lines.append(f"   {preview.get('reason')}")
    return "\n".join(lines) + "\n"



def _validate_best_short_render_contract(
    *,
    short_schedule: dict,
    mute_regions: list[dict[str, float]],
    duration: float,
    candidate_id: str,
    require_mute_regions: bool = True,
) -> None:
    mappings = [mapping for mapping in short_schedule.get("mappings", []) if mapping.get("enabled", True)]
    if not mappings:
        raise ValueError(f"Best Short candidate {candidate_id} has no mappings inside the extracted segment; refusing to render unchanged source audio.")
    if require_mute_regions:
        if not mute_regions:
            raise ValueError(f"Best Short candidate {candidate_id} produced no mute regions; refusing to render unchanged source audio.")
        mute_duration = sum(float(region.get("duration", 0.0) or 0.0) for region in mute_regions)
        if mute_duration <= 0.05:
            raise ValueError(f"Best Short candidate {candidate_id} produced negligible mute coverage; refusing to render unchanged source audio.")
    for mapping in mappings:
        start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        render_duration = float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)) or 0.0)
        if start < -0.001 or start >= float(duration) + 0.001:
            raise ValueError(f"Best Short candidate {candidate_id} has an out-of-segment mapping at {start:.3f}s.")
        if start + render_duration > float(duration) + 0.5:
            raise ValueError(f"Best Short candidate {candidate_id} has a mapping extending beyond the extracted segment.")


def _speech_mute_regions(
    schedule: dict,
    *,
    padding: float = 0.0,
    merge_gap: float = 0.0,
    duration: float | None = None,
) -> list[dict[str, float]]:
    regions = []
    for start, end in _speech_slot_regions(schedule):
        if end > start:
            regions.append((start, end))
    if not regions:
        for row in schedule.get("destination_performance_fills", []):
            start = float(row.get("start", 0.0) or 0.0)
            region_duration = float(row.get("duration", 0.0) or 0.0)
            if region_duration > 0:
                regions.append((start, start + region_duration))
    if not regions:
        for mapping in schedule.get("mappings", []):
            if not mapping.get("enabled", True):
                continue
            start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
            region_duration = float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)) or 0.0)
            if region_duration > 0:
                regions.append((start, start + region_duration))
    return _merge_mute_regions(regions, padding=padding, merge_gap=merge_gap, duration=duration)


def _speech_slot_regions(schedule: dict) -> list[tuple[float, float]]:
    regions_by_key: dict[tuple[float, float], tuple[float, float]] = {}
    for mapping in schedule.get("mappings", []):
        if not mapping.get("enabled", True):
            continue
        if mapping.get("alignment_mode") != "speech_window_snap":
            continue
        slot_start = mapping.get("alignment_slot_start")
        slot_end = mapping.get("alignment_slot_end")
        if slot_start is None or slot_end is None:
            continue
        start = float(slot_start)
        end = float(slot_end)
        if end <= start:
            continue
        key = (round(start, 3), round(end, 3))
        regions_by_key[key] = (start, end)
    return sorted(regions_by_key.values(), key=lambda item: item[0])


def _attach_performance_speech_windows(
    performance_rows: list[dict],
    timeline_windows: list[dict],
) -> list[dict]:
    all_timeline_by_id = {str(window.get("id")): window for window in timeline_windows}
    usable_timeline_by_id = {str(window.get("id")): window for window in usable_rows(timeline_windows)}
    enriched = []
    for row in performance_rows:
        item = dict(row)
        speech_windows = []
        for window_id in item.get("speaking_window_ids", []):
            key = str(window_id)
            source = usable_timeline_by_id.get(key)
            source_kind = "detected_speech_window"
            if source is None:
                source = all_timeline_by_id.get(key)
                source_kind = "recovered_filtered_speech_window"
            if not source:
                continue
            start = float(source.get("start", 0.0) or 0.0)
            duration = float(source.get("duration", 0.0) or 0.0)
            end = float(source.get("end", start + duration) or start + duration)
            duration = max(0.0, end - start)
            if duration <= 0.0:
                continue
            speech_windows.append(
                {
                    "id": str(source.get("id")),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(duration, 3),
                    "confidence": source.get("confidence", item.get("confidence", 0.7)),
                    "source_kind": source_kind,
                    "recovered": source_kind != "detected_speech_window",
                    "reject_reason": source.get("reject_reason"),
                }
            )
        if speech_windows:
            item["speech_windows"] = speech_windows
        enriched.append(item)
    return enriched


def _merge_mute_regions(
    regions: list[tuple[float, float]],
    *,
    padding: float,
    merge_gap: float,
    duration: float | None,
) -> list[dict[str, float]]:
    padded = []
    for start, end in regions:
        padded_start = max(0.0, float(start) - max(0.0, padding))
        padded_end = float(end) + max(0.0, padding)
        if duration is not None:
            padded_end = min(float(duration), padded_end)
        if padded_end > padded_start:
            padded.append((padded_start, padded_end))
    if not padded:
        return []

    padded.sort(key=lambda item: item[0])
    merged = [padded[0]]
    for start, end in padded[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + max(0.0, merge_gap):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return [
        {"start": round(start, 3), "duration": round(end - start, 3)}
        for start, end in merged
        if end > start
    ]


def _is_stale(output_path: Path, input_path: Path) -> bool:
    if not output_path.exists() or not input_path.exists():
        return True
    return input_path.stat().st_mtime > output_path.stat().st_mtime













