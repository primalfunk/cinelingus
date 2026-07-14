from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import __version__
from .config import AppConfig
from .speakers import speaker_mapping_summary, speaker_preservation_summary
from .util import utc_now, write_json


def build_run_report(
    *,
    config: AppConfig,
    source_hash: str,
    destination_hash: str,
    destination_movie: dict[str, Any],
    source_movie: dict[str, Any],
    source_events: dict[str, Any],
    filtered_source_events: dict[str, Any],
    clip_library: dict[str, Any],
    destination_timeline: dict[str, Any],
    filtered_destination_timeline: dict[str, Any],
    schedule: dict[str, Any],
    visual_schedule_report: dict[str, Any] | None = None,
    review_notes: dict[str, Any] | None = None,
    review_analysis: dict[str, Any] | None = None,
    audio_output: Path,
    video_output: Path,
    source_performances: dict[str, Any] | None = None,
    destination_performances: dict[str, Any] | None = None,
    performance_placement_report: dict[str, Any] | None = None,
    problem_region_report: dict[str, Any] | None = None,
    editorial_highlights: dict[str, Any] | None = None,
    source_speaker_map: dict[str, Any] | None = None,
    destination_speaker_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_stats = filtered_source_events.get("filter_stats", {})
    timeline_stats = filtered_destination_timeline.get("filter_stats", {})
    mappings = schedule.get("mappings", [])
    enabled_mappings = [mapping for mapping in mappings if mapping.get("enabled", True)]
    disabled_mappings = len(mappings) - len(enabled_mappings)
    usable_windows = int(timeline_stats.get("usable_count", len(filtered_destination_timeline.get("windows", []))))
    skipped_windows = max(0, usable_windows - len(enabled_mappings))
    rendered_dialogue_duration = round(sum(float(m.get("planned_render_duration", 0.0)) for m in enabled_mappings), 3)
    alignment_summary = _alignment_summary(schedule)
    bed_summary = _soundtrack_bed_summary(destination_movie, alignment_summary, config.original_duck_db)
    speaker_summary = _speaker_summary(
        source_events=source_events,
        destination_timeline=destination_timeline,
        filtered_source_events=filtered_source_events,
        filtered_destination_timeline=filtered_destination_timeline,
        schedule=schedule,
        source_speaker_map=source_speaker_map,
        destination_speaker_map=destination_speaker_map,
    )

    warnings = []
    if config.quick_test_seconds is not None:
        warnings.append(f"quick_test_seconds={config.quick_test_seconds}; report reflects a preview analysis window")
    if skipped_windows:
        warnings.append(f"{skipped_windows} usable destination windows were not scheduled")
    if not audio_output.exists():
        warnings.append(f"audio output missing: {audio_output}")
    if not video_output.exists():
        warnings.append(f"video output missing: {video_output}")
    if not mappings:
        warnings.append("replacement schedule has no mappings")
    for role, diagnostics in (("source", speaker_summary.get("source_diagnostics", {})), ("destination", speaker_summary.get("destination_diagnostics", {}))):
        if diagnostics.get("fallback_used"):
            warnings.append(f"{role} speaker diarization used fallback: {diagnostics.get('fallback_reason') or diagnostics.get('effective_backend')}")
        elif diagnostics and diagnostics.get("status") in {"weak", "unavailable"}:
            warnings.append(f"{role} speaker coverage is {diagnostics.get('status')}: {diagnostics.get('labeled_item_count', 0)} / {diagnostics.get('speech_item_count', 0)} items labeled")

    return {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "inputs": {
            "destination_video": {
                "path": str(config.destination_video),
                "media_hash": destination_hash,
                "duration": destination_movie.get("duration"),
                "resolution": destination_movie.get("resolution"),
                "video_codec": destination_movie.get("video_codec"),
                "audio_codec": destination_movie.get("audio_codec"),
            },
            "source_dialogue": {
                "path": str(config.source_dialogue),
                "media_hash": source_hash,
                "duration": source_movie.get("duration"),
                "resolution": source_movie.get("resolution"),
                "video_codec": source_movie.get("video_codec"),
                "audio_codec": source_movie.get("audio_codec"),
            },
        },
        "configuration": {
            "speech_backend": config.speech_backend,
            "transcription_mode": config.transcription_mode,
            "whisper_model": config.whisper_model,
            "whisper_language": config.whisper_language,
            "quick_test_seconds": config.quick_test_seconds,
            "scheduling_mode": config.scheduling_mode,
            "shot_boundary_mode": config.shot_boundary_mode,
            "best_fit_lookahead": config.best_fit_lookahead,
            "max_time_stretch": config.max_time_stretch,
            "target_lufs": config.target_lufs,
            "audio_fade_duration": config.audio_fade_duration,
            "original_duck_db": config.original_duck_db,
            "cinematic_filter": config.cinematic_filter,
        },
        "counts": {
            "raw_source_events": len(source_events.get("events", [])),
            "filtered_source_events": int(source_stats.get("usable_count", 0)),
            "rejected_source_events": int(source_stats.get("rejected_count", 0)),
            "source_clips": len(clip_library.get("clips", [])),
            "source_performances": len((source_performances or {}).get("performances", [])),
            "raw_destination_windows": len(destination_timeline.get("windows", [])),
            "destination_performances": len((destination_performances or {}).get("performances", [])),
            "filtered_destination_windows": usable_windows,
            "rejected_destination_windows": int(timeline_stats.get("rejected_count", 0)),
            "scheduled_mappings": len(mappings),
            "enabled_mappings": len(enabled_mappings),
            "disabled_mappings": disabled_mappings,
            "skipped_windows": skipped_windows,
        },
        "transformation": {
            "name": schedule.get("transformation_name", "movie_masher"),
            "history": schedule.get("transformation_history", []),
        },
        "soundtrack_bed": bed_summary,
        "speakers": speaker_summary,
        "schedule": {
            "mode": schedule.get("scheduling_mode"),
            "shot_boundary_mode": schedule.get("shot_boundary_mode", config.shot_boundary_mode),
            "average_score": _average([m.get("score") for m in enabled_mappings]),
            "average_visual_fit_score": _average([m.get("visual_fit_score") for m in enabled_mappings]),
            "crossing_mappings": sum(1 for m in enabled_mappings if m.get("mapping_crosses_shot_boundary")),
            "rendered_dialogue_duration": rendered_dialogue_duration,
            "active_filter": schedule.get("active_filter", config.cinematic_filter),
            "active_filter_display_name": schedule.get("active_filter_display_name"),
            "average_performance_similarity": _average([m.get("performance_similarity_score") for m in enabled_mappings]),
            "average_baseline_similarity": _average([m.get("baseline_similarity_score") for m in enabled_mappings]),
            "alignment": alignment_summary,
        },
        "performances": {
            "source": source_performances or {},
            "destination": destination_performances or {},
        },
        "visual_schedule": visual_schedule_report or {},
        "performance_placement_report": performance_placement_report or {},
        "problem_region_report": problem_region_report or {},
        "editorial_highlights": editorial_highlights or {},
        "review_notes": review_notes or {},
        "review_analysis": review_analysis or {},
        "outputs": {
            "audio_path": str(audio_output),
            "audio_exists": audio_output.exists(),
            "video_path": str(video_output),
            "video_exists": video_output.exists(),
            "run_report_json": str(config.output_dir / "run_report.json"),
            "run_report_txt": str(config.output_dir / "run_report.txt"),
            "schedule_report_csv": str(config.output_dir / "schedule_report.csv"),
            "cinematic_index": str(config.output_dir / "cinematic_index.json"),
            "performance_placement_report": str(config.output_dir / "performance_placement_report.json"),
            "performance_placement_report_txt": str(config.output_dir / "performance_placement_report.txt"),
            "performance_placement_report_csv": str(config.output_dir / "performance_placement_report.csv"),
            "problem_regions": str(config.output_dir / "problem_regions.json"),
            "problem_regions_txt": str(config.output_dir / "problem_regions.txt"),
            "problem_regions_csv": str(config.output_dir / "problem_regions.csv"),
            "editorial_highlights": str(config.output_dir / "editorial_highlights.json"),
            "taste_profile": str(config.output_dir / "taste_profile.json"),
        },
        "warnings": warnings,
        "errors": [],
    }


def _speaker_summary(
    *,
    source_events: dict[str, Any],
    destination_timeline: dict[str, Any],
    filtered_source_events: dict[str, Any],
    filtered_destination_timeline: dict[str, Any],
    schedule: dict[str, Any],
    source_speaker_map: dict[str, Any] | None = None,
    destination_speaker_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_rows = _prefer_speaker_rows(filtered_source_events.get("events", []), source_events.get("events", []))
    destination_rows = _prefer_speaker_rows(filtered_destination_timeline.get("windows", []), destination_timeline.get("windows", []))
    source_diagnostics = (source_speaker_map or {}).get("diagnostics") or filtered_source_events.get("speaker_diagnostics") or source_events.get("speaker_diagnostics") or {}
    destination_diagnostics = (destination_speaker_map or {}).get("diagnostics") or filtered_destination_timeline.get("speaker_diagnostics") or destination_timeline.get("speaker_diagnostics") or {}
    return {
        "source_dialogue": _speaker_counts(source_rows),
        "destination_video": _speaker_counts(destination_rows),
        "source_diagnostics": source_diagnostics,
        "destination_diagnostics": destination_diagnostics,
        "source_warnings": (source_speaker_map or {}).get("warnings") or filtered_source_events.get("speaker_warnings", []),
        "destination_warnings": (destination_speaker_map or {}).get("warnings") or filtered_destination_timeline.get("speaker_warnings", []),
        "schedule_preservation": speaker_preservation_summary(schedule),
        "speaker_mapping": speaker_mapping_summary(schedule),
    }

def _prefer_speaker_rows(primary: list[dict[str, Any]], fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(row.get("speaker_id") or row.get("speaker") for row in primary):
        return primary
    if any(row.get("speaker_id") or row.get("speaker") for row in fallback):
        return fallback
    return primary or fallback

def _speaker_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations: dict[str, float] = {}
    for row in rows:
        speaker_id = row.get("speaker_id") or row.get("speaker")
        if not speaker_id:
            continue
        duration = _coerce_float(row.get("duration"), 0.0)
        durations[str(speaker_id)] = durations.get(str(speaker_id), 0.0) + duration
    return {
        "speaker_count": len(durations),
        "duration_by_speaker": {key: round(value, 3) for key, value in sorted(durations.items())},
    }


def _alignment_summary(schedule: dict[str, Any]) -> dict[str, Any]:
    mappings = [mapping for mapping in schedule.get("mappings", []) if mapping.get("enabled", True)]
    snapped = [mapping for mapping in mappings if mapping.get("alignment_mode") == "speech_window_snap"]
    fallback = [mapping for mapping in mappings if mapping.get("alignment_mode") != "speech_window_snap"]
    detected_snaps = [mapping for mapping in snapped if mapping.get("alignment_source_kind", "detected_speech_window") == "detected_speech_window"]
    recovered_snaps = [mapping for mapping in snapped if mapping.get("alignment_source_kind") == "recovered_filtered_speech_window"]
    synthetic_snaps = [mapping for mapping in snapped if mapping.get("alignment_source_kind") == "synthetic_speech_slot"]
    unique_slots: dict[tuple[float, float], None] = {}
    for mapping in snapped:
        slot_start = mapping.get("alignment_slot_start")
        slot_end = mapping.get("alignment_slot_end")
        if slot_start is None or slot_end is None:
            continue
        start = float(slot_start)
        end = float(slot_end)
        if end > start:
            unique_slots[(round(start, 3), round(end, 3))] = None
    speech_slot_duration = sum(end - start for start, end in unique_slots)
    merged_duck_regions = _merge_report_intervals(list(unique_slots), padding=0.35, merge_gap=0.25)
    merged_duck_duration = sum(end - start for start, end in merged_duck_regions)
    fills = schedule.get("destination_performance_fills", [])
    return {
        "speech_snapped_mappings": len(snapped),
        "detected_speech_snaps": len(detected_snaps),
        "recovered_speech_snaps": len(recovered_snaps),
        "synthetic_speech_snaps": len(synthetic_snaps),
        "fallback_mappings": len(fallback),
        "speech_snap_rate": round(len(snapped) / len(mappings), 4) if mappings else 0.0,
        "unique_speech_slots": len(unique_slots),
        "speech_slot_duration": round(speech_slot_duration, 3),
        "merged_duck_region_count": len(merged_duck_regions),
        "merged_ducked_duration": round(merged_duck_duration, 3),
        "average_alignment_spillover": _average([mapping.get("alignment_spillover_seconds") for mapping in snapped]),
        "speech_basis_performances": sum(1 for row in fills if row.get("coverage_basis") == "speech_windows"),
        "performance_basis_performances": sum(1 for row in fills if row.get("coverage_basis") != "speech_windows"),
        "underfilled_performances": sum(1 for row in fills if float(row.get("coverage", 0.0) or 0.0) < float(row.get("target_coverage", 0.0) or 0.0)),
    }


def _merge_report_intervals(
    intervals: list[tuple[float, float]],
    *,
    padding: float,
    merge_gap: float,
) -> list[tuple[float, float]]:
    padded = [(max(0.0, start - padding), end + padding) for start, end in intervals if end > start]
    if not padded:
        return []
    padded.sort(key=lambda item: item[0])
    merged = [padded[0]]
    for start, end in padded[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + merge_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _soundtrack_bed_summary(destination_movie: dict[str, Any], alignment_summary: dict[str, Any], duck_db: float) -> dict[str, Any]:
    destination_duration = _coerce_float(destination_movie.get("duration"), 0.0)
    ducked_duration = min(destination_duration, _coerce_float(alignment_summary.get("merged_ducked_duration"), 0.0))
    preserved_full_volume = max(0.0, destination_duration - ducked_duration)
    return {
        "strategy": "duck_original_under_replacement_dialogue",
        "duck_db": round(float(duck_db), 3),
        "destination_duration": round(destination_duration, 3),
        "duck_region_count": int(alignment_summary.get("merged_duck_region_count") or 0),
        "ducked_duration": round(ducked_duration, 3),
        "preserved_full_volume_duration": round(preserved_full_volume, 3),
        "continuous_original_bed": True,
    }


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_report_files(report: dict[str, Any], schedule: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "run_report.json"
    txt_path = output_dir / "run_report.txt"
    csv_path = output_dir / "schedule_report.csv"

    write_json(json_path, report)
    txt_path.write_text(_format_text_report(report), encoding="utf-8")
    _write_schedule_csv(schedule, csv_path)
    return {"json": json_path, "txt": txt_path, "csv": csv_path}


def _format_text_report(report: dict[str, Any]) -> str:
    counts = report["counts"]
    outputs = report["outputs"]
    config = report["configuration"]
    schedule = report["schedule"]
    transformation = report.get("transformation", {})
    destination = report["inputs"]["destination_video"]
    source = report["inputs"]["source_dialogue"]
    lines = [
        "Cinelingus Transposition Run Report",
        "=====================================",
        f"Created: {report['creation_timestamp']}",
        "",
        "Inputs",
        f"  destination_video: {destination['path']}",
        f"  destination hash: {destination['media_hash']}",
        f"  source_dialogue: {source['path']}",
        f"  source hash: {source['media_hash']}",
        "",
        "Configuration",
        f"  speech backend: {config['speech_backend']}",
        f"  transcription mode: {config['transcription_mode']}",
        f"  whisper model: {config['whisper_model']}",
        f"  quick seconds: {config['quick_test_seconds']}",
        f"  scheduling mode: {config['scheduling_mode']}",
        f"  shot boundary mode: {config.get('shot_boundary_mode')}",
        f"  target LUFS: {config['target_lufs']}",
        f"  original duck dB: {config.get('original_duck_db')}",
        "",
        "Counts",
        f"  raw source events: {counts['raw_source_events']}",
        f"  filtered source events: {counts['filtered_source_events']}",
        f"  rejected source events: {counts['rejected_source_events']}",
        f"  source clips: {counts['source_clips']}",
        f"  source performances: {counts.get('source_performances', 0)}",
        f"  raw destination windows: {counts['raw_destination_windows']}",
        f"  destination performances: {counts.get('destination_performances', 0)}",
        f"  filtered destination windows: {counts['filtered_destination_windows']}",
        f"  scheduled mappings: {counts['scheduled_mappings']}",
        f"  enabled mappings: {counts['enabled_mappings']}",
        f"  disabled mappings: {counts['disabled_mappings']}",
        f"  skipped windows: {counts['skipped_windows']}",
        "",
        "Transformation",
        f"  name: {transformation.get('name')}",
        f"  steps: {len(transformation.get('history', []))}",
        "",
        "Schedule",
        f"  average score: {schedule['average_score']}",
        f"  average visual fit: {schedule.get('average_visual_fit_score')}",
        f"  crossing mappings: {schedule.get('crossing_mappings')}",
        f"  rendered dialogue duration: {schedule['rendered_dialogue_duration']}s",
    ]
    alignment = schedule.get("alignment") or {}
    if alignment:
        lines.extend([
            f"  speech-snapped mappings: {alignment.get('speech_snapped_mappings')}",
            f"  detected speech snaps: {alignment.get('detected_speech_snaps')}",
            f"  recovered speech snaps: {alignment.get('recovered_speech_snaps')}",
            f"  synthetic speech snaps: {alignment.get('synthetic_speech_snaps')}",
            f"  fallback mappings: {alignment.get('fallback_mappings')}",
            f"  unique muted speech spans: {alignment.get('unique_speech_slots')}",
            f"  raw speech-span duration: {alignment.get('speech_slot_duration')}s",
            f"  merged duck regions: {alignment.get('merged_duck_region_count')}",
            f"  merged ducked duration: {alignment.get('merged_ducked_duration')}s",
            f"  performance fills using speech basis: {alignment.get('speech_basis_performances')}",
            f"  performance fills using broad basis: {alignment.get('performance_basis_performances')}",
        ])
    speakers = report.get("speakers") or {}
    if speakers:
        source = speakers.get("source_dialogue") or {}
        destination = speakers.get("destination_video") or {}
        preservation = speakers.get("schedule_preservation") or {}
        mapping = speakers.get("speaker_mapping") or {}
        source_diag = speakers.get("source_diagnostics") or {}
        destination_diag = speakers.get("destination_diagnostics") or {}
        lines.extend([
            "",
            "Speakers",
            f"  source speakers: {source.get('speaker_count')}",
            f"  source speaker status: {source_diag.get('status', 'unknown')} via {source_diag.get('effective_backend', 'unknown')}",
            f"  source speaker coverage: {source_diag.get('labeled_item_count', 0)} / {source_diag.get('speech_item_count', 0)} items",
            f"  destination speakers: {destination.get('speaker_count')}",
            f"  destination speaker status: {destination_diag.get('status', 'unknown')} via {destination_diag.get('effective_backend', 'unknown')}",
            f"  destination speaker coverage: {destination_diag.get('labeled_item_count', 0)} / {destination_diag.get('speech_item_count', 0)} items",
            f"  same-speaker placements: {preservation.get('same_speaker_count')} / {preservation.get('speaker_aware_mapping_count')}",
            f"  same-speaker rate: {preservation.get('same_speaker_rate')}",
            f"  speaker-map placements: {mapping.get('speaker_mapping_followed_count')} / {mapping.get('speaker_mapping_aware_count')}",
            f"  speaker-map rate: {mapping.get('speaker_mapping_followed_rate')}",
        ])
    bed = report.get("soundtrack_bed") or {}
    if bed:
        lines.extend([
            "",
            "Soundtrack Bed",
            f"  strategy: {bed.get('strategy')}",
            f"  duck dB: {bed.get('duck_db')}",
            f"  duck regions: {bed.get('duck_region_count')}",
            f"  ducked duration: {bed.get('ducked_duration')}s",
            f"  preserved full-volume duration: {bed.get('preserved_full_volume_duration')}s",
        ])
    lines.extend([
        "",
        "Performance Placements",
        f"  placements: {(report.get('performance_placement_report') or {}).get('placement_count')}",
        f"  average quality: {((report.get('performance_placement_report') or {}).get('summary') or {}).get('average_quality_score')}",
        f"  warnings: {((report.get('performance_placement_report') or {}).get('summary') or {}).get('warning_count')}",
    ])
    problem_report = report.get("problem_region_report") or {}
    if problem_report:
        summary = problem_report.get("summary") or {}
        lines.extend([
            "",
            "Problem Regions",
            f"  total problems: {problem_report.get('problem_count')}",
            f"  fallback mappings: {summary.get('fallback_mapping_count')}",
            f"  underfilled performances: {summary.get('underfilled_performance_count')}",
            f"  low-fit mappings: {summary.get('low_fit_mapping_count')}",
        ])
    editorial = report.get("editorial_highlights") or {}
    if editorial:
        summary = editorial.get("summary") or {}
        highlights = editorial.get("highlights") or {}
        lines.extend([
            "",
            "Editorial Highlights",
            f"  evaluated performances: {summary.get('evaluated_performances')}",
            f"  average editorial score: {summary.get('average_editorial_score')}",
            f"  positive highlights: {summary.get('positive_highlight_count')}",
            f"  needs review: {summary.get('needs_review_count')}",
            f"  convincing picks: {len(highlights.get('most_convincing', []))}",
            f"  funny picks: {len(highlights.get('funniest', []))}",
            f"  awkward picks: {len(highlights.get('most_awkward', []))}",
        ])
    lines.extend([
        "",
        "Outputs",
        f"  audio: {outputs['audio_path']}",
        f"  video: {outputs['video_path']}",
        f"  schedule CSV: {outputs['schedule_report_csv']}",
        f"  CIR index: {outputs['cinematic_index']}",
    ])
    visual_schedule = report.get("visual_schedule") or {}
    if visual_schedule:
        lines.extend([
            "",
            "Visual Schedule",
            f"  total shots: {visual_schedule.get('total_shots')}",
            f"  empty dialogue shots: {len(visual_schedule.get('empty_dialogue_shots', []))}",
            f"  overloaded shots: {len(visual_schedule.get('overloaded_shots', []))}",
            f"  crossing mappings: {len(visual_schedule.get('crossing_mappings', []))}",
        ])
    review_notes = report.get("review_notes") or {}
    if review_notes:
        lines.extend([
            "",
            "Review Notes",
            f"  reviewed mappings: {review_notes.get('reviewed_mappings')}",
        ])
        for label, count in (review_notes.get("label_counts") or {}).items():
            if count:
                lines.append(f"  {label}: {count}")
    review_analysis = report.get("review_analysis") or {}
    if review_analysis:
        lines.extend([
            "",
            "Review Analysis",
            f"  good mappings: {review_analysis.get('good_mappings')}",
            f"  bad mappings: {review_analysis.get('bad_mappings')}",
        ])
        for item in review_analysis.get("recommendations", [])[:5]:
            lines.append(f"  - {item}")
    if report.get("warnings"):
        lines.extend(["", "Warnings"])
        lines.extend(f"  - {warning}" for warning in report["warnings"])
    if report.get("errors"):
        lines.extend(["", "Errors"])
        lines.extend(f"  - {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def _write_schedule_csv(schedule: dict[str, Any], output_path: Path) -> None:
    fields = [
        "window_id",
        "clip_id",
        "enabled",
        "destination_timestamp",
        "clip_trim_duration",
        "planned_render_duration",
        "stretch_factor",
        "score",
        "selection_reason",
        "timing_strategy",
        "skipped_source_clips",
        "shot_id",
        "crosses_shot_boundary",
        "mapping_crosses_shot_boundary",
        "boundary_overrun_seconds",
        "visual_fit_score",
        "source_performance_id",
        "source_performance_type",
        "destination_performance_id",
        "source_transcript",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mapping in schedule.get("mappings", []):
            writer.writerow({field: mapping.get(field, "") for field in fields})


def _average(values: list[Any]) -> float | None:
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            pass
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)




