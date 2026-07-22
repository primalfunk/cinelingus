from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..pipeline import Pipeline
from ..util import read_json, stable_hash, utc_now, write_json
from ..render_verification import evaluate_rendered_dialogue
from ..whisper_backend import transcribe_words_with_whisper
from .review import build_blinded_semantic_review_package
from .acoustic_preflight import require_accepted_semantic_preflight


def run_semantic_render_proof(
    *, config: AppConfig, screen_dir: Path, output_dir: Path,
    control_variant: str = "control", semantic_variant: str = "assisted_005",
    force: bool = False, preflight_path: Path | None = None,
) -> dict[str, Any]:
    """Render and verify one schedule-selected control/semantic pair under identical settings."""
    screen = read_json(screen_dir / "semantic_schedule_screen.json")
    selected = set(screen.get("render_selection") or [])
    if control_variant not in selected or semantic_variant not in selected:
        raise ValueError("Requested proof variants were not nominated by the schedule screen")
    if preflight_path is None:
        raise ValueError("An accepted semantic acoustic preflight is required before rendering")
    preflight = require_accepted_semantic_preflight(
        preflight_path, screen_signature=screen["experiment_signature"], semantic_variant=semantic_variant,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for variant_id in (control_variant, semantic_variant):
        variant_dir = output_dir / variant_id
        variant_config = replace(config, output_dir=variant_dir)
        pipeline = Pipeline(variant_config)
        requested_schedule = read_json(screen_dir / f"{variant_id}_schedule.json")
        completed = _completed_variant(variant_dir, requested_schedule)
        if completed is not None and not force:
            schedule, audio, video = completed
            schedule = _refine_rendered_verification_with_words(
                schedule=schedule, audio=audio, variant_dir=variant_dir, config=config, force=False,
            )
            variants.append(_variant_result(variant_id, variant_dir, schedule, audio, video))
            continue
        schedule = requested_schedule
        destination_movie = pipeline._inspect_one(pipeline.destination, force=False)
        audio = pipeline.render_audio_from_schedule(
            schedule=schedule,
            dest_movie=destination_movie,
            force=force,
            output_path=variant_dir / "replacement_dialogue.wav",
            persist_schedule=False,
        )
        video = pipeline.render_video_from_audio(
            audio=audio,
            force=force,
            output_path=variant_dir / "translation_output.mp4",
            duration=float(destination_movie["duration"]),
        )
        write_json(variant_dir / "final_schedule.json", schedule)
        schedule = _refine_rendered_verification_with_words(
            schedule=schedule, audio=audio, variant_dir=variant_dir, config=config, force=force,
        )
        variants.append(_variant_result(variant_id, variant_dir, schedule, audio, video))
    intervention_repair = {"state": "NOT_REQUIRED", "rolled_back_placement_keys": []}
    preliminary_intervention = assess_semantic_intervention(
        control_requested=read_json(screen_dir / f"{control_variant}_schedule.json"),
        semantic_requested=read_json(screen_dir / f"{semantic_variant}_schedule.json"),
        control_final=read_json(output_dir / control_variant / "final_schedule.json"),
        semantic_final=read_json(output_dir / semantic_variant / "final_schedule.json"),
    )
    failed_keys = [
        str(row["placement_key"]) for row in preliminary_intervention.get("mappings", [])
        if row.get("semantic_donor_survived_repair") and row.get("semantic_status") != "PASS"
    ]
    if failed_keys:
        control_final = read_json(output_dir / control_variant / "final_schedule.json")
        semantic_final = read_json(output_dir / semantic_variant / "final_schedule.json")
        repaired_schedule = _rollback_placements_to_control(
            semantic_final, control_final, set(failed_keys),
        )
        repaired_schedule["semantic_intervention_repair"] = {
            "strategy": "failed_changed_placement_control_rollback_v1",
            "rolled_back_placement_keys": failed_keys,
        }
        variant_dir = output_dir / semantic_variant
        pipeline = Pipeline(replace(config, output_dir=variant_dir))
        destination_movie = pipeline._inspect_one(pipeline.destination, force=False)
        audio = pipeline.render_audio_from_schedule(
            schedule=repaired_schedule, dest_movie=destination_movie, force=True,
            output_path=variant_dir / "replacement_dialogue.wav", persist_schedule=False,
        )
        video = pipeline.render_video_from_audio(
            audio=audio, force=True, output_path=variant_dir / "translation_output.mp4",
            duration=float(destination_movie["duration"]),
        )
        write_json(variant_dir / "final_schedule.json", repaired_schedule)
        repaired_schedule = _refine_rendered_verification_with_words(
            schedule=repaired_schedule, audio=audio, variant_dir=variant_dir, config=config, force=True,
        )
        variants[1] = _variant_result(semantic_variant, variant_dir, repaired_schedule, audio, video)
        intervention_repair = {"state": "APPLIED", "rolled_back_placement_keys": failed_keys}
    invariant_config = {
        "destination_media_hash": variants[0]["destination_media_hash"],
        "render_sample_rate": config.render_sample_rate,
        "render_channels": config.render_channels,
        "target_lufs": config.target_lufs,
        "audio_fade_duration": config.audio_fade_duration,
        "dialogue_suppression": config.dialogue_suppression,
        "suppression_padding": config.suppression_padding,
        "background_reconstruction": config.background_reconstruction,
        "whisper_model": config.whisper_model,
        "whisper_language": config.whisper_language,
        "transcription_mode": config.transcription_mode,
        "residue_correction_passes": config.residue_correction_passes,
    }
    technical_ready = all(row["audio_ready"] and row["video_ready"] for row in variants)
    verification_available = all(
        row["voice_residue_status"] != "UNAVAILABLE" and row["rendered_dialogue_status"] != "UNAVAILABLE"
        for row in variants
    )
    intervention = assess_semantic_intervention(
        control_requested=read_json(screen_dir / f"{control_variant}_schedule.json"),
        semantic_requested=read_json(screen_dir / f"{semantic_variant}_schedule.json"),
        control_final=read_json(output_dir / control_variant / "final_schedule.json"),
        semantic_final=read_json(output_dir / semantic_variant / "final_schedule.json"),
    )
    acceptance = assess_render_acceptance(variants, intervention=intervention)
    report = {
        "schema_version": "1.0", "proof_version": "semantic_render_proof_v1",
        "creation_timestamp": utc_now(),
        "screen_signature": screen["experiment_signature"],
        "proof_signature": stable_hash({"screen": screen["experiment_signature"], "config": invariant_config, "variants": variants}),
        "control_variant": control_variant, "semantic_variant": semantic_variant,
        "acoustic_preflight": str(preflight_path),
        "acoustic_preflight_signature": preflight["preflight_signature"],
        "invariant_configuration": invariant_config, "variants": variants,
        "technical_render_state": "COMPLETE" if technical_ready else "INCOMPLETE",
        "verification_state": "AVAILABLE" if verification_available else "PARTIAL_OR_UNAVAILABLE",
        "render_acceptance": acceptance,
        "semantic_intervention_verification": intervention,
        "semantic_intervention_repair": intervention_repair,
        "repair_lineage": screen.get("acoustic_repair") or {},
        "editorial_repair_state": "NOT_RUN_NON_PERFORMANCE_SCHEDULE",
        "claim_scope": "Rendered technical and verification evidence only; semantic relatedness and preference require blinded human review.",
    }
    write_json(output_dir / "semantic_render_proof.json", report)
    if acceptance["state"] == "ACCEPTED_FOR_HUMAN_REVIEW":
        changed_count = _changed_mapping_count(
            read_json(screen_dir / f"{control_variant}_schedule.json"),
            read_json(screen_dir / f"{semantic_variant}_schedule.json"),
        )
        review_case = {
            "case_id": f"{screen['source_hash'][:12]}_to_{screen['destination_hash'][:12]}_{semantic_variant}",
            "control_media": output_dir / control_variant / "translation_output.mp4",
            "semantic_media": output_dir / semantic_variant / "translation_output.mp4",
            "destination_context": (
                f"Full Translation render of {config.destination_video.name}; "
                f"semantic-assisted schedule changes {changed_count} donor placement(s)."
            ),
        }
        build_blinded_semantic_review_package([review_case], output_dir / "blinded_review")
        report["review_case"] = {
            "case_id": review_case["case_id"],
            "destination_context": review_case["destination_context"],
            "changed_mapping_count": changed_count,
        }
        report["blinded_review_manifest"] = "blinded_review/review_manifest.json"
        write_json(output_dir / "semantic_render_proof.json", report)
    else:
        report["human_review_state"] = "WITHHELD_TECHNICAL_FAILURE"
        write_json(output_dir / "semantic_render_proof.json", report)
    return report


def _refine_rendered_verification_with_words(
    *, schedule: dict[str, Any], audio: Path, variant_dir: Path,
    config: AppConfig, force: bool,
) -> dict[str, Any]:
    """Use word timestamps to avoid broad Whisper-window attribution bleed."""
    words_path = variant_dir / "rendered_dialogue_words.json"
    signature = stable_hash({
        "version": "semantic_render_word_verification_v1",
        "audio_sha256": _digest(audio), "model": config.whisper_model,
        "language": config.whisper_language, "mode": config.transcription_mode,
    })
    cached = read_json(words_path) if words_path.is_file() else {}
    if not force and cached.get("media_hash") == signature:
        word_timeline = cached
    else:
        word_timeline = transcribe_words_with_whisper(
            audio_path=audio, media_hash=signature, output_path=words_path,
            model_name=config.whisper_model, language=config.whisper_language,
            transcription_mode=config.transcription_mode,
        )
    windows = _word_timeline_as_windows(word_timeline)
    if not windows:
        return schedule
    refined = evaluate_rendered_dialogue(
        schedule=schedule, rendered_timeline={"windows": windows},
    )
    refined["attribution_basis"] = "whisper_word_timestamps"
    refined["segment_level_verification"] = schedule.get("rendered_dialogue_verification") or {}
    write_json(variant_dir / "rendered_dialogue_verification_words.json", refined)
    updated = {**schedule, "rendered_dialogue_verification": refined}
    write_json(variant_dir / "final_schedule.json", updated)
    return updated


def _word_timeline_as_windows(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for index, row in enumerate(timeline.get("words") or [], start=1):
        start, end = float(row.get("start", 0.0) or 0.0), float(row.get("end", 0.0) or 0.0)
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        windows.append({
            "id": str(row.get("id") or f"word_{index:06d}"),
            "start": max(0.0, midpoint - 0.001), "end": midpoint + 0.001,
            "transcript": str(row.get("text") or "").strip(),
            "confidence": float(row.get("probability", 0.0) or 0.0),
            "word_start": start, "word_end": end,
        })
    return windows


def _completed_variant(
    variant_dir: Path, requested_schedule: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path] | None:
    final_schedule_path = variant_dir / "final_schedule.json"
    audio = variant_dir / "replacement_dialogue.wav"
    video = variant_dir / "translation_output.mp4"
    if not final_schedule_path.is_file() or not audio.is_file() or not video.is_file():
        return None
    if audio.stat().st_size <= 44 or video.stat().st_size <= 0:
        return None
    completed_schedule = read_json(final_schedule_path)
    requested_destination = requested_schedule.get("destination_media_hash") or requested_schedule.get("media_hash")
    completed_destination = completed_schedule.get("destination_media_hash") or completed_schedule.get("media_hash")
    if requested_destination != completed_destination:
        return None
    if not completed_schedule.get("voice_residue_verification") or not completed_schedule.get("rendered_dialogue_verification"):
        return None
    return completed_schedule, audio, video


def _variant_result(variant_id: str, variant_dir: Path, schedule: dict[str, Any], audio: Path, video: Path) -> dict[str, Any]:
    residue = schedule.get("voice_residue_verification") or {}
    rendered = schedule.get("rendered_dialogue_verification") or {}
    admission = schedule.get("semantic_pareto_admission") or {}
    guarded = int(admission.get("admission_count", 0) or 0) > 0
    return {
        "variant_id": variant_id,
        "semantic_mode": "SEMANTIC_ASSISTED" if guarded else (schedule.get("semantic_scoring") or {}).get("mode", "SEMANTIC_DISABLED"),
        "semantic_weight": float((schedule.get("semantic_scoring") or {}).get("configured_weight", 0.0) or 0.0),
        "semantic_selection_policy": admission.get("policy") if guarded else None,
        "destination_media_hash": schedule.get("destination_media_hash") or schedule.get("media_hash"),
        "mapping_count": len(schedule.get("mappings") or []),
        "audio": audio.relative_to(variant_dir.parent).as_posix(),
        "video": video.relative_to(variant_dir.parent).as_posix(),
        "final_schedule": (variant_dir / "final_schedule.json").relative_to(variant_dir.parent).as_posix(),
        "audio_ready": audio.is_file() and audio.stat().st_size > 44,
        "video_ready": video.is_file() and video.stat().st_size > 0,
        "audio_bytes": audio.stat().st_size if audio.is_file() else 0,
        "video_bytes": video.stat().st_size if video.is_file() else 0,
        "audio_sha256": _digest(audio) if audio.is_file() else None,
        "video_sha256": _digest(video) if video.is_file() else None,
        "voice_residue_status": residue.get("status", "UNAVAILABLE"),
        "rendered_dialogue_status": rendered.get("status", "UNAVAILABLE"),
        "rendered_dialogue_average_word_coverage_percentage": rendered.get("average_word_coverage_percentage"),
        "rendered_dialogue_failed_mapping_count": rendered.get("failed_mapping_count"),
        "rendered_dialogue_warning_mapping_count": rendered.get("warning_mapping_count"),
        "rendered_dialogue_measurable_mapping_count": rendered.get("measurable_mapping_count"),
        "residue_correction_report": schedule.get("residue_correction_report", {}),
        "editorial_refinement": schedule.get("editorial_refinement"),
    }


def assess_render_acceptance(
    variants: list[dict[str, Any]], *, intervention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(variants) != 2:
        return {"state": "REJECTED", "reasons": ["exactly_two_variants_required"]}
    control, semantic = variants
    reasons = []
    if not all(row.get("audio_ready") and row.get("video_ready") for row in variants):
        reasons.append("render_artifact_incomplete")
    if any(row.get("voice_residue_status") != "NONE_DETECTED" for row in variants):
        reasons.append("voice_residue_not_clear")
    if any(row.get("rendered_dialogue_status") in {"UNAVAILABLE", None} for row in variants):
        reasons.append("rendered_dialogue_verification_not_passing")
    control_failed = int(control.get("rendered_dialogue_failed_mapping_count") or 0)
    semantic_failed = int(semantic.get("rendered_dialogue_failed_mapping_count") or 0)
    if semantic_failed > control_failed:
        reasons.append("semantic_failed_mapping_count_regressed")
    control_coverage = float(control.get("rendered_dialogue_average_word_coverage_percentage") or 0.0)
    semantic_coverage = float(semantic.get("rendered_dialogue_average_word_coverage_percentage") or 0.0)
    if semantic_coverage + 1e-9 < control_coverage:
        reasons.append("semantic_word_coverage_regressed")
    if intervention and int(intervention.get("semantic_failed_mapping_count", 0) or 0) > 0:
        reasons.append("semantic_intervention_mapping_not_passing")
    return {
        "state": "ACCEPTED_FOR_HUMAN_REVIEW" if not reasons else "REJECTED",
        "reasons": reasons,
        "control_average_word_coverage_percentage": control_coverage,
        "semantic_average_word_coverage_percentage": semantic_coverage,
        "control_failed_mapping_count": control_failed,
        "semantic_failed_mapping_count": semantic_failed,
        "shared_baseline_failed_mapping_count": min(control_failed, semantic_failed),
        "shared_baseline_failures_disclosed": control_failed > 0 and semantic_failed <= control_failed,
    }


def assess_semantic_intervention(
    *, control_requested: dict[str, Any], semantic_requested: dict[str, Any],
    control_final: dict[str, Any], semantic_final: dict[str, Any],
) -> dict[str, Any]:
    requested_control = _by_placement(control_requested.get("mappings") or [])
    requested_semantic = _by_placement(semantic_requested.get("mappings") or [])
    changed_keys = sorted(
        key for key, row in requested_semantic.items()
        if key not in requested_control or _donor(row) != _donor(requested_control[key])
    )
    final_control = _by_placement(control_final.get("mappings") or [])
    final_semantic = _by_placement(semantic_final.get("mappings") or [])
    control_verification = _by_placement((control_final.get("rendered_dialogue_verification") or {}).get("mappings") or [])
    semantic_verification = _by_placement((semantic_final.get("rendered_dialogue_verification") or {}).get("mappings") or [])
    rows = []
    rejected_sources: set[str] = set()
    for key in changed_keys:
        requested = requested_semantic[key]
        semantic_mapping = final_semantic.get(key, {})
        semantic_check = semantic_verification.get(key, {})
        control_check = control_verification.get(key, {})
        survived = _donor(requested) == _donor(semantic_mapping)
        semantic_status = str(semantic_check.get("status") or "UNAVAILABLE").upper() if survived else "ROLLED_BACK"
        requested_source = str(requested.get("source_performance_id") or "")
        if semantic_status not in {"PASS", "ROLLED_BACK"} and requested_source:
            rejected_sources.add(requested_source)
        rows.append({
            "placement_key": key,
            "destination_performance_id": requested.get("destination_performance_id") or requested.get("window_id"),
            "requested_source_performance_id": requested_source or None,
            "requested_clip_id": requested.get("clip_id"),
            "final_source_performance_id": semantic_mapping.get("source_performance_id"),
            "final_clip_id": semantic_mapping.get("clip_id"),
            "semantic_donor_survived_repair": survived,
            "control_status": str(control_check.get("status") or "UNAVAILABLE").upper(),
            "control_word_coverage_percentage": control_check.get("word_coverage_percentage"),
            "semantic_status": semantic_status,
            "semantic_word_coverage_percentage": semantic_check.get("word_coverage_percentage"),
            "semantic_rendered_transcript": semantic_check.get("rendered_transcript"),
        })
    failed = [row for row in rows if row["semantic_status"] not in {"PASS", "ROLLED_BACK"}]
    rolled_back = [row for row in rows if row["semantic_status"] == "ROLLED_BACK"]
    return {
        "policy": "CHANGED_PLACEMENT_RENDER_VERIFICATION_V1",
        "changed_mapping_count": len(rows),
        "semantic_passed_mapping_count": sum(row["semantic_status"] == "PASS" for row in rows),
        "semantic_failed_mapping_count": len(failed),
        "semantic_rolled_back_mapping_count": len(rolled_back),
        "rejected_source_performance_ids": sorted(rejected_sources),
        "mappings": rows,
        "state": "PASS" if rows and not failed else "FAIL" if rows else "NOT_APPLICABLE",
    }


def _rollback_placements_to_control(
    semantic: dict[str, Any], control: dict[str, Any], placement_keys: set[str],
) -> dict[str, Any]:
    control_by_key = _by_placement(control.get("mappings") or [])
    repaired = []
    for index, row in enumerate(semantic.get("mappings") or []):
        key = str(row.get("editorial_placement_id") or row.get("window_id") or index)
        repaired.append(dict(control_by_key[key]) if key in placement_keys and key in control_by_key else dict(row))
    return {**semantic, "mappings": repaired}


def _changed_mapping_count(control: dict[str, Any], semantic: dict[str, Any]) -> int:
    control_by_placement = {
        str(row.get("editorial_placement_id") or row.get("window_id") or index): row
        for index, row in enumerate(control.get("mappings") or [])
    }
    changed = 0
    for index, row in enumerate(semantic.get("mappings") or []):
        key = str(row.get("editorial_placement_id") or row.get("window_id") or index)
        before = control_by_placement.get(key)
        if before is None or (
            before.get("clip_id"), before.get("source_performance_id")
        ) != (
            row.get("clip_id"), row.get("source_performance_id")
        ):
            changed += 1
    return changed


def _by_placement(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("editorial_placement_id") or row.get("window_id") or index): row
        for index, row in enumerate(rows)
    }


def _donor(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("clip_id"), row.get("source_performance_id")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
