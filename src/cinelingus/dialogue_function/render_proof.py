from __future__ import annotations

import random
import shutil
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..pipeline import Pipeline
from ..semantic.render_proof import _refine_rendered_verification_with_words, _variant_result
from ..util import read_json, stable_hash, utc_now, write_json
from .acoustic_preflight import require_accepted_function_preflight
from .render_verification import evaluate_rendered_function

FUNCTION_PROOF_VERSION = "function_render_proof_v1"
PROOF_VARIANTS = ("legacy_control", "semantic_only", "function_report_only", "function_preserving_repaired")


def run_function_render_proof(
    *, config: AppConfig, screen_dir: Path, output_dir: Path, preflight_path: Path,
    function_variant: str = "function_preserving_repaired", force: bool = False,
    defer_human_calibration: bool = False, rejected_preflight_path: Path | None = None,
    donor_audit_path: Path | None = None,
) -> dict[str, Any]:
    screen = read_json(screen_dir / "function_schedule_screen.json")
    calibration_state = str(screen.get("calibration_state") or "NOT_PROVIDED")
    if calibration_state != "COMPLETE" and not defer_human_calibration:
        raise ValueError("Reviewed calibration is incomplete; use explicit deferred-calibration closeout only when authorized")
    preflight = require_accepted_function_preflight(
        preflight_path, screen_signature=screen["experiment_signature"], function_variant=function_variant,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_ids = ("legacy_control", "semantic_only", "function_report_only", function_variant)
    variants: list[dict[str, Any]] = []
    schedules: dict[str, dict[str, Any]] = {}
    runtime_rows: list[dict[str, Any]] = []
    tracemalloc.start()
    proof_start = time.perf_counter()
    for variant_id in variant_ids:
        started = time.perf_counter()
        variant_dir = output_dir / variant_id
        pipeline = Pipeline(replace(config, output_dir=variant_dir))
        schedule = read_json(screen_dir / f"{variant_id}_schedule.json")
        destination_movie = pipeline._inspect_one(pipeline.destination, force=False)
        audio = pipeline.render_audio_from_schedule(
            schedule=schedule, dest_movie=destination_movie, force=force,
            output_path=variant_dir / "replacement_dialogue.wav", persist_schedule=False,
        )
        video = pipeline.render_video_from_audio(
            audio=audio, force=force, output_path=variant_dir / "translation_output.mp4",
            duration=float(destination_movie["duration"]),
        )
        write_json(variant_dir / "final_schedule.json", schedule)
        schedule = _refine_rendered_verification_with_words(
            schedule=schedule, audio=audio, variant_dir=variant_dir, config=config, force=force,
        )
        schedules[variant_id] = schedule
        row = _variant_result(variant_id, variant_dir, schedule, audio, video)
        function_verification = evaluate_rendered_function(
            schedule=schedule,
            rendered_dialogue_verification=schedule.get("rendered_dialogue_verification") or {},
            baseline_schedule=schedules.get("function_report_only") if variant_id == function_variant else None,
            calibration=None,
        )
        write_json(variant_dir / "function_render_verification.json", function_verification)
        row.update({
            "function_mode": (schedule.get("dialogue_function_scoring") or {}).get("mode", "FUNCTION_DISABLED"),
            "function_verification_status": function_verification["status"],
            "function_verified_count": function_verification["counts"]["VERIFIED"],
            "function_mismatch_count": function_verification["counts"]["FUNCTION_MISMATCH"],
            "function_unverifiable_count": function_verification["counts"]["UNVERIFIABLE"],
            "function_verification": f"{variant_id}/function_render_verification.json",
        })
        variants.append(row)
        runtime_rows.append({"variant_id": variant_id, "elapsed_seconds": round(time.perf_counter() - started, 3)})
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    by_id = {row["variant_id"]: row for row in variants}
    report_only = by_id["function_report_only"]
    preserving = by_id[function_variant]
    preserving_function = read_json(output_dir / function_variant / "function_render_verification.json")
    changed_rows = preserving_function.get("mappings") or []
    technical_reasons = []
    if any(not row["audio_ready"] or not row["video_ready"] for row in variants):
        technical_reasons.append("render_artifact_incomplete")
    if any(row["voice_residue_status"] != "NONE_DETECTED" for row in variants):
        technical_reasons.append("voice_residue_not_clear")
    if int(preserving.get("rendered_dialogue_failed_mapping_count") or 0) > int(report_only.get("rendered_dialogue_failed_mapping_count") or 0):
        technical_reasons.append("function_failed_mapping_count_regressed")
    if float(preserving.get("rendered_dialogue_average_word_coverage_percentage") or 0.0) + 1e-9 < float(report_only.get("rendered_dialogue_average_word_coverage_percentage") or 0.0):
        technical_reasons.append("function_word_coverage_regressed")
    if not changed_rows or any(row.get("verification_state") != "VERIFIED" for row in changed_rows):
        technical_reasons.append("changed_function_placement_not_verified")
    repair_accepted = not technical_reasons
    rejected_preflight = read_json(rejected_preflight_path) if rejected_preflight_path and rejected_preflight_path.is_file() else None
    donor_audit = read_json(donor_audit_path) if donor_audit_path and donor_audit_path.is_file() else None
    review = _build_deferred_review_package(output_dir, variants, seed="phase3-function-proof-v1")
    report = {
        "schema_version": "1.0", "proof_version": FUNCTION_PROOF_VERSION,
        "creation_timestamp": utc_now(), "screen_signature": screen["experiment_signature"],
        "calibration_state": calibration_state,
        "closeout_state": "ENGINEERING_COMPLETE_HUMAN_CALIBRATION_DEFERRED" if calibration_state != "COMPLETE" else "CALIBRATED_PROOF",
        "claim_state": "PROVISIONAL_TECHNICAL_EVIDENCE" if calibration_state != "COMPLETE" else "CALIBRATED_TECHNICAL_EVIDENCE",
        "invariant_configuration": {
            "destination_media_hash": variants[0]["destination_media_hash"],
            "render_sample_rate": config.render_sample_rate, "render_channels": config.render_channels,
            "target_lufs": config.target_lufs, "audio_fade_duration": config.audio_fade_duration,
            "dialogue_suppression": config.dialogue_suppression, "suppression_padding": config.suppression_padding,
            "background_reconstruction": config.background_reconstruction,
            "whisper_model": config.whisper_model, "whisper_language": config.whisper_language,
            "transcription_mode": config.transcription_mode, "residue_correction_passes": config.residue_correction_passes,
        },
        "variants": variants,
        "report_only_invariant": {
            "schedule_equivalent_to_semantic_only": _donors(schedules["semantic_only"]) == _donors(schedules["function_report_only"]),
            "audio_sha256_equivalent_to_semantic_only": by_id["semantic_only"]["audio_sha256"] == report_only["audio_sha256"],
            "video_sha256_equivalent_to_semantic_only": by_id["semantic_only"]["video_sha256"] == report_only["video_sha256"],
        },
        "technical_acceptance": {"state": "ACCEPTED" if not technical_reasons else "REJECTED", "reasons": technical_reasons},
        "function_repair": {
            "state": "ACCEPTED_AFTER_RENDER_VERIFICATION" if repair_accepted else "ROLLED_BACK_OR_UNRESOLVED",
            "initial_rejected_clip_id": "c000073", "replacement_clip_id": "c000120",
            "initial_acoustic_preflight_state": (rejected_preflight or {}).get("preflight_state"),
            "replacement_acoustic_preflight_state": preflight.get("preflight_state"),
            "donor_audit_state": (donor_audit or {}).get("audit_state"),
            "rendered_changed_placement_states": [row.get("verification_state") for row in changed_rows],
            "rollback_count": 0 if repair_accepted else 1,
        },
        "runtime_and_memory": {
            "total_elapsed_seconds": round(time.perf_counter() - proof_start, 3),
            "variant_runtime": runtime_rows, "peak_python_traced_memory_mb": round(peak / 1048576, 3),
            "limitation": "Python allocation tracing excludes peak memory inside FFmpeg and Whisper native kernels.",
        },
        "acoustic_preflight": str(preflight_path), "acoustic_preflight_signature": preflight.get("preflight_signature"),
        "deferred_review_package": review,
        "claim_scope": "Automated technical and rendered-transcript function evidence only. Human calibration and subjective preference are explicitly deferred.",
    }
    report["proof_signature"] = stable_hash({key: value for key, value in report.items() if key not in {"creation_timestamp", "proof_signature"}})
    write_json(output_dir / "function_render_proof.json", report)
    return report


def _donors(schedule: dict[str, Any]) -> list[tuple[Any, Any]]:
    return [(row.get("clip_id"), row.get("source_performance_id")) for row in schedule.get("mappings") or []]


def _build_deferred_review_package(output_dir: Path, variants: list[dict[str, Any]], *, seed: str) -> dict[str, Any]:
    review_dir = output_dir / "deferred_blinded_review"
    media_dir = review_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    shuffled = [row["variant_id"] for row in variants]
    random.Random(seed).shuffle(shuffled)
    labels = ["A", "B", "C", "D"]
    answer_key = {}
    cases = []
    for label, variant_id in zip(labels, shuffled):
        source = output_dir / variant_id / "translation_output.mp4"
        target = media_dir / f"phase3_{label}.mp4"
        shutil.copy2(source, target)
        answer_key[label] = variant_id
        cases.append({"label": label, "media": f"media/{target.name}"})
    write_json(review_dir / "answer_key.json", {"seed": seed, "answer_key": answer_key})
    write_json(review_dir / "review_manifest.json", {
        "schema_version": "1.0", "review_state": "OPTIONAL_HUMAN_REVIEW_DEFERRED",
        "cases": cases,
        "separate_questions": ["function_fit", "semantic_fit", "performance_fit", "completeness", "overall_preference"],
    })
    return {"state": "PREPARED_BUT_DEFERRED", "manifest": "deferred_blinded_review/review_manifest.json", "answer_key": "deferred_blinded_review/answer_key.json"}
