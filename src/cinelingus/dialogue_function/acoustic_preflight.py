from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..render_verification import evaluate_rendered_dialogue
from ..semantic.acoustic_preflight import _build_reel, _changed_mappings, _digest, _mapping_decision
from ..util import read_json, stable_hash, utc_now, write_json
from ..whisper_backend import transcribe_with_whisper

Transcriber = Callable[..., dict[str, Any]]
FUNCTION_PREFLIGHT_VERSION = "function_acoustic_preflight_v1"


def run_function_acoustic_preflight(
    *, screen_dir: Path, output_dir: Path,
    baseline_variant: str = "function_report_only", function_variant: str = "function_preserving",
    model_name: str = "medium", language: str | None = "en", transcription_mode: str = "quality",
    minimum_word_coverage: float = 0.72, force: bool = False,
    transcriber: Transcriber = transcribe_with_whisper,
) -> dict[str, Any]:
    screen = read_json(screen_dir / "function_schedule_screen.json")
    baseline = read_json(screen_dir / f"{baseline_variant}_schedule.json")
    assisted = read_json(screen_dir / f"{function_variant}_schedule.json")
    changed = _changed_mappings(baseline.get("mappings") or [], assisted.get("mappings") or [])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "function_acoustic_preflight.json"
    if not changed:
        raise ValueError("Function-preserving proof requires at least one changed donor")
    reel_path = output_dir / "changed_function_donor_reel.wav"
    verification_schedule, reel_manifest = _build_reel(changed, reel_path)
    manifest_path = output_dir / "changed_function_donor_reel_manifest.json"
    write_json(manifest_path, reel_manifest)
    signature = stable_hash({
        "version": FUNCTION_PREFLIGHT_VERSION, "screen_signature": screen["experiment_signature"],
        "baseline_variant": baseline_variant, "function_variant": function_variant,
        "model_name": model_name, "language": language, "transcription_mode": transcription_mode,
        "minimum_word_coverage": minimum_word_coverage, "reel_sha256": _digest(reel_path),
        "mappings": verification_schedule["mappings"],
    })
    transcription_path = output_dir / "changed_function_donor_transcription.json"
    cached = read_json(transcription_path) if transcription_path.is_file() else {}
    if not force and cached.get("media_hash") == signature:
        timeline, cache_state = cached, "REUSED"
    else:
        timeline = transcriber(
            audio_path=reel_path, media_hash=signature, output_path=transcription_path,
            model_name=model_name, language=language, artifact_type="timeline",
            transcription_mode=transcription_mode,
        )
        if not transcription_path.is_file():
            write_json(transcription_path, timeline)
        cache_state = "CREATED"
    verification = evaluate_rendered_dialogue(schedule=verification_schedule, rendered_timeline=timeline)
    source_by_key = {
        (str(row.get("clip_id") or ""), str(row.get("destination_performance_id") or row.get("window_id") or "")): row
        for row in changed
    }
    decisions = [
        _mapping_decision(
            row, minimum_word_coverage,
            source_mapping=source_by_key.get((str(row.get("clip_id") or ""), str(row.get("window_id") or ""))),
        )
        for row in verification.get("mappings") or []
    ]
    rejected = [row for row in decisions if row["state"] == "REJECTED"]
    report = {
        "schema_version": "1.0", "preflight_version": FUNCTION_PREFLIGHT_VERSION,
        "creation_timestamp": utc_now(), "screen_signature": screen["experiment_signature"],
        "baseline_variant": baseline_variant, "function_variant": function_variant,
        "preflight_signature": signature,
        "whisper_configuration": {"requested_model": model_name, "language": language, "transcription_mode": transcription_mode},
        "minimum_word_coverage": minimum_word_coverage, "changed_mapping_count": len(changed),
        "reel": reel_path.name, "reel_manifest": manifest_path.name,
        "transcription": transcription_path.name, "transcription_cache_state": cache_state,
        "verification": verification, "mapping_decisions": decisions,
        "preflight_state": "REJECTED_ACOUSTIC_INTEGRITY" if rejected else "ACCEPTED_FOR_RENDER",
        "rejection_count": len(rejected),
        "calibration_state": screen.get("calibration_state"),
        "claim_scope": "Changed function-donor acoustic transcript integrity only; human calibration and preference remain deferred.",
    }
    write_json(output_path, report)
    return report


def require_accepted_function_preflight(path: Path, *, screen_signature: str, function_variant: str) -> dict[str, Any]:
    report = read_json(path)
    reasons = []
    if report.get("preflight_state") != "ACCEPTED_FOR_RENDER":
        reasons.append("preflight_not_accepted")
    if report.get("screen_signature") != screen_signature:
        reasons.append("screen_signature_mismatch")
    if report.get("function_variant") != function_variant:
        reasons.append("function_variant_mismatch")
    if reasons:
        raise ValueError(f"Function acoustic preflight does not admit render: {', '.join(reasons)}")
    return report
