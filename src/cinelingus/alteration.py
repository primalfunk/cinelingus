from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .audio_provenance import analyze_wav_activity, compare_wav_audio
from .cache import media_hash
from .tools import ToolError, run
from .util import utc_now, write_json


ALTERATION_POLICY_VERSION = "alteration_guarantee_v1"
MINIMUM_AUTHORED_TIMELINE_RATIO = 0.05
MINIMUM_PLACEMENT_COUNT = 3
MINIMUM_OCCUPIED_TIMELINE_BUCKETS = 2
MINIMUM_CHANGED_SAMPLE_RATIO = 0.60
SAMPLE_COUNT = 5
SAMPLE_DURATION_SECONDS = 5.0
SAMPLE_RATE = 8000
SAMPLE_CHANNELS = 1


class OutputAlterationError(ValueError):
    """Raised when usable media is not measurably altered from its anchor."""

    def __init__(self, message: str, acceptance: dict[str, Any]):
        super().__init__(message)
        self.acceptance = acceptance


def evaluate_requested_alteration(
    *,
    result,
    anchor: Path,
    output: Path,
    expected_duration: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate requested-filter evidence without confusing a valid remux for alteration."""
    acceptance_path = result.artifacts.get("filter_acceptance")
    filter_acceptance = _read_json(Path(acceptance_path)) if acceptance_path else None
    measurements = dict((filter_acceptance or {}).get("measurements") or {})
    render_duration = _positive_float(measurements.get("render_duration")) or expected_duration
    mapped_duration = _positive_float(measurements.get("mapped_dialogue_duration")) or 0.0
    dialogue_coverage = _bounded_ratio(measurements.get("dialogue_coverage"))
    authored_timeline_ratio = min(1.0, mapped_duration / max(0.001, render_duration))
    effective_alteration_ratio = max(dialogue_coverage, authored_timeline_ratio)
    placement_count = int(measurements.get("source_placement_count") or 0)
    occupied_buckets = int(measurements.get("occupied_timeline_buckets") or 0)
    timeline_bucket_count = int(measurements.get("timeline_bucket_count") or 0)
    required_buckets = min(MINIMUM_OCCUPIED_TIMELINE_BUCKETS, timeline_bucket_count)
    checks = {
        "filter_acceptance_present": filter_acceptance is not None,
        "filter_acceptance_passed": str((filter_acceptance or {}).get("status", "")).lower() == "pass",
        "minimum_authored_timeline_ratio": effective_alteration_ratio >= MINIMUM_AUTHORED_TIMELINE_RATIO,
        "minimum_placement_count": placement_count >= MINIMUM_PLACEMENT_COUNT,
        "minimum_timeline_distribution": required_buckets > 0 and occupied_buckets >= required_buckets,
    }
    report = {
        "schema_version": "1.0",
        "policy_version": ALTERATION_POLICY_VERSION,
        "creation_timestamp": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "basis": "REQUESTED_FILTER_ACCEPTANCE",
        "checks": checks,
        "thresholds": alteration_thresholds(),
        "measurements": {
            "expected_duration": round(expected_duration, 3),
            "render_duration": round(render_duration, 3),
            "mapped_dialogue_duration": round(mapped_duration, 3),
            "dialogue_coverage": round(dialogue_coverage, 4),
            "authored_timeline_ratio": round(authored_timeline_ratio, 4),
            "effective_alteration_ratio": round(effective_alteration_ratio, 4),
            "source_placement_count": placement_count,
            "occupied_timeline_buckets": occupied_buckets,
            "timeline_bucket_count": timeline_bucket_count,
        },
        "provenance": {
            "anchor": str(anchor),
            "output": str(output),
            "filter_acceptance": str(acceptance_path) if acceptance_path else None,
        },
        "renderer_manifest": None,
        "sampled_audio_difference": None,
    }
    return _persist_alteration_report(report, output_dir, "requested")


def render_universal_alteration(
    films: Iterable[Path],
    output_dir: Path,
    expected_duration: float,
) -> tuple[Path, str, dict[str, Any]]:
    """Render a full-timeline alteration using only deterministic FFmpeg primitives."""
    paths = tuple(Path(path) for path in films)
    if not paths:
        raise ValueError("Universal alteration requires at least one film.")
    if expected_duration <= 0:
        raise ValueError("Universal alteration requires a positive duration.")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output = _unique_output(output_root, "cinelingus_altered-result")
    if len(paths) >= 2:
        strategy = "MULTI_INPUT_COMPLETE_SUPPORTING_AUDIO"
        audio_filter, audio_map = _multi_input_audio_filter(len(paths) - 1)
    else:
        strategy = "SINGLE_INPUT_FULL_TRACK_AUDIO_TRANSFORMATION"
        audio_filter = (
            "[0:a:0]highpass=f=120,lowpass=f=7500,"
            "equalizer=f=1800:t=q:w=1:g=6,aecho=0.8:0.7:90:0.45,"
            "alimiter=limit=0.95[aout]"
        )
        audio_map = "[aout]"
    inputs: list[str] = []
    for path in paths:
        inputs.extend(["-i", str(path)])
    common = [
        "ffmpeg", "-y", "-v", "error", *inputs,
        "-filter_complex", audio_filter,
        "-map", "0:v:0", "-map", audio_map,
    ]
    duration_args = ["-t", f"{expected_duration:.3f}", "-shortest", "-movflags", "+faststart"]
    try:
        run([*common, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", *duration_args, str(output)])
        method = f"{strategy}_VIDEO_COPY"
    except (ToolError, OSError):
        run([
            *common, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k", *duration_args, str(output),
        ])
        method = f"{strategy}_VIDEO_TRANSCODE"
    hashes = [media_hash(path) for path in paths]
    manifest = {
        "schema_version": "1.0",
        "policy_version": ALTERATION_POLICY_VERSION,
        "creation_timestamp": utc_now(),
        "strategy": strategy,
        "method": method,
        "anchor": {"path": str(paths[0]), "media_hash": hashes[0]},
        "contributors": [
            {"path": str(path), "media_hash": hash_value, "role": "supporting_audio" if index else "transformed_anchor"}
            for index, (path, hash_value) in enumerate(zip(paths, hashes))
        ],
        "supporting_contributor_count": max(0, len(paths) - 1),
        "audio_coverage_ratio": 1.0,
        "audio_filter": audio_filter,
        "expected_duration": round(expected_duration, 3),
        "output": str(output),
    }
    manifest_path = output_root / "universal_alteration_manifests" / f"{output.stem}.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return output, method, manifest


def evaluate_universal_alteration(
    *,
    manifest: dict[str, Any],
    duration_acceptance: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    anchor = Path(manifest["anchor"]["path"])
    output = Path(manifest["output"])
    sampled = measure_sampled_audio_difference(
        anchor=anchor,
        output=output,
        duration=float(manifest["expected_duration"]),
        working_dir=Path(output_dir) / "alteration_samples" / output.stem,
    )
    contributors = list(manifest.get("contributors") or [])
    anchor_hash = str(manifest.get("anchor", {}).get("media_hash") or "")
    supporting_hashes = {
        str(row.get("media_hash"))
        for row in contributors
        if row.get("role") == "supporting_audio" and row.get("media_hash")
    }
    multi_input = manifest.get("strategy") == "MULTI_INPUT_COMPLETE_SUPPORTING_AUDIO"
    provenance_check = bool(supporting_hashes - {anchor_hash}) if multi_input else bool(manifest.get("audio_filter"))
    duration_checks = dict((duration_acceptance.get("duration_contract") or {}).get("checks") or {})
    checks = {
        "duration_acceptance_passed": duration_acceptance.get("status") == "PASS" and all(duration_checks.values()),
        "full_timeline_audio_coverage": float(manifest.get("audio_coverage_ratio") or 0.0) >= 0.999,
        "alteration_provenance_declared": provenance_check,
        "sampled_audio_is_measurably_different": sampled["changed_sample_ratio"] >= MINIMUM_CHANGED_SAMPLE_RATIO,
        "output_exists": output.is_file() and output.stat().st_size > 0,
    }
    report = {
        "schema_version": "1.0",
        "policy_version": ALTERATION_POLICY_VERSION,
        "creation_timestamp": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "basis": "UNIVERSAL_RENDERER_AND_SAMPLED_AUDIO",
        "checks": checks,
        "thresholds": alteration_thresholds(),
        "measurements": {
            "expected_duration": round(float(manifest["expected_duration"]), 3),
            "render_duration": duration_acceptance.get("duration"),
            "mapped_dialogue_duration": 0.0,
            "dialogue_coverage": 0.0,
            "authored_timeline_ratio": float(manifest.get("audio_coverage_ratio") or 0.0),
            "effective_alteration_ratio": sampled["changed_sample_ratio"],
            "source_placement_count": len(supporting_hashes) if multi_input else 1,
            "occupied_timeline_buckets": SAMPLE_COUNT,
            "timeline_bucket_count": SAMPLE_COUNT,
        },
        "provenance": {
            "anchor": str(anchor),
            "output": str(output),
            "filter_acceptance": None,
        },
        "renderer_manifest": manifest,
        "sampled_audio_difference": sampled,
    }
    return _persist_alteration_report(report, output_dir, "universal")


def build_unaltered_recovery_acceptance(
    *,
    anchor: Path,
    output: Path,
    output_dir: Path,
    reason: str,
) -> dict[str, Any]:
    report = {
        "schema_version": "1.0",
        "policy_version": ALTERATION_POLICY_VERSION,
        "creation_timestamp": utc_now(),
        "status": "FAIL",
        "basis": "UNALTERED_RECOVERY",
        "checks": {"significant_alteration_proven": False},
        "thresholds": alteration_thresholds(),
        "measurements": {
            "expected_duration": 0.0,
            "render_duration": 0.0,
            "mapped_dialogue_duration": 0.0,
            "dialogue_coverage": 0.0,
            "authored_timeline_ratio": 0.0,
            "effective_alteration_ratio": 0.0,
            "source_placement_count": 0,
            "occupied_timeline_buckets": 0,
            "timeline_bucket_count": 0,
        },
        "provenance": {"anchor": str(anchor), "output": str(output), "filter_acceptance": None},
        "renderer_manifest": {"reason": reason},
        "sampled_audio_difference": None,
    }
    return _persist_alteration_report(report, output_dir, "unaltered_recovery")


def measure_sampled_audio_difference(
    *,
    anchor: Path,
    output: Path,
    duration: float,
    working_dir: Path,
    runner: Callable[..., Any] = run,
) -> dict[str, Any]:
    working_dir.mkdir(parents=True, exist_ok=True)
    starts = _sample_starts(duration)
    rows = []
    changed = 0
    comparable = 0
    for index, start in enumerate(starts):
        anchor_wav = working_dir / f"anchor_{index:02d}.wav"
        output_wav = working_dir / f"output_{index:02d}.wav"
        for source, destination in ((anchor, anchor_wav), (output, output_wav)):
            runner([
                "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
                "-t", f"{SAMPLE_DURATION_SECONDS:.3f}", "-i", str(source), "-vn",
                "-ac", str(SAMPLE_CHANNELS), "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_s16le", str(destination),
            ])
        anchor_activity = analyze_wav_activity(anchor_wav)
        output_activity = analyze_wav_activity(output_wav)
        comparison = compare_wav_audio(left_path=anchor_wav, right_path=output_wav)
        reference_rms = max(float(anchor_activity.get("rms") or 0.0), float(output_activity.get("rms") or 0.0))
        active = reference_rms >= 200.0
        diff_ratio = float(comparison["diff_rms"]) / max(200.0, reference_rms)
        is_changed = active and diff_ratio >= 0.35
        if active:
            comparable += 1
            changed += int(is_changed)
        rows.append({
            "start": round(start, 3),
            "duration": SAMPLE_DURATION_SECONDS,
            "anchor_rms": anchor_activity.get("rms"),
            "output_rms": output_activity.get("rms"),
            "diff_rms": comparison.get("diff_rms"),
            "diff_ratio": round(diff_ratio, 4),
            "active": active,
            "changed": is_changed,
        })
    return {
        "sample_count": len(rows),
        "comparable_sample_count": comparable,
        "changed_sample_count": changed,
        "changed_sample_ratio": round(changed / max(1, comparable), 4),
        "samples": rows,
    }


def alteration_thresholds() -> dict[str, Any]:
    return {
        "minimum_authored_timeline_ratio": MINIMUM_AUTHORED_TIMELINE_RATIO,
        "minimum_placement_count": MINIMUM_PLACEMENT_COUNT,
        "minimum_occupied_timeline_buckets": MINIMUM_OCCUPIED_TIMELINE_BUCKETS,
        "minimum_changed_sample_ratio": MINIMUM_CHANGED_SAMPLE_RATIO,
    }


def _multi_input_audio_filter(supporting_count: int) -> tuple[str, str]:
    labels = []
    parts = []
    for index in range(1, supporting_count + 1):
        label = f"support{index}"
        parts.append(f"[{index}:a:0]aresample=48000[{label}]")
        labels.append(f"[{label}]")
    if supporting_count == 1:
        parts.append(f"{labels[0]}alimiter=limit=0.95[aout]")
    else:
        parts.append(
            f"{''.join(labels)}amix=inputs={supporting_count}:duration=shortest:normalize=1,"
            "alimiter=limit=0.95[aout]"
        )
    return ";".join(parts), "[aout]"


def _sample_starts(duration: float) -> list[float]:
    maximum_start = max(0.0, duration - SAMPLE_DURATION_SECONDS)
    if maximum_start <= 0:
        return [0.0]
    return [maximum_start * index / max(1, SAMPLE_COUNT - 1) for index in range(SAMPLE_COUNT)]


def _persist_alteration_report(report: dict[str, Any], output_dir: Path, label: str) -> dict[str, Any]:
    root = Path(output_dir) / "alteration_acceptance"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{time.time_ns()}_{label}.json"
    write_json(path, report)
    report["artifact_path"] = str(path)
    if report["status"] != "PASS" and report["basis"] != "UNALTERED_RECOVERY":
        failed = ", ".join(name for name, passed in report["checks"].items() if not passed)
        report["failure_summary"] = f"Significant alteration was not proven: {failed}."
    else:
        report["failure_summary"] = None
    write_json(path, report)
    return report


def _unique_output(output_dir: Path, stem: str) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output = output_dir / f"{stem}_{stamp}.mp4"
    suffix = 2
    while output.exists():
        output = output_dir / f"{stem}_{stamp}_{suffix}.mp4"
        suffix += 1
    return output


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bounded_ratio(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
