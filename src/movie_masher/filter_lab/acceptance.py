from __future__ import annotations

from pathlib import Path
from typing import Any

from movie_masher.audio_provenance import MINIMUM_ACTIVE_AUDIO_RATIO, analyze_wav_activity
from movie_masher.intervals import covered_speech_duration
from movie_masher.tools import ToolError, ffprobe_json
from movie_masher.util import write_json
from movie_masher.validation import validate_artifact

from .contracts import FilterContract, default_contract_catalog


class FilterAcceptanceError(RuntimeError):
    pass


FULL_LENGTH_DIALOGUE_ACCEPTANCE_REQUIREMENTS = {
    "minimum_dialogue_coverage": 0.08,
    "timeline_bucket_count": 4,
    "minimum_occupied_timeline_buckets": 3,
    "minimum_unique_source_ratio": 0.8,
    "maximum_source_reuse": 2,
}


def apply_full_length_dialogue_requirements(schedule: dict[str, Any], *, render_duration: float) -> dict[str, Any]:
    schedule["render_duration"] = round(float(render_duration), 3)
    schedule["audio_activity_basis"] = "rendered_mix"
    schedule["acceptance_requirements"] = dict(FULL_LENGTH_DIALOGUE_ACCEPTANCE_REQUIREMENTS)
    return schedule


def validate_schedule_quality(schedule: dict[str, Any]) -> dict[str, Any]:
    quality = _schedule_quality(schedule)
    failed = [name for name, passed in quality["checks"].items() if not passed]
    if failed:
        raise FilterAcceptanceError(
            f"Filter schedule acceptance failed before render: {', '.join(failed)}."
        )
    return quality


def validate_filter_output(
    *,
    filter_id: str,
    schedule: dict[str, Any],
    final_video: Path,
    replacement_audio: Path,
    output_path: Path,
    schemas_dir: Path,
    audio_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = default_contract_catalog().get(filter_id)
    enabled = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    audio = analyze_wav_activity(replacement_audio)
    render_duration = float(schedule.get("render_duration") or audio.get("duration") or 0.0)
    mapped_duration = covered_speech_duration(enabled, [])
    coverage = min(1.0, mapped_duration / render_duration) if render_duration > 0 else 0.0
    schedule_quality = _schedule_quality(schedule)
    audio_stream = _audio_stream(final_video)
    provenance = _provenance_check(schedule, enabled, audio_provenance)
    invariants = [_evaluate_invariant(contract, row, schedule, provenance) for row in contract.data["hard_invariants"]]
    checks = {
        "final_mp4_produced": final_video.exists() and final_video.suffix.lower() == ".mp4" and final_video.stat().st_size > 0,
        "replacement_audio_provenance": provenance["passed"],
        "dialogue_coverage_measured": render_duration > 0 and bool(enabled),
        **schedule_quality["checks"],
        "replacement_audio_has_sufficient_activity": float(audio.get("active_ratio", 0.0) or 0.0) >= MINIMUM_ACTIVE_AUDIO_RATIO,
        "audio_stream_verified": bool(audio_stream),
        "contract_invariants_pass": bool(invariants) and all(row["passed"] for row in invariants),
    }
    report = {
        "schema_version": "1.0",
        "filter_id": contract.filter_id,
        "contract_version": contract.data["contract_version"],
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "measurements": {
            "dialogue_coverage": round(coverage, 4),
            "mapped_dialogue_duration": round(mapped_duration, 3),
            "render_duration": round(render_duration, 3),
            **schedule_quality["measurements"],
            "active_ratio": round(float(audio.get("active_ratio", 0.0) or 0.0), 4),
            "silence_ratio": round(float(audio.get("silent_ratio", 1.0)), 4),
        },
        "outputs": {
            "final_video": str(final_video),
            "replacement_audio": str(replacement_audio),
            "replacement_audio_retained": replacement_audio.exists(),
            "audio_activity_basis": str(schedule.get("audio_activity_basis") or "rendered_mix"),
            "audio_stream": audio_stream,
            "provenance": provenance,
        },
        "invariants": invariants,
    }
    write_json(output_path, report)
    validate_artifact("filter_acceptance", output_path, schemas_dir)
    if report["status"] != "pass":
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise FilterAcceptanceError(f"Filter output acceptance failed: {failed}. See {output_path}")
    return report


def _schedule_quality(schedule: dict[str, Any]) -> dict[str, Any]:
    enabled = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    requirements = dict(schedule.get("acceptance_requirements") or {})
    render_duration = float(schedule.get("render_duration") or 0.0)
    mapped_duration = covered_speech_duration(enabled, [])
    coverage = min(1.0, mapped_duration / render_duration) if render_duration > 0 else 0.0
    bucket_count = max(1, int(requirements.get("timeline_bucket_count", 4) or 4))
    occupied_buckets = {
        min(bucket_count - 1, max(0, int(float(row.get("destination_timestamp", 0.0)) * bucket_count / render_duration)))
        for row in enabled
    } if render_duration > 0 else set()
    source_ids = [str(row.get("clip_id") or row.get("clip_path") or "") for row in enabled]
    source_ids = [value for value in source_ids if value]
    counts = {value: source_ids.count(value) for value in set(source_ids)}
    unique_ratio = len(counts) / len(source_ids) if source_ids else 0.0
    maximum_reuse = max(counts.values(), default=0)
    minimum_coverage = float(requirements.get("minimum_dialogue_coverage", 0.0) or 0.0)
    minimum_buckets = int(requirements.get("minimum_occupied_timeline_buckets", 0) or 0)
    minimum_unique_ratio = float(requirements.get("minimum_unique_source_ratio", 0.0) or 0.0)
    maximum_source_reuse = int(requirements.get("maximum_source_reuse", 0) or 0)
    checks = {
            "dialogue_coverage_sufficient": coverage >= minimum_coverage,
            "timeline_distribution_sufficient": len(occupied_buckets) >= minimum_buckets,
            "source_repetition_within_limit": (
                unique_ratio >= minimum_unique_ratio
                and (maximum_source_reuse <= 0 or maximum_reuse <= maximum_source_reuse)
            ),
    }
    identity_quality = schedule.get("identity_quality")
    if isinstance(identity_quality, dict):
        checks["identity_quality_sufficient"] = identity_quality.get("passed") is True
    return {
        "checks": checks,
        "measurements": {
            "occupied_timeline_buckets": len(occupied_buckets),
            "timeline_bucket_count": bucket_count,
            "unique_source_clip_count": len(counts),
            "source_placement_count": len(source_ids),
            "unique_source_ratio": round(unique_ratio, 4),
            "maximum_source_reuse": maximum_reuse,
            "minimum_dialogue_coverage": round(minimum_coverage, 4),
            "minimum_occupied_timeline_buckets": minimum_buckets,
            "minimum_unique_source_ratio": round(minimum_unique_ratio, 4),
            "maximum_allowed_source_reuse": maximum_source_reuse,
        },
    }


def _audio_stream(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        probe = ffprobe_json(path)
    except (ToolError, OSError, ValueError):
        return None
    stream = next((row for row in probe.get("streams", []) if row.get("codec_type") == "audio"), None)
    if not stream:
        return None
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": int(stream["channels"]) if stream.get("channels") else None,
    }


def _provenance_check(
    schedule: dict[str, Any],
    mappings: list[dict[str, Any]],
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    if report is not None:
        return {"passed": report.get("status") == "pass", "basis": "audio_provenance_report"}
    expected_many = sorted(str(value) for value in schedule.get("source_media_hashes", []) if value)
    if expected_many:
        roots = sorted({_clip_cache_root(row.get("clip_path")) for row in mappings if row.get("clip_path")})
        roots = [row for row in roots if row]
        mapping_hashes = sorted({str(row.get("source_media_hash")) for row in mappings if row.get("source_media_hash")})
        return {
            "passed": bool(roots and roots == expected_many and mapping_hashes == expected_many),
            "basis": "multiworld_schedule_clip_cache_roots",
            "expected_source_hashes": expected_many,
            "observed_source_hashes": roots,
            "mapping_source_hashes": mapping_hashes,
        }
    expected = str(schedule.get("source_media_hash") or schedule.get("media_hash") or "")
    roots = sorted({_clip_cache_root(row.get("clip_path")) for row in mappings if row.get("clip_path")})
    roots = [row for row in roots if row]
    return {
        "passed": bool(expected and roots and set(roots) == {expected}),
        "basis": "schedule_clip_cache_roots",
        "expected_source_hash": expected,
        "observed_source_hashes": roots,
    }


def _evaluate_invariant(
    contract: FilterContract,
    invariant: dict[str, Any],
    schedule: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    validator = str(invariant["validator"])
    value = _resolve_path(schedule, validator)
    if value is None:
        value = _computed_invariant(invariant["id"], schedule, provenance)
    return {
        "id": invariant["id"],
        "statement": invariant["statement"],
        "validator": validator,
        "passed": value is True,
    }


def _resolve_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _computed_invariant(invariant_id: str, schedule: dict[str, Any], provenance: dict[str, Any]) -> bool:
    mappings = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    duration = float(schedule.get("render_duration") or 0.0)
    if invariant_id in {"all_enabled_clips_come_from_source_media", "all_clips_come_from_same_film"}:
        return bool(provenance.get("passed"))
    if invariant_id == "destination_chronology_is_preserved":
        return bool(mappings) and all(0 <= float(row.get("destination_timestamp", 0.0)) <= duration for row in mappings)
    if invariant_id == "source_dialogue_is_not_reused":
        clip_ids = [str(row.get("clip_id")) for row in mappings]
        return bool(clip_ids) and len(clip_ids) == len(set(clip_ids))
    return False


def _clip_cache_root(path: Any) -> str | None:
    if not path:
        return None
    parts = Path(str(path)).parts
    for index, part in enumerate(parts):
        if part == "cache" and index + 1 < len(parts):
            return parts[index + 1]
    return None
