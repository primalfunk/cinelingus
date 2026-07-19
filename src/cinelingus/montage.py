from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract_runtime import (
    active_run_contract,
    record_multi_input_guarantee,
    record_schedule_qualification,
)
from .montage_implementation import *  # noqa: F401,F403 - preserve the public planning API
from .montage_implementation import build_full_timeline_plan as _build_full_timeline_plan
from .montage_implementation import build_montage_render_acceptance as _build_render_acceptance
from .filter_lab.registry import default_filter_registry
from .multi_input_guarantee import (
    MultiInputGuaranteeStatus,
    certify_multi_input_schedule,
)
from .qualification import QualificationStatus, qualify_schedule
from .util import write_json


class ScheduleQualificationError(ValueError):
    pass


def build_full_timeline_plan(**kwargs: Any) -> dict[str, Any]:
    """Compile a full timeline from the active run contract before rendering."""
    contract = active_run_contract()
    qualification = None
    multi_input_guarantee = None
    if contract is not None and str(kwargs.get("filter_id")) == contract.filter_id:
        schedule = kwargs.get("schedule")
        if isinstance(schedule, dict):
            definition = default_filter_registry().get(contract.filter_id)
            if len(contract.media) >= 2 and definition.sparse_schedule:
                requirements = dict(schedule.get("acceptance_requirements") or {})
                requirements.update({
                    "minimum_dialogue_coverage": 0.0,
                    "minimum_occupied_timeline_buckets": 1,
                    "minimum_unique_source_ratio": 1.0,
                    "maximum_source_reuse": 1,
                })
                schedule["acceptance_requirements"] = requirements
            qualification = qualify_schedule(schedule, contract)
            record_schedule_qualification(qualification)
            if qualification.status == QualificationStatus.BLOCKED:
                detail = "; ".join(qualification.reasons) or "schedule qualification failed"
                raise ScheduleQualificationError(f"{contract.filter_id} is blocked before render: {detail}")
            if len(contract.media) >= 2:
                multi_input_guarantee = certify_multi_input_schedule(
                    contract=contract,
                    schedule=schedule,
                    qualification=qualification,
                )
                record_multi_input_guarantee(multi_input_guarantee)
                if multi_input_guarantee.status == MultiInputGuaranteeStatus.REJECTED:
                    detail = "; ".join(multi_input_guarantee.reasons) or "multi-input guarantee failed"
                    raise ScheduleQualificationError(
                        f"{contract.filter_id} is rejected before render: {detail}"
                    )
        kwargs["anchor_duration"] = contract.timeline.anchor_video_duration
        kwargs["supporting_audio_durations"] = list(contract.timeline.required_audio_durations)

    plan = _build_full_timeline_plan(**kwargs)
    if contract is not None and str(kwargs.get("filter_id")) == contract.filter_id:
        plan["provenance"]["run_contract_id"] = contract.contract_id
        plan["provenance"]["contract_kernel_version"] = contract.kernel_version
        plan["provenance"]["canonical_extent_authority"] = contract.timeline.authority
        if qualification is not None:
            plan["provenance"]["schedule_qualification"] = {
                "status": qualification.status.value,
                "schedule_signature": qualification.schedule_signature,
                "disabled_mapping_count": qualification.measurements["disabled_mapping_count"],
            }
        if multi_input_guarantee is not None:
            plan["provenance"]["multi_input_guarantee"] = {
                "status": multi_input_guarantee.status.value,
                "film_count": multi_input_guarantee.film_count,
                "all_checks_passed": all(multi_input_guarantee.checks.values()),
            }
    return plan


def build_montage_render_acceptance(
    *,
    plan: dict[str, Any],
    encoded_probe: dict[str, Any],
    output_path: Path,
    timing_tolerance_seconds: float = 0.05,
) -> dict[str, Any]:
    """Require one intentional video/audio pair and contract duration conformance."""
    contract = active_run_contract()
    if contract is not None and plan.get("filter_id") == contract.filter_id:
        timing_tolerance_seconds = float(contract.acceptance["duration_tolerance_seconds"])
    artifact = _build_render_acceptance(
        plan=plan,
        encoded_probe=encoded_probe,
        output_path=output_path,
        timing_tolerance_seconds=timing_tolerance_seconds,
    )
    streams = list(encoded_probe.get("streams") or [])
    video_streams = [row for row in streams if row.get("codec_type") == "video"]
    audio_streams = [row for row in streams if row.get("codec_type") == "audio"]
    format_duration = _positive_duration((encoded_probe.get("format") or {}).get("duration"))
    video_duration = _stream_duration(video_streams, format_duration)
    audio_duration = _stream_duration(audio_streams, format_duration)
    planned_duration = float(artifact["planned_duration"])
    video_tolerance = (
        float(contract.acceptance["video_packet_tolerance_seconds"])
        if contract is not None and plan.get("filter_id") == contract.filter_id
        else max(timing_tolerance_seconds, 0.25)
    )

    artifact["checks"].update({
        "exactly_one_video_stream": len(video_streams) == 1,
        "exactly_one_replacement_audio_stream": len(audio_streams) == 1,
        "encoded_video_duration_matches_plan": (
            video_duration is not None
            and abs(video_duration - planned_duration) <= video_tolerance
        ),
        "encoded_audio_duration_matches_plan": (
            audio_duration is not None
            and abs(audio_duration - planned_duration) <= timing_tolerance_seconds
        ),
    })
    artifact["provenance"]["encoded_stream_contract"] = {
        "video_stream_count": len(video_streams),
        "audio_stream_count": len(audio_streams),
        "video_duration": round(video_duration, 3) if video_duration is not None else None,
        "audio_duration": round(audio_duration, 3) if audio_duration is not None else None,
        "video_packet_tolerance_seconds": video_tolerance,
        "audio_source_policy": "REPLACEMENT_AUDIO_ONLY",
        "run_contract_id": contract.contract_id if contract is not None else None,
    }
    artifact["acceptance_status"] = "PASS" if all(artifact["checks"].values()) else "FAIL"
    write_json(output_path, artifact)
    return artifact


def _stream_duration(streams: list[dict[str, Any]], fallback: float | None) -> float | None:
    if len(streams) != 1:
        return None
    return _positive_duration(streams[0].get("duration")) or fallback


def _positive_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None

