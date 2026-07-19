from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .alteration import (
    ALTERATION_POLICY_VERSION,
    OutputAlterationError,
    alteration_thresholds,
    build_unaltered_recovery_acceptance,
    evaluate_requested_alteration,
    evaluate_universal_alteration,
    render_universal_alteration,
)
from .filter_lab.registry import default_filter_registry
from .reliable_inputs import preflight_media_inputs
from .tools import ToolError, ffprobe_json, run
from .transformations.base import TransformationResult
from .util import utc_now, write_json


OUTCOME_GUARANTEE_VERSION = "configuration_outcome_v3"
FORMAT_DURATION_TOLERANCE_SECONDS = 0.25
VIDEO_DURATION_TOLERANCE_SECONDS = 0.25
AUDIO_DURATION_TOLERANCE_SECONDS = 0.05
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".mpg", ".mpeg"}
BLOOM_ID = "experimental.bloom"


class OutputDurationError(ValueError):
    """Raised when usable media violates the audio-supported duration contract."""

    def __init__(self, message: str, acceptance: dict[str, Any]):
        super().__init__(message)
        self.acceptance = acceptance


def discover_default_videos(directory: Path) -> tuple[Path, ...]:
    """Return deterministic complete-file candidates from the configured media directory."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return ()
    return tuple(sorted(
        (path.resolve() for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    ))


def normalize_parameters_tolerantly(definition, supplied: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    supplied = dict(supplied or {})
    normalized: dict[str, Any] = {}
    adjustments: list[dict[str, Any]] = []
    known = {parameter.id: parameter for parameter in definition.parameters}
    for unknown in sorted(set(supplied) - set(known)):
        adjustments.append({"parameter": unknown, "requested": supplied[unknown], "used": None, "reason": "unknown_parameter_ignored"})
    for parameter in definition.parameters:
        requested = supplied.get(parameter.id, parameter.default)
        try:
            normalized[parameter.id] = parameter.validate(requested)
        except (TypeError, ValueError):
            normalized[parameter.id] = parameter.validate(parameter.default)
            adjustments.append({
                "parameter": parameter.id,
                "requested": requested,
                "used": normalized[parameter.id],
                "reason": "invalid_value_replaced_with_default",
            })
    return normalized, adjustments


def resolve_configuration(
    filter_id: str,
    *,
    selected_filter_stack: Iterable[str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = default_filter_registry()
    requested = list(selected_filter_stack or [filter_id])
    resolved: list[str] = []
    resolution_reasons: list[str] = []
    for item in requested:
        try:
            resolved.append(registry.get(item).id)
        except ValueError as exc:
            resolution_reasons.append(str(exc))
    executable_filter_id: str | None = None
    stack_status = "SINGLE_FILTER"
    if len(resolved) == 1:
        definition = registry.get(resolved[0])
        if definition.implemented:
            executable_filter_id = definition.id
        else:
            stack_status = "PASSTHROUGH_REQUIRED"
            resolution_reasons.append(f"{definition.name} is not implemented.")
    elif resolved:
        try:
            registry.validate_stack(resolved)
        except ValueError as exc:
            stack_status = "PRIMARY_ONLY"
            resolution_reasons.append(str(exc))
            primary = next(
                (registry.get(item) for item in resolved if item != BLOOM_ID and registry.get(item).implemented),
                None,
            )
            executable_filter_id = primary.id if primary is not None else None
            if executable_filter_id is None:
                stack_status = "PASSTHROUGH_REQUIRED"
        else:
            executable_filter_id = resolved[0]
            stack_status = "CERTIFIED_STACK"
    else:
        stack_status = "PASSTHROUGH_REQUIRED"

    normalized_parameters: dict[str, Any] = {}
    parameter_adjustments: list[dict[str, Any]] = []
    if executable_filter_id is not None:
        definition = registry.get(executable_filter_id)
        normalized_parameters, parameter_adjustments = normalize_parameters_tolerantly(definition, parameters)
    return {
        "requested_filter_id": filter_id,
        "requested_filter_stack": requested,
        "resolved_filter_stack": resolved,
        "stack_status": stack_status,
        "executable_filter_id": executable_filter_id,
        "normalized_parameters": normalized_parameters,
        "parameter_adjustments": parameter_adjustments,
        "resolution_reasons": resolution_reasons,
    }


def execute_reliably(
    pipeline,
    filter_id: str,
    *,
    force: bool = False,
    parameters: dict[str, Any] | None = None,
    selected_filter_stack: Iterable[str] | None = None,
    run_filter: Callable[..., TransformationResult] | None = None,
    fallback_renderer: Callable[[Path, Path], tuple[Path, str]] | None = None,
    duration_repairer: Callable[[Path, Path, float], tuple[Path, str]] | None = None,
    requested_alteration_evaluator: Callable[..., dict[str, Any]] | None = None,
    altered_renderer: Callable[..., tuple[Path, str, dict[str, Any]]] | None = None,
    universal_alteration_evaluator: Callable[..., dict[str, Any]] | None = None,
    probe: Callable[[Path], dict[str, Any]] = ffprobe_json,
) -> TransformationResult:
    """Return a duration-correct, measurably altered artifact whenever rendering permits."""
    resolution = resolve_configuration(
        filter_id,
        selected_filter_stack=selected_filter_stack,
        parameters=parameters,
    )
    input_preflight = preflight_media_inputs(
        pipeline.config.films,
        output_dir=Path(pipeline.config.output_dir),
        probe=probe,
    )
    expected_duration = float(input_preflight["predicted_output_duration"])
    duration_contract = {
        "policy": input_preflight["duration_policy"],
        "expected_duration": expected_duration,
        "format_tolerance_seconds": FORMAT_DURATION_TOLERANCE_SECONDS,
        "video_tolerance_seconds": VIDEO_DURATION_TOLERANCE_SECONDS,
        "audio_tolerance_seconds": AUDIO_DURATION_TOLERANCE_SECONDS,
        "anchor_curtailed": bool(input_preflight["anchor_curtailed"]),
        "input_preflight": input_preflight,
    }
    alteration_contract = {
        "policy_version": ALTERATION_POLICY_VERSION,
        "thresholds": alteration_thresholds(),
        "success_requires_significant_alteration": True,
        "unaltered_recovery_is_success": False,
    }
    run_filter = run_filter or getattr(pipeline, "execute_transformation", None)
    using_default_fallback = fallback_renderer is None
    fallback_renderer = fallback_renderer or render_passthrough
    duration_repairer = duration_repairer or repair_output_duration
    requested_alteration_evaluator = requested_alteration_evaluator or evaluate_requested_alteration
    altered_renderer = altered_renderer or render_universal_alteration
    universal_alteration_evaluator = universal_alteration_evaluator or evaluate_universal_alteration
    result: TransformationResult | None = None
    execution_error: str | None = None
    attempted_filter_id = resolution["executable_filter_id"]
    fallback_maximum_duration: float | None = None
    fallback_method: str | None = None
    duration_repair_method: str | None = None
    duration_repair_source: str | None = None
    altered_fallback_method: str | None = None
    altered_fallback_error: str | None = None
    alteration_acceptance: dict[str, Any] | None = None
    if attempted_filter_id is not None:
        try:
            result = run_filter(
                attempted_filter_id,
                force=force,
                parameters=resolution["normalized_parameters"],
            )
            output = _result_video(result)
            try:
                acceptance = validate_usable_output(
                    output,
                    expected_duration=expected_duration,
                    probe=probe,
                )
            except OutputDurationError as exc:
                if not _duration_can_be_repaired(exc.acceptance):
                    raise
                duration_repair_source = str(output)
                output, duration_repair_method = duration_repairer(
                    output,
                    Path(pipeline.config.output_dir),
                    expected_duration,
                )
                acceptance = validate_usable_output(
                    output,
                    expected_duration=expected_duration,
                    probe=probe,
                )
                result.outputs["video"] = output
                result.warnings.append(
                    f"Final media duration was repaired to the audio-supported extent using {duration_repair_method}."
                )
            alteration_acceptance = requested_alteration_evaluator(
                result=result,
                anchor=Path(pipeline.config.anchor_film),
                output=output,
                expected_duration=expected_duration,
                output_dir=Path(pipeline.config.output_dir),
            )
            if alteration_acceptance.get("status") != "PASS":
                raise OutputAlterationError(
                    alteration_acceptance.get("failure_summary") or "Requested output was not significantly altered.",
                    alteration_acceptance,
                )
        except Exception as exc:
            if _is_cancelled(pipeline, exc):
                raise
            execution_error = f"{type(exc).__name__}: {exc}"
            result = None

    if result is None:
        try:
            output, altered_fallback_method, renderer_manifest = altered_renderer(
                pipeline.config.films,
                Path(pipeline.config.output_dir),
                expected_duration,
            )
            acceptance = validate_usable_output(
                output,
                expected_duration=expected_duration,
                probe=probe,
            )
            alteration_acceptance = universal_alteration_evaluator(
                manifest=renderer_manifest,
                duration_acceptance=acceptance,
                output_dir=Path(pipeline.config.output_dir),
            )
            if alteration_acceptance.get("status") != "PASS":
                raise OutputAlterationError(
                    alteration_acceptance.get("failure_summary") or "Universal output was not significantly altered.",
                    alteration_acceptance,
                )
            result = TransformationResult(
                transformation_id=str(attempted_filter_id or filter_id),
                outputs={"video": output},
                warnings=[message for message in [*resolution["resolution_reasons"], execution_error] if message],
            )
            fallback_method = altered_fallback_method
            outcome_status = "ALTERED_FALLBACK_SUCCESS"
        except Exception as exc:
            if _is_cancelled(pipeline, exc):
                raise
            altered_fallback_error = f"{type(exc).__name__}: {exc}"
            if using_default_fallback:
                fallback_maximum_duration = expected_duration
                output, fallback_method = render_passthrough(
                    Path(pipeline.config.anchor_film),
                    Path(pipeline.config.output_dir),
                    maximum_duration=fallback_maximum_duration,
                )
            else:
                output, fallback_method = fallback_renderer(
                    Path(pipeline.config.anchor_film),
                    Path(pipeline.config.output_dir),
                )
            acceptance = validate_usable_output(
                output,
                expected_duration=expected_duration,
                probe=probe,
            )
            alteration_acceptance = build_unaltered_recovery_acceptance(
                anchor=Path(pipeline.config.anchor_film),
                output=output,
                output_dir=Path(pipeline.config.output_dir),
                reason=altered_fallback_error,
            )
            result = TransformationResult(
                transformation_id=str(attempted_filter_id or filter_id),
                outputs={"video": output},
                warnings=[
                    message
                    for message in [
                        *resolution["resolution_reasons"],
                        execution_error,
                        altered_fallback_error,
                        "Significant alteration could not be proven; returned unaltered recovery media.",
                    ]
                    if message
                ],
            )
            outcome_status = "UNALTERED_RECOVERY"
    else:
        if duration_repair_method is not None:
            outcome_status = "DURATION_REPAIRED_SUCCESS"
        elif resolution["stack_status"] == "PRIMARY_ONLY":
            outcome_status = "PRIMARY_ONLY_SUCCESS"
        elif resolution["parameter_adjustments"]:
            outcome_status = "NORMALIZED_SUCCESS"
        else:
            outcome_status = "REQUESTED_SUCCESS"

    outcome = {
        "schema_version": "1.0",
        "guarantee_version": OUTCOME_GUARANTEE_VERSION,
        "creation_timestamp": utc_now(),
        "status": outcome_status,
        "requested_configuration": {
            "filter_id": filter_id,
            "filter_stack": resolution["requested_filter_stack"],
            "parameters": dict(parameters or {}),
            "films": [str(path) for path in pipeline.config.films],
        },
        "resolution": resolution,
        "duration_contract": duration_contract,
        "alteration_contract": alteration_contract,
        "execution": {
            "attempted_filter_id": attempted_filter_id,
            "execution_error": execution_error,
            "fallback_method": fallback_method,
            "fallback_maximum_duration": fallback_maximum_duration,
            "duration_repair_method": duration_repair_method,
            "duration_repair_source": duration_repair_source,
            "altered_fallback_method": altered_fallback_method,
            "altered_fallback_error": altered_fallback_error,
        },
        "output": {
            "video": str(_result_video(result)),
            "acceptance": acceptance,
            "alteration_acceptance": alteration_acceptance,
        },
        "filter_id": str(attempted_filter_id or (resolution["resolved_filter_stack"][0] if resolution["resolved_filter_stack"] else filter_id)),
    }
    outcome_path = _outcome_path(Path(pipeline.config.output_dir), filter_id)
    write_json(outcome_path, outcome)
    result.artifacts["configuration_outcome"] = outcome_path
    alteration_path = (alteration_acceptance or {}).get("artifact_path")
    if alteration_path:
        result.artifacts["alteration_acceptance"] = Path(alteration_path)
    return result

def validate_usable_output(
    path: Path,
    *,
    expected_duration: float | None = None,
    format_tolerance_seconds: float = FORMAT_DURATION_TOLERANCE_SECONDS,
    video_tolerance_seconds: float = VIDEO_DURATION_TOLERANCE_SECONDS,
    audio_tolerance_seconds: float = AUDIO_DURATION_TOLERANCE_SECONDS,
    probe: Callable[[Path], dict[str, Any]] = ffprobe_json,
) -> dict[str, Any]:
    output = Path(path)
    if not output.is_file() or output.stat().st_size <= 0:
        raise ValueError(f"No usable output media was produced: {output}")
    document = probe(output)
    streams = list(document.get("streams") or [])
    video_streams = [row for row in streams if row.get("codec_type") == "video"]
    audio_streams = [row for row in streams if row.get("codec_type") == "audio"]
    container_duration = _positive_duration((document.get("format") or {}).get("duration")) or 0.0
    video_duration, video_duration_source = _measured_stream_duration(video_streams, container_duration)
    audio_duration, audio_duration_source = _measured_stream_duration(audio_streams, container_duration)
    checks = {
        "file_nonempty": output.stat().st_size > 0,
        "video_stream_present": bool(video_streams),
        "audio_stream_present": bool(audio_streams),
        "positive_duration": container_duration > 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Output media failed usability checks: {checks}")

    duration_evidence = None
    if expected_duration is not None:
        if expected_duration <= 0:
            raise ValueError("Expected output duration must be positive.")
        duration_checks = {
            "container_duration_matches_expected": abs(container_duration - expected_duration) <= format_tolerance_seconds,
            "video_duration_matches_expected": (
                video_duration is not None
                and abs(video_duration - expected_duration) <= video_tolerance_seconds
            ),
            "audio_duration_matches_expected": (
                audio_duration is not None
                and abs(audio_duration - expected_duration) <= audio_tolerance_seconds
            ),
        }
        checks.update(duration_checks)
        duration_evidence = {
            "policy": "FULL_SOURCE_TIMELINE_LIMITED_BY_SUPPORTING_AUDIO",
            "expected_duration": round(expected_duration, 3),
            "container_duration": round(container_duration, 3),
            "video_duration": round(video_duration, 3) if video_duration is not None else None,
            "audio_duration": round(audio_duration, 3) if audio_duration is not None else None,
            "video_duration_source": video_duration_source,
            "audio_duration_source": audio_duration_source,
            "container_delta_seconds": round(container_duration - expected_duration, 3),
            "video_delta_seconds": round(video_duration - expected_duration, 3) if video_duration is not None else None,
            "audio_delta_seconds": round(audio_duration - expected_duration, 3) if audio_duration is not None else None,
            "format_tolerance_seconds": format_tolerance_seconds,
            "video_tolerance_seconds": video_tolerance_seconds,
            "audio_tolerance_seconds": audio_tolerance_seconds,
            "checks": duration_checks,
        }

    acceptance = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "duration": round(container_duration, 3),
        "video_stream_count": len(video_streams),
        "audio_stream_count": len(audio_streams),
        "duration_contract": duration_evidence,
    }
    if acceptance["status"] != "PASS":
        raise OutputDurationError(
            f"Output media violated the audio-supported duration contract: {duration_evidence}",
            acceptance,
        )
    return acceptance


def _positive_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _measured_stream_duration(
    streams: list[dict[str, Any]],
    container_duration: float,
) -> tuple[float | None, str | None]:
    durations = [_positive_duration(row.get("duration")) for row in streams]
    durations = [duration for duration in durations if duration is not None]
    if durations:
        return min(durations), "stream"
    if streams and container_duration > 0:
        return container_duration, "container_fallback"
    return None, None


def _duration_can_be_repaired(acceptance: dict[str, Any]) -> bool:
    evidence = acceptance.get("duration_contract") or {}
    expected = _positive_duration(evidence.get("expected_duration"))
    container = _positive_duration(evidence.get("container_duration"))
    video = _positive_duration(evidence.get("video_duration"))
    audio = _positive_duration(evidence.get("audio_duration"))
    if expected is None or container is None or video is None or audio is None:
        return False
    return (
        container >= expected - float(evidence.get("format_tolerance_seconds") or 0.0)
        and video >= expected - float(evidence.get("video_tolerance_seconds") or 0.0)
        and audio >= expected - float(evidence.get("audio_tolerance_seconds") or 0.0)
    )


def repair_output_duration(
    source: Path,
    output_dir: Path,
    expected_duration: float,
) -> tuple[Path, str]:
    if expected_duration <= 0:
        raise ValueError("Duration repair requires a positive expected duration.")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{Path(source).stem}_duration-repaired.mp4"
    suffix = 2
    while output.exists():
        output = output_root / f"{Path(source).stem}_duration-repaired_{suffix}.mp4"
        suffix += 1
    duration_args = ["-t", f"{expected_duration:.3f}"]
    try:
        run([
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", *duration_args,
            "-movflags", "+faststart", str(output),
        ])
        return output, "STREAM_COPY_DURATION_CAP"
    except (ToolError, OSError):
        run([
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-c:a", "aac", "-b:a", "160k", *duration_args,
            "-movflags", "+faststart", str(output),
        ])
        return output, "TRANSCODE_DURATION_CAP"

def render_passthrough(
    anchor: Path,
    output_dir: Path,
    *,
    maximum_duration: float | None = None,
) -> tuple[Path, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output = output_root / f"cinelingus_safe-result_{stamp}.mp4"
    suffix = 2
    while output.exists():
        output = output_root / f"cinelingus_safe-result_{stamp}_{suffix}.mp4"
        suffix += 1
    duration_args = []
    curtailed_suffix = ""
    if maximum_duration is not None:
        if maximum_duration <= 0:
            raise ValueError("Passthrough duration must be positive.")
        duration_args = ["-t", f"{maximum_duration:.3f}"]
        curtailed_suffix = "_CURTAILED"
    try:
        run([
            "ffmpeg", "-y", "-v", "error", "-i", str(anchor),
            "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", *duration_args,
            "-movflags", "+faststart", str(output),
        ])
        return output, f"STREAM_COPY_MP4{curtailed_suffix}"
    except (ToolError, OSError):
        try:
            run([
                "ffmpeg", "-y", "-v", "error", "-i", str(anchor),
                "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "22", "-c:a", "aac", "-b:a", "160k", *duration_args,
                "-movflags", "+faststart", str(output),
            ])
            return output, f"TRANSCODE_MP4{curtailed_suffix}"
        except (ToolError, OSError):
            copied = output.with_suffix(anchor.suffix.lower())
            shutil.copy2(anchor, copied)
            return copied, "SOURCE_CONTAINER_COPY_UNCURTAILED" if maximum_duration is not None else "SOURCE_CONTAINER_COPY"


def _result_video(result: TransformationResult) -> Path:
    try:
        return Path(result.outputs["video"])
    except KeyError as exc:
        raise ValueError("Transformation returned no video output.") from exc


def _outcome_path(output_dir: Path, filter_id: str) -> Path:
    slug = "".join(character if character.isalnum() else "_" for character in filter_id).strip("_") or "unknown"
    return output_dir / "configuration_outcomes" / f"{time.time_ns()}_{slug}.json"


def install_reliable_executor(pipeline_class: type) -> None:
    """Expose the outcome-guaranteed user-facing execution service."""
    if getattr(pipeline_class, "_reliable_executor_installed", False):
        return

    def execute_configuration(
        self,
        filter_id: str,
        *,
        force: bool = False,
        parameters: dict[str, Any] | None = None,
        selected_filter_stack: Iterable[str] | None = None,
    ) -> TransformationResult:
        return execute_reliably(
            self,
            filter_id,
            force=force,
            parameters=parameters,
            selected_filter_stack=selected_filter_stack,
        )

    pipeline_class.execute_configuration = execute_configuration
    pipeline_class._reliable_executor_installed = True

def _is_cancelled(pipeline, exc: Exception) -> bool:
    check = getattr(pipeline, "cancel_check", None)
    return bool(check is not None and check()) or "cancel" in str(exc).lower()
