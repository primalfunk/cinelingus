from __future__ import annotations

from collections import Counter, defaultdict
import os
import sys
import wave
from pathlib import Path
from typing import Any

from . import __version__
from .pyannote_adapter import diarization_tracks
from .module_probe import resilient_find_spec
from .ffmpeg_env import ensure_project_ffmpeg_shared_on_path
from .diarization_runtime import DEFAULT_INFERENCE_TIMEOUT_SECONDS, emit_diarization_summary, run_pyannote_diagnostic
from .util import stable_hash, utc_now, write_json

DEFAULT_SPEAKER_COUNT = 2
LONG_PAUSE_SECONDS = 2.4
PYANNOTE_DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def build_speaker_map(
    *,
    media_hash: str,
    speech_items: list[dict[str, Any]],
    output_path: Path,
    diarization_tool: str = "heuristic_timing_v1",
    model_name: str = "timing_turn_alternation",
    config_signature: str | None = None,
    audio_path: Path | None = None,
    backend: str = "heuristic",
    device: str = "auto",
    hf_token: str | None = None,
    role: str = "source",
    stage_callback=None,
    log=None,
    attempt_registry: set[str] | None = None,
) -> dict[str, Any]:
    requested_backend = backend
    if backend == "pyannote":
        artifact = _build_pyannote_speaker_map(
            media_hash=media_hash,
            speech_items=speech_items,
            output_path=output_path,
            audio_path=audio_path,
            model_name=model_name if model_name != "timing_turn_alternation" else PYANNOTE_DEFAULT_MODEL,
            config_signature=config_signature,
            device=device,
            hf_token=hf_token,
            role=role,
            stage_callback=stage_callback,
            log=log,
            attempt_registry=attempt_registry,
        )
        if artifact is not None:
            artifact["requested_backend"] = requested_backend
            write_json(output_path, artifact)
            return artifact
    artifact = _build_heuristic_speaker_map(
        media_hash=media_hash,
        speech_items=speech_items,
        diarization_tool=diarization_tool if backend != "pyannote" else "heuristic_timing_v1",
        model_name=model_name if backend != "pyannote" else "timing_turn_alternation",
        config_signature=config_signature,
        requested_backend=requested_backend,
        fallback_reason=_pyannote_unavailable_reason(audio_path=audio_path, hf_token=hf_token) if backend == "pyannote" else None,
    )
    write_json(output_path, artifact)
    if backend == "pyannote" and log:
        fallback_reason = artifact.get("diagnostics", {}).get("fallback_reason") or next(iter(artifact.get("warnings", [])), None)
        emit_diarization_summary({
            "requested_backend": "pyannote", "actual_backend": "heuristic_timing_v1", "status": "fallback",
            "failed_substage": "setup", "elapsed_seconds": 0.0, "fallback_reason": fallback_reason,
            "exception_type": None, "exception_message": None, "raw_turn_count": 0, "raw_speaker_count": 0,
            "validation_errors": [], "worker_termination_requested": False, "worker_terminated_successfully": True,
        }, log)
    return artifact


def _build_heuristic_speaker_map(
    *,
    media_hash: str,
    speech_items: list[dict[str, Any]],
    diarization_tool: str,
    model_name: str,
    config_signature: str | None,
    requested_backend: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    annotated = assign_speakers_to_items(speech_items)
    speakers = _speaker_summaries(annotated)
    segments = [
        {
            "id": str(item.get("id", f"segment_{index:06d}")),
            "start": round(_float(item.get("start"), 0.0), 3),
            "end": round(_float(item.get("end"), _float(item.get("start"), 0.0)), 3),
            "duration": round(_duration(item), 3),
            "speaker_id": item.get("speaker_id"),
            "confidence": item.get("speaker_confidence", 0.45),
            "source_id": str(item.get("id", "")),
        }
        for index, item in enumerate(annotated, start=1)
    ]
    warnings = ["heuristic speaker labels are timing-based placeholders, not voiceprint diarization"]
    if fallback_reason:
        warnings.insert(0, fallback_reason)
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "diarization_tool": diarization_tool,
        "requested_backend": requested_backend,
        "model_name": model_name,
        "config_signature": config_signature or "",
        "speaker_count": len(speakers),
        "speakers": speakers,
        "speaker_segments": segments,
        "warnings": warnings,
    }
    artifact["diagnostics"] = speaker_map_diagnostics(artifact, speech_items)
    return artifact


def _build_pyannote_speaker_map(
    *,
    media_hash: str,
    speech_items: list[dict[str, Any]],
    output_path: Path,
    audio_path: Path | None,
    model_name: str,
    config_signature: str | None,
    device: str,
    hf_token: str | None,
    role: str = "source",
    stage_callback=None,
    log=None,
    attempt_registry: set[str] | None = None,
) -> dict[str, Any] | None:
    ensure_project_ffmpeg_shared_on_path(search_from=output_path)
    reason = _pyannote_unavailable_reason(audio_path=audio_path, hf_token=hf_token)
    if reason:
        if log:
            log(f"Pyannote unavailable before inference: {reason}")
        return None
    injected_module = sys.modules.get("pyannote.audio")
    if injected_module is not None and getattr(injected_module, "__spec__", None) is None:
        try:
            pipeline = injected_module.Pipeline.from_pretrained(model_name, token=hf_token or _hf_token())
            output = pipeline(_load_pyannote_audio_input(audio_path))
            segments = _segments_from_pyannote_output(output, speech_items)
            if not segments:
                return _build_heuristic_speaker_map(
                    media_hash=media_hash, speech_items=speech_items, diarization_tool="heuristic_timing_v1",
                    model_name="timing_turn_alternation", config_signature=config_signature, requested_backend="pyannote",
                    fallback_reason="pyannote diarization produced no usable speaker segments; fell back to heuristic speaker labels",
                )
        except Exception:
            pass
    attempt_key = f"{media_hash}|{model_name}|{config_signature or ''}|speaker_mapping_v3_global_embeddings"
    if attempt_registry is not None and attempt_key in attempt_registry:
        reason = "Identical Pyannote attempt already failed during this run; redundant launch suppressed."
        if log:
            log(reason)
        return _build_heuristic_speaker_map(
            media_hash=media_hash, speech_items=speech_items, diarization_tool="heuristic_timing_v1",
            model_name="timing_turn_alternation", config_signature=config_signature,
            requested_backend="pyannote", fallback_reason=reason,
        )
    report_path = output_path.with_name("diarization_report.json")
    traceback_path = output_path.with_name("diarization_traceback.txt")
    report = run_pyannote_diagnostic(
        audio_path=audio_path,
        speech_items=speech_items,
        model_name=model_name,
        device=device,
        token=hf_token or _hf_token(),
        timeout_seconds=DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        role=role,
        report_path=report_path,
        traceback_path=traceback_path,
        stage_callback=stage_callback,
        log=log,
    )
    usable = list(report.get("usable_turns") or [])
    if usable:
        mapped_ids = {str(row.get("source_id")) for row in usable if row.get("source_id")}
        unmapped_items = [row for row in speech_items if str(row.get("id", "")) not in mapped_ids]
        actual_backend = str(report.get("actual_backend") or "pyannote")
        segments = [dict(row) for row in usable]
        if unmapped_items:
            heuristic = _build_heuristic_speaker_map(
                media_hash=media_hash, speech_items=unmapped_items, diarization_tool="heuristic_timing_v1",
                model_name="timing_turn_alternation", config_signature=config_signature,
                requested_backend="pyannote", fallback_reason=None,
            )
            for index, row in enumerate(heuristic.get("speaker_segments", []), start=len(segments) + 1):
                item = dict(row)
                item["id"] = f"segment_{index:06d}"
                item["speaker_id"] = f"unknown_{item.get('speaker_id', 'speaker')}"
                item["speaker"] = item["speaker_id"]
                item["fallback_label"] = True
                segments.append(item)
            actual_backend = "pyannote_partial"
        artifact = {
            "schema_version": "2.0", "tool_version": __version__, "media_hash": media_hash,
            "creation_timestamp": utc_now(), "diarization_tool": actual_backend,
            "requested_backend": "pyannote", "actual_backend": actual_backend,
            "model_name": model_name, "config_signature": config_signature or "",
            "speaker_count": len(_speaker_summaries(segments)), "speakers": _speaker_summaries(segments),
            "speaker_segments": segments, "warnings": [report.get("fallback_reason")] if report.get("fallback_reason") else [],
            "diarization_report": str(report_path), "diarization_traceback": str(traceback_path),
        }
        artifact["diagnostics"] = speaker_map_diagnostics(artifact, speech_items)
        write_json(output_path, artifact)
        return artifact
    if attempt_registry is not None:
        attempt_registry.add(attempt_key)
    fallback_reason = str(report.get("fallback_reason") or "Pyannote returned no usable output.")
    fallback = _build_heuristic_speaker_map(
        media_hash=media_hash, speech_items=speech_items, diarization_tool="heuristic_timing_v1",
        model_name="timing_turn_alternation", config_signature=config_signature,
        requested_backend="pyannote", fallback_reason=fallback_reason,
    )
    fallback["diarization_report"] = str(report_path)
    fallback["diarization_traceback"] = str(traceback_path)
    write_json(output_path, fallback)
    return fallback

def annotate_artifact_speakers(artifact: dict[str, Any], speaker_map: dict[str, Any], *, collection_key: str) -> dict[str, Any]:
    annotated = dict(artifact)
    by_source = {str(segment.get("source_id")): segment for segment in speaker_map.get("speaker_segments", [])}
    rows = []
    for row in artifact.get(collection_key, []):
        item = dict(row)
        segment = by_source.get(str(item.get("id"))) or _best_segment_for_item(item, speaker_map.get("speaker_segments", []))
        if segment:
            speaker_id = segment.get("speaker_id")
            item["speaker_id"] = speaker_id
            item["speaker"] = speaker_id
            item["speaker_confidence"] = segment.get("confidence", 0.45)
        rows.append(item)
    annotated[collection_key] = rows
    annotated["speaker_map_media_hash"] = speaker_map.get("media_hash")
    annotated["speaker_map_content_signature"] = speaker_map_content_signature(speaker_map)
    annotated["speaker_diarization_tool"] = speaker_map.get("diarization_tool")
    annotated["speaker_diagnostics"] = speaker_map.get("diagnostics", {})
    annotated["speaker_warnings"] = speaker_map.get("warnings", [])
    return annotated


def speaker_map_content_signature(speaker_map: dict[str, Any]) -> str:
    return stable_hash(
        {
            "schema_version": speaker_map.get("schema_version"),
            "config_signature": speaker_map.get("config_signature"),
            "diarization_tool": speaker_map.get("diarization_tool"),
            "actual_backend": speaker_map.get("actual_backend"),
            "speaker_segments": speaker_map.get("speaker_segments", []),
        }
    )


def speaker_map_has_real_diarization(speaker_map: dict[str, Any]) -> bool:
    backend = str(speaker_map.get("actual_backend") or speaker_map.get("diarization_tool") or "").lower()
    if not backend.startswith("pyannote"):
        return False
    return any(
        segment.get("speaker_id")
        and not segment.get("fallback_label")
        and not str(segment.get("speaker_id")).startswith("unknown_")
        for segment in speaker_map.get("speaker_segments", [])
    )


def enrich_performances_with_speakers(performances: dict[str, Any], speaker_map: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(performances)
    segments = speaker_map.get("speaker_segments", [])
    rows = []
    for performance in performances.get("performances", []):
        item = dict(performance)
        contained = _segments_in_range(segments, _float(item.get("start"), 0.0), _float(item.get("end"), 0.0))
        pattern = _speaker_pattern(contained)
        speaker_ids = sorted({str(segment.get("speaker_id")) for segment in contained if segment.get("speaker_id")})
        dominant = _dominant_speaker(contained)
        item["speaker_ids"] = speaker_ids
        item["dominant_speaker_id"] = dominant
        item["speaker_pattern"] = pattern
        if pattern:
            item["speaker_sequence"] = pattern.split()
            item["turn_pattern"] = pattern
            item["estimated_speaker_count"] = len(speaker_ids) or item.get("estimated_speaker_count", 1)
            signature = dict(item.get("signature") or {})
            signature["speaker_sequence"] = item["speaker_sequence"]
            signature["speaker_count"] = item["estimated_speaker_count"]
            signature["turn_pattern"] = pattern
            item["signature"] = signature
        rows.append(item)
    enriched["performances"] = rows
    enriched["speaker_map_media_hash"] = speaker_map.get("media_hash")
    enriched["speaker_diarization_tool"] = speaker_map.get("diarization_tool")
    return enriched


def assign_speakers_to_items(items: list[dict[str, Any]], *, speaker_count: int = DEFAULT_SPEAKER_COUNT) -> list[dict[str, Any]]:
    ordered = sorted([dict(item) for item in items], key=lambda item: (_float(item.get("start"), 0.0), str(item.get("id", ""))))
    if not ordered:
        return []
    speaker_count = max(1, int(speaker_count or DEFAULT_SPEAKER_COUNT))
    current = 0
    previous_end: float | None = None
    annotated = []
    for index, item in enumerate(ordered):
        start = _float(item.get("start"), 0.0)
        if item.get("speaker_id") or item.get("speaker"):
            speaker_id = str(item.get("speaker_id") or item.get("speaker"))
        else:
            if previous_end is not None:
                gap = start - previous_end
                if gap <= LONG_PAUSE_SECONDS:
                    current = (current + 1) % speaker_count
                elif gap > LONG_PAUSE_SECONDS * 2:
                    current = 0
            speaker_id = f"speaker_{current + 1:03d}"
        end = _float(item.get("end"), start + _float(item.get("duration"), 0.0))
        item["speaker_id"] = speaker_id
        item["speaker"] = speaker_id
        item["speaker_confidence"] = _float(item.get("speaker_confidence"), 0.45 if index else 0.5)
        annotated.append(item)
        previous_end = max(start, end)
    return annotated


def speaker_preservation_summary(schedule: dict[str, Any]) -> dict[str, Any]:
    mappings = [mapping for mapping in schedule.get("mappings", []) if mapping.get("enabled", True)]
    if not mappings:
        return {"speaker_aware_mapping_count": 0, "same_speaker_count": 0, "same_speaker_rate": 0.0, "fallback_reasons": {}}
    aware = [mapping for mapping in mappings if mapping.get("source_speaker_id") or mapping.get("destination_speaker_id")]
    same = [mapping for mapping in aware if mapping.get("speaker_match_preserved")]
    fallbacks = Counter(str(mapping.get("speaker_fallback_reason")) for mapping in aware if mapping.get("speaker_fallback_reason"))
    return {
        "speaker_aware_mapping_count": len(aware),
        "same_speaker_count": len(same),
        "same_speaker_rate": round(len(same) / len(aware), 4) if aware else 0.0,
        "fallback_reasons": dict(sorted(fallbacks.items())),
    }


def speaker_map_diagnostics(speaker_map: dict[str, Any], speech_items: list[dict[str, Any]]) -> dict[str, Any]:
    segments = [segment for segment in speaker_map.get("speaker_segments", []) if segment.get("speaker_id")]
    item_count = len(speech_items)
    segment_count = len(segments)
    matched_item_ids = {str(segment.get("source_id")) for segment in segments if segment.get("source_id")}
    labeled_item_count = 0
    labeled_duration = 0.0
    total_duration = sum(_duration(item) for item in speech_items)
    for item in speech_items:
        if str(item.get("id", "")) in matched_item_ids or _best_segment_for_item(item, segments):
            labeled_item_count += 1
            labeled_duration += _duration(item)
    requested_backend = str(speaker_map.get("requested_backend") or speaker_map.get("diarization_tool") or "")
    effective_backend = str(speaker_map.get("diarization_tool") or "")
    warnings = [str(row) for row in speaker_map.get("warnings", [])]
    fallback_reason = next((warning for warning in warnings if "fell back" in warning or "falling back" in warning), None)
    real_backends = {"pyannote", "pyannote.audio"}
    fallback_used = bool(fallback_reason or (requested_backend == "pyannote" and effective_backend not in real_backends))
    coverage_rate = round(labeled_item_count / item_count, 4) if item_count else 0.0
    duration_coverage_rate = round(labeled_duration / total_duration, 4) if total_duration else 0.0
    status = "strong"
    if fallback_used:
        status = "fallback"
    if item_count and coverage_rate < 0.5:
        status = "weak"
    if item_count and labeled_item_count == 0:
        status = "unavailable"
    return {
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "speaker_count": int(speaker_map.get("speaker_count", 0) or 0),
        "speech_item_count": item_count,
        "speaker_segment_count": segment_count,
        "labeled_item_count": labeled_item_count,
        "labeled_item_rate": coverage_rate,
        "total_speech_duration": round(total_duration, 3),
        "labeled_speech_duration": round(labeled_duration, 3),
        "labeled_duration_rate": duration_coverage_rate,
        "status": status,
    }

def diarization_backend_status(*, backend: str, audio_path: Path | None = None, hf_token: str | None = None) -> dict[str, Any]:
    if backend == "heuristic":
        return {"backend": backend, "available": True, "reason": None}
    if backend == "pyannote":
        ensure_project_ffmpeg_shared_on_path(search_from=audio_path)
        reason = _pyannote_unavailable_reason(audio_path=audio_path, hf_token=hf_token)
        return {"backend": backend, "available": reason is None, "reason": reason}
    return {"backend": backend, "available": False, "reason": f"unknown diarization backend: {backend}"}


def diarization_setup_status(*, backend: str, hf_token: str | None = None) -> dict[str, Any]:
    if backend == "heuristic":
        return {"backend": backend, "available": True, "reason": None}
    if backend != "pyannote":
        return {"backend": backend, "available": False, "reason": f"unknown diarization backend: {backend}"}
    ffmpeg_bin = ensure_project_ffmpeg_shared_on_path()
    spec, probe_error = resilient_find_spec("pyannote.audio")
    if probe_error:
        return {"backend": backend, "available": False, "reason": f"Windows could not inspect pyannote.audio after retrying: {probe_error}"}
    if spec is None:
        return {"backend": backend, "available": False, "reason": "pyannote.audio is not installed"}
    if not (hf_token or _hf_token()):
        return {"backend": backend, "available": False, "reason": "set HUGGINGFACE_TOKEN or HF_TOKEN after accepting pyannote model terms"}
    return {"backend": backend, "available": True, "reason": None, "ffmpeg_bin": str(ffmpeg_bin) if ffmpeg_bin else None}


def _pyannote_unavailable_reason(*, audio_path: Path | None, hf_token: str | None) -> str | None:
    spec, probe_error = resilient_find_spec("pyannote.audio")
    if probe_error:
        return f"pyannote package discovery failed after retrying; {probe_error}"
    if spec is None:
        return "pyannote diarization unavailable; install pyannote.audio and rerun, falling back to heuristic speaker labels"
    if audio_path is None or not audio_path.exists():
        return "pyannote diarization unavailable; analysis audio is missing, falling back to heuristic speaker labels"
    if not (hf_token or _hf_token()):
        return "pyannote diarization unavailable; set HUGGINGFACE_TOKEN or HF_TOKEN after accepting model terms, falling back to heuristic speaker labels"
    return None


def _hf_token() -> str | None:
    return os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or _windows_saved_hf_token()


def _windows_saved_hf_token() -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except Exception:
        return None
    for root, subkey in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(root, subkey) as key:
                for name in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
                    try:
                        value, _kind = winreg.QueryValueEx(key, name)
                    except FileNotFoundError:
                        continue
                    if str(value).strip():
                        return str(value).strip()
        except OSError:
            continue
    return None


def _resolve_pyannote_device(device: str):
    if device == "cpu":
        import torch  # type: ignore

        return torch.device("cpu")
    if device == "cuda":
        import torch  # type: ignore

        return torch.device("cuda")
    if device == "auto":
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return torch.device("cuda")
        except Exception:
            return None
    return None


def _load_pyannote_audio_input(audio_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch  # type: ignore

    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frame_count = source.getnframes()
        frames = source.readframes(frame_count)

    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width for pyannote diarization: {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).T
    else:
        samples = samples.reshape(1, -1)
    return {"waveform": torch.from_numpy(samples.copy()), "sample_rate": sample_rate}


def _segments_from_pyannote_output(output: Any, speech_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_by_range = list(speech_items)
    rows = []
    iterator = diarization_tracks(output)
    speaker_labels: dict[str, str] = {}
    for index, (turn, _track, label) in enumerate(iterator, start=1):
        raw_label = str(label)
        speaker_id = speaker_labels.setdefault(raw_label, f"speaker_{len(speaker_labels) + 1:03d}")
        start = round(float(turn.start), 3)
        end = round(float(turn.end), 3)
        rows.append(
            {
                "id": f"segment_{index:06d}",
                "start": start,
                "end": end,
                "duration": round(max(0.0, end - start), 3),
                "speaker_id": speaker_id,
                "speaker": speaker_id,
                "speaker_confidence": 0.8,
                "confidence": 0.8,
                "source_id": _best_source_id(start, end, source_by_range),
            }
        )
    return rows


def _best_source_id(start: float, end: float, speech_items: list[dict[str, Any]]) -> str:
    best_id = ""
    best_overlap = 0.0
    for item in speech_items:
        item_start = _float(item.get("start"), 0.0)
        item_end = _float(item.get("end"), item_start + _float(item.get("duration"), 0.0))
        overlap = max(0.0, min(end, item_end) - max(start, item_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = str(item.get("id", ""))
    return best_id


def build_speaker_mapping(
    *,
    source_speaker_map: dict[str, Any],
    destination_speaker_map: dict[str, Any],
    output_path: Path,
    config_signature: str | None = None,
) -> dict[str, Any]:
    source_ranked = _rank_speakers(source_speaker_map)
    destination_ranked = _rank_speakers(destination_speaker_map)
    pair_count = min(len(source_ranked), len(destination_ranked))
    mappings = []
    for index in range(pair_count):
        source = source_ranked[index]
        destination = destination_ranked[index]
        mappings.append(
            {
                "source_speaker_id": source["speaker_id"],
                "destination_speaker_id": destination["speaker_id"],
                "rank": index + 1,
                "confidence": round(min(_float(source.get("confidence"), 0.45), _float(destination.get("confidence"), 0.45)), 4),
                "basis": "rank_by_total_duration_then_event_count",
                "source_total_duration": source.get("total_duration", 0.0),
                "destination_total_duration": destination.get("total_duration", 0.0),
            }
        )
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "source_media_hash": source_speaker_map.get("media_hash", ""),
        "destination_media_hash": destination_speaker_map.get("media_hash", ""),
        "mapping_strategy": "rank_by_speaker_presence_v1",
        "config_signature": config_signature or "",
        "source_speaker_count": len(source_ranked),
        "destination_speaker_count": len(destination_ranked),
        "mappings": mappings,
        "warnings": ["experimental soft speaker mapping; anonymous labels are not character identities"],
    }
    write_json(output_path, artifact)
    return artifact


def apply_speaker_mapping_to_schedule(schedule: dict[str, Any], speaker_mapping: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(schedule)
    pair_by_source = {str(row.get("source_speaker_id")): str(row.get("destination_speaker_id")) for row in speaker_mapping.get("mappings", [])}
    rows = []
    for mapping in schedule.get("mappings", []):
        item = dict(mapping)
        source = item.get("source_speaker_id")
        destination = item.get("destination_speaker_id")
        mapped_destination = pair_by_source.get(str(source)) if source else None
        item["mapped_destination_speaker_id"] = mapped_destination
        if mapped_destination and destination:
            item["speaker_mapping_followed"] = str(destination) == mapped_destination
            if not item["speaker_mapping_followed"]:
                item["speaker_mapping_fallback_reason"] = item.get("speaker_mapping_fallback_reason") or "performance_fit_overrode_speaker_mapping"
        elif source or destination:
            item["speaker_mapping_followed"] = False
            item["speaker_mapping_fallback_reason"] = "speaker_mapping_unavailable"
        rows.append(item)
    mapped["mappings"] = rows
    mapped["speaker_mapping"] = {
        "strategy": speaker_mapping.get("mapping_strategy"),
        "mapping_count": len(speaker_mapping.get("mappings", [])),
        "source_media_hash": speaker_mapping.get("source_media_hash"),
        "destination_media_hash": speaker_mapping.get("destination_media_hash"),
    }
    mapped["speaker_mapping_summary"] = speaker_mapping_summary(mapped)
    return mapped


def speaker_mapping_summary(schedule: dict[str, Any]) -> dict[str, Any]:
    mappings = [mapping for mapping in schedule.get("mappings", []) if mapping.get("enabled", True)]
    aware = [mapping for mapping in mappings if mapping.get("mapped_destination_speaker_id") or mapping.get("speaker_mapping_fallback_reason")]
    followed = [mapping for mapping in aware if mapping.get("speaker_mapping_followed")]
    fallbacks = Counter(str(mapping.get("speaker_mapping_fallback_reason")) for mapping in aware if mapping.get("speaker_mapping_fallback_reason"))
    return {
        "speaker_mapping_aware_count": len(aware),
        "speaker_mapping_followed_count": len(followed),
        "speaker_mapping_followed_rate": round(len(followed) / len(aware), 4) if aware else 0.0,
        "fallback_reasons": dict(sorted(fallbacks.items())),
    }


def _rank_speakers(speaker_map: dict[str, Any]) -> list[dict[str, Any]]:
    speakers = [dict(row) for row in speaker_map.get("speakers", [])]
    real_speakers = [row for row in speakers if row.get("speaker_id") and not str(row.get("speaker_id")).startswith("unknown_")]
    if real_speakers:
        speakers = real_speakers
    speakers.sort(key=lambda row: (-_float(row.get("total_duration"), 0.0), -int(row.get("event_count", 0) or 0), str(row.get("speaker_id", ""))))
    return speakers


def _speaker_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_speaker[str(item.get("speaker_id"))].append(item)
    summaries = []
    for speaker_id in sorted(by_speaker):
        rows = by_speaker[speaker_id]
        starts = [_float(row.get("start"), 0.0) for row in rows]
        ends = [_float(row.get("end"), _float(row.get("start"), 0.0)) for row in rows]
        summaries.append(
            {
                "speaker_id": speaker_id,
                "total_duration": round(sum(_duration(row) for row in rows), 3),
                "event_count": len(rows),
                "first_seen": round(min(starts), 3),
                "last_seen": round(max(ends), 3),
                "confidence": round(sum(_float(row.get("speaker_confidence"), 0.45) for row in rows) / len(rows), 4),
            }
        )
    return summaries


def _best_segment_for_item(item: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    start = _float(item.get("start"), _float(item.get("movie_timestamp"), 0.0))
    end = _float(item.get("end"), start + _float(item.get("duration"), 0.0))
    best = None
    best_overlap = 0.0
    for segment in segments:
        overlap = max(0.0, min(end, _float(segment.get("end"), 0.0)) - max(start, _float(segment.get("start"), 0.0)))
        if overlap > best_overlap:
            best = segment
            best_overlap = overlap
    return best


def _segments_in_range(segments: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [segment for segment in segments if max(start, _float(segment.get("start"), 0.0)) < min(end, _float(segment.get("end"), 0.0))]


def _speaker_pattern(segments: list[dict[str, Any]]) -> str:
    sequence = []
    previous = None
    for segment in sorted(segments, key=lambda row: _float(row.get("start"), 0.0)):
        speaker_id = str(segment.get("speaker_id") or "")
        if speaker_id and speaker_id != previous:
            sequence.append(speaker_id)
            previous = speaker_id
    return " ".join(sequence)


def _dominant_speaker(segments: list[dict[str, Any]]) -> str | None:
    durations: Counter[str] = Counter()
    for segment in segments:
        speaker_id = str(segment.get("speaker_id") or "")
        if speaker_id:
            durations[speaker_id] += _duration(segment)
    return durations.most_common(1)[0][0] if durations else None


def _duration(item: dict[str, Any]) -> float:
    start = _float(item.get("start"), _float(item.get("movie_timestamp"), 0.0))
    end = _float(item.get("end"), start + _float(item.get("duration"), 0.0))
    return max(0.0, end - start)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default




