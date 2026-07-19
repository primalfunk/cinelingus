from __future__ import annotations

import gc
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
import warnings
import wave
from pathlib import Path
from typing import Any, Callable

from .ffmpeg_env import ensure_project_ffmpeg_shared_on_path
from .pyannote_adapter import diarization_tracks
from .util import utc_now, write_json

DIARIZATION_SCHEMA_VERSION = "2.2"
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 180
DEFAULT_TOTAL_INFERENCE_TIMEOUT_SECONDS = 3600
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 6.0
DEFAULT_CUDA_MEMORY_FRACTION = 0.65
DEFAULT_SPEAKER_MATCH_MAX_COSINE_DISTANCE = 0.35


def _run_pipeline_without_empty_cluster_warning_noise(pipeline: Any, audio: Any) -> Any:
    """Run Pyannote while hiding known non-fatal empty-cluster warnings.

    Community-1 can evaluate empty intermediate cluster candidates even when
    the final diarization is valid. Result validation below remains authoritative.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
        return _resolve_pipeline_output(pipeline(audio))


def _worker(result_queue, audio_path: str, model_name: str, device: str, token: str | None) -> None:
    try:
        ensure_project_ffmpeg_shared_on_path(search_from=Path(audio_path))
        from pyannote.audio import Pipeline  # type: ignore
        from .speakers import _load_pyannote_audio_input, _resolve_pyannote_device

        selected_device = _resolve_pyannote_device(device)
        actual_device = str(selected_device or "cpu")
        cuda_memory_fraction = _configure_cuda_memory_limit(selected_device)
        pipeline = Pipeline.from_pretrained(model_name, token=token)
        if selected_device:
            pipeline.to(selected_device)
        result_queue.put({"kind": "progress", "phase": "model_loaded", "actual_device": actual_device})
        with wave.open(audio_path, "rb") as source:
            duration = source.getnframes() / max(1, source.getframerate())
        if duration > DEFAULT_CHUNK_SECONDS:
            audio = _load_pyannote_audio_input(Path(audio_path))
            turns, chunk_count = _run_chunked_pipeline(
                pipeline,
                audio,
                duration,
                chunk_seconds=DEFAULT_CHUNK_SECONDS,
                overlap_seconds=DEFAULT_CHUNK_OVERLAP_SECONDS,
                cleanup_cuda=actual_device.startswith("cuda"),
                progress_callback=lambda completed, total: result_queue.put(
                    {"kind": "progress", "phase": "chunk_complete", "chunk_index": completed, "chunk_count": total}
                ),
            )
            output_debug = {
                "mode": "overlapping_chunks",
                "chunk_count": chunk_count,
                "chunk_seconds": DEFAULT_CHUNK_SECONDS,
                "overlap_seconds": DEFAULT_CHUNK_OVERLAP_SECONDS,
                "speaker_stitching": "overlap_then_global_embedding_centroid",
                "speaker_match_max_cosine_distance": _speaker_match_max_cosine_distance(),
                "cuda_memory_fraction": cuda_memory_fraction,
            }
        else:
            output = _run_pipeline_without_empty_cluster_warning_noise(pipeline, audio_path)
            turns = _turn_rows(output)
            chunk_count = 1
            output_debug = _describe_output(output)
            output_debug["cuda_memory_fraction"] = cuda_memory_fraction
        result_queue.put({"kind": "result", "ok": True, "turns": turns, "actual_device": actual_device, "output_debug": output_debug, "chunk_count": chunk_count})
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "result",
                "ok": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _turn_rows(output: Any) -> list[dict[str, Any]]:
    return [{"start": float(turn.start), "end": float(turn.end), "raw_speaker": str(label)} for turn, _track, label in diarization_tracks(output)]


def _run_chunked_pipeline(
    pipeline: Any,
    audio: dict[str, Any],
    duration: float,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    cleanup_cuda: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("chunk_seconds must be positive and overlap_seconds must be in [0, chunk_seconds)")
    audio_duration = waveform.shape[-1] / sample_rate
    effective_duration = min(float(duration), audio_duration)
    windows = _chunk_windows(effective_duration, chunk_seconds, overlap_seconds)
    global_turns: list[dict[str, Any]] = []
    previous_context: list[dict[str, Any]] = []
    global_embeddings: dict[str, dict[str, Any]] = {}
    next_global = 1
    for chunk_index, (chunk_start, chunk_end) in enumerate(windows, start=1):
        start_sample = max(0, int(round(chunk_start * sample_rate)))
        end_sample = min(waveform.shape[-1], int(round(chunk_end * sample_rate)))
        chunk_waveform = waveform[..., start_sample:end_sample].clone()
        output: Any = None
        try:
            output = _run_pipeline_without_empty_cluster_warning_noise(
                pipeline,
                {"waveform": chunk_waveform, "sample_rate": sample_rate},
            )
            local_turns = _turn_rows(output)
            local_embeddings = _speaker_embedding_rows(output)
        finally:
            del output
            del chunk_waveform
            _release_chunk_memory(cleanup_cuda)
        for row in local_turns:
            row["start"] += chunk_start
            row["end"] += chunk_start

        label_map = _overlap_label_mapping(local_turns, previous_context)
        embedding_map = _embedding_label_mapping(
            local_turns,
            local_embeddings,
            global_embeddings,
            claimed_globals=set(label_map.values()),
            max_cosine_distance=_speaker_match_max_cosine_distance(),
        )
        for local_label, global_label in embedding_map.items():
            label_map.setdefault(local_label, global_label)
        for local_label in sorted({str(row["raw_speaker"]) for row in local_turns}):
            if local_label not in label_map:
                label_map[local_label] = f"GLOBAL_{next_global:03d}"
                next_global += 1
        _update_global_embeddings(global_embeddings, local_turns, local_embeddings, label_map)
        for row in local_turns:
            row["raw_speaker"] = label_map[str(row["raw_speaker"])]
        previous_context = [dict(row) for row in local_turns]

        keep_start = 0.0 if chunk_index == 1 else chunk_start + overlap_seconds / 2.0
        keep_end = effective_duration if chunk_index == len(windows) else chunk_end - overlap_seconds / 2.0
        for row in local_turns:
            owned = dict(row)
            owned["start"] = max(float(owned["start"]), keep_start)
            owned["end"] = min(float(owned["end"]), keep_end)
            if owned["end"] > owned["start"]:
                global_turns.append(owned)
        if progress_callback:
            progress_callback(chunk_index, len(windows))
    global_turns.sort(key=lambda row: (row["start"], row["end"]))
    return _merge_boundary_turns(global_turns), len(windows)


def _chunk_windows(duration: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    step = chunk_seconds - overlap_seconds
    windows = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        windows.append((start, end))
        if end >= duration:
            break
        start += step
    return windows


def _overlap_label_mapping(current_turns: list[dict[str, Any]], previous_turns: list[dict[str, Any]]) -> dict[str, str]:
    scores: dict[tuple[str, str], float] = {}
    for current in current_turns:
        local_label = str(current["raw_speaker"])
        for previous in previous_turns:
            overlap = max(
                0.0,
                min(float(current["end"]), float(previous["end"]))
                - max(float(current["start"]), float(previous["start"])),
            )
            if overlap > 0:
                key = (local_label, str(previous["raw_speaker"]))
                scores[key] = scores.get(key, 0.0) + overlap

    # Local labels are distinct clusters. A one-to-one match prevents two new
    # speakers from being collapsed into the same global speaker at a boundary.
    mapping: dict[str, str] = {}
    claimed_globals: set[str] = set()
    for (local_label, global_label), score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if score < 0.25 or local_label in mapping or global_label in claimed_globals:
            continue
        mapping[local_label] = global_label
        claimed_globals.add(global_label)
    return mapping


def _speaker_embedding_rows(output: Any) -> dict[str, list[float]]:
    embeddings = output.get("speaker_embeddings") if isinstance(output, dict) else getattr(output, "speaker_embeddings", None)
    if embeddings is None:
        return {}
    annotation = output.get("speaker_diarization") if isinstance(output, dict) else getattr(output, "speaker_diarization", None)
    if annotation is None or not hasattr(annotation, "labels"):
        return {}
    try:
        labels = [str(label) for label in annotation.labels()]
        rows = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
    except (TypeError, ValueError):
        return {}
    if len(labels) != len(rows):
        return {}
    result: dict[str, list[float]] = {}
    for label, row in zip(labels, rows):
        try:
            vector = [float(value) for value in row]
        except (TypeError, ValueError):
            continue
        if vector and all(math.isfinite(value) for value in vector) and _vector_norm(vector) > 0:
            result[label] = vector
    return result


def _embedding_label_mapping(
    local_turns: list[dict[str, Any]],
    local_embeddings: dict[str, list[float]],
    global_embeddings: dict[str, dict[str, Any]],
    *,
    claimed_globals: set[str],
    max_cosine_distance: float,
) -> dict[str, str]:
    local_labels = sorted({str(row["raw_speaker"]) for row in local_turns})
    candidates: list[tuple[float, str, str]] = []
    for local_label in local_labels:
        local_vector = local_embeddings.get(local_label)
        if local_vector is None:
            continue
        for global_label, state in global_embeddings.items():
            if global_label in claimed_globals:
                continue
            distance = _cosine_distance(local_vector, state.get("centroid", []))
            if distance <= max_cosine_distance:
                candidates.append((distance, local_label, global_label))
    mapping: dict[str, str] = {}
    for _distance, local_label, global_label in sorted(candidates):
        if local_label in mapping or global_label in claimed_globals:
            continue
        mapping[local_label] = global_label
        claimed_globals.add(global_label)
    return mapping


def _update_global_embeddings(
    global_embeddings: dict[str, dict[str, Any]],
    local_turns: list[dict[str, Any]],
    local_embeddings: dict[str, list[float]],
    label_map: dict[str, str],
) -> None:
    durations: dict[str, float] = {}
    for row in local_turns:
        label = str(row["raw_speaker"])
        durations[label] = durations.get(label, 0.0) + max(0.0, float(row["end"]) - float(row["start"]))
    for local_label, global_label in label_map.items():
        vector = local_embeddings.get(local_label)
        if vector is None:
            continue
        weight = max(0.001, durations.get(local_label, 0.0))
        state = global_embeddings.get(global_label)
        if state is None or len(state.get("centroid", [])) != len(vector):
            global_embeddings[global_label] = {"centroid": list(vector), "weight": weight}
            continue
        previous_weight = float(state.get("weight", 0.0))
        total_weight = previous_weight + weight
        state["centroid"] = [
            (float(old) * previous_weight + float(new) * weight) / total_weight
            for old, new in zip(state["centroid"], vector)
        ]
        state["weight"] = total_weight


def _speaker_match_max_cosine_distance() -> float:
    raw = os.environ.get(
        "CINELINGUS_SPEAKER_MATCH_MAX_COSINE_DISTANCE",
        str(DEFAULT_SPEAKER_MATCH_MAX_COSINE_DISTANCE),
    )
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError("CINELINGUS_SPEAKER_MATCH_MAX_COSINE_DISTANCE must be between 0 and 1")
    return value


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return math.inf
    denominator = _vector_norm(left) * _vector_norm(right)
    if denominator <= 0:
        return math.inf
    similarity = sum(a * b for a, b in zip(left, right)) / denominator
    return max(0.0, min(2.0, 1.0 - similarity))


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _merge_boundary_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in turns:
        if (
            merged
            and merged[-1]["raw_speaker"] == row["raw_speaker"]
            and abs(float(merged[-1]["end"]) - float(row["start"])) <= 1e-6
        ):
            merged[-1]["end"] = max(float(merged[-1]["end"]), float(row["end"]))
        else:
            merged.append(dict(row))
    return merged


def _release_chunk_memory(cleanup_cuda: bool) -> None:
    gc.collect()
    if not cleanup_cuda:
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _configure_cuda_memory_limit(selected_device: Any) -> float | None:
    if selected_device is None or not str(selected_device).startswith("cuda"):
        return None
    raw_fraction = os.environ.get("CINELINGUS_CUDA_MEMORY_FRACTION", str(DEFAULT_CUDA_MEMORY_FRACTION))
    fraction = float(raw_fraction)
    if not 0.1 <= fraction <= 0.9:
        raise ValueError("CINELINGUS_CUDA_MEMORY_FRACTION must be between 0.1 and 0.9")
    import torch

    normalized_device = torch.device(selected_device)
    device_index = normalized_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, int(device_index))
    return fraction

def _resolve_pipeline_output(result: Any) -> Any:
    if not hasattr(result, "__next__"):
        return result
    while True:
        try:
            next(result)
        except StopIteration as completed:
            return completed.value

def _describe_output(output: Any) -> dict[str, Any]:
    debug: dict[str, Any] = {"output_type": f"{type(output).__module__}.{type(output).__name__}"}
    debug["output_attributes"] = sorted(name for name in dir(output) if not name.startswith("_"))
    for name in ("speaker_diarization", "exclusive_speaker_diarization"):
        value = getattr(output, name, None)
        row = {"present": value is not None, "type": f"{type(value).__module__}.{type(value).__name__}" if value is not None else None}
        if value is not None:
            try:
                direct = list(value)
                row["direct_length"] = len(direct)
                row["first_direct_repr"] = repr(direct[:3])[:1000]
            except Exception as exc:
                row["direct_error"] = f"{type(exc).__name__}: {exc}"
            if hasattr(value, "itertracks"):
                try:
                    tracks = list(value.itertracks(yield_label=True))
                    row["itertracks_length"] = len(tracks)
                    row["first_itertracks_repr"] = repr(tracks[:3])[:1000]
                except Exception as exc:
                    row["itertracks_error"] = f"{type(exc).__name__}: {exc}"
        debug[name] = row
    return debug


def _receive_worker_result(
    worker: Any,
    result_queue: Any,
    *,
    inactivity_timeout_seconds: int,
    total_timeout_seconds: int,
    progress_callback: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any] | None, str | None]:
    started = time.monotonic()
    last_activity = started
    while True:
        now = time.monotonic()
        if now - started >= total_timeout_seconds:
            return None, f"Pyannote inference reached the {total_timeout_seconds}-second total safety limit."
        if now - last_activity >= inactivity_timeout_seconds:
            return None, f"Pyannote made no progress for {inactivity_timeout_seconds} seconds."
        wait_seconds = min(
            0.5,
            total_timeout_seconds - (now - started),
            inactivity_timeout_seconds - (now - last_activity),
        )
        try:
            message = result_queue.get(timeout=max(0.01, wait_seconds))
        except queue.Empty:
            if worker.is_alive():
                continue
            try:
                message = result_queue.get(timeout=2)
            except queue.Empty:
                return None, None
        if message.get("kind") == "progress":
            last_activity = time.monotonic()
            progress_callback(message)
            continue
        return message, None


def _terminate_worker(worker: Any) -> None:
    if not worker.is_alive():
        return
    worker.terminate()
    worker.join(5)
    if worker.is_alive():
        try:
            worker.kill()
        except (AttributeError, OSError):
            worker.terminate()
        worker.join(5)

def run_pyannote_diagnostic(
    *,
    audio_path: Path,
    speech_items: list[dict[str, Any]],
    model_name: str,
    device: str,
    token: str | None,
    timeout_seconds: int = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    total_timeout_seconds: int = DEFAULT_TOTAL_INFERENCE_TIMEOUT_SECONDS,
    role: str = "source",
    report_path: Path,
    traceback_path: Path,
    stage_callback: Callable[[str], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    emit = log or (lambda _message: None)
    stage = stage_callback or (lambda _stage: None)
    report: dict[str, Any] = {
        "schema_version": DIARIZATION_SCHEMA_VERSION,
        "created_at": utc_now(),
        "requested_backend": "pyannote",
        "actual_backend": "none",
        "status": "starting",
        "failed_substage": None,
        "inference_completed": False,
        "validation_passed": False,
        "timeout_seconds": int(timeout_seconds),
        "total_timeout_seconds": int(total_timeout_seconds),
        "timed_out": False,
        "worker_termination_requested": False,
        "worker_terminated_successfully": None,
        "exception_type": None,
        "exception_message": None,
        "raw_turn_count": 0,
        "raw_speaker_count": 0,
        "first_ten_normalized_turns": [],
        "non_finite_value_count": 0,
        "event_mapping_coverage": 0.0,
        "validation_errors": [],
        "actual_device": _resolve_actual_device_label(device),
        "worker_pid": None,
        "worker_alive_after_cleanup": None,
        "fallback_reason": None,
    }
    stage("loading_speaker_model")
    emit(f"Pyannote inactivity timeout: {timeout_seconds} seconds")
    emit(f"Pyannote total timeout: {total_timeout_seconds} seconds")
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    worker = context.Process(target=_worker, args=(result_queue, str(audio_path), model_name, device, token), daemon=False)
    stage(f"diarizing_{role}")
    worker.start()
    report["worker_pid"] = worker.pid

    def record_progress(message: dict[str, Any]) -> None:
        if message.get("actual_device"):
            report["actual_device"] = message["actual_device"]
        if message.get("phase") != "chunk_complete":
            return
        completed = int(message.get("chunk_index") or 0)
        total = int(message.get("chunk_count") or 0)
        report["chunks_completed"] = completed
        report["chunk_count"] = total
        stage(f"diarizing_{role}_chunk_{completed}_of_{total}")
        if completed == 1 or completed == total or completed % 5 == 0:
            emit(f"Pyannote chunk progress: {completed}/{total}")

    payload, timeout_reason = _receive_worker_result(
        worker,
        result_queue,
        inactivity_timeout_seconds=timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
        progress_callback=record_progress,
    )
    if timeout_reason:
        report.update(
            {
                "status": "timeout",
                "failed_substage": "inference",
                "timed_out": True,
                "worker_termination_requested": True,
                "fallback_reason": timeout_reason,
            }
        )
        emit(timeout_reason)
        emit("Worker termination requested.")
        _terminate_worker(worker)
    else:
        worker.join(15)
        if worker.is_alive():
            report["worker_termination_requested"] = True
            _terminate_worker(worker)
        if payload is None:
            payload = {"ok": False, "exception_type": "WorkerResultMissing", "exception_message": "worker exited without returning a result"}
    report["worker_terminated_successfully"] = not worker.is_alive()
    report["worker_alive_after_cleanup"] = worker.is_alive()
    emit(f"Worker terminated successfully: {str(not worker.is_alive()).lower()}.")
    result_queue.close()
    if worker.is_alive():
        result_queue.cancel_join_thread()
    else:
        result_queue.join_thread()

    if payload is not None:
        if not payload.get("ok"):
            report.update(
                {
                    "status": "failed",
                    "failed_substage": "inference",
                    "exception_type": payload.get("exception_type"),
                    "exception_message": payload.get("exception_message"),
                    "fallback_reason": f"Pyannote inference failed: {payload.get('exception_type')}: {payload.get('exception_message')}",
                }
            )
            traceback_path.parent.mkdir(parents=True, exist_ok=True)
            traceback_path.write_text(str(payload.get("traceback") or ""), encoding="utf-8")
        else:
            report["inference_completed"] = True
            report["actual_device"] = payload.get("actual_device")
            report["output_debug"] = payload.get("output_debug")
            report["chunk_count"] = payload.get("chunk_count", 1)
            raw_turns = list(payload.get("turns") or [])
            report["raw_turn_count"] = len(raw_turns)
            report["raw_speaker_count"] = len({str(row.get("raw_speaker")) for row in raw_turns})
            stage(f"mapping_{role}_speakers")
            normalized, non_finite = _normalize_turns(raw_turns, speech_items)
            report["normalized_turns"] = normalized
            report["first_ten_normalized_turns"] = normalized[:10]
            report["non_finite_value_count"] = non_finite
            stage(f"validating_{role}_speakers")
            coverage = _event_mapping_coverage(normalized, speech_items)
            report["event_mapping_coverage"] = coverage
            errors = _validation_errors(normalized, non_finite, coverage)
            report["validation_errors"] = errors
            valid_turns = [row for row in normalized if row.get("valid")]
            if valid_turns and coverage >= 0.5:
                report["validation_passed"] = not errors
                report["actual_backend"] = "pyannote" if not errors else "pyannote_partial"
                report["status"] = "success" if not errors else "partial"
                report["usable_turns"] = valid_turns
                if errors:
                    report["fallback_reason"] = "Partial Pyannote output retained; unmapped events require fallback labels."
            else:
                report["status"] = "fallback"
                report["failed_substage"] = "validation"
                report["fallback_reason"] = "; ".join(errors) or "Pyannote returned no usable turns."
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    stage("applying_speaker_fallback" if report["actual_backend"] in {"none", "pyannote_partial"} else f"validating_{role}_speakers")
    write_json(report_path, report)
    if not traceback_path.exists():
        traceback_path.parent.mkdir(parents=True, exist_ok=True)
        traceback_path.write_text("No exception traceback was produced.\n", encoding="utf-8")
    emit_diarization_summary(report, emit)
    return report


def emit_diarization_summary(report: dict[str, Any], emit: Callable[[str], None]) -> None:
    fields = (
        "requested_backend", "actual_backend", "status", "failed_substage", "elapsed_seconds",
        "fallback_reason", "exception_type", "exception_message", "raw_turn_count",
        "raw_speaker_count", "validation_errors", "worker_termination_requested",
        "worker_terminated_successfully",
    )
    emit("Diarization attempt summary:")
    for field in fields:
        emit(f"  {field}: {report.get(field)}")


def _resolve_actual_device_label(device: str) -> str:
    if device in {"cpu", "cuda"}:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _normalize_turns(raw_turns: list[dict[str, Any]], speech_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    labels: dict[str, str] = {}
    normalized = []
    non_finite = 0
    for index, row in enumerate(raw_turns, start=1):
        start = float(row.get("start", 0.0))
        end = float(row.get("end", 0.0))
        finite = math.isfinite(start) and math.isfinite(end)
        if not finite:
            non_finite += int(not math.isfinite(start)) + int(not math.isfinite(end))
        raw_speaker = str(row.get("raw_speaker") or "unknown")
        speaker_id = labels.setdefault(raw_speaker, f"speaker_{len(labels) + 1:03d}")
        source_id = _best_source_id(start, end, speech_items) if finite else ""
        normalized.append(
            {
                "id": f"segment_{index:06d}", "start": round(start, 3) if math.isfinite(start) else 0.0,
                "end": round(end, 3) if math.isfinite(end) else 0.0,
                "duration": round(max(0.0, end - start), 3) if finite else 0.0,
                "speaker_id": speaker_id, "speaker": speaker_id, "confidence": 0.8,
                "speaker_confidence": 0.8, "source_id": source_id,
                "valid": bool(finite and end > start),
            }
        )
    return normalized, non_finite


def _best_source_id(start: float, end: float, speech_items: list[dict[str, Any]]) -> str:
    best_id, best_overlap = "", 0.0
    for item in speech_items:
        item_start = float(item.get("start", 0.0) or 0.0)
        item_end = float(item.get("end", item_start + float(item.get("duration", 0.0) or 0.0)) or item_start)
        overlap = max(0.0, min(end, item_end) - max(start, item_start))
        if overlap > best_overlap:
            best_id, best_overlap = str(item.get("id", "")), overlap
    return best_id


def _event_mapping_coverage(turns: list[dict[str, Any]], speech_items: list[dict[str, Any]]) -> float:
    if not speech_items:
        return 0.0
    valid_turns = [row for row in turns if row.get('valid')]
    mapped_count = 0
    for item in speech_items:
        item_start = float(item.get('start', 0.0) or 0.0)
        item_end = float(item.get('end', item_start + float(item.get('duration', 0.0) or 0.0)) or item_start)
        if any(
            max(item_start, float(turn.get('start', 0.0) or 0.0))
            < min(item_end, float(turn.get('end', 0.0) or 0.0))
            for turn in valid_turns
        ):
            mapped_count += 1
    return round(mapped_count / len(speech_items), 4)


def _validation_errors(turns: list[dict[str, Any]], non_finite: int, coverage: float) -> list[str]:
    errors = []
    if not turns:
        errors.append("no speaker turns returned")
    if non_finite:
        errors.append(f"{non_finite} non-finite timing values")
    if not any(row.get("valid") for row in turns):
        errors.append("no valid positive-duration turns")
    if coverage < 0.5:
        errors.append(f"event mapping coverage {coverage:.4f} is below 0.5000")
    return errors
