from __future__ import annotations

from pathlib import Path

from . import __version__
from .util import utc_now, write_json


def whisper_runtime() -> dict:
    try:
        import torch
        import whisper  # noqa: F401
    except ImportError:
        return {"available": False, "cuda_available": False, "device": None}
    cuda = torch.cuda.is_available()
    return {"available": True, "cuda_available": cuda, "device": "cuda" if cuda else "cpu"}


def transcribe_with_whisper(
    *,
    audio_path: Path,
    media_hash: str,
    output_path: Path,
    model_name: str,
    language: str | None,
    artifact_type: str,
    transcription_mode: str,
) -> dict:
    try:
        import torch
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai-whisper is not installed") from exc

    transcription_audio = audio_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_model_name, result, fallback_warning = _transcribe_with_model_fallback(
        whisper=whisper,
        transcription_audio=transcription_audio,
        requested_model=model_name,
        device=device,
        language=language,
    )
    segments = result.get("segments", [])
    detected_language = result.get("language")
    if artifact_type == "dialogue_events":
        key = "events"
        rows = _rows_for_events(segments)
    elif artifact_type == "timeline":
        key = "windows"
        rows = _rows_for_windows(segments)
    else:
        raise ValueError(f"Unknown Whisper artifact type: {artifact_type}")

    data = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "media_hash": media_hash,
        "creation_timestamp": utc_now(),
        "detector": f"openai_whisper:{actual_model_name}:{device}",
        "speech_backend": "whisper",
        "transcription_mode": transcription_mode,
        "requested_whisper_model": model_name,
        "whisper_model": actual_model_name,
        "whisper_model_fallback": actual_model_name != model_name,
        "whisper_model_warning": fallback_warning,
        "whisper_device": device,
        "cuda_available": torch.cuda.is_available(),
        "configured_language": language,
        "detected_language": detected_language,
        key: rows,
    }
    write_json(output_path, data)
    return data


MODEL_FALLBACK_ORDER = ("medium", "small", "base", "tiny")


def _transcribe_with_model_fallback(
    *,
    whisper,
    transcription_audio: Path,
    requested_model: str,
    device: str,
    language: str | None,
) -> tuple[str, dict, str | None]:
    errors: list[str] = []
    for candidate in _fallback_candidates(requested_model):
        try:
            model = whisper.load_model(candidate, device=device)
            options = {"verbose": False, "fp16": device == "cuda"}
            if language:
                options["language"] = language
            result = model.transcribe(str(transcription_audio), **options)
            if candidate == requested_model:
                return candidate, result, None
            warning = (
                f"Requested Whisper model '{requested_model}' could not be used on {device}; "
                f"fell back to '{candidate}'. Previous errors: {'; '.join(errors)}"
            )
            return candidate, result, warning
        except (RuntimeError, MemoryError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(f"Unable to load any Whisper fallback model for '{requested_model}'. Errors: {'; '.join(errors)}")


def _fallback_candidates(requested_model: str) -> list[str]:
    if requested_model in MODEL_FALLBACK_ORDER:
        return list(MODEL_FALLBACK_ORDER[MODEL_FALLBACK_ORDER.index(requested_model) :])
    return [requested_model, "small", "base", "tiny"]


def _rows_for_events(segments: list[dict]) -> list[dict]:
    rows = []
    for i, segment in enumerate(segments, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        rows.append(
            {
                "id": f"e{i:06d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "transcript": str(segment.get("text", "")).strip(),
                "confidence": _segment_confidence(segment),
                "speaker": None,
            }
        )
    return rows


def _rows_for_windows(segments: list[dict]) -> list[dict]:
    rows = []
    for i, segment in enumerate(segments, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        rows.append(
            {
                "id": f"w{i:06d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "confidence": _segment_confidence(segment),
                "transcript": str(segment.get("text", "")).strip(),
            }
        )
    return rows


def _segment_confidence(segment: dict) -> float:
    no_speech = float(segment.get("no_speech_prob", 0.0) or 0.0)
    return round(max(0.0, min(1.0, 1.0 - no_speech)), 4)
