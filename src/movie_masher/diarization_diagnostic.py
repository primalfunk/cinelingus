from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .build_info import build_identification, format_build_identification
from .cache import media_hash
from .config import load_config
from .detect import detect_voice_windows, extract_analysis_audio
from .diarization_runtime import DEFAULT_INFERENCE_TIMEOUT_SECONDS, run_pyannote_diagnostic
from .media import inspect_media
from .speakers import _hf_token
from .util import write_json


def diagnose_diarization(root: Path, media_path: Path) -> tuple[dict, Path, Path]:
    media_path = media_path.expanduser().resolve()
    if not media_path.exists():
        raise FileNotFoundError(f"Media file does not exist: {media_path}")
    config = load_config(root).with_overrides(destination_video=media_path, source_dialogue=media_path)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = root / "temp" / "diarization_diagnostic" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "analysis_audio.wav"
    media_info_path = output_dir / "movie.json"
    media_info = inspect_media(media_path, media_hash(media_path), media_info_path)
    extract_analysis_audio(media_path, audio_path)
    speech_items = detect_voice_windows(
        audio_path,
        float(media_info["duration"]),
        noise_db=config.silence_noise_db,
        min_silence=config.silence_min_duration,
        min_speech=config.min_speech_duration,
        merge_gap=config.merge_gap,
    )
    for index, item in enumerate(speech_items, start=1):
        item.setdefault("id", f"speech_{index:06d}")
    report_path = output_dir / "diarization_report.json"
    traceback_path = output_dir / "diarization_traceback.txt"
    for line in format_build_identification(root):
        print(line)
    print(f"Diagnostic media: {media_path}")
    report = run_pyannote_diagnostic(
        audio_path=audio_path,
        speech_items=speech_items,
        model_name=config.speaker_diarization_model,
        device=config.speaker_diarization_device,
        token=_hf_token(),
        timeout_seconds=DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        report_path=report_path,
        traceback_path=traceback_path,
        stage_callback=lambda stage: print(f"Diarization stage: {stage}"),
        log=print,
    )
    report["build_identification"] = build_identification(root)
    report["media_path"] = str(media_path)
    report["speech_event_count"] = len(speech_items)
    write_json(report_path, report)
    print(f"Inference completed: {str(bool(report.get('inference_completed'))).lower()}")
    print(f"Validation passed: {str(bool(report.get('validation_passed'))).lower()}")
    print(f"Fallback reason: {report.get('fallback_reason')}")
    print(f"Exact elapsed time: {report.get('elapsed_seconds')} seconds")
    print(f"Actual device: {report.get('actual_device')}")
    print(f"Worker cleanup result: {report.get('worker_terminated_successfully')}")
    print(f"Diarization report: {report_path}")
    print(f"Traceback artifact: {traceback_path}")
    return report, report_path, traceback_path
