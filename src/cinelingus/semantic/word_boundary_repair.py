from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..render_verification import evaluate_rendered_dialogue, lexically_equivalent_transcript
from ..tools import run
from ..util import read_json, stable_hash, utc_now, write_json
from ..whisper_backend import transcribe_words_with_whisper, transcribe_with_whisper
from .acoustic_preflight import _digest


WordTranscriber = Callable[..., dict[str, Any]]
Verifier = Callable[..., dict[str, Any]]
Extractor = Callable[[Path, float, float, Path, int, int], None]
REPAIR_VERSION = "semantic_word_boundary_repair_v2_timing_preserving_silence"
TOKEN = re.compile(r"[a-z0-9']+")


def repair_semantic_word_boundaries(
    *, clip_library: dict[str, Any], rejection_evidence: dict[str, Any],
    analysis_audio: Path, output_dir: Path, model_name: str = "medium",
    language: str | None = "en", transcription_mode: str = "quality",
    context_padding: float = 4.0, leading_padding: float = 0.08,
    trailing_padding: float = 0.12, force: bool = False,
    word_transcriber: WordTranscriber = transcribe_words_with_whisper,
    verifier: Verifier = transcribe_with_whisper, extractor: Extractor | None = None,
) -> dict[str, Any]:
    """Replace coarse rejected clips with verified word-aligned derived clips."""
    if not analysis_audio.is_file():
        raise FileNotFoundError(f"Analysis audio is unavailable: {analysis_audio}")
    extractor = extractor or _extract
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir, repaired_dir = output_dir / "word_contexts", output_dir / "repaired_clips"
    contexts_dir.mkdir(exist_ok=True); repaired_dir.mkdir(exist_ok=True)
    rejected_ids = _rejected_clip_ids(rejection_evidence)
    clips = [dict(row) for row in clip_library.get("clips", [])]
    clip_by_id = {str(row.get("id")): row for row in clips}
    audio_duration = _wav_duration(analysis_audio)
    repaired_by_id: dict[str, dict[str, Any]] = {}
    results = []
    for clip_id in rejected_ids:
        clip = clip_by_id.get(clip_id)
        if clip is None:
            results.append({"clip_id": clip_id, "repair_state": "MISSING_CLIP", "reason": "clip_not_present_in_library"})
            continue
        original_start = float(clip.get("movie_timestamp", 0.0) or 0.0)
        original_duration = float(clip.get("duration", 0.0) or 0.0)
        context_start = max(0.0, original_start - context_padding)
        context_end = min(audio_duration, original_start + original_duration + context_padding)
        context_path = contexts_dir / f"{clip_id}.wav"
        if force or not context_path.is_file():
            extractor(analysis_audio, context_start, context_end - context_start, context_path, 16000, 1)
        word_path = contexts_dir / f"{clip_id}_words.json"
        signature = stable_hash({
            "version": REPAIR_VERSION, "context_sha256": _digest(context_path),
            "intended": clip.get("transcript"), "model": model_name,
            "language": language, "mode": transcription_mode,
        })
        cached = read_json(word_path) if word_path.is_file() else {}
        if not force and cached.get("media_hash") == signature:
            word_timeline = cached
        else:
            word_timeline = word_transcriber(
                audio_path=context_path, media_hash=signature, output_path=word_path,
                model_name=model_name, language=language, transcription_mode=transcription_mode,
            )
            if not word_path.is_file():
                write_json(word_path, word_timeline)
        alignment = _align_words(str(clip.get("transcript") or ""), word_timeline.get("words") or [])
        if alignment is None:
            results.append({
                "clip_id": clip_id, "intended_transcript": clip.get("transcript", ""),
                "repair_state": "UNRECOVERED", "reason": "intended_words_not_found_in_context",
                "context_start": round(context_start, 3), "observed_context_text": word_timeline.get("text", ""),
            })
            continue
        aligned_start = max(context_start, context_start + float(alignment[0]["start"]) - leading_padding)
        aligned_end = min(context_end, context_start + float(alignment[-1]["end"]) + trailing_padding)
        repaired_path = repaired_dir / f"{clip_id}.wav"
        extractor(analysis_audio, aligned_start, aligned_end - aligned_start, repaired_path, 48000, 2)
        content_duration = aligned_end - aligned_start
        delivery_duration = max(content_duration, original_duration)
        if delivery_duration > content_duration + 0.001:
            _pad_wav_to_duration(repaired_path, delivery_duration)
        verification_path = repaired_dir / f"{clip_id}_verification.json"
        verify_signature = stable_hash({
            "version": REPAIR_VERSION, "clip_sha256": _digest(repaired_path),
            "intended": clip.get("transcript"), "model": model_name,
            "language": language, "mode": transcription_mode,
        })
        cached_verification = read_json(verification_path) if verification_path.is_file() else {}
        if not force and cached_verification.get("media_hash") == verify_signature:
            timeline = cached_verification
        else:
            timeline = verifier(
                audio_path=repaired_path, media_hash=verify_signature, output_path=verification_path,
                model_name=model_name, language=language, artifact_type="timeline",
                transcription_mode=transcription_mode,
            )
            if not verification_path.is_file():
                write_json(verification_path, timeline)
        verification = evaluate_rendered_dialogue(
            schedule={"mappings": [{
                "id": clip_id, "clip_id": clip_id, "destination_timestamp": 0.0,
                "planned_render_duration": round(delivery_duration, 3),
                "source_transcript": clip.get("transcript", ""), "enabled": True,
            }]}, rendered_timeline=timeline,
        )["mappings"][0]
        accepted = _accepted(verification)
        if accepted:
            repaired = {
                **clip, "path": str(repaired_path),
                "movie_timestamp": round(aligned_start, 3),
                "duration": round(delivery_duration, 3),
                "speech_rate": round(len(_tokens(clip.get("transcript"))) / max(delivery_duration, 0.001), 3),
                "word_boundary_repair": {
                    "repair_version": REPAIR_VERSION,
                    "original_path": clip.get("path"),
                    "original_movie_timestamp": clip.get("movie_timestamp"),
                    "original_duration": clip.get("duration"),
                    "content_duration": round(content_duration, 3),
                    "delivery_duration": round(delivery_duration, 3),
                    "timing_padding": "trailing_silence" if delivery_duration > content_duration + 0.001 else "none",
                    "aligned_words": [row.get("text") for row in alignment],
                    "verification_observed_transcript": verification.get("rendered_transcript", ""),
                },
            }
            repaired_by_id[clip_id] = repaired
        results.append({
            "clip_id": clip_id, "intended_transcript": clip.get("transcript", ""),
            "repair_state": "REPAIRED" if accepted else "UNRECOVERED",
            "reason": None if accepted else "derived_clip_failed_independent_verification",
            "original_start": round(original_start, 3), "original_duration": round(original_duration, 3),
            "repaired_start": round(aligned_start, 3), "content_duration": round(content_duration, 3),
            "repaired_duration": round(delivery_duration, 3),
            "aligned_words": [row.get("text") for row in alignment],
            "verification": verification,
        })
    overlay = {
        **clip_library, "tool_version": __version__, "creation_timestamp": utc_now(),
        "clip_library_role": "derived_word_boundary_repair_overlay",
        "parent_clip_library_signature": stable_hash(clip_library),
        "word_boundary_repair_version": REPAIR_VERSION,
        "clips": [repaired_by_id.get(str(row.get("id")), row) for row in clips],
    }
    overlay_path = output_dir / "repaired_clip_library.json"
    write_json(overlay_path, overlay)
    repaired_count = len(repaired_by_id)
    report = {
        "schema_version": "1.0", "repair_version": REPAIR_VERSION,
        "creation_timestamp": utc_now(),
        "repair_signature": stable_hash({
            "version": REPAIR_VERSION, "audio_sha256": _digest(analysis_audio),
            "evidence": rejection_evidence.get("preflight_signature") or rejection_evidence.get("audit_signature"),
            "clips": rejected_ids, "model": model_name, "context_padding": context_padding,
        }),
        "source_evidence_signature": rejection_evidence.get("preflight_signature") or rejection_evidence.get("audit_signature"),
        "rejected_clip_count": len(rejected_ids), "repaired_clip_count": repaired_count,
        "unrecovered_clip_count": len(rejected_ids) - repaired_count,
        "repaired_clip_ids": sorted(repaired_by_id), "clips": results,
        "repaired_clip_library": str(overlay_path),
        "repair_state": "ALL_REPAIRED" if rejected_ids and repaired_count == len(rejected_ids) else "PARTIAL_REPAIR" if repaired_count else "NO_REPAIR",
        "claim_scope": "Word-timestamp boundary recovery with independent lexical verification; no speaker, visual-fit, semantic-preference, or phoneme-perfect-boundary claim.",
    }
    write_json(output_dir / "semantic_word_boundary_repair.json", report)
    return report


def _rejected_clip_ids(evidence: dict[str, Any]) -> list[str]:
    ids = {
        str(row.get("clip_id")) for row in evidence.get("mapping_decisions", [])
        if row.get("clip_id") and row.get("state") == "REJECTED"
    }
    ids.update(
        str(row.get("clip_id")) for row in evidence.get("clips", [])
        if row.get("clip_id") and row.get("health_state") != "ACCEPTED"
    )
    return sorted(ids)


def _align_words(intended: str, observed_words: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    intended_tokens = _tokens(intended)
    observed = [(row, _tokens(row.get("text"))) for row in observed_words]
    flattened = [(row, token) for row, tokens in observed for token in tokens]
    if not intended_tokens or len(flattened) < len(intended_tokens):
        return None
    for start in range(len(flattened) - len(intended_tokens) + 1):
        span = flattened[start:start + len(intended_tokens)]
        if all(lexically_equivalent_transcript(expected, actual) for expected, (_, actual) in zip(intended_tokens, span)):
            return [row for row, _ in span]
    return None


def _accepted(row: dict[str, Any]) -> bool:
    lexical = lexically_equivalent_transcript(row.get("intended_transcript"), row.get("rendered_transcript"))
    return (
        (lexical or (
            float(row.get("word_coverage_percentage", 0.0) or 0.0) >= 90.0
            and not row.get("missing_sentence_beginning") and not row.get("missing_sentence_ending")
        ))
        and not row.get("adjacent_dialogue_before") and not row.get("adjacent_dialogue_after")
    )


def _tokens(value: Any) -> list[str]:
    return TOKEN.findall(str(value or "").lower())


def _extract(source: Path, start: float, duration: float, target: Path, rate: int, channels: int) -> None:
    run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source), "-vn", "-ac", str(channels), "-ar", str(rate), str(target)])


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _pad_wav_to_duration(path: Path, duration: float) -> None:
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        payload = source.readframes(source.getnframes())
    current_frames = len(payload) // (params.nchannels * params.sampwidth)
    target_frames = max(current_frames, int(round(duration * params.framerate)))
    payload += b"\0" * ((target_frames - current_frames) * params.nchannels * params.sampwidth)
    with wave.open(str(path), "wb") as target:
        target.setparams(params)
        target.writeframes(payload)
