from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..render_verification import evaluate_rendered_dialogue, lexically_equivalent_transcript
from ..tools import run
from ..util import read_json, stable_hash, utc_now, write_json
from ..whisper_backend import transcribe_with_whisper
from .acoustic_preflight import _build_reel, _digest


Transcriber = Callable[..., dict[str, Any]]
Extractor = Callable[[Path, float, float, Path], None]
REPAIR_VERSION = "semantic_clip_boundary_repair_v1"
SEARCH_POLICY = "fixed_duration_quarter_second_shifts_neighbor_bounded_v1"
CONFIRMATION_POLICY = "independent_reel_lexical_transcript_no_adjacent_words_v2"
DEFAULT_OFFSETS = (0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, -1.0, 1.0)


def repair_semantic_clip_boundaries(
    *, clip_library: dict[str, Any], dialogue_events: dict[str, Any],
    acoustic_audit: dict[str, Any], analysis_audio: Path, output_dir: Path,
    model_name: str = "medium", language: str | None = "en",
    transcription_mode: str = "quality", offsets: tuple[float, ...] = DEFAULT_OFFSETS,
    batch_size: int = 24, force: bool = False,
    transcriber: Transcriber = transcribe_with_whisper,
    extractor: Extractor | None = None,
) -> dict[str, Any]:
    """Recover rejected clip boundaries without mutating the canonical cache.

    A candidate is admitted only when an exact normalized transcript is observed
    in both discovery and confirmation contexts, with no neighboring words.
    """
    if not analysis_audio.is_file():
        raise FileNotFoundError(f"Analysis audio is unavailable: {analysis_audio}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "boundary_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    extractor = extractor or _extract_wav
    clips = [dict(row) for row in clip_library.get("clips", [])]
    rejected_ids = _rejected_clip_ids(acoustic_audit)
    clip_by_id = {str(row.get("id")): row for row in clips}
    events = sorted(
        (dict(row) for row in dialogue_events.get("events", [])),
        key=lambda row: float(row.get("start", 0.0) or 0.0),
    )
    event_bounds = _event_search_bounds(events, _wav_duration(analysis_audio))
    candidates: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for clip_id in rejected_ids:
        clip = clip_by_id.get(clip_id)
        if clip is None:
            missing_ids.append(clip_id)
            continue
        candidates.extend(_make_candidates(
            clip=clip, event_bounds=event_bounds, analysis_audio=analysis_audio,
            output_dir=candidate_dir, offsets=offsets, extractor=extractor,
        ))

    discovery_rows, discovery_batches = _transcribe_candidates(
        candidates, output_dir=output_dir, prefix="boundary_discovery",
        batch_size=batch_size, model_name=model_name, language=language,
        transcription_mode=transcription_mode, transcriber=transcriber, force=force,
    )
    discovery_exact = [
        row for row in discovery_rows if _is_exact(row.get("verification") or {})
    ]
    # Confirm one best candidate per source clip. Repeating many overlapping
    # versions of the same short line in one reel triggers Whisper's repetition
    # suppression and is not an independent acoustic check.
    confirmation_candidates = []
    exact_by_clip: dict[str, list[dict[str, Any]]] = {}
    for row in discovery_exact:
        exact_by_clip.setdefault(row["original_clip_id"], []).append(row)
    for rows in exact_by_clip.values():
        confirmation_candidates.append(min(rows, key=_candidate_rank))
    confirmation_rows, confirmation_batches = _transcribe_candidates(
        confirmation_candidates, output_dir=output_dir, prefix="boundary_confirmation",
        batch_size=batch_size, model_name=model_name, language=language,
        transcription_mode=transcription_mode, transcriber=transcriber, force=force,
    )
    confirmed_keys = {
        row["candidate_id"] for row in confirmation_rows
        if _is_exact(row.get("verification") or {})
    }
    confirmation_by_key = {row["candidate_id"]: row for row in confirmation_rows}
    discovery_by_clip: dict[str, list[dict[str, Any]]] = {}
    for row in discovery_rows:
        discovery_by_clip.setdefault(row["original_clip_id"], []).append(row)

    repaired_by_id: dict[str, dict[str, Any]] = {}
    results = []
    for clip_id in rejected_ids:
        clip = clip_by_id.get(clip_id)
        rows = discovery_by_clip.get(clip_id, [])
        admitted = [row for row in rows if row["candidate_id"] in confirmed_keys]
        selected = min(admitted, key=_candidate_rank) if admitted else None
        if clip is None:
            state, reason = "MISSING_CLIP", "clip_not_present_in_library"
        elif selected is None:
            state, reason = "UNRECOVERED", "no_repeatable_exact_boundary_candidate"
        else:
            state, reason = "REPAIRED", None
            confirmation = (confirmation_by_key.get(selected["candidate_id"]) or {}).get("verification") or {}
            repaired = {
                **clip,
                "path": selected["clip_path"],
                "movie_timestamp": selected["candidate_start"],
                "duration": selected["candidate_duration"],
                "boundary_repair": {
                    "repair_version": REPAIR_VERSION,
                    "original_path": clip.get("path"),
                    "original_movie_timestamp": clip.get("movie_timestamp"),
                    "original_duration": clip.get("duration"),
                    "offset_seconds": selected["offset_seconds"],
                    "candidate_id": selected["candidate_id"],
                    "evidence_state": "REPEATABLE_LEXICAL_TRANSCRIPT",
                    "discovery_observed_transcript": (selected.get("verification") or {}).get("rendered_transcript", ""),
                    "confirmation_observed_transcript": confirmation.get("rendered_transcript", ""),
                },
            }
            repaired_by_id[clip_id] = repaired
        results.append({
            "clip_id": clip_id, "intended_transcript": (clip or {}).get("transcript", ""),
            "repair_state": state, "reason": reason,
            "candidate_count": len(rows),
            "discovery_exact_count": sum(_is_exact(row.get("verification") or {}) for row in rows),
            "confirmed_exact_count": len(admitted),
            "selected_candidate": _public_candidate(
                selected,
                confirmation=(confirmation_by_key.get(selected["candidate_id"]) or {}).get("verification") if selected else None,
            ) if selected else None,
            "candidates": [_public_candidate(row) for row in rows],
        })
    overlay = {
        **clip_library,
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "clip_library_role": "derived_boundary_repair_overlay",
        "parent_clip_library_signature": stable_hash(clip_library),
        "boundary_repair_version": REPAIR_VERSION,
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
            "audit_signature": acoustic_audit.get("audit_signature"),
            "clip_library_signature": stable_hash(clip_library), "offsets": offsets,
            "model_name": model_name, "language": language,
            "transcription_mode": transcription_mode,
        }),
        "source_audit_signature": acoustic_audit.get("audit_signature"),
        "search_policy": SEARCH_POLICY, "confirmation_policy": CONFIRMATION_POLICY,
        "whisper_configuration": {
            "requested_model": model_name, "language": language,
            "transcription_mode": transcription_mode,
        },
        "rejected_clip_count": len(rejected_ids), "candidate_count": len(candidates),
        "discovery_batch_count": discovery_batches,
        "discovery_exact_candidate_count": len(discovery_exact),
        "confirmation_candidate_count": len(confirmation_candidates),
        "confirmation_batch_count": confirmation_batches,
        "repaired_clip_count": repaired_count,
        "unrecovered_clip_count": len(rejected_ids) - repaired_count,
        "missing_clip_ids": missing_ids, "clips": results,
        "repaired_clip_library": str(overlay_path),
        "repair_state": (
            "ALL_REJECTED_CLIPS_REPAIRED" if rejected_ids and repaired_count == len(rejected_ids)
            else "PARTIAL_REPAIR" if repaired_count
            else "NO_REPAIR_CANDIDATE" if rejected_ids
            else "NO_REJECTED_CLIPS"
        ),
        "claim_scope": "Repeatable lexical recovery under bounded fixed-duration shifts only; no claim of phoneme-perfect cuts, speaker identity, visual fit, or semantic preference.",
    }
    write_json(output_dir / "semantic_clip_boundary_repair.json", report)
    return report


def _rejected_clip_ids(audit: dict[str, Any]) -> list[str]:
    return sorted({
        str(row.get("clip_id")) for row in audit.get("clips", [])
        if row.get("clip_id") and row.get("health_state") != "ACCEPTED"
    })


def _event_search_bounds(events: list[dict[str, Any]], audio_duration: float) -> dict[str, tuple[float, float]]:
    result = {}
    for index, event in enumerate(events):
        prior_end = float(events[index - 1].get("end", 0.0) or 0.0) if index else 0.0
        next_start = float(events[index + 1].get("start", audio_duration) or audio_duration) if index + 1 < len(events) else audio_duration
        result[str(event.get("id"))] = (max(0.0, prior_end), min(audio_duration, next_start))
    return result


def _make_candidates(
    *, clip: dict[str, Any], event_bounds: dict[str, tuple[float, float]],
    analysis_audio: Path, output_dir: Path, offsets: tuple[float, ...], extractor: Extractor,
) -> list[dict[str, Any]]:
    clip_id = str(clip["id"])
    original_start = float(clip.get("movie_timestamp", 0.0) or 0.0)
    duration = float(clip.get("duration", 0.0) or 0.0)
    if duration <= 0:
        return []
    bounds = [event_bounds[str(value)] for value in (clip.get("event_ids") or [clip.get("event_id")]) if str(value) in event_bounds]
    lower = min((row[0] for row in bounds), default=0.0)
    upper = max((row[1] for row in bounds), default=_wav_duration(analysis_audio))
    latest_start = max(lower, upper - duration)
    seen: set[float] = set()
    rows = []
    for offset in offsets:
        start = round(min(latest_start, max(lower, original_start + float(offset))), 3)
        if start in seen:
            continue
        seen.add(start)
        actual_offset = round(start - original_start, 3)
        token = f"{len(rows) + 1:02d}_{actual_offset:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        path = output_dir / f"{clip_id}_{token}.wav"
        if not path.is_file():
            extractor(analysis_audio, start, duration, path)
        rows.append({
            "candidate_id": f"{clip_id}:{start:.3f}:{duration:.3f}",
            "original_clip_id": clip_id, "clip_id": f"{clip_id}@{start:.3f}",
            "clip_path": str(path), "candidate_start": start,
            "candidate_duration": round(duration, 3), "offset_seconds": actual_offset,
            "intended_transcript": str(clip.get("transcript") or "").strip(),
        })
    return rows


def _transcribe_candidates(
    candidates: list[dict[str, Any]], *, output_dir: Path, prefix: str,
    batch_size: int, model_name: str, language: str | None,
    transcription_mode: str, transcriber: Transcriber, force: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    batches = [candidates[index:index + batch_size] for index in range(0, len(candidates), batch_size)]
    for batch_index, batch in enumerate(batches, start=1):
        mappings = [{
            "id": row["candidate_id"], "window_id": row["candidate_id"],
            "clip_id": row["clip_id"], "clip_path": row["clip_path"],
            "clip_trim_start": 0.0, "clip_trim_duration": row["candidate_duration"],
            "source_transcript": row["intended_transcript"], "enabled": True,
        } for row in batch]
        reel_path = output_dir / f"{prefix}_reel_{batch_index:03d}.wav"
        schedule, manifest = _build_reel(mappings, reel_path)
        write_json(output_dir / f"{prefix}_manifest_{batch_index:03d}.json", manifest)
        signature = stable_hash({
            "repair_version": REPAIR_VERSION, "prefix": prefix,
            "model_name": model_name, "language": language,
            "transcription_mode": transcription_mode, "reel_sha256": _digest(reel_path),
            "candidate_ids": [row["candidate_id"] for row in batch],
        })
        transcription_path = output_dir / f"{prefix}_transcription_{batch_index:03d}.json"
        cached = read_json(transcription_path) if transcription_path.is_file() else {}
        if not force and cached.get("media_hash") == signature:
            timeline = cached
        else:
            timeline = transcriber(
                audio_path=reel_path, media_hash=signature, output_path=transcription_path,
                model_name=model_name, language=language, artifact_type="timeline",
                transcription_mode=transcription_mode,
            )
            if not transcription_path.is_file():
                write_json(transcription_path, timeline)
        verification = evaluate_rendered_dialogue(schedule=schedule, rendered_timeline=timeline)
        checked = {str(row.get("clip_id")): row for row in verification.get("mappings", [])}
        for candidate in batch:
            rows.append({**candidate, "verification": checked.get(candidate["clip_id"], {})})
    return rows, len(batches)


def _is_exact(row: dict[str, Any]) -> bool:
    intended = _normalized_words(row.get("intended_transcript"))
    observed = _normalized_words(row.get("rendered_transcript"))
    return (
        bool(str(row.get("rendered_transcript") or "").strip())
        and not row.get("adjacent_dialogue_before")
        and not row.get("adjacent_dialogue_after")
        and lexically_equivalent_transcript(" ".join(intended), " ".join(observed))
    )


def _normalized_words(value: Any) -> tuple[str, ...]:
    import re
    words = re.findall(r"[a-z0-9']+", str(value or "").lower())
    return tuple(re.sub(r"(.)\1{2,}", r"\1", word) for word in words)


def _candidate_rank(row: dict[str, Any]) -> tuple[float, float, str]:
    # Once transcript completeness is established, prefer the latest viable
    # fixed-duration window. It removes the greatest amount of leading context,
    # where short Whisper segments most often retain a previous utterance.
    return -float(row["candidate_start"]), abs(float(row["offset_seconds"])), str(row["candidate_id"])


def _public_candidate(
    row: dict[str, Any] | None, *, confirmation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    verification = row.get("verification") or {}
    return {
        "candidate_id": row["candidate_id"], "clip_path": row["clip_path"],
        "candidate_start": row["candidate_start"], "candidate_duration": row["candidate_duration"],
        "offset_seconds": row["offset_seconds"],
        "observed_transcript": verification.get("rendered_transcript", ""),
        "word_coverage_percentage": verification.get("word_coverage_percentage", 0.0),
        "adjacent_dialogue_before": verification.get("adjacent_dialogue_before", False),
        "adjacent_dialogue_after": verification.get("adjacent_dialogue_after", False),
        "discovery_state": "EXACT" if _is_exact(verification) else "REJECTED",
        "confirmation_observed_transcript": (confirmation or {}).get("rendered_transcript"),
        "confirmation_state": "CORROBORATED" if confirmation and _is_exact(confirmation) else "NOT_CORROBORATED",
    }


def _extract_wav(source: Path, start: float, duration: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(source), "-vn", "-ac", "2", "-ar", "48000", str(target),
    ])


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()
