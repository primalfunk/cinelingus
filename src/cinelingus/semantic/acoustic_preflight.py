from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Any, Callable

from ..render_verification import evaluate_rendered_dialogue, lexically_equivalent_transcript
from ..util import read_json, stable_hash, utc_now, write_json
from ..whisper_backend import transcribe_with_whisper


Transcriber = Callable[..., dict[str, Any]]


def run_semantic_acoustic_preflight(
    *, screen_dir: Path, output_dir: Path, semantic_variant: str,
    model_name: str = "medium", language: str | None = "en",
    transcription_mode: str = "quality", minimum_word_coverage: float = 0.72,
    force: bool = False, transcriber: Transcriber = transcribe_with_whisper,
) -> dict[str, Any]:
    """Retranscribe only donor clips introduced by a semantic schedule change."""
    screen = read_json(screen_dir / "semantic_schedule_screen.json")
    if semantic_variant not in set(screen.get("render_selection") or []):
        raise ValueError("Semantic variant was not nominated by the schedule screen")
    control = read_json(screen_dir / "control_schedule.json")
    assisted = read_json(screen_dir / f"{semantic_variant}_schedule.json")
    changed = _changed_mappings(control.get("mappings") or [], assisted.get("mappings") or [])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "semantic_acoustic_preflight.json"
    if not changed:
        report = _empty_report(screen, semantic_variant, minimum_word_coverage)
        write_json(output_path, report)
        return report

    reel_path = output_dir / "changed_donor_reel.wav"
    reel_manifest_path = output_dir / "changed_donor_reel_manifest.json"
    verification_schedule, reel_manifest = _build_reel(changed, reel_path)
    write_json(reel_manifest_path, reel_manifest)
    signature = stable_hash({
        "version": "semantic_acoustic_preflight_v1",
        "screen_signature": screen["experiment_signature"],
        "semantic_variant": semantic_variant,
        "model_name": model_name, "language": language,
        "transcription_mode": transcription_mode,
        "minimum_word_coverage": minimum_word_coverage,
        "reel_sha256": _digest(reel_path),
        "mappings": verification_schedule["mappings"],
    })
    transcription_path = output_dir / "changed_donor_transcription.json"
    cached = read_json(transcription_path) if transcription_path.exists() else {}
    if not force and cached.get("media_hash") == signature:
        timeline = cached
        transcription_cache_state = "REUSED"
    else:
        timeline = transcriber(
            audio_path=reel_path, media_hash=signature, output_path=transcription_path,
            model_name=model_name, language=language, artifact_type="timeline",
            transcription_mode=transcription_mode,
        )
        if not transcription_path.exists():
            write_json(transcription_path, timeline)
        transcription_cache_state = "CREATED"
    verification = evaluate_rendered_dialogue(schedule=verification_schedule, rendered_timeline=timeline)
    changed_by_key = {
        (str(row.get("clip_id") or ""), str(row.get("destination_performance_id") or row.get("window_id") or "")): row
        for row in changed
    }
    decisions = [
        _mapping_decision(
            row, minimum_word_coverage,
            source_mapping=changed_by_key.get((str(row.get("clip_id") or ""), str(row.get("window_id") or ""))),
        )
        for row in verification["mappings"]
    ]
    rejected = [row for row in decisions if row["state"] == "REJECTED"]
    report = {
        "schema_version": "1.0",
        "preflight_version": "semantic_acoustic_preflight_v1",
        "creation_timestamp": utc_now(),
        "screen_signature": screen["experiment_signature"],
        "semantic_variant": semantic_variant,
        "preflight_signature": signature,
        "whisper_configuration": {
            "requested_model": model_name, "language": language,
            "transcription_mode": transcription_mode,
        },
        "minimum_word_coverage": round(float(minimum_word_coverage), 4),
        "changed_mapping_count": len(changed),
        "reel": reel_path.name,
        "reel_manifest": reel_manifest_path.name,
        "transcription": transcription_path.name,
        "transcription_cache_state": transcription_cache_state,
        "verification": verification,
        "mapping_decisions": decisions,
        "repair_lineage": screen.get("acoustic_repair") or {},
        "rejected_source_performance_ids": sorted({
            str(row["source_performance_id"])
            for row in rejected if row.get("source_performance_id")
        }),
        "preflight_state": "REJECTED_ACOUSTIC_INTEGRITY" if rejected else "ACCEPTED_FOR_RENDER",
        "rejection_count": len(rejected),
        "claim_scope": "Selected-donor acoustic transcript integrity only; this does not establish render quality, semantic preference, voice separation, or visual fit.",
    }
    write_json(output_path, report)
    return report


def require_accepted_semantic_preflight(
    path: Path, *, screen_signature: str, semantic_variant: str,
) -> dict[str, Any]:
    report = read_json(path)
    reasons = []
    if report.get("preflight_state") != "ACCEPTED_FOR_RENDER":
        reasons.append("preflight_not_accepted")
    if report.get("screen_signature") != screen_signature:
        reasons.append("screen_signature_mismatch")
    if report.get("semantic_variant") != semantic_variant:
        reasons.append("semantic_variant_mismatch")
    if reasons:
        raise ValueError(f"Semantic acoustic preflight does not admit render: {', '.join(reasons)}")
    return report


def _changed_mappings(control: list[dict[str, Any]], assisted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control_by_placement = {_placement_key(row, index): row for index, row in enumerate(control)}
    changed = []
    for index, row in enumerate(assisted):
        before = control_by_placement.get(_placement_key(row, index))
        if before is None or _donor_key(before) != _donor_key(row):
            changed.append(row)
    return changed


def _placement_key(row: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        row.get("editorial_placement_id") or row.get("window_id") or index,
        round(float(row.get("destination_timestamp", row.get("start", 0.0)) or 0.0), 3),
    )


def _donor_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return row.get("clip_id"), row.get("source_performance_id")


def _build_reel(changed: list[dict[str, Any]], output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    gap_seconds = 1.0
    params: tuple[int, int, int] | None = None
    chunks: list[bytes] = []
    reel_cursor = gap_seconds
    synthetic = []
    segments = []
    for index, mapping in enumerate(changed, start=1):
        clip_path = Path(str(mapping.get("clip_path") or ""))
        if not clip_path.is_file():
            raise FileNotFoundError(f"Changed semantic donor clip is unavailable: {clip_path}")
        with wave.open(str(clip_path), "rb") as source:
            current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            if params is None:
                params = current
                chunks.append(b"\0" * int(gap_seconds * current[2]) * current[0] * current[1])
            elif current != params:
                raise ValueError("Changed donor clips have incompatible WAV formats")
            trim_start = max(0.0, float(mapping.get("clip_trim_start", 0.0) or 0.0))
            available = max(0.0, source.getnframes() / current[2] - trim_start)
            trim_duration = min(available, float(mapping.get("clip_trim_duration", available) or available))
            source.setpos(min(source.getnframes(), int(round(trim_start * current[2]))))
            frame_count = min(source.getnframes() - source.tell(), int(round(trim_duration * current[2])))
            payload = source.readframes(max(0, frame_count))
            actual_duration = frame_count / current[2]
        start, end = reel_cursor, reel_cursor + actual_duration
        chunks.append(payload)
        chunks.append(b"\0" * int(gap_seconds * params[2]) * params[0] * params[1])
        synthetic.append({
            "id": f"preflight_{index:03d}", "mapping_id": mapping.get("id"),
            "window_id": mapping.get("window_id"), "clip_id": mapping.get("clip_id"),
            "source_transcript": mapping.get("source_transcript", ""),
            "destination_timestamp": round(start, 3),
            "planned_render_duration": round(actual_duration, 3), "enabled": True,
        })
        segments.append({
            "preflight_mapping_id": f"preflight_{index:03d}", "clip_id": mapping.get("clip_id"),
            "clip_path": str(clip_path), "reel_start": round(start, 3), "reel_end": round(end, 3),
            "trim_start": round(trim_start, 3), "trim_duration": round(actual_duration, 3),
            "intended_transcript": mapping.get("source_transcript", ""),
        })
        reel_cursor = end + gap_seconds
    assert params is not None
    with wave.open(str(output_path), "wb") as target:
        target.setnchannels(params[0]); target.setsampwidth(params[1]); target.setframerate(params[2])
        target.writeframes(b"".join(chunks))
    return {"mappings": synthetic}, {
        "schema_version": "1.0", "strategy": "changed_semantic_donor_reel_v1",
        "gap_seconds": gap_seconds, "duration": round(reel_cursor, 3), "segments": segments,
    }


def _mapping_decision(
    row: dict[str, Any], minimum_word_coverage: float,
    source_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = float(row.get("word_coverage_percentage", 0.0) or 0.0) / 100.0
    lexical_match = lexically_equivalent_transcript(
        row.get("intended_transcript"), row.get("rendered_transcript"),
    )
    reasons = []
    if coverage < minimum_word_coverage and not lexical_match:
        reasons.append("word_coverage_below_threshold")
    if row.get("missing_sentence_beginning") and not lexical_match:
        reasons.append("sentence_beginning_missing")
    if row.get("missing_sentence_ending") and not lexical_match:
        reasons.append("sentence_ending_missing")
    if not str(row.get("rendered_transcript") or "").strip():
        reasons.append("no_speech_observed")
    if row.get("adjacent_dialogue_before"):
        reasons.append("adjacent_dialogue_before")
    if row.get("adjacent_dialogue_after"):
        reasons.append("adjacent_dialogue_after")
    source_mapping = source_mapping or {}
    return {
        "clip_id": row.get("clip_id"), "window_id": row.get("window_id"),
        "source_performance_id": source_mapping.get("source_performance_id"),
        "destination_performance_id": source_mapping.get("destination_performance_id") or row.get("window_id"),
        "word_coverage_percentage": row.get("word_coverage_percentage"),
        "missing_sentence_beginning": row.get("missing_sentence_beginning"),
        "missing_sentence_ending": row.get("missing_sentence_ending"),
        "state": "REJECTED" if reasons else "ACCEPTED", "reasons": reasons,
    }


def _empty_report(screen: dict[str, Any], semantic_variant: str, minimum_word_coverage: float) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "preflight_version": "semantic_acoustic_preflight_v1",
        "creation_timestamp": utc_now(), "screen_signature": screen["experiment_signature"],
        "semantic_variant": semantic_variant,
        "preflight_signature": stable_hash({"screen": screen["experiment_signature"], "variant": semantic_variant, "changed": []}),
        "whisper_configuration": {}, "minimum_word_coverage": minimum_word_coverage,
        "changed_mapping_count": 0, "reel": None, "reel_manifest": None, "transcription": None,
        "transcription_cache_state": "NOT_APPLICABLE", "verification": {}, "mapping_decisions": [],
        "rejected_source_performance_ids": [],
        "repair_lineage": screen.get("acoustic_repair") or {},
        "preflight_state": "NO_CHANGED_MAPPINGS", "rejection_count": 0,
        "claim_scope": "No changed semantic donor mappings were available for acoustic preflight.",
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
