from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..render_verification import RENDER_VERIFICATION_VERSION, evaluate_rendered_dialogue, lexically_equivalent_transcript
from ..util import read_json, stable_hash, utc_now, write_json
from ..whisper_backend import transcribe_with_whisper
from .acoustic_preflight import _build_reel, _digest


Transcriber = Callable[..., dict[str, Any]]
AUDIT_VERSION = "semantic_opportunity_acoustic_audit_v2_adjacent_dialogue_rejection"
CACHE_VERSION = "semantic_opportunity_acoustic_health_cache_v2_adjacent_dialogue_rejection"
CONTEXT_POLICY = "ordered_batches_max3_gap1_reject_adjacent_dialogue_v2"
CONFIRMATION_POLICY = "isolated_confirmation_for_batch_rejections_v1"


def audit_semantic_opportunity_audio(
    *, screen_path: Path, clips: list[dict[str, Any]],
    source_performances: dict[str, Any], output_dir: Path,
    cache_path: Path | None = None, model_name: str = "medium",
    language: str | None = "en", transcription_mode: str = "quality",
    minimum_word_coverage: float = 0.72, max_source_performances: int = 24,
    transcription_batch_size: int = 3,
    force: bool = False, transcriber: Transcriber = transcribe_with_whisper,
) -> dict[str, Any]:
    """Batch-check Pareto opportunity clips before guarded nomination."""
    screen = read_json(screen_path)
    opportunities = [
        row for row in (screen.get("semantic_opportunity_audit") or {}).get("opportunities", [])
        if row.get("globally_admissible")
    ]
    candidates = _opportunity_candidates(
        opportunities, clips, source_performances, max_source_performances=max_source_performances,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path or output_dir / "acoustic_health_cache.json"
    cache = read_json(cache_path) if cache_path.is_file() else {
        "schema_version": "1.0", "cache_version": CACHE_VERSION, "entries": [],
    }
    cached_entries = {
        str(row.get("cache_key")): row for row in cache.get("entries", [])
        if isinstance(row, dict) and row.get("cache_key")
    }
    clip_by_id = {str(row.get("id")): row for row in clips if row.get("id")}
    requested: list[dict[str, Any]] = []
    for candidate in candidates:
        for clip_id in candidate["clip_ids"]:
            clip = clip_by_id.get(clip_id)
            if clip is None:
                requested.append(_missing_clip_entry(candidate["source_performance_id"], clip_id))
                continue
            requested.append(_request_entry(
                candidate["source_performance_id"], clip, model_name=model_name,
                language=language, transcription_mode=transcription_mode,
                minimum_word_coverage=minimum_word_coverage,
            ))
    pending_by_key = {
        row["cache_key"]: row for row in requested
        if row.get("health_state") != "MISSING_CLIP"
        and (force or row["cache_key"] not in cached_entries)
    }
    pending = list(pending_by_key.values())
    transcription_cache_state = "REUSED"
    transcription_batch_count = 0
    confirmation_transcription_count = 0
    if pending:
        transcription_cache_state = "CREATED" if not cached_entries else "PARTIAL_REBUILD"
        batches = [pending[index:index + transcription_batch_size] for index in range(0, len(pending), transcription_batch_size)]
        transcription_batch_count = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            mappings = [_reel_mapping(row, index) for index, row in enumerate(batch, start=1)]
            suffix = f"_{batch_index:03d}"
            reel_path = output_dir / f"opportunity_donor_reel{suffix}.wav"
            verification_schedule, manifest = _build_reel(mappings, reel_path)
            write_json(output_dir / f"opportunity_donor_reel_manifest{suffix}.json", manifest)
            transcription_signature = stable_hash({
                "audit_version": AUDIT_VERSION, "context_policy": CONTEXT_POLICY,
                "model_name": model_name, "language": language,
                "transcription_mode": transcription_mode,
                "reel_sha256": _digest(reel_path),
                "cache_keys": [row["cache_key"] for row in batch],
            })
            transcription_path = output_dir / f"opportunity_donor_transcription{suffix}.json"
            timeline = transcriber(
                audio_path=reel_path, media_hash=transcription_signature,
                output_path=transcription_path, model_name=model_name,
                language=language, artifact_type="timeline",
                transcription_mode=transcription_mode,
            )
            if not transcription_path.is_file():
                write_json(transcription_path, timeline)
            verification = evaluate_rendered_dialogue(
                schedule=verification_schedule, rendered_timeline=timeline,
            )
            verification_by_clip = {
                str(row.get("clip_id")): row for row in verification.get("mappings", [])
            }
            for row in batch:
                checked = verification_by_clip.get(row["clip_id"], {})
                cached_entries[row["cache_key"]] = {
                    **row,
                    "observed_transcript": checked.get("rendered_transcript", ""),
                    "word_coverage_percentage": checked.get("word_coverage_percentage", 0.0),
                    "missing_sentence_beginning": checked.get("missing_sentence_beginning", True),
                    "missing_sentence_ending": checked.get("missing_sentence_ending", True),
                    "verification_status": checked.get("status", "fail"),
                    "health_state": "ACCEPTED" if _accepted(checked, minimum_word_coverage) else "REJECTED",
                    "batch_health_state": "ACCEPTED" if _accepted(checked, minimum_word_coverage) else "REJECTED",
                    "checked_timestamp": utc_now(),
                }
    confirmation_rows = [
        cached_entries[row["cache_key"]] for row in requested
        if row.get("health_state") != "MISSING_CLIP"
        and cached_entries[row["cache_key"]].get("health_state") == "REJECTED"
        and cached_entries[row["cache_key"]].get("confirmation_policy") != CONFIRMATION_POLICY
    ]
    if confirmation_rows:
        confirmation_transcription_count = len(confirmation_rows)
        transcription_cache_state = "PARTIAL_REBUILD" if cached_entries else "CREATED"
        for confirmation_index, row in enumerate(confirmation_rows, start=1):
            reel_path = output_dir / f"opportunity_confirmation_reel_{confirmation_index:03d}.wav"
            verification_schedule, manifest = _build_reel([_reel_mapping(row, 1)], reel_path)
            write_json(output_dir / f"opportunity_confirmation_manifest_{confirmation_index:03d}.json", manifest)
            transcription_signature = stable_hash({
                "audit_version": AUDIT_VERSION, "confirmation_policy": CONFIRMATION_POLICY,
                "model_name": model_name, "language": language,
                "transcription_mode": transcription_mode,
                "reel_sha256": _digest(reel_path), "cache_key": row["cache_key"],
            })
            transcription_path = output_dir / f"opportunity_confirmation_transcription_{confirmation_index:03d}.json"
            timeline = transcriber(
                audio_path=reel_path, media_hash=transcription_signature,
                output_path=transcription_path, model_name=model_name,
                language=language, artifact_type="timeline",
                transcription_mode=transcription_mode,
            )
            if not transcription_path.is_file():
                write_json(transcription_path, timeline)
            verification = evaluate_rendered_dialogue(
                schedule=verification_schedule, rendered_timeline=timeline,
            )
            checked = (verification.get("mappings") or [{}])[0]
            cached_entries[row["cache_key"]] = {
                **row,
                "observed_transcript": checked.get("rendered_transcript", ""),
                "word_coverage_percentage": checked.get("word_coverage_percentage", 0.0),
                "missing_sentence_beginning": checked.get("missing_sentence_beginning", True),
                "missing_sentence_ending": checked.get("missing_sentence_ending", True),
                "verification_status": checked.get("status", "fail"),
                "health_state": "ACCEPTED" if _accepted(checked, minimum_word_coverage) else "REJECTED",
                "confirmation_policy": CONFIRMATION_POLICY,
                "confirmation_state": "ACCEPTED" if _accepted(checked, minimum_word_coverage) else "REJECTED",
                "checked_timestamp": utc_now(),
            }
    resolved = []
    for row in requested:
        if row.get("health_state") == "MISSING_CLIP":
            resolved.append(row)
        else:
            resolved.append({
                **cached_entries[row["cache_key"]],
                "source_performance_id": row["source_performance_id"],
            })
    performance_rows = _performance_results(candidates, resolved)
    rejected_sources = sorted(
        row["source_performance_id"] for row in performance_rows
        if row["health_state"] != "ACCEPTED"
    )
    cache_report = {
        "schema_version": "1.0", "cache_version": CACHE_VERSION,
        "updated_timestamp": utc_now(),
        "entries": sorted(cached_entries.values(), key=lambda row: str(row.get("cache_key"))),
    }
    write_json(cache_path, cache_report)
    report = {
        "schema_version": "1.0", "audit_version": AUDIT_VERSION,
        "creation_timestamp": utc_now(), "screen_signature": screen.get("experiment_signature"),
        "audit_signature": stable_hash({
            "version": AUDIT_VERSION, "screen": screen.get("experiment_signature"),
            "model_name": model_name, "language": language,
            "transcription_mode": transcription_mode,
            "minimum_word_coverage": minimum_word_coverage,
            "max_source_performances": max_source_performances,
            "context_policy": CONTEXT_POLICY,
            "cache_keys": sorted(row["cache_key"] for row in requested if row.get("cache_key")),
        }),
        "whisper_configuration": {
            "requested_model": model_name, "language": language,
            "transcription_mode": transcription_mode,
        },
        "verifier_version": RENDER_VERIFICATION_VERSION,
        "minimum_word_coverage": round(float(minimum_word_coverage), 4),
        "maximum_source_performances": max_source_performances,
        "audited_source_performance_count": len(performance_rows),
        "audited_clip_count": len(resolved),
        "accepted_source_performance_count": len(performance_rows) - len(rejected_sources),
        "rejected_source_performance_count": len(rejected_sources),
        "rejected_source_performance_ids": rejected_sources,
        "repair_lineage": screen.get("acoustic_repair") or {},
        "transcription_cache_state": transcription_cache_state,
        "transcription_batch_count": transcription_batch_count,
        "confirmation_transcription_count": confirmation_transcription_count,
        "context_policy": CONTEXT_POLICY,
        "confirmation_policy": CONFIRMATION_POLICY,
        "cache_path": str(cache_path),
        "source_performances": performance_rows,
        "clips": resolved,
        "audit_state": "REJECTIONS_AVAILABLE" if rejected_sources else "ALL_AUDITED_CANDIDATES_ACCEPTED",
        "claim_scope": "Acoustic transcript integrity of bounded Pareto opportunity clips only; not semantic relatedness, performance fit, or in-context render quality.",
    }
    write_json(output_dir / "semantic_opportunity_acoustic_audit.json", report)
    return report


def _opportunity_candidates(
    opportunities: list[dict[str, Any]], clips: list[dict[str, Any]],
    source_performances: dict[str, Any], *, max_source_performances: int,
) -> list[dict[str, Any]]:
    performance_clip_ids = _performance_clip_ids(clips, source_performances)
    candidates: dict[str, dict[str, Any]] = {}
    for opportunity in sorted(
        opportunities,
        key=lambda row: (-float(row.get("semantic_delta", 0.0) or 0.0), str(row.get("source_performance_id") or "")),
    ):
        source_id = str(opportunity.get("source_performance_id") or "")
        if source_id and source_id not in candidates and len(candidates) < max_source_performances:
            candidates[source_id] = {
                "source_performance_id": source_id,
                "clip_ids": sorted(set(str(value) for value in opportunity.get("clip_ids", []) if value)),
                "semantic_delta": float(opportunity.get("semantic_delta", 0.0) or 0.0),
                "admission_role": "PRIMARY",
            }
        swap = opportunity.get("two_cycle_swap") or {}
        if opportunity.get("global_admission_mode") == "TWO_CYCLE" and swap.get("state") == "ADMISSIBLE_TWO_CYCLE":
            partner = str(swap.get("replacement_source_performance_id") or "")
            if partner and partner not in candidates and len(candidates) < max_source_performances:
                candidates[partner] = {
                    "source_performance_id": partner,
                    "clip_ids": performance_clip_ids.get(partner, []),
                    "semantic_delta": float(swap.get("net_semantic_delta", 0.0) or 0.0),
                    "admission_role": "TWO_CYCLE_PARTNER",
                }
    for row in candidates.values():
        if not row["clip_ids"]:
            row["clip_ids"] = performance_clip_ids.get(row["source_performance_id"], [])
    return list(candidates.values())


def _performance_clip_ids(
    clips: list[dict[str, Any]], source_performances: dict[str, Any],
) -> dict[str, list[str]]:
    by_event: dict[str, list[str]] = {}
    for clip in clips:
        for event_id in clip.get("event_ids") or [clip.get("event_id")]:
            if event_id:
                by_event.setdefault(str(event_id), []).append(str(clip.get("id")))
    result = {}
    for performance in source_performances.get("performances", []):
        clip_ids = {
            clip_id for event_id in performance.get("dialogue_event_ids", [])
            for clip_id in by_event.get(str(event_id), [])
        }
        result[str(performance.get("id"))] = sorted(clip_ids)
    return result


def _request_entry(
    source_performance_id: str, clip: dict[str, Any], *, model_name: str,
    language: str | None, transcription_mode: str, minimum_word_coverage: float,
) -> dict[str, Any]:
    path = Path(str(clip.get("path") or ""))
    if not path.is_file():
        return _missing_clip_entry(source_performance_id, str(clip.get("id") or ""))
    transcript = str(clip.get("transcript") or "").strip()
    duration = float(clip.get("duration", 0.0) or 0.0)
    identity = {
        "cache_version": CACHE_VERSION, "clip_sha256": _digest(path),
        "clip_id": str(clip.get("id")), "trim_start": 0.0,
        "trim_duration": round(duration, 3), "intended_transcript": transcript,
        "model_name": model_name, "language": language,
        "transcription_mode": transcription_mode,
        "minimum_word_coverage": minimum_word_coverage,
        "verifier_version": RENDER_VERIFICATION_VERSION,
        "context_policy": CONTEXT_POLICY,
    }
    return {
        "cache_key": stable_hash(identity), "source_performance_id": source_performance_id,
        "clip_id": str(clip.get("id")), "clip_path": str(path),
        "trim_start": 0.0, "trim_duration": round(duration, 3),
        "intended_transcript": transcript, "identity": identity,
    }


def _missing_clip_entry(source_performance_id: str, clip_id: str) -> dict[str, Any]:
    return {
        "cache_key": stable_hash({"missing_clip": clip_id, "source_performance_id": source_performance_id}),
        "source_performance_id": source_performance_id, "clip_id": clip_id,
        "clip_path": None, "trim_start": 0.0, "trim_duration": 0.0,
        "intended_transcript": "", "health_state": "MISSING_CLIP",
        "verification_status": "UNAVAILABLE",
    }


def _reel_mapping(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"opportunity_{index:03d}", "window_id": row["source_performance_id"],
        "source_performance_id": row["source_performance_id"],
        "clip_id": row["clip_id"], "clip_path": row["clip_path"],
        "clip_trim_start": row["trim_start"], "clip_trim_duration": row["trim_duration"],
        "source_transcript": row["intended_transcript"], "enabled": True,
    }


def _accepted(row: dict[str, Any], minimum_word_coverage: float) -> bool:
    lexical_match = lexically_equivalent_transcript(
        row.get("intended_transcript"), row.get("rendered_transcript"),
    )
    return (
        (lexical_match or (
            float(row.get("word_coverage_percentage", 0.0) or 0.0) / 100.0 >= minimum_word_coverage
            and not row.get("missing_sentence_beginning")
            and not row.get("missing_sentence_ending")
        ))
        and bool(str(row.get("rendered_transcript") or "").strip())
        and not row.get("adjacent_dialogue_before")
        and not row.get("adjacent_dialogue_after")
    )


def _performance_results(
    candidates: list[dict[str, Any]], entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_source.setdefault(str(entry.get("source_performance_id")), []).append(entry)
    rows = []
    for candidate in candidates:
        source_id = candidate["source_performance_id"]
        clips = by_source.get(source_id, [])
        accepted = bool(clips) and all(row.get("health_state") == "ACCEPTED" for row in clips)
        rows.append({
            **candidate,
            "audited_clip_count": len(clips),
            "accepted_clip_count": sum(row.get("health_state") == "ACCEPTED" for row in clips),
            "health_state": "ACCEPTED" if accepted else "REJECTED",
        })
    return rows
