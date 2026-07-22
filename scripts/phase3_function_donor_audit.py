from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cinelingus.dialogue_function.classifier import RuleDialogueFunctionClassifier
from cinelingus.dialogue_function.render_verification import _compare
from cinelingus.render_verification import evaluate_rendered_dialogue
from cinelingus.semantic.acoustic_preflight import _build_reel, _digest
from cinelingus.util import read_json, stable_hash, utc_now, write_json
from cinelingus.whisper_backend import transcribe_with_whisper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--mapping-index", type=int, required=True)
    parser.add_argument("--clip-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clip_artifact = read_json(args.clips)
    clips = {str(row.get("id")): row for row in clip_artifact.get("clips") or []}
    requested = [value.strip() for value in args.clip_ids.split(",") if value.strip()]
    missing = [value for value in requested if value not in clips]
    if missing:
        raise ValueError(f"Missing clips: {', '.join(missing)}")
    mapping = (read_json(args.schedule).get("mappings") or [])[args.mapping_index]
    destination = (mapping.get("dialogue_function_compatibility") or {}).get("destination_distribution")
    if not destination:
        raise ValueError("Selected schedule mapping has no destination function distribution")
    classifier = RuleDialogueFunctionClassifier()
    rows: list[dict[str, Any]] = []
    for batch_index in range(0, len(requested), 3):
        batch = requested[batch_index:batch_index + 3]
        checked = _transcribe_batch(batch, clips, args.output, batch_index // 3 + 1, args.model, args.language)
        for clip_id in batch:
            clip = clips[clip_id]
            observed = checked.get(clip_id) or {}
            function = classifier.classify(str(observed.get("rendered_transcript") or ""))
            comparison = _compare(function, destination, classifier.config.confidence_threshold)
            score = float((comparison or {}).get("normalized_function_contribution", 0.0) or 0.0)
            confidence = float(observed.get("confidence", 0.0) or 0.0)
            coverage = float(observed.get("word_coverage_percentage", 0.0) or 0.0)
            acoustic_reliable = confidence >= 0.45 or (confidence >= 0.2 and coverage >= 72.0 and observed.get("status") != "fail")
            eligible = bool(
                observed.get("rendered_transcript") and acoustic_reliable
                and not function.get("abstention", {}).get("abstained")
                and score >= 0.5 and not observed.get("adjacent_dialogue_before")
                and not observed.get("adjacent_dialogue_after")
            )
            rows.append({
                "clip_id": clip_id, "clip_path": clip.get("path"), "metadata_transcript": clip.get("transcript"),
                "observed_transcript": observed.get("rendered_transcript", ""),
                "transcript_confidence": confidence, "metadata_word_coverage_percentage": observed.get("word_coverage_percentage"),
                "observed_function": function, "destination_function_compatibility": comparison,
                "function_score": score, "eligible_for_isolated_confirmation": eligible,
            })
    eligible = sorted((row for row in rows if row["eligible_for_isolated_confirmation"]), key=lambda row: (-row["function_score"], -float(row.get("metadata_word_coverage_percentage") or 0.0), -row["transcript_confidence"], row["clip_id"]))
    confirmation = None
    if eligible:
        candidate = eligible[0]
        checked = _transcribe_batch([candidate["clip_id"]], clips, args.output, 999, args.model, args.language, prefix="confirmation")
        observed = checked.get(candidate["clip_id"]) or {}
        function = classifier.classify(str(observed.get("rendered_transcript") or ""))
        comparison = _compare(function, destination, classifier.config.confidence_threshold)
        score = float((comparison or {}).get("normalized_function_contribution", 0.0) or 0.0)
        confidence = float(observed.get("confidence", 0.0) or 0.0)
        coverage = float(observed.get("word_coverage_percentage", 0.0) or 0.0)
        acoustic_reliable = confidence >= 0.45 or (confidence >= 0.2 and coverage >= 72.0 and observed.get("status") != "fail")
        confirmed = bool(observed.get("rendered_transcript") and acoustic_reliable and not function.get("abstention", {}).get("abstained") and score >= 0.5)
        confirmation = {
            "clip_id": candidate["clip_id"], "observed_transcript": observed.get("rendered_transcript", ""),
            "transcript_confidence": confidence, "metadata_word_coverage_percentage": coverage, "observed_function": function,
            "destination_function_compatibility": comparison, "function_score": score,
            "confirmation_state": "CONFIRMED" if confirmed else "REJECTED",
        }
    report = {
        "schema_version": "1.0", "audit_version": "phase3_function_donor_audit_v1",
        "creation_timestamp": utc_now(), "candidate_count": len(rows), "candidates": rows,
        "isolated_confirmation": confirmation,
        "audit_state": "CONFIRMED_DONOR_AVAILABLE" if confirmation and confirmation["confirmation_state"] == "CONFIRMED" else "NO_CONFIRMED_DONOR",
        "claim_scope": "Acoustic transcript and dialogue-function evidence for a bounded legal donor shortlist; not preference or narrative interpretation.",
    }
    report["audit_signature"] = stable_hash({key: value for key, value in report.items() if key not in {"creation_timestamp", "audit_signature"}})
    write_json(args.output / "function_donor_audit.json", report)
    print(f"Function donor audit: {report['audit_state']}")
    if confirmation:
        print(f"Confirmed candidate: {confirmation['clip_id']} — {confirmation['observed_transcript']}")
    return 0 if report["audit_state"] == "CONFIRMED_DONOR_AVAILABLE" else 2


def _transcribe_batch(
    ids: list[str], clips: dict[str, dict[str, Any]], output: Path, index: int,
    model: str, language: str, *, prefix: str = "batch",
) -> dict[str, dict[str, Any]]:
    mappings = [{
        "window_id": clip_id, "clip_id": clip_id, "clip_path": clips[clip_id]["path"],
        "source_transcript": clips[clip_id].get("transcript", ""), "clip_trim_start": 0.0,
        "clip_trim_duration": clips[clip_id].get("duration"),
    } for clip_id in ids]
    reel = output / f"{prefix}_{index:03d}.wav"
    schedule, manifest = _build_reel(mappings, reel)
    write_json(output / f"{prefix}_{index:03d}_manifest.json", manifest)
    signature = stable_hash({"version": "phase3_function_donor_audit_v1", "ids": ids, "reel": _digest(reel), "model": model, "language": language})
    timeline = transcribe_with_whisper(
        audio_path=reel, media_hash=signature, output_path=output / f"{prefix}_{index:03d}_transcription.json",
        model_name=model, language=language, artifact_type="timeline", transcription_mode="quality",
    )
    verification = evaluate_rendered_dialogue(schedule=schedule, rendered_timeline=timeline)
    return {str(row.get("clip_id")): row for row in verification.get("mappings") or []}


if __name__ == "__main__":
    raise SystemExit(main())
