from __future__ import annotations

import argparse
from pathlib import Path

from cinelingus.dialogue_function.scheduling import FunctionMode, FunctionScheduleContext
from cinelingus.editorial.repair_engine import _preserve_context
from cinelingus.schedule import build_editorial_repair_mapping, prepare_editorial_repair_candidates, score_editorial_repair_candidate
from cinelingus.semantic.config import SemanticMode
from cinelingus.semantic.scheduling import SemanticScheduleContext
from cinelingus.util import read_json, stable_hash, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--source-performances", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--destination-model", type=Path, required=True)
    parser.add_argument("--source-semantic", type=Path, required=True)
    parser.add_argument("--destination-semantic", type=Path, required=True)
    parser.add_argument("--source-function", type=Path, required=True)
    parser.add_argument("--destination-function", type=Path, required=True)
    parser.add_argument("--mapping-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = read_json(args.audit)
    confirmation = audit.get("isolated_confirmation") or {}
    if confirmation.get("confirmation_state") != "CONFIRMED":
        raise ValueError("Function donor audit has no isolated confirmed candidate")
    source_model, destination_model = read_json(args.source_model), read_json(args.destination_model)
    function_context = FunctionScheduleContext.from_bundles(
        mode=FunctionMode.PRESERVING, weight=0.15,
        source_model=source_model, source_bundle=read_json(args.source_function),
        destination_model=destination_model, destination_bundle=read_json(args.destination_function),
    )
    semantic_context = SemanticScheduleContext.from_bundles(
        mode=SemanticMode.ASSISTED, weight=0.05,
        source_model=source_model, source_bundle=read_json(args.source_semantic), source_dir=args.source_semantic.parent,
        destination_model=destination_model, destination_bundle=read_json(args.destination_semantic), destination_dir=args.destination_semantic.parent,
    )
    clip_artifact = read_json(args.clips)
    candidates = prepare_editorial_repair_candidates(clip_artifact.get("clips") or [], read_json(args.source_performances))
    candidates = semantic_context.annotate_clips(function_context.annotate_clips(candidates))
    donor = next(row for row in candidates if str(row.get("id")) == str(confirmation["clip_id"]))
    window_artifact = read_json(args.windows)
    windows = semantic_context.annotate_windows(function_context.annotate_windows(window_artifact.get("windows") or []))
    schedule = read_json(args.screen / "function_preserving_schedule.json")
    current = schedule["mappings"][args.mapping_index]
    window = next(row for row in windows if str(row.get("id")) == str(current.get("window_id")))
    window["active_filter"] = schedule.get("active_filter", "balanced")
    score = score_editorial_repair_candidate(window, donor, max_time_stretch=0.1, shot_boundary_mode="soft")
    replacement = build_editorial_repair_mapping(
        window, donor, score, max_time_stretch=0.1, shot_boundary_mode="soft", cinematic_filter="balanced",
    )
    replacement = _preserve_context(current, replacement)
    replacement["source_transcript"] = confirmation["observed_transcript"]
    replacement["function_repair"] = {
        "repair_version": "function_mismatch_repair_v1", "repair_role": "ALTERNATIVE_LEGAL_DONOR",
        "rejected_clip_id": current.get("clip_id"), "confirmed_clip_id": donor.get("id"),
        "donor_audit_signature": audit.get("audit_signature"),
        "acoustic_confirmation_state": confirmation.get("confirmation_state"),
        "function_score": confirmation.get("function_score"),
        "acceptance_state": "PENDING_RENDER_VERIFICATION",
    }
    repaired = {**schedule, "mappings": [dict(row) for row in schedule.get("mappings") or []]}
    repaired["mappings"][args.mapping_index] = replacement
    repaired["function_repair"] = {
        "repair_version": "function_mismatch_repair_v1", "creation_timestamp": utc_now(),
        "mapping_index": args.mapping_index, "old_clip_id": current.get("clip_id"), "new_clip_id": donor.get("id"),
        "donor_audit_signature": audit.get("audit_signature"), "acceptance_state": "PENDING_RENDER_VERIFICATION",
    }
    repaired["function_repair"]["repair_signature"] = stable_hash(repaired["function_repair"])
    write_json(args.output, repaired)
    print(f"Confirmed function repair schedule: {current.get('clip_id')} -> {donor.get('id')}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
