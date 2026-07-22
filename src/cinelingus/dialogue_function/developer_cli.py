from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..util import read_json, write_json
from ..validation import validate_artifact
from .bundle import build_function_bundle, validate_function_bundle
from .classifier import FunctionClassifierConfig, RuleDialogueFunctionClassifier
from .reports import render_function_bundle_report, write_function_bundle_report
from .taxonomy import TAXONOMY_PATH, load_taxonomy
from .calibration import prepare_calibration_review, finalize_calibration_review
from .experiment import run_function_schedule_screen
from .render_verification import evaluate_rendered_function
from .acoustic_preflight import run_function_acoustic_preflight
from .render_proof import run_function_render_proof
from ..config import load_config
from .scheduling import FunctionMode, FunctionScheduleContext
from ..semantic.config import SemanticMode
from ..semantic.scheduling import SemanticScheduleContext
from ..performance import attach_performance_speech_windows, performance_windows

FUNCTION_COMMANDS = frozenset({"validate-function-taxonomy", "build-function-bundle", "validate-function-bundle", "report-function-bundle", "prepare-function-calibration", "finalize-function-calibration", "screen-function-schedules", "verify-rendered-function", "preflight-function-schedule", "render-function-proof"})


def add_function_parsers(subparsers: argparse._SubParsersAction) -> None:
    taxonomy = subparsers.add_parser("validate-function-taxonomy", help="Validate the Phase 3 dialogue-function taxonomy contract.")
    taxonomy.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)

    build = subparsers.add_parser("build-function-bundle", help="Build a separately cached SpeechPassage dialogue-function bundle.")
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--context-mode", choices=["passage_alone", "adjacent_passages", "dialogue_turn"], default="passage_alone")
    build.add_argument("--confidence-threshold", type=float, default=0.62)
    build.add_argument("--no-resume", action="store_true")

    validate = subparsers.add_parser("validate-function-bundle", help="Validate function coverage, identity, and classification structure.")
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--model", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    report = subparsers.add_parser("report-function-bundle", help="Render the disclosed dialogue-function capability report.")
    report.add_argument("bundle", type=Path)
    report.add_argument("--output", type=Path)

    calibration = subparsers.add_parser("prepare-function-calibration", help="Prepare a provenance-preserving human annotation package from real function bundles.")
    calibration.add_argument("--manifest", type=Path, required=True, help="JSON object with model_path, function_bundle_path, case_id, and media_class cases.")
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--maximum-samples", type=int, default=36)

    finalize = subparsers.add_parser("finalize-function-calibration", help="Validate human function annotations and compute calibration statistics.")
    finalize.add_argument("--package", type=Path, required=True)
    finalize.add_argument("--annotations", type=Path)
    finalize.add_argument("--output", type=Path)

    screen = subparsers.add_parser("screen-function-schedules", help="Run legacy/semantic/report-only/function-preserving schedule comparison.")
    screen.add_argument("--clips", type=Path, required=True)
    screen.add_argument("--windows", type=Path, required=True)
    screen.add_argument("--speech-windows", type=Path)
    screen.add_argument("--source-model", type=Path, required=True)
    screen.add_argument("--source-semantic-bundle", type=Path, required=True)
    screen.add_argument("--source-function-bundle", type=Path, required=True)
    screen.add_argument("--destination-model", type=Path, required=True)
    screen.add_argument("--destination-semantic-bundle", type=Path, required=True)
    screen.add_argument("--destination-function-bundle", type=Path, required=True)
    screen.add_argument("--calibration", type=Path)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--semantic-weight", type=float, default=0.05)
    screen.add_argument("--function-weight", type=float, default=0.15)
    screen.add_argument("--scheduling-mode", choices=["strict_order", "best_fit", "window_fill", "whole_line_fill", "performance_fill"], default="best_fit")
    screen.add_argument("--best-fit-lookahead", type=int, default=8)
    screen.add_argument("--max-time-stretch", type=float, default=0.1)
    screen.add_argument("--shot-boundary-mode", choices=["off", "soft", "strict"], default="off")
    screen.add_argument("--cinematic-filter", default="balanced")
    screen.add_argument("--source-performances", type=Path)
    screen.add_argument("--speaker-mapping", type=Path)

    verify = subparsers.add_parser("verify-rendered-function", help="Reclassify actual rendered transcripts and compare donor/destination function evidence.")
    verify.add_argument("--schedule", type=Path, required=True)
    verify.add_argument("--rendered-verification", type=Path, required=True)
    verify.add_argument("--baseline-schedule", type=Path)
    verify.add_argument("--calibration", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--minimum-transcript-confidence", type=float, default=0.45)
    verify.add_argument("--minimum-function-compatibility", type=float, default=0.5)

    preflight = subparsers.add_parser("preflight-function-schedule", help="Acoustically verify changed function-preserving donors before rendering.")
    preflight.add_argument("--screen", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--baseline-variant", default="function_report_only")
    preflight.add_argument("--function-variant", default="function_preserving")
    preflight.add_argument("--model", default="medium")
    preflight.add_argument("--language", default="en")
    preflight.add_argument("--minimum-word-coverage", type=float, default=0.72)
    preflight.add_argument("--force", action="store_true")

    proof = subparsers.add_parser("render-function-proof", help="Render and verify the provisional four-way Phase 3 proof.")
    proof.add_argument("--screen", type=Path, required=True)
    proof.add_argument("--destination-video", type=Path, required=True)
    proof.add_argument("--source-dialogue", type=Path, required=True)
    proof.add_argument("--preflight", type=Path, required=True)
    proof.add_argument("--rejected-preflight", type=Path)
    proof.add_argument("--donor-audit", type=Path)
    proof.add_argument("--output", type=Path, required=True)
    proof.add_argument("--function-variant", default="function_preserving_repaired")
    proof.add_argument("--defer-human-calibration", action="store_true")
    proof.add_argument("--force", action="store_true")


def run_function_command(args: argparse.Namespace, root: Path) -> int:
    if args.command == "validate-function-taxonomy":
        taxonomy = load_taxonomy(args.taxonomy)
        validate_artifact("dialogue_function_taxonomy", args.taxonomy, root / "schemas")
        print(f"Dialogue-function taxonomy: VALID ({len(taxonomy['axes']['interaction_function']['labels'])} interaction labels)")
        return 0
    if args.command == "build-function-bundle":
        model = read_json(args.model)
        classifier = RuleDialogueFunctionClassifier(FunctionClassifierConfig(
            context_mode=args.context_mode, confidence_threshold=args.confidence_threshold,
        ))
        bundle = build_function_bundle(model, args.output, classifier, resume=not args.no_resume)
        validate_artifact("dialogue_function_bundle", args.output / "dialogue_function_bundle.json", root / "schemas")
        write_function_bundle_report(args.output / "dialogue_function_report.txt", bundle)
        print(f"Dialogue-function bundle: {bundle['construction_state']} ({bundle['coverage']['accounted_entity_count']} passage(s))")
        print(args.output / "dialogue_function_bundle.json")
        return 0 if bundle["construction_state"] == "READY" else 1
    if args.command == "validate-function-bundle":
        model, bundle = read_json(args.model), read_json(args.bundle)
        result = validate_function_bundle(bundle, model)
        if args.output:
            write_json(args.output, result)
        print(f"Dialogue-function bundle validation: {result['status']} ({result['error_count']} errors, {result['warning_count']} warnings)")
        return 0 if result["status"] == "VALID" else 1
    if args.command == "report-function-bundle":
        bundle = read_json(args.bundle)
        if args.output:
            write_function_bundle_report(args.output, bundle)
            print(args.output)
        else:
            print(render_function_bundle_report(bundle), end="")
        return 0
    if args.command == "prepare-function-calibration":
        source = read_json(args.manifest)
        cases = source.get("cases", []) if isinstance(source, dict) else source
        result = prepare_calibration_review(cases, args.output, maximum_samples=args.maximum_samples)
        print(f"Function calibration package: PENDING_HUMAN_ANNOTATION ({result['sample_count']} sample(s))")
        print(args.output / "calibration_manifest.json")
        return 0
    if args.command == "finalize-function-calibration":
        result = finalize_calibration_review(args.package, annotations_path=args.annotations, output_path=args.output)
        destination = args.output or args.package / "reviewed_calibration_set.json"
        print(f"Function calibration: {result['review_state']} ({result['reviewed_sample_count']}/{result['sample_count']} reviewed)")
        print(destination)
        return 0 if result["review_state"] == "COMPLETE" else 2
    if args.command == "screen-function-schedules":
        source_model, destination_model = read_json(args.source_model), read_json(args.destination_model)
        semantic = SemanticScheduleContext.from_bundles(
            mode=SemanticMode.REPORT_ONLY, weight=0.0,
            source_model=source_model, source_bundle=read_json(args.source_semantic_bundle), source_dir=args.source_semantic_bundle.parent,
            destination_model=destination_model, destination_bundle=read_json(args.destination_semantic_bundle), destination_dir=args.destination_semantic_bundle.parent,
        )
        functions = FunctionScheduleContext.from_bundles(
            mode=FunctionMode.REPORT_ONLY, weight=0.0,
            source_model=source_model, source_bundle=read_json(args.source_function_bundle),
            destination_model=destination_model, destination_bundle=read_json(args.destination_function_bundle),
        )
        clip_artifact, window_artifact = read_json(args.clips), read_json(args.windows)
        clips = clip_artifact.get("clips", []) if isinstance(clip_artifact, dict) else clip_artifact
        if isinstance(window_artifact, dict) and "performances" in window_artifact:
            if not args.speech_windows:
                raise ValueError("--speech-windows is required when --windows is a performance artifact")
            speech_artifact = read_json(args.speech_windows)
            speech_rows = _artifact_rows(speech_artifact, "windows", "events")
            windows = attach_performance_speech_windows(performance_windows(window_artifact), speech_rows)
        else:
            windows = window_artifact.get("windows", []) if isinstance(window_artifact, dict) else window_artifact
        report = run_function_schedule_screen(
            clips=clips, windows=windows, semantic_evidence=semantic, function_evidence=functions,
            output_dir=args.output,
            source_hash=str(clip_artifact.get("media_hash") or source_model["film_id"]) if isinstance(clip_artifact, dict) else source_model["film_id"],
            destination_hash=str(window_artifact.get("media_hash") or destination_model["film_id"]) if isinstance(window_artifact, dict) else destination_model["film_id"],
            max_time_stretch=args.max_time_stretch, semantic_weight=args.semantic_weight, function_weight=args.function_weight,
            scheduling_mode=args.scheduling_mode, best_fit_lookahead=args.best_fit_lookahead,
            shot_boundary_mode=args.shot_boundary_mode, cinematic_filter=args.cinematic_filter,
            source_performances=read_json(args.source_performances) if args.source_performances else None,
            speaker_mapping=read_json(args.speaker_mapping) if args.speaker_mapping else None,
            calibration=read_json(args.calibration) if args.calibration else None,
        )
        validate_artifact("function_schedule_screen", args.output / "function_schedule_screen.json", root / "schemas")
        print(f"Function schedule screen: {report['render_selection_state']}")
        print(args.output / "function_schedule_screen.json")
        return 0
    if args.command == "verify-rendered-function":
        result = evaluate_rendered_function(
            schedule=read_json(args.schedule),
            rendered_dialogue_verification=read_json(args.rendered_verification),
            baseline_schedule=read_json(args.baseline_schedule) if args.baseline_schedule else None,
            calibration=read_json(args.calibration) if args.calibration else None,
            minimum_transcript_confidence=args.minimum_transcript_confidence,
            minimum_function_compatibility=args.minimum_function_compatibility,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        validate_artifact("function_render_verification", args.output, root / "schemas")
        print(f"Rendered function verification: {result['status']} / {result['claim_state']}")
        print(args.output)
        return 0 if result["status"] in {"PASS", "WARN", "INCONCLUSIVE"} else 1
    if args.command == "preflight-function-schedule":
        result = run_function_acoustic_preflight(
            screen_dir=args.screen, output_dir=args.output,
            baseline_variant=args.baseline_variant, function_variant=args.function_variant,
            model_name=args.model, language=args.language,
            minimum_word_coverage=args.minimum_word_coverage, force=args.force,
        )
        validate_artifact("function_acoustic_preflight", args.output / "function_acoustic_preflight.json", root / "schemas")
        print(f"Function acoustic preflight: {result['preflight_state']}")
        print(args.output / "function_acoustic_preflight.json")
        return 0 if result["preflight_state"] == "ACCEPTED_FOR_RENDER" else 2
    if args.command == "render-function-proof":
        config = load_config(root, args.config).with_overrides(
            destination_video=args.destination_video, source_dialogue=args.source_dialogue, output_dir=args.output,
        )
        result = run_function_render_proof(
            config=config, screen_dir=args.screen, output_dir=args.output, preflight_path=args.preflight,
            function_variant=args.function_variant, force=args.force,
            defer_human_calibration=args.defer_human_calibration,
            rejected_preflight_path=args.rejected_preflight, donor_audit_path=args.donor_audit,
        )
        validate_artifact("function_render_proof", args.output / "function_render_proof.json", root / "schemas")
        print(f"Function render proof: {result['technical_acceptance']['state']} / {result['closeout_state']}")
        print(args.output / "function_render_proof.json")
        return 0 if result["technical_acceptance"]["state"] == "ACCEPTED" else 2
    raise ValueError(f"Unsupported dialogue-function command: {args.command}")


def _artifact_rows(artifact: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = artifact.get(key)
        if isinstance(value, list):
            return value
    return []
