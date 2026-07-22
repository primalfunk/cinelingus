from __future__ import annotations

import argparse
from pathlib import Path

from ..util import read_json, write_json
from ..config import load_config
from ..performance import attach_performance_speech_windows, performance_windows
from ..validation import validate_artifact
from .bundle import build_semantic_bundle, validate_semantic_bundle
from .config import SemanticConfig
from .providers import DeterministicFakeProvider, LocalE5Provider, UnavailableProvider
from .reports import render_semantic_report, write_semantic_report
from .experiment import DEFAULT_WEIGHT_GRID, run_semantic_schedule_screen
from .scheduling import SemanticScheduleContext
from .config import SemanticMode
from .review import build_blinded_semantic_review_package, finalize_blinded_semantic_review
from .render_proof import run_semantic_render_proof
from .corpus_experiment import aggregate_semantic_schedule_screens
from .acoustic_preflight import run_semantic_acoustic_preflight
from .opportunity_acoustics import audit_semantic_opportunity_audio
from .clip_boundary_repair import repair_semantic_clip_boundaries
from .word_boundary_repair import repair_semantic_word_boundaries

SEMANTIC_COMMANDS = frozenset({"build-semantic-bundle", "validate-semantic-bundle", "report-semantic-bundle", "screen-semantic-schedules", "aggregate-semantic-screens", "preflight-semantic-screen", "audit-semantic-opportunities", "repair-semantic-clip-boundaries", "repair-semantic-word-boundaries", "prepare-semantic-review", "finalize-semantic-review", "render-semantic-proof"})


def add_semantic_parsers(subparsers: argparse._SubParsersAction) -> None:
    build = subparsers.add_parser("build-semantic-bundle", help="Build a separately cached SpeechPassage semantic bundle.")
    build.add_argument("--model", type=Path, required=True, help="Validated FilmModel JSON.")
    build.add_argument("--output", type=Path, required=True, help="Semantic bundle directory.")
    build.add_argument("--provider", choices=["local", "fake", "unavailable"], default="local")
    build.add_argument("--device", default="cpu")
    build.add_argument("--asset-dir", type=Path, help="Verified local E5 asset directory; defaults beneath models/semantic.")
    build.add_argument("--batch-size", type=int, default=16)
    build.add_argument("--entity-type", choices=["speech_passage", "dialogue_turn", "turn_sequence"], default="speech_passage", help="Turn and sequence bundles are experimental and require existing ordered structure.")
    build.add_argument("--no-resume", action="store_true")

    validate = subparsers.add_parser("validate-semantic-bundle", help="Validate semantic metadata, coverage, and vector digests.")
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--model", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    report = subparsers.add_parser("report-semantic-bundle", help="Render the semantic subsystem's disclosed capability report.")
    report.add_argument("bundle", type=Path)
    report.add_argument("--output", type=Path)

    screen = subparsers.add_parser("screen-semantic-schedules", help="Run deterministic control/report-only/assisted schedule screening.")
    screen.add_argument("--clips", type=Path, required=True, help="Clip-library JSON or a JSON clip array.")
    screen.add_argument("--windows", type=Path, required=True, help="Filtered-timeline JSON or a JSON window array.")
    screen.add_argument("--speech-windows", type=Path, help="Underlying filtered timeline when --windows is a performance artifact.")
    screen.add_argument("--source-model", type=Path, required=True)
    screen.add_argument("--source-bundle", type=Path, required=True)
    screen.add_argument("--destination-model", type=Path, required=True)
    screen.add_argument("--destination-bundle", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True, help="Directory for schedules and comparison report.")
    screen.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHT_GRID))
    screen.add_argument("--scheduling-mode", choices=["strict_order", "best_fit", "window_fill", "whole_line_fill", "performance_fill"], default="best_fit")
    screen.add_argument("--best-fit-lookahead", type=int, default=8)
    screen.add_argument("--max-time-stretch", type=float, default=0.1)
    screen.add_argument("--shot-boundary-mode", choices=["off", "soft", "strict"], default="off")
    screen.add_argument("--cinematic-filter", default="balanced")
    screen.add_argument("--source-performances", type=Path)
    screen.add_argument("--speaker-mapping", type=Path)
    screen.add_argument(
        "--repair-preflight", type=Path,
        help="Rejected acoustic preflight whose failed source performances must be quarantined during guarded admission.",
    )
    screen.add_argument(
        "--repair-render-proof", type=Path,
        help="Rejected semantic render proof whose failed intervention donors and prior quarantine must be excluded.",
    )
    screen.add_argument(
        "--opportunity-audio-audit", type=Path,
        help="Bounded opportunity acoustic audit whose rejected donor performances must be excluded.",
    )

    aggregate = subparsers.add_parser("aggregate-semantic-screens", help="Aggregate schedule screens with explicit coverage and restraint accounting.")
    aggregate.add_argument("--manifest", type=Path, required=True, help="JSON object with a cases array containing screen paths and optional corpus classes.")
    aggregate.add_argument("--output", type=Path, required=True, help="Output directory for semantic_corpus_screen.json.")

    preflight = subparsers.add_parser("preflight-semantic-screen", help="Retranscribe only changed semantic donor clips before rendering.")
    preflight.add_argument("--screen", type=Path, required=True)
    preflight.add_argument("--semantic-variant", default="assisted_005")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--minimum-word-coverage", type=float)

    opportunity_audio = subparsers.add_parser(
        "audit-semantic-opportunities",
        help="Batch-transcribe and cache acoustic health for globally admissible Pareto opportunity donors.",
    )
    opportunity_audio.add_argument("--screen", type=Path, required=True)
    opportunity_audio.add_argument("--clips", type=Path, required=True)
    opportunity_audio.add_argument("--source-performances", type=Path, required=True)
    opportunity_audio.add_argument("--output", type=Path, required=True)
    opportunity_audio.add_argument("--cache", type=Path)
    opportunity_audio.add_argument("--max-source-performances", type=int, default=24)
    opportunity_audio.add_argument("--minimum-word-coverage", type=float)

    boundary_repair = subparsers.add_parser(
        "repair-semantic-clip-boundaries",
        help="Search bounded source-audio shifts for repeatable exact transcripts and write a derived clip-library overlay.",
    )
    boundary_repair.add_argument("--audit", type=Path, required=True)
    boundary_repair.add_argument("--clips", type=Path, required=True)
    boundary_repair.add_argument("--events", type=Path, required=True)
    boundary_repair.add_argument("--analysis-audio", type=Path, required=True)
    boundary_repair.add_argument("--output", type=Path, required=True)
    boundary_repair.add_argument("--batch-size", type=int, default=24)

    word_repair = subparsers.add_parser(
        "repair-semantic-word-boundaries",
        help="Recover rejected coarse clips from word timestamps and independently verify a derived overlay.",
    )
    word_repair.add_argument("--evidence", type=Path, required=True)
    word_repair.add_argument("--clips", type=Path, required=True)
    word_repair.add_argument("--analysis-audio", type=Path, required=True)
    word_repair.add_argument("--output", type=Path, required=True)
    word_repair.add_argument("--context-padding", type=float, default=4.0)
    word_repair.add_argument("--leading-padding", type=float, default=0.08)
    word_repair.add_argument("--trailing-padding", type=float, default=0.12)

    review = subparsers.add_parser("prepare-semantic-review", help="Build a blinded A/B package from completed control and semantic renders.")
    review.add_argument("--cases", type=Path, required=True, help="JSON array, or object with a cases array, containing completed render pairs.")
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--seed", default="phase2-semantic-review-v1")

    finalize_review = subparsers.add_parser("finalize-semantic-review", help="Validate and unblind a completed semantic A/B review.")
    finalize_review.add_argument("--package", type=Path, required=True)
    finalize_review.add_argument("--responses", type=Path)
    finalize_review.add_argument("--output", type=Path)

    proof = subparsers.add_parser("render-semantic-proof", help="Render, verify, and blind one nominated control/semantic schedule pair.")
    proof.add_argument("--screen", type=Path, required=True, help="Directory containing semantic_schedule_screen.json and schedules.")
    proof.add_argument("--destination-video", type=Path, required=True)
    proof.add_argument("--source-dialogue", type=Path, required=True)
    proof.add_argument("--output", type=Path, required=True)
    proof.add_argument("--control-variant", default="control")
    proof.add_argument("--semantic-variant", default="assisted_005")
    proof.add_argument("--preflight", type=Path, required=True, help="Accepted semantic_acoustic_preflight.json for this screen and variant.")


def run_semantic_command(args: argparse.Namespace, root: Path) -> int:
    if args.command == "build-semantic-bundle":
        model = read_json(args.model)
        config = SemanticConfig(device=args.device)
        if args.provider == "fake":
            provider = DeterministicFakeProvider(config)
        elif args.provider == "unavailable":
            provider = UnavailableProvider(config, state="UNAVAILABLE", reason="Explicit unavailable-provider developer test.")
        else:
            asset_dir = args.asset_dir or root / "models" / "semantic" / "intfloat-multilingual-e5-small" / config.model_revision
            provider = LocalE5Provider(config, asset_dir=asset_dir, allow_download=False)
        result = build_semantic_bundle(
            model, args.output, provider, config, batch_size=args.batch_size,
            resume=not (args.no_resume or args.force), entity_type=args.entity_type,
        )
        write_json(args.output / "validation_report.json", result.validation_report)
        write_json(args.output / "cache_report.json", result.cache_report)
        write_semantic_report(args.output / "semantic_report.txt", result.bundle, result.cache_report)
        print(f"Semantic bundle: {result.bundle['construction_state']} ({result.validation_report['status']})")
        print(args.output / "semantic_bundle.json")
        return 0 if result.validation_report["status"] == "VALID" else 1
    if args.command == "validate-semantic-bundle":
        bundle = read_json(args.bundle)
        report = validate_semantic_bundle(bundle, args.bundle.parent, read_json(args.model))
        if args.output:
            write_json(args.output, report)
        print(f"Semantic bundle validation: {report['status']} ({report['error_count']} errors, {report['warning_count']} warnings)")
        return 0 if report["status"] == "VALID" else 1
    if args.command == "report-semantic-bundle":
        bundle = read_json(args.bundle)
        if args.output:
            write_semantic_report(args.output, bundle)
            print(args.output)
        else:
            print(render_semantic_report(bundle))
        return 0
    if args.command == "screen-semantic-schedules":
        source_model, destination_model = read_json(args.source_model), read_json(args.destination_model)
        context = SemanticScheduleContext.from_bundles(
            mode=SemanticMode.REPORT_ONLY,
            weight=0.0,
            source_model=source_model,
            source_bundle=read_json(args.source_bundle),
            source_dir=args.source_bundle.parent,
            destination_model=destination_model,
            destination_bundle=read_json(args.destination_bundle),
            destination_dir=args.destination_bundle.parent,
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
        if sum(bool(value) for value in (args.repair_preflight, args.repair_render_proof, args.opportunity_audio_audit)) > 1:
            raise ValueError("Use only one repair evidence source per rescreen")
        repair_preflight = read_json(args.repair_preflight) if args.repair_preflight else None
        if repair_preflight and repair_preflight.get("preflight_state") != "REJECTED_ACOUSTIC_INTEGRITY":
            raise ValueError("--repair-preflight requires a rejected acoustic-integrity report")
        if repair_preflight:
            repair_preflight = {**repair_preflight, "evidence_type": "acoustic_preflight"}
        repair_proof = read_json(args.repair_render_proof) if args.repair_render_proof else None
        if repair_proof:
            intervention = repair_proof.get("semantic_intervention_verification") or {}
            lineage = repair_proof.get("repair_lineage") or {}
            quarantined = set(intervention.get("rejected_source_performance_ids") or [])
            quarantined.update(lineage.get("quarantined_source_performance_ids") or [])
            repair_preflight = {
                "evidence_type": "semantic_render_proof",
                "screen_signature": repair_proof.get("screen_signature"),
                "render_proof_signature": repair_proof.get("proof_signature"),
                "rejected_source_performance_ids": sorted(quarantined),
            }
        opportunity_audit = read_json(args.opportunity_audio_audit) if args.opportunity_audio_audit else None
        if opportunity_audit:
            lineage = opportunity_audit.get("repair_lineage") or {}
            quarantined = set(opportunity_audit.get("rejected_source_performance_ids") or [])
            quarantined.update(lineage.get("quarantined_source_performance_ids") or [])
            repair_preflight = {
                "evidence_type": "opportunity_acoustic_audit",
                "screen_signature": opportunity_audit.get("screen_signature"),
                "opportunity_audit_signature": opportunity_audit.get("audit_signature"),
                "rejected_source_performance_ids": sorted(quarantined),
            }
        quarantined_sources = set((repair_preflight or {}).get("rejected_source_performance_ids") or [])
        quarantined_sources.update(
            ((repair_preflight or {}).get("repair_lineage") or {}).get("quarantined_source_performance_ids") or []
        )
        if repair_preflight:
            repair_preflight = {
                **repair_preflight,
                "rejected_source_performance_ids": sorted(quarantined_sources),
            }
        if repair_preflight and not quarantined_sources:
            raise ValueError("Repair evidence does not identify source performances to quarantine")
        report = run_semantic_schedule_screen(
            clips=clips,
            windows=windows,
            semantic_evidence=context,
            output_dir=args.output,
            source_hash=str(clip_artifact.get("media_hash") or source_model["film_id"]) if isinstance(clip_artifact, dict) else str(source_model["film_id"]),
            destination_hash=str(window_artifact.get("media_hash") or destination_model["film_id"]) if isinstance(window_artifact, dict) else str(destination_model["film_id"]),
            max_time_stretch=args.max_time_stretch,
            weights=tuple(args.weights),
            scheduling_mode=args.scheduling_mode,
            best_fit_lookahead=args.best_fit_lookahead,
            shot_boundary_mode=args.shot_boundary_mode,
            cinematic_filter=args.cinematic_filter,
            source_performances=read_json(args.source_performances) if args.source_performances else None,
            speaker_mapping=read_json(args.speaker_mapping) if args.speaker_mapping else None,
            prohibited_source_performance_ids=quarantined_sources,
            repair_preflight=repair_preflight,
        )
        print(f"Semantic schedule screen: {report['render_selection_state']}")
        print(args.output / "semantic_schedule_screen.json")
        return 0
    if args.command == "prepare-semantic-review":
        source = read_json(args.cases)
        cases = source.get("cases", []) if isinstance(source, dict) else source
        manifest = build_blinded_semantic_review_package(cases, args.output, seed=args.seed)
        print(f"Semantic review package: BLINDED ({manifest['case_count']} case(s))")
        print(args.output / "review_manifest.json")
        return 0
    if args.command == "finalize-semantic-review":
        result = finalize_blinded_semantic_review(
            args.package, responses_path=args.responses, output_path=args.output,
        )
        destination = args.output or args.package / "semantic_review_result.json"
        print(f"Semantic review: {result['review_state']} ({len(result['cases'])} case(s))")
        print(destination)
        return 0 if result["review_state"] == "COMPLETE" else 2
    if args.command == "preflight-semantic-screen":
        config = load_config(root, args.config)
        report = run_semantic_acoustic_preflight(
            screen_dir=args.screen, output_dir=args.output, semantic_variant=args.semantic_variant,
            model_name=config.whisper_model, language=config.whisper_language,
            transcription_mode=config.transcription_mode,
            minimum_word_coverage=(args.minimum_word_coverage if args.minimum_word_coverage is not None else config.editorial_min_word_coverage),
            force=args.force,
        )
        validate_artifact("semantic_acoustic_preflight", args.output / "semantic_acoustic_preflight.json", root / "schemas")
        print(f"Semantic acoustic preflight: {report['preflight_state']} ({report['changed_mapping_count']} changed mapping(s))")
        print(args.output / "semantic_acoustic_preflight.json")
        return 0
    if args.command == "audit-semantic-opportunities":
        config = load_config(root, args.config)
        screen_path = args.screen / "semantic_schedule_screen.json" if args.screen.is_dir() else args.screen
        clip_artifact = read_json(args.clips)
        clips = clip_artifact.get("clips", []) if isinstance(clip_artifact, dict) else clip_artifact
        report = audit_semantic_opportunity_audio(
            screen_path=screen_path, clips=clips,
            source_performances=read_json(args.source_performances),
            output_dir=args.output, cache_path=args.cache,
            model_name=config.whisper_model, language=config.whisper_language,
            transcription_mode=config.transcription_mode,
            minimum_word_coverage=(args.minimum_word_coverage if args.minimum_word_coverage is not None else config.editorial_min_word_coverage),
            max_source_performances=args.max_source_performances,
            force=args.force,
        )
        validate_artifact(
            "semantic_opportunity_acoustic_audit",
            args.output / "semantic_opportunity_acoustic_audit.json", root / "schemas",
        )
        print(
            f"Semantic opportunity acoustic audit: {report['audit_state']} "
            f"({report['audited_source_performance_count']} performance(s))"
        )
        print(args.output / "semantic_opportunity_acoustic_audit.json")
        return 0
    if args.command == "repair-semantic-clip-boundaries":
        config = load_config(root, args.config)
        report = repair_semantic_clip_boundaries(
            clip_library=read_json(args.clips), dialogue_events=read_json(args.events),
            acoustic_audit=read_json(args.audit), analysis_audio=args.analysis_audio,
            output_dir=args.output, model_name=config.whisper_model,
            language=config.whisper_language, transcription_mode=config.transcription_mode,
            batch_size=args.batch_size, force=args.force,
        )
        validate_artifact(
            "semantic_clip_boundary_repair",
            args.output / "semantic_clip_boundary_repair.json", root / "schemas",
        )
        validate_artifact("clip_library", args.output / "repaired_clip_library.json", root / "schemas")
        print(
            f"Semantic clip-boundary repair: {report['repair_state']} "
            f"({report['repaired_clip_count']}/{report['rejected_clip_count']} repaired)"
        )
        print(args.output / "semantic_clip_boundary_repair.json")
        return 0
    if args.command == "repair-semantic-word-boundaries":
        config = load_config(root, args.config)
        report = repair_semantic_word_boundaries(
            clip_library=read_json(args.clips), rejection_evidence=read_json(args.evidence),
            analysis_audio=args.analysis_audio, output_dir=args.output,
            model_name=config.whisper_model, language=config.whisper_language,
            transcription_mode=config.transcription_mode,
            context_padding=args.context_padding, leading_padding=args.leading_padding,
            trailing_padding=args.trailing_padding, force=args.force,
        )
        validate_artifact(
            "semantic_word_boundary_repair",
            args.output / "semantic_word_boundary_repair.json", root / "schemas",
        )
        validate_artifact("clip_library", args.output / "repaired_clip_library.json", root / "schemas")
        print(
            f"Semantic word-boundary repair: {report['repair_state']} "
            f"({report['repaired_clip_count']}/{report['rejected_clip_count']} repaired)"
        )
        print(args.output / "semantic_word_boundary_repair.json")
        return 0
    if args.command == "aggregate-semantic-screens":
        manifest = read_json(args.manifest)
        cases = manifest.get("cases", []) if isinstance(manifest, dict) else manifest
        resolved = []
        for case in cases:
            item = dict(case)
            path = Path(item["screen"])
            item["screen"] = path if path.is_absolute() else (args.manifest.parent / path).resolve()
            resolved.append(item)
        output_path = args.output / "semantic_corpus_screen.json"
        report = aggregate_semantic_schedule_screens(resolved, output_path=output_path, schemas_dir=root / "schemas")
        print(f"Semantic corpus screen: {report['corpus_state']} ({report['case_count']} case(s))")
        print(output_path)
        return 0
    if args.command == "render-semantic-proof":
        config = load_config(root, args.config).with_overrides(
            destination_video=args.destination_video,
            source_dialogue=args.source_dialogue,
            output_dir=args.output,
        )
        report = run_semantic_render_proof(
            config=config,
            screen_dir=args.screen,
            output_dir=args.output,
            control_variant=args.control_variant,
            semantic_variant=args.semantic_variant,
            force=args.force,
            preflight_path=args.preflight,
        )
        print(f"Semantic render proof: {report['technical_render_state']} / {report['verification_state']}")
        print(args.output / "semantic_render_proof.json")
        return 0
    raise ValueError(f"Unsupported semantic command: {args.command}")


def _artifact_rows(artifact, *keys: str):
    if not isinstance(artifact, dict):
        return artifact
    for key in keys:
        if key in artifact:
            return artifact.get(key) or []
    return []
