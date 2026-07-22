from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..util import read_json, write_json
from .builder import SUPPORTED_ARTIFACTS, build_film_model
from .cache import evaluate_model_cache
from .reports import compare_models, render_model_report, write_model_bundle, write_model_report
from .schedule_bridge import compare_schedule_equivalence, ingest_schedule, reconstruct_schedule
from .validation import validate_film_model

MODEL_COMMANDS = frozenset({
    "build-film-model", "validate-film-model", "report-film-model", "compare-film-model",
    "trace-schedule", "reconstruct-schedule", "compare-schedule",
})


def add_model_parsers(subparsers: argparse._SubParsersAction) -> None:
    build = subparsers.add_parser("build-film-model", help="Build the opt-in Phase 1 FilmModel from existing analysis artifacts.")
    identity = build.add_mutually_exclusive_group(required=True)
    identity.add_argument("--artifact-dir", type=Path, help="Existing cache role directory containing movie.json and analysis artifacts.")
    identity.add_argument("--cache-root", type=Path, help="Existing cache root; requires --media-hash and --role.")
    build.add_argument("--media-hash", help="Media hash beneath --cache-root.")
    build.add_argument("--role", help="Cache role beneath the media hash, such as destination_video.")
    build.add_argument("--output", type=Path, help="Model bundle directory; defaults to <artifact-dir>/cinematic_model.")
    build.add_argument("--include-editorial-run", type=Path, help="Optional run output directory containing editorial_report.json.")
    build.add_argument(
        "--include-speech-role", type=Path, action="append", default=[],
        help="Optional additional source_dialogue or destination_video cache directory whose speech evidence is merged into the FilmModel.",
    )
    build.add_argument("--model-force", "--force", dest="model_force", action="store_true", help="Rebuild even when a compatible cached model exists.")
    build.add_argument("--validation-only", action="store_true", help="Build and validate in memory without writing a bundle.")
    build.add_argument("--report-only", action="store_true", help="Build and print the report without writing a bundle.")
    build.add_argument("--strict", action="store_true", help="Require VALID rather than VALID_WITH_WARNINGS.")
    build.add_argument("--deterministic-diagnostic", action="store_true", help="Print deterministic artifact and cache diagnostics as JSON.")

    validate = subparsers.add_parser("validate-film-model", help="Validate an existing FilmModel without modifying it.")
    validate.add_argument("model", type=Path)
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--output", type=Path, help="Optional validation-report output path.")

    report = subparsers.add_parser("report-film-model", help="Render a human-readable report for an existing FilmModel.")
    report.add_argument("model", type=Path)
    report.add_argument("--output", type=Path)

    compare = subparsers.add_parser("compare-film-model", help="Compare two canonical FilmModel artifacts.")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", type=Path)

    trace = subparsers.add_parser("trace-schedule", help="Ingest a Translation schedule and trace it through source and destination FilmModels.")
    trace.add_argument("--schedule", type=Path, required=True)
    trace.add_argument("--source-model", type=Path, required=True)
    trace.add_argument("--destination-model", type=Path, required=True)
    trace.add_argument("--verification", type=Path)
    trace.add_argument("--output", type=Path, required=True)
    trace.add_argument("--strict", action="store_true", help="Reject partial traceability warnings.")

    reconstruct = subparsers.add_parser("reconstruct-schedule", help="Reconstruct the exact legacy schedule payload from a bridge artifact.")
    reconstruct.add_argument("bridge", type=Path)
    reconstruct.add_argument("--output", type=Path, required=True)

    schedule_compare = subparsers.add_parser("compare-schedule", help="Compare original and reconstructed Translation schedules.")
    schedule_compare.add_argument("original", type=Path)
    schedule_compare.add_argument("reconstructed", type=Path)
    schedule_compare.add_argument("--bridge", type=Path)
    schedule_compare.add_argument("--output", type=Path)


def run_model_command(args: argparse.Namespace, root: Path) -> int:
    schemas_dir = root / "schemas"
    if args.command == "build-film-model":
        artifact_dir = _artifact_dir(args)
        artifacts = _discover_artifacts(artifact_dir, args.include_editorial_run, args.include_speech_role)
        result = build_film_model(artifacts, schemas_dir=schemas_dir)
        valid = result.validation_report["status"] == "VALID" if args.strict else result.validation_report["status"] in {"VALID", "VALID_WITH_WARNINGS"}
        if args.deterministic_diagnostic:
            print(json.dumps({
                "artifact_dir": artifact_dir.resolve().as_posix(),
                "artifact_types": result.build_report["artifact_types_used"],
                "created_from_signature": result.model["created_from_signature"],
                "validation_status": result.validation_report["status"],
            }, indent=2, sort_keys=True))
        if not valid:
            raise ValueError(f"FilmModel validation did not satisfy {'strict ' if args.strict else ''}requirements: {result.validation_report['status']}")
        if args.validation_only:
            print(f"FilmModel validation: {result.validation_report['status']}")
            return 0
        if args.report_only:
            print(render_model_report(result.model))
            return 0
        output = args.output or artifact_dir / "cinematic_model"
        cache = evaluate_model_cache(output / "film_model.json", result.model["created_from_signature"], force=args.model_force or args.force)
        if cache.reuse:
            print(f"FilmModel cache: {cache.status}")
            print(output / "film_model.json")
            return 0
        paths = write_model_bundle(output, result)
        print(f"FilmModel cache: {cache.status} — {'; '.join(cache.reasons)}")
        print(paths["model"])
        return 0
    if args.command == "validate-film-model":
        model = read_json(args.model)
        report = validate_film_model(model, schemas_dir)
        if args.output:
            write_json(args.output, report)
        print(f"FilmModel validation: {report['status']} ({report['error_count']} errors, {report['warning_count']} warnings)")
        accepted = report["status"] == "VALID" if args.strict else report["status"] in {"VALID", "VALID_WITH_WARNINGS"}
        return 0 if accepted else 1
    if args.command == "report-film-model":
        model = read_json(args.model)
        if args.output:
            write_model_report(args.output, model)
            print(args.output)
        else:
            print(render_model_report(model))
        return 0
    if args.command == "compare-film-model":
        comparison = compare_models(read_json(args.left), read_json(args.right))
        if args.output:
            write_json(args.output, comparison)
            print(args.output)
        else:
            print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0 if comparison["equivalent"] else 1
    if args.command == "trace-schedule":
        bridge = ingest_schedule(
            read_json(args.schedule), source_model=read_json(args.source_model),
            destination_model=read_json(args.destination_model),
            rendered_verification=read_json(args.verification) if args.verification else None,
        )
        write_json(args.output, bridge)
        print(f"Schedule trace: {bridge['validation_state']['schedule_trace_readiness']} ({bridge['placement_count']} placements)")
        print(args.output)
        accepted = bridge["validation_state"]["status"] == "VALID" if args.strict else bridge["validation_state"]["status"] in {"VALID", "VALID_WITH_WARNINGS"}
        return 0 if accepted else 1
    if args.command == "reconstruct-schedule":
        reconstructed = reconstruct_schedule(read_json(args.bridge))
        write_json(args.output, reconstructed)
        print(args.output)
        return 0
    if args.command == "compare-schedule":
        comparison = compare_schedule_equivalence(
            read_json(args.original), read_json(args.reconstructed),
            bridge=read_json(args.bridge) if args.bridge else None,
        )
        if args.output:
            write_json(args.output, comparison)
            print(args.output)
        else:
            print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0 if comparison["equivalent"] else 1
    raise ValueError(f"Unsupported FilmModel command: {args.command}")


def _artifact_dir(args: argparse.Namespace) -> Path:
    if args.artifact_dir:
        return args.artifact_dir
    if not args.media_hash or not args.role:
        raise ValueError("--cache-root requires both --media-hash and --role")
    return args.cache_root / args.media_hash / args.role


def _discover_artifacts(
    artifact_dir: Path,
    editorial_dir: Path | None,
    speech_role_dirs: list[Path] | None = None,
) -> dict[str, Path]:
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
    artifacts = {name: artifact_dir / f"{name}.json" for name in SUPPORTED_ARTIFACTS if (artifact_dir / f"{name}.json").is_file()}
    if editorial_dir:
        for name in ("editorial_report", "editorial_decisions"):
            candidate = editorial_dir / f"{name}.json"
            if candidate.is_file():
                artifacts[name] = candidate
    for speech_dir in speech_role_dirs or []:
        if not speech_dir.is_dir():
            raise FileNotFoundError(f"Additional speech role directory does not exist: {speech_dir}")
        role = speech_dir.name
        if role not in {"source_dialogue", "destination_video"}:
            raise ValueError(
                "--include-speech-role must name a source_dialogue or destination_video cache directory"
            )
        for artifact_type in ("dialogue_events", "timeline"):
            candidate = speech_dir / f"{artifact_type}.json"
            if not candidate.is_file():
                continue
            if any(path.resolve() == candidate.resolve() for path in artifacts.values()):
                continue
            artifacts[f"{role}_{artifact_type}"] = candidate
    if "movie" not in artifacts:
        raise FileNotFoundError(f"Required movie.json is missing from {artifact_dir}")
    return artifacts
