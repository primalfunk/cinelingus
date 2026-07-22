from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .cinematic_model.developer_cli import MODEL_COMMANDS, add_model_parsers, run_model_command
from .semantic.developer_cli import SEMANTIC_COMMANDS, add_semantic_parsers, run_semantic_command
from .dialogue_function.developer_cli import FUNCTION_COMMANDS, add_function_parsers, run_function_command
from .pipeline import Pipeline
from .publish import publish_single_video
from .presets import list_presets, load_preset
from .run_guard import exclusive_output_run, verify_filter_execution
from .tools import ToolError
from .validation import ValidationError
from .whisper_backend import whisper_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cinelingus")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate artifacts instead of reusing valid cache entries.")
    parser.add_argument("--mode", choices=["fast_preview", "balanced", "quality"], default=None, help="Override transcription quality mode.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder where rendered outputs and reports are written.")
    parser.add_argument("--semantic-mode", choices=["SEMANTIC_DISABLED", "SEMANTIC_REPORT_ONLY", "SEMANTIC_ASSISTED"], default=None, help="Optional transcript-vector scheduling mode.")
    parser.add_argument("--semantic-weight", type=float, default=None, help="Bounded assisted semantic contribution (0.0 to 1.0).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    sub.add_parser("extract-source")
    sub.add_parser("filter-source")
    sub.add_parser("clips")
    sub.add_parser("timeline")
    sub.add_parser("filter-timeline")
    sub.add_parser("visual")
    sub.add_parser("schedule")
    sub.add_parser("render-audio")
    sub.add_parser("render-video")
    sub.add_parser("report")
    sub.add_parser("presets")
    preset = sub.add_parser("preset")
    preset.add_argument("preset_id", help="Preset id to run, for example translation or self_shuffle (legacy translation remains accepted).")
    preset.add_argument("--seed", type=int, default=None, help="Preset seed parameter when supported.")
    self_shuffle = sub.add_parser("self-shuffle")
    self_shuffle.add_argument("--seed", type=int, default=1, help="Deterministic shuffle seed.")
    sub.add_parser("validate")
    sub.add_parser("whisper-info")
    sub.add_parser("run")
    add_model_parsers(sub)
    add_semantic_parsers(sub)
    add_function_parsers(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path.cwd()
    pipeline: Pipeline | None = None
    run_guard = None
    run_lease = None
    try:
        if args.command in MODEL_COMMANDS:
            return run_model_command(args, root)
        if args.command in SEMANTIC_COMMANDS:
            return run_semantic_command(args, root)
        if args.command in FUNCTION_COMMANDS:
            return run_function_command(args, root)
        config = load_config(root, args.config).with_overrides(
            mode=args.mode,
            output_dir=args.output_dir,
            semantic_mode=args.semantic_mode,
            semantic_weight=args.semantic_weight,
        )
        if args.command not in {"presets", "validate", "whisper-info"}:
            requested_filter = _command_filter_id(args, root)
            candidate_guard = exclusive_output_run(config.output_dir, requested_filter)
            run_lease = candidate_guard.__enter__()
            run_guard = candidate_guard
        pipeline = Pipeline(config)
        if args.command == "inspect":
            dest, source = pipeline.inspect(force=args.force)
            print(f"destination_video: {dest['duration']:.2f}s {dest.get('resolution')}")
            print(f"source_dialogue: {source['duration']:.2f}s {source.get('resolution')}")
            print(f"transcription_mode: {config.transcription_mode}")
            print(f"whisper_model: {config.whisper_model}")
        elif args.command == "extract-source":
            data = pipeline.extract_source_dialogue(force=args.force)
            print(f"source dialogue events: {len(data['events'])}")
        elif args.command == "filter-source":
            data = pipeline.filter_source_dialogue(force=args.force)
            print(_filter_summary("source dialogue", data))
        elif args.command == "clips":
            data = pipeline.build_clip_library(force=args.force)
            print(f"clips: {len(data['clips'])}")
        elif args.command == "timeline":
            data = pipeline.detect_destination_timeline(force=args.force)
            print(f"destination windows: {len(data['windows'])}")
        elif args.command == "filter-timeline":
            data = pipeline.filter_destination_timeline(force=args.force)
            print(_filter_summary("destination timeline", data))
        elif args.command == "visual":
            data = pipeline.analyze_visual(force=args.force)
            report = data["visual_report"]
            print(f"visual shots: {report['total_shots']}")
            print(f"average shot duration: {report['average_shot_duration']:.2f}s")
        elif args.command == "schedule":
            data = pipeline.schedule(force=args.force)
            print(f"schedule mappings: {len(data['mappings'])}")
        elif args.command == "render-audio":
            print(pipeline.render_audio(force=args.force))
        elif args.command == "render-video":
            print(pipeline.render_video(force=args.force))
        elif args.command == "report":
            paths = pipeline.generate_reports()
            print(f"run report: {paths['txt']}")
            print(f"run report json: {paths['json']}")
            print(f"schedule csv: {paths['csv']}")
            print(f"cinematic index: {paths['cir']}")
        elif args.command == "presets":
            for preset in list_presets(root):
                print(f"{preset.id}: {preset.name} - {preset.description}")
        elif args.command == "preset":
            params = {}
            if args.seed is not None:
                params["seed"] = args.seed
            paths = pipeline.run_preset(args.preset_id, force=args.force, parameters=params)
            video = publish_single_video(video=paths["video"], output_dir=config.output_dir, process=args.preset_id)
            assert run_lease is not None
            requested_filter = _command_filter_id(args, root)
            verify_filter_execution(
                run_lease,
                requested_filter_id=requested_filter,
                evidence_paths=[config.output_dir / requested_filter / "filter_recipe.json"],
                output=video,
            )
            print(video)
        elif args.command == "self-shuffle":
            paths = pipeline.run_self_shuffle(seed=args.seed, force=args.force)
            assert run_lease is not None
            verify_filter_execution(
                run_lease,
                requested_filter_id="self_shuffle",
                evidence_paths=[config.output_dir / "self_shuffle" / "filter_recipe.json"],
                output=paths["video"],
            )
            print(f"self-shuffle schedule: {paths['schedule']}")
            print(f"self-shuffle audio: {paths['audio']}")
            print(f"self-shuffle video: {paths['video']}")
        elif args.command == "validate":
            checks = pipeline.validate_existing()
            for name, exists in checks.items():
                print(f"{name}: {'ok' if exists else 'missing'}")
        elif args.command == "whisper-info":
            runtime = whisper_runtime()
            print(f"available: {runtime['available']}")
            print(f"cuda_available: {runtime['cuda_available']}")
            print(f"device: {runtime['device']}")
            print(f"mode: {config.transcription_mode}")
            print(f"model: {config.whisper_model}")
        elif args.command == "run":
            transformation_result = pipeline.execute_configuration("multiworld.translation", force=args.force)
            video = transformation_result.outputs["video"]
            assert run_lease is not None
            verify_filter_execution(
                run_lease,
                requested_filter_id="translation",
                evidence_paths=[
                    config.output_dir / "translation" / "filter_acceptance.json",
                    config.output_dir / "translation" / "filter_recipe.json",
                    transformation_result.artifacts["alteration_acceptance"],
                    transformation_result.artifacts["configuration_outcome"],
                ],
                output=video,
            )
            print(video)
        return 0
    except (FileNotFoundError, ToolError, ValueError, ValidationError, RuntimeError) as exc:
        if pipeline is not None:
            pipeline.logger.error(str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if run_guard is not None:
            run_guard.__exit__(*sys.exc_info())


def _filter_summary(label: str, data: dict) -> str:
    stats = data["filter_stats"]
    return f"{label}: {stats['usable_count']} usable / {stats['raw_count']} raw / {stats['rejected_count']} rejected"


def _command_filter_id(args, root: Path) -> str:
    if args.command == "self-shuffle":
        return "self_shuffle"
    if args.command == "preset":
        return load_preset(root, args.preset_id).transformation_strategy
    return "translation"


if __name__ == "__main__":
    raise SystemExit(main())
