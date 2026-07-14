from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_masher.config import load_config
from movie_masher.gui import main as gui_main
from movie_masher.mutations import MUTATION_CHOICES
from movie_masher.pipeline import Pipeline
from movie_masher.validation import ValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Cinelingus transformation laboratory by default; developer actions are available as subcommands.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="gui",
        choices=["gui", "run", "mutate", "short", "schedule", "preview", "problem-previews", "report", "validate", "open-output"],
        help="Workflow action to execute. Default: gui.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder for rendered outputs and reports.")
    parser.add_argument("--input-video", type=Path, default=None, help="Input film for single-film mutation developer runs.")
    parser.add_argument("--destination-video", type=Path, default=None, help="Destination film for two-input Transposition developer runs.")
    parser.add_argument("--source-dialogue", type=Path, default=None, help="Replacement dialogue source for two-input Transposition developer runs.")
    parser.add_argument("--mutation", choices=MUTATION_CHOICES, default="echo", help="Mutation filter for the mutate developer action.")
    parser.add_argument("--preference", choices=["balanced", "funniest", "cleanest"], default="balanced", help="Dialogue Reel scoring preference.")
    parser.add_argument("--mode", choices=["fast_preview", "balanced", "quality"], default=None)
    parser.add_argument("--quick", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--force", action="store_true", help="Regenerate cached/rendered outputs.")
    parser.add_argument(
        "--mapping",
        type=int,
        action="append",
        default=None,
        help="Schedule mapping index to preview. Can be repeated. Default for preview: first cross-shot mapping, else 0.",
    )
    parser.add_argument("--audio-only", action="store_true", help="For preview, render WAV only.")
    parser.add_argument("--max-regions", type=int, default=None, help="For problem-previews, cap the number of problem regions rendered.")
    parser.add_argument("--open", action="store_true", help="Open the produced file or output folder when done.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "gui":
        return gui_main()

    if args.action == "mutate" and args.input_video is None:
        print("error: mutate requires --input-video so project defaults are not used accidentally", file=sys.stderr)
        return 1

    try:
        destination_video = args.input_video or args.destination_video
        source_dialogue = args.input_video or args.source_dialogue
        config = load_config(ROOT, args.config).with_overrides(
            mode=args.mode,
            quick_seconds=args.quick,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            destination_video=destination_video.resolve() if destination_video else None,
            source_dialogue=source_dialogue.resolve() if source_dialogue else None,
        )
        pipeline = Pipeline(config)

        if args.action == "run":
            output = pipeline.run_all(force=args.force)
            print(f"finished video: {output}")
            maybe_open(output, args.open)
        elif args.action == "mutate":
            outputs = pipeline.run_mutation(args.mutation, force=args.force)
            print(f"finished mutation video: {outputs['video']}")
            maybe_open(outputs["video"], args.open)
        elif args.action == "short":
            app_mode = "Movie Masher" if args.source_dialogue else "Cinelingus"
            outputs = pipeline.run_best_short_remix(
                app_mode=app_mode,
                mutation_id=args.mutation,
                preference=args.preference,
                force=args.force,
            )
            print(f"finished dialogue reel: {outputs['video']}")
            print(f"dialogue reel report: {outputs['report']}")
            maybe_open(outputs["video"], args.open)
        elif args.action == "schedule":
            schedule = pipeline.schedule(force=args.force)
            print(f"schedule mappings: {len(schedule.get('mappings', []))}")
            print(f"schedule path: {pipeline.destination.cache_dir / 'replacement_schedule.json'}")
        elif args.action == "preview":
            indices = args.mapping or default_preview_indices(pipeline)
            result = pipeline.render_preview(indices, video=not args.audio_only)
            for key, value in result.items():
                print(f"{key}: {value}")
            target = result.get("video") or result.get("audio")
            if isinstance(target, Path):
                maybe_open(target, args.open)
        elif args.action == "problem-previews":
            result = pipeline.render_problem_region_previews(max_regions=args.max_regions)
            print(f"problem preview count: {len(result['previews'])}")
            print(f"manifest: {result['manifest']}")
            print(f"text: {result['text']}")
            print(f"directory: {result['directory']}")
            maybe_open(result["directory"], args.open)
        elif args.action == "report":
            paths = pipeline.generate_reports()
            for key, value in paths.items():
                print(f"{key}: {value}")
            maybe_open(paths["txt"], args.open)
        elif args.action == "validate":
            checks = pipeline.validate_existing()
            failed = False
            for name, exists in checks.items():
                status = "ok" if exists else "missing"
                print(f"{name}: {status}")
                failed = failed or not exists
            return 1 if failed else 0
        elif args.action == "open-output":
            config.output_dir.mkdir(parents=True, exist_ok=True)
            maybe_open(config.output_dir, True)
            print(f"output folder: {config.output_dir}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def default_preview_indices(pipeline: Pipeline) -> list[int]:
    schedule = pipeline.schedule(force=False)
    mappings = schedule.get("mappings", [])
    for index, mapping in enumerate(mappings):
        if mapping.get("mapping_crosses_shot_boundary"):
            return [index]
    if mappings:
        return [0]
    raise ValueError("No schedule mappings are available to preview.")


def maybe_open(path: Path, should_open: bool) -> None:
    if not should_open:
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
