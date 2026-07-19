from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cinelingus.config import load_config
from cinelingus.pipeline import Pipeline
from cinelingus.reliable_inputs import preflight_media_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one full-source filter for reliability evaluation.")
    parser.add_argument("filter_id")
    parser.add_argument("media", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", default="balanced", choices=("fast_preview", "balanced", "quality"))
    parser.add_argument("--seed", type=int, default=180726)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(root).with_films([args.media]).with_overrides(
        mode=args.mode,
        output_dir=args.output_dir,
    )
    preflight = preflight_media_inputs(config.films, output_dir=config.output_dir)
    print("RUN_START", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("RUN_PREFLIGHT " + json.dumps(preflight, ensure_ascii=False), flush=True)
    result = Pipeline(config).execute_transformation(
        args.filter_id,
        force=args.force,
        parameters={"intensity": "Moderate", "seed": args.seed},
    )
    print("RUN_RESULT " + json.dumps({
        "transformation_id": result.transformation_id,
        "outputs": {key: str(value) for key, value in result.outputs.items()},
        "artifacts": {key: str(value) for key, value in result.artifacts.items()},
    }, ensure_ascii=False), flush=True)
    print("RUN_END", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
