from __future__ import annotations

import argparse
from collections import Counter
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"TensorFloat-32 \(TF32\) has been disabled.*",
    category=UserWarning,
    module=r"pyannote\.audio\.utils\.reproducibility",
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cinelingus.config import load_config
from cinelingus.corpus_calibration import build_calibration_followup_plan, build_calibration_plan, execute_calibration_plan
from cinelingus.editorial import EditorialMemory, build_repair_batch, evaluate_editorial_decisions
from cinelingus.evaluation_corpus import build_corpus_manifest, build_evaluation_plan, build_excerpt_plan
from cinelingus.gui import main as gui_main
from cinelingus.mutations import MUTATION_CHOICES
from cinelingus.pipeline import Pipeline
from cinelingus.phase0_benchmarks import build_observed_failure_strategy_plan, build_rendered_strategy_coverage, build_strategy_isolation_plan, run_strategy_contract_benchmarks
from cinelingus.quality_corpus import evaluate_quality_corpus
from cinelingus.schedule import build_editorial_repair_mapping, prepare_editorial_repair_candidates, score_editorial_repair_candidate
from cinelingus.util import read_json, write_json
from cinelingus.validation import ValidationError, validate_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Cinelingus transformation laboratory by default; developer actions are available as subcommands.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="gui",
        choices=["gui", "run", "mutate", "schedule", "repair-preflight", "problem-previews", "performance-previews", "quality-corpus", "corpus-inventory", "corpus-plan", "corpus-excerpts", "corpus-calibrate", "corpus-calibrate-followup", "phase0-strategy-audit", "phase0-strategy-plan", "phase0-observed-plan", "report", "validate", "open-output"],
        help="Workflow action to execute. Default: gui.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder for rendered outputs and reports.")
    parser.add_argument("--input-video", type=Path, default=None, help="Input film for single-film mutation developer runs.")
    parser.add_argument("--destination-video", type=Path, default=None, help="Anchor film for two-input Translation developer runs.")
    parser.add_argument("--source-dialogue", type=Path, default=None, help="Donor dialogue film for two-input Translation developer runs.")
    parser.add_argument("--mutation", choices=MUTATION_CHOICES, default="echo", help="Mutation filter for the mutate developer action.")
    parser.add_argument("--mode", choices=["fast_preview", "balanced", "quality"], default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate cached/rendered outputs.")
    parser.add_argument("--max-regions", type=int, default=None, help="For problem-previews, cap the number of problem regions rendered.")
    parser.add_argument("--corpus-manifest", type=Path, default=None, help="Portable quality-corpus manifest for the quality-corpus action.")
    parser.add_argument("--runs-root", type=Path, default=None, help="Root containing completed corpus run-report folders.")
    parser.add_argument("--movies-root", type=Path, default=Path.home() / "Downloads" / "Movies", help="Read-only local movie corpus root.")
    parser.add_argument("--inventory-cache", type=Path, default=None, help="Workspace-owned cache for corpus hashes and ffprobe results.")
    parser.add_argument("--tier", choices=["smoke", "standard", "extended"], default="smoke", help="Corpus evaluation tier.")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic corpus-selection seed.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum corpus files to inventory or select.")
    parser.add_argument("--max-pairings", type=int, default=None, help="Maximum deterministic benchmark pairings.")
    parser.add_argument("--max-total-duration", type=float, default=None, help="Maximum selected source duration in seconds.")
    parser.add_argument("--max-rendered-duration", type=float, default=None, help="Maximum planned rendered duration in seconds.")
    parser.add_argument("--max-disk-gb", type=float, default=None, help="Maximum generated-artifact disk budget in GiB.")
    parser.add_argument("--max-runtime-minutes", type=float, default=None, help="Maximum benchmark runtime budget in minutes.")
    parser.add_argument("--max-excerpts", type=int, default=None, help="Maximum evidence-selected corpus excerpts.")
    parser.add_argument("--prepare-only", action="store_true", help="Build a calibration plan without extracting or rendering cases.")
    parser.add_argument("--prior-plan", type=Path, default=None, help="Prior calibration plan used to build a fallback-only follow-up.")
    parser.add_argument("--prior-report", type=Path, default=None, help="Prior calibration report used to identify uninformative cases.")
    parser.add_argument("--calibration-plan", type=Path, default=None, help="Execute an existing calibration plan without rebuilding it.")
    parser.add_argument("--calibration-report", type=Path, action="append", default=[], help="Completed calibration report to include in the Phase 0 strategy audit; repeat as needed.")
    parser.add_argument("--strategy-coverage", type=Path, default=None, help="Phase 0 rendered-strategy coverage report used to build an isolation plan.")
    parser.add_argument("--schedule-artifact", type=Path, default=None, help="Replacement schedule containing placements referenced by an observed editorial report.")
    parser.add_argument("--max-strategy-variants", type=int, default=1, help="Maximum observed placement variants per missing repair strategy.")
    parser.add_argument("--refresh-inventory", action="store_true", help="Rehash and reprobe corpus files instead of reusing inventory metadata.")
    parser.add_argument("--open", action="store_true", help="Open the produced file or output folder when done.")
    parser.add_argument("--windowed", action="store_true", help="Open the Qt interface at canonical size instead of maximized.")
    parser.add_argument("--legacy-tk", action="store_true", help="Explicitly launch the retired Tk compatibility interface.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "gui":
        gui_args = []
        if args.windowed:
            gui_args.append("--windowed")
        if args.legacy_tk:
            gui_args.append("--legacy-tk")
        return gui_main(gui_args)

    if args.action == "mutate" and args.input_video is None:
        print("error: mutate requires --input-video so project defaults are not used accidentally", file=sys.stderr)
        return 1

    if args.action == "quality-corpus":
        if args.corpus_manifest is None or args.runs_root is None:
            print("error: quality-corpus requires --corpus-manifest and --runs-root", file=sys.stderr)
            return 1
        output_dir = args.output_dir.resolve() if args.output_dir else ROOT / "output"
        try:
            result = evaluate_quality_corpus(
                manifest_path=args.corpus_manifest.resolve(),
                runs_root=args.runs_root.resolve(),
                output_path=output_dir / "performance_quality_corpus_report.json",
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"quality corpus: {result['passed_case_count']}/{result['case_count']} passed")
        print(f"report: {output_dir / 'performance_quality_corpus_report.json'}")
        return 0 if result["passed"] else 1

    if args.action == "corpus-inventory":
        evaluation_root = ROOT / "evaluation"
        manifest_path = args.corpus_manifest.resolve() if args.corpus_manifest else evaluation_root / "corpus_manifest.json"
        inventory_cache = args.inventory_cache.resolve() if args.inventory_cache else evaluation_root / "inventory_cache.json"
        try:
            result = build_corpus_manifest(
                source_root=args.movies_root.resolve(), output_path=manifest_path,
                inventory_cache_path=inventory_cache, pipeline_cache_root=ROOT / "cache",
                max_files=args.max_files, refresh=args.refresh_inventory,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"corpus inventory: {result['compatible_file_count']} compatible / {result['excluded_file_count']} excluded")
        print(f"manifest: {manifest_path}")
        return 0

    if args.action == "corpus-plan":
        evaluation_root = ROOT / "evaluation"
        manifest_path = args.corpus_manifest.resolve() if args.corpus_manifest else evaluation_root / "corpus_manifest.json"
        plan_path = (args.output_dir.resolve() if args.output_dir else evaluation_root) / f"{args.tier}_plan.json"
        try:
            result = build_evaluation_plan(
                manifest_path=manifest_path, output_path=plan_path, tier=args.tier, seed=args.seed,
                max_files=args.max_files, max_pairings=args.max_pairings,
                max_total_source_duration=args.max_total_duration,
                max_rendered_duration=args.max_rendered_duration,
                max_disk_bytes=int(args.max_disk_gb * 1024 ** 3) if args.max_disk_gb is not None else None,
                max_runtime_seconds=args.max_runtime_minutes * 60 if args.max_runtime_minutes is not None else None,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"corpus plan: tier={result['tier']} files={result['selected_file_count']} pairings={result['selected_pairing_count']}")
        print(f"plan: {plan_path}")
        return 0

    if args.action == "corpus-excerpts":
        evaluation_root = ROOT / "evaluation"
        manifest_path = args.corpus_manifest.resolve() if args.corpus_manifest else evaluation_root / "corpus_manifest.json"
        plan_path = (args.output_dir.resolve() if args.output_dir else evaluation_root) / f"{args.tier}_excerpt_plan.json"
        try:
            result = build_excerpt_plan(
                manifest_path=manifest_path, cache_root=ROOT / "cache", output_path=plan_path,
                tier=args.tier, seed=args.seed, max_excerpts=args.max_excerpts,
                max_total_duration=args.max_rendered_duration,
            )
            validate_artifact("corpus_excerpt_plan", plan_path, ROOT / "schemas")
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"corpus excerpts: tier={result['tier']} selected={result['selected_excerpt_count']} candidates={result['candidate_count']}")
        print(f"plan: {plan_path}")
        return 0

    if args.action == "corpus-calibrate":
        evaluation_root = ROOT / "evaluation"
        excerpt_plan = evaluation_root / f"{args.tier}_excerpt_plan.json"
        calibration_plan = args.calibration_plan.resolve() if args.calibration_plan else evaluation_root / f"{args.tier}_calibration_plan.json"
        output_root = args.output_dir.resolve() if args.output_dir else ROOT / "output" / "corpus_calibration" / args.tier
        try:
            result = (
                read_json(calibration_plan)
                if args.calibration_plan
                else build_calibration_plan(
                    excerpt_plan_path=excerpt_plan, output_path=calibration_plan,
                    max_cases=args.max_pairings,
                )
            )
            validate_artifact("corpus_calibration_plan", calibration_plan, ROOT / "schemas")
            if args.prepare_only:
                print(f"calibration plan: tier={result['tier']} cases={result['case_count']}")
                print(f"plan: {calibration_plan}")
                return 0
            report = execute_calibration_plan(
                root=ROOT, plan_path=calibration_plan, output_root=output_root,
                base_config_path=(args.config.resolve() if args.config else ROOT / "config" / "default.json"),
                force=args.force,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"calibration: {report['completed_case_count']}/{report['case_count']} completed")
        print(f"report: {output_root / 'calibration_report.json'}")
        return 0 if report["failed_case_count"] == 0 else 1

    if args.action == "corpus-calibrate-followup":
        if args.prior_plan is None or args.prior_report is None:
            print("error: corpus-calibrate-followup requires --prior-plan and --prior-report", file=sys.stderr)
            return 1
        evaluation_root = ROOT / "evaluation"
        candidate_plan = evaluation_root / f"{args.tier}_calibration_plan.json"
        followup_plan = evaluation_root / f"{args.tier}_calibration_followup_plan.json"
        output_root = args.output_dir.resolve() if args.output_dir else ROOT / "output" / "corpus_calibration" / f"{args.tier}_followup"
        try:
            result = build_calibration_followup_plan(
                candidate_plan_path=candidate_plan, prior_plan_path=args.prior_plan.resolve(),
                prior_report_path=args.prior_report.resolve(), output_path=followup_plan,
            )
            validate_artifact("corpus_calibration_plan", followup_plan, ROOT / "schemas")
            if args.prepare_only:
                print(f"calibration follow-up plan: tier={result['tier']} cases={result['case_count']}")
                print(f"plan: {followup_plan}")
                return 0
            report = execute_calibration_plan(
                root=ROOT, plan_path=followup_plan, output_root=output_root,
                base_config_path=(args.config.resolve() if args.config else ROOT / "config" / "default.json"),
                force=args.force,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"calibration follow-up: {report['completed_case_count']}/{report['case_count']} completed")
        print(f"report: {output_root / 'calibration_report.json'}")
        return 0 if report["failed_case_count"] == 0 else 1

    if args.action == "phase0-strategy-audit":
        output_root = args.output_dir.resolve() if args.output_dir else ROOT / "evaluation"
        output_root.mkdir(parents=True, exist_ok=True)
        contract_path = output_root / "phase0_strategy_contract_benchmarks.json"
        contract = run_strategy_contract_benchmarks(output_path=contract_path)
        validate_artifact("phase0_strategy_benchmarks", contract_path, ROOT / "schemas")
        print(f"Phase 0 strategy contracts: {contract['passed_case_count']}/{contract['case_count']} passed")
        print(f"contract report: {contract_path}")
        if args.calibration_report:
            coverage_path = output_root / "phase0_rendered_strategy_coverage.json"
            coverage = build_rendered_strategy_coverage(
                contract_report_path=contract_path,
                calibration_report_paths=[path.resolve() for path in args.calibration_report],
                output_path=coverage_path,
            )
            validate_artifact("phase0_strategy_coverage", coverage_path, ROOT / "schemas")
            print(
                "rendered strategy evidence: "
                f"{coverage['evidence_satisfied_strategy_count']}/{coverage['declared_strategy_count']}"
            )
            print(f"coverage report: {coverage_path}")
        return 0 if contract["failed_case_count"] == 0 else 1

    if args.action == "phase0-strategy-plan":
        if args.prior_plan is None or args.prior_report is None or args.strategy_coverage is None:
            print("error: phase0-strategy-plan requires --prior-plan, --prior-report, and --strategy-coverage", file=sys.stderr)
            return 1
        output_root = args.output_dir.resolve() if args.output_dir else ROOT / "evaluation"
        output_root.mkdir(parents=True, exist_ok=True)
        plan_path = output_root / "phase0_strategy_isolation_plan.json"
        try:
            plan = build_strategy_isolation_plan(
                prior_plan_path=args.prior_plan.resolve(),
                prior_report_path=args.prior_report.resolve(),
                coverage_report_path=args.strategy_coverage.resolve(),
                output_path=plan_path, max_cases=args.max_pairings,
            )
            validate_artifact("corpus_calibration_plan", plan_path, ROOT / "schemas")
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Phase 0 strategy-isolation plan: {plan['case_count']} case(s)")
        print(f"unfilled strategies: {len(plan['unfilled_target_strategies'])}")
        print(f"plan: {plan_path}")
        return 0

    if args.action == "phase0-observed-plan":
        if args.config is None or args.prior_report is None or args.schedule_artifact is None or args.strategy_coverage is None:
            print("error: phase0-observed-plan requires --config, --prior-report, --schedule-artifact, and --strategy-coverage", file=sys.stderr)
            return 1
        output_root = args.output_dir.resolve() if args.output_dir else ROOT / "evaluation"
        output_root.mkdir(parents=True, exist_ok=True)
        plan_path = output_root / "phase0_observed_strategy_plan.json"
        try:
            plan = build_observed_failure_strategy_plan(
                config_path=args.config.resolve(), editorial_report_path=args.prior_report.resolve(),
                schedule_path=args.schedule_artifact.resolve(), coverage_report_path=args.strategy_coverage.resolve(),
                output_path=plan_path, max_cases=args.max_pairings,
                variants_per_strategy=max(1, args.max_strategy_variants),
            )
            validate_artifact("corpus_calibration_plan", plan_path, ROOT / "schemas")
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Phase 0 observed-failure plan: {plan['case_count']} case(s)")
        print(f"unfilled strategies: {len(plan['unfilled_target_strategies'])}")
        print(f"plan: {plan_path}")
        return 0

    try:
        destination_video = args.input_video or args.destination_video
        source_dialogue = args.input_video or args.source_dialogue
        config = load_config(ROOT, args.config).with_overrides(
            mode=args.mode,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            destination_video=destination_video.resolve() if destination_video else None,
            source_dialogue=source_dialogue.resolve() if source_dialogue else None,
        )
        pipeline = Pipeline(config)

        if args.action == "run":
            result = pipeline.execute_configuration("multiworld.translation", force=args.force)
            output = result.outputs["video"]
            print(f"finished video: {output}")
            maybe_open(output, args.open)
        elif args.action == "mutate":
            result = pipeline.execute_configuration(args.mutation, force=args.force)
            print(f"finished mutation video: {result.outputs['video']}")
            maybe_open(result.outputs["video"], args.open)
        elif args.action == "schedule":
            schedule = pipeline.schedule(force=args.force)
            print(f"schedule mappings: {len(schedule.get('mappings', []))}")
            print(f"schedule path: {pipeline.destination.cache_dir / 'replacement_schedule.json'}")
        elif args.action == "repair-preflight":
            schedule = pipeline.schedule(force=False)
            verification = read_json(config.output_dir / "rendered_dialogue_verification.json")
            residue_path = config.output_dir / "voice_residue_verification.json"
            problem_path = config.output_dir / "problem_regions.json"
            decisions = evaluate_editorial_decisions(
                schedule=schedule, rendered_verification=verification,
                residue_verification=read_json(residue_path) if residue_path.exists() else {},
                acceptance_threshold=getattr(config, "editorial_acceptance_threshold", 0.72),
                minimum_word_coverage=getattr(config, "editorial_min_word_coverage", 0.72),
                max_time_stretch=config.max_time_stretch,
                problem_report=read_json(problem_path) if problem_path.exists() else {},
            )
            candidates = prepare_editorial_repair_candidates(
                pipeline.build_clip_library(force=False).get("clips", []),
                pipeline.build_source_performances(force=False),
            )
            batch = build_repair_batch(
                schedule=schedule, decisions=decisions, donor_candidates=candidates,
                memory=EditorialMemory(),
                score_candidate=lambda window, clip: score_editorial_repair_candidate(
                    window, clip, max_time_stretch=config.max_time_stretch,
                    shot_boundary_mode=str(schedule.get("shot_boundary_mode", "off")),
                ),
                build_mapping=lambda window, clip, score: build_editorial_repair_mapping(
                    window, clip, score, max_time_stretch=config.max_time_stretch,
                    shot_boundary_mode=str(schedule.get("shot_boundary_mode", "off")),
                    cinematic_filter=str(schedule.get("active_filter", "balanced")),
                ),
                maximum_repairs=getattr(config, "editorial_max_repairs_per_pass", 24),
            )
            report = {
                "schema_version": "1.0", "report_version": "repair_preflight_v1",
                "dry_run": True, "rendered": False,
                "decision_failure_counts": decisions.get("failure_counts", {}),
                "repairable_placement_count": decisions.get("repair_count", 0),
                "attempted_placement_count": batch.get("attempted_count", 0),
                "proposed_candidate_count": batch.get("repaired_count", 0),
                "candidate_family_counts": dict(sorted(Counter(
                    str(row.get("candidate_family") or "no_candidate") for row in batch.get("attempts", [])
                ).items())),
                "strategy_counts": dict(sorted(Counter(
                    str(row.get("repair_strategy") or "unspecified") for row in batch.get("attempts", [])
                ).items())),
                "candidate_loss_stage_counts": dict(sorted(Counter(
                    str(row.get("candidate_loss_stage") or "proposed") for row in batch.get("attempts", [])
                ).items())),
                "repair_neighborhood_count": len(batch.get("repair_neighborhoods", [])),
                "coordinated_neighborhood_count": int(batch.get("coordinated_neighborhood_count", 0)),
                "repair_neighborhoods": batch.get("repair_neighborhoods", []),
                "repairs": batch.get("repairs", []), "attempts": batch.get("attempts", []),
            }
            output_path = config.output_dir / "repair_preflight.json"
            write_json(output_path, report)
            validate_artifact("repair_preflight", output_path, ROOT / "schemas")
            print(f"repair preflight: {report['proposed_candidate_count']}/{report['attempted_placement_count']} proposed")
            print(f"report: {output_path}")
        elif args.action == "problem-previews":
            result = pipeline.render_problem_region_previews(max_regions=args.max_regions)
            print(f"problem preview count: {len(result['previews'])}")
            print(f"manifest: {result['manifest']}")
            print(f"text: {result['text']}")
            print(f"directory: {result['directory']}")
            maybe_open(result["directory"], args.open)
        elif args.action == "performance-previews":
            result = pipeline.render_performance_curation_previews(max_per_stratum=args.max_regions or 2)
            print(f"performance preview count: {len(result['previews'])}")
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
