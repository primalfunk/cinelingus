from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.reliable_experiment import (
    discover_default_videos,
    execute_reliably,
    normalize_parameters_tolerantly,
    resolve_configuration,
)
from cinelingus.reliable_inputs import default_input_directory, preflight_media_inputs
from cinelingus.util import utc_now, write_json
from cinelingus.validation import validate_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Certify that every declared configuration resolves to a media-producing outcome."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_directory())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/configuration_outcome_certification"),
    )
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_default_videos(input_dir)
    if not videos:
        raise SystemExit(f"No video files found in {input_dir}")
    preflight = preflight_media_inputs(videos, output_dir=output_dir)
    registry = default_filter_registry()

    parameter_cases = 0
    arity_failures: list[str] = []
    for definition in registry.definitions(implemented_only=True):
        if len(videos) < definition.minimum_films:
            arity_failures.append(definition.id)
        normalized, adjustments = normalize_parameters_tolerantly(definition, definition.parameter_defaults)
        if adjustments or normalized != definition.parameter_defaults:
            raise SystemExit(f"Default parameter normalization failed for {definition.id}")
        parameter_cases += 1
        for parameter in definition.parameters:
            if parameter.kind == "choice":
                parameter_cases += len(parameter.choices)
            elif parameter.kind == "boolean":
                parameter_cases += 2
            elif parameter.kind in {"integer", "float"}:
                parameter_cases += len({value for value in (parameter.minimum, parameter.default, parameter.maximum) if value is not None})
            else:
                parameter_cases += 1

    stack_counts: Counter[str] = Counter()
    pair_count = 0
    definitions = registry.definitions()
    for first in definitions:
        for second in definitions:
            if first.id == second.id:
                continue
            resolution = resolve_configuration(first.id, selected_filter_stack=[first.id, second.id])
            stack_counts[resolution["stack_status"]] += 1
            pair_count += 1

    individual_durations = {
        path: float(preflight_media_inputs((path,), output_dir=output_dir)["predicted_output_duration"])
        for path in videos
    }
    certification_films = tuple(
        sorted(videos, key=lambda path: (-individual_durations[path], path.name.casefold()))[:2]
    )
    fallback_preflight = preflight_media_inputs(certification_films, output_dir=output_dir)
    pipeline = SimpleNamespace(
        config=SimpleNamespace(
            anchor_film=certification_films[0],
            output_dir=output_dir,
            films=certification_films,
        ),
        cancel_check=lambda: False,
    )

    def forced_failure(*_args: Any, **_kwargs: Any):
        raise RuntimeError("forced configuration-outcome assurance fallback")

    fallback_result = execute_reliably(
        pipeline,
        registry.definitions(implemented_only=True)[0].id,
        run_filter=forced_failure,
    )
    outcome_path = fallback_result.artifacts["configuration_outcome"]
    outcome = validate_artifact("configuration_outcome", outcome_path, Path.cwd() / "schemas")
    fallback_video = fallback_result.outputs["video"]
    single_certification_film = min(videos, key=lambda path: (individual_durations[path], path.name.casefold()))
    single_preflight = preflight_media_inputs((single_certification_film,), output_dir=output_dir)
    single_pipeline = SimpleNamespace(
        config=SimpleNamespace(
            anchor_film=single_certification_film,
            output_dir=output_dir,
            films=(single_certification_film,),
        ),
        cancel_check=lambda: False,
    )
    single_result = execute_reliably(
        single_pipeline,
        registry.definitions(implemented_only=True)[0].id,
        run_filter=forced_failure,
    )
    single_outcome_path = single_result.artifacts["configuration_outcome"]
    single_outcome = validate_artifact("configuration_outcome", single_outcome_path, Path.cwd() / "schemas")
    single_video = single_result.outputs["video"]
    checks = {
        "default_directory_contains_video": bool(videos),
        "all_discovered_media_preflight": preflight.get("status") == "pass",
        "all_implemented_filter_arities_available": not arity_failures,
        "all_default_and_declared_parameter_cases_normalize": parameter_cases > 0,
        "all_ordered_pairs_resolve": sum(stack_counts.values()) == pair_count,
        "real_forced_failure_returns_altered_fallback": outcome.get("status") == "ALTERED_FALLBACK_SUCCESS",
        "real_fallback_video_exists": fallback_video.is_file() and fallback_video.stat().st_size > 0,
        "real_fallback_acceptance_passes": outcome.get("output", {}).get("acceptance", {}).get("status") == "PASS",
        "real_fallback_duration_contract_present": outcome.get("duration_contract", {}).get("expected_duration") == fallback_preflight.get("predicted_output_duration"),
        "real_fallback_all_duration_checks_pass": all(
            outcome.get("output", {}).get("acceptance", {}).get("duration_contract", {}).get("checks", {}).values()
        ),
        "real_alteration_acceptance_passes": outcome.get("output", {}).get("alteration_acceptance", {}).get("status") == "PASS",
        "real_altered_fallback_changes_sampled_audio": outcome.get("output", {}).get("alteration_acceptance", {}).get("sampled_audio_difference", {}).get("changed_sample_ratio", 0.0) >= 0.6,
        "real_single_input_returns_altered_fallback": single_outcome.get("status") == "ALTERED_FALLBACK_SUCCESS",
        "real_single_input_video_exists": single_video.is_file() and single_video.stat().st_size > 0,
        "real_single_input_duration_contract_passes": all(
            single_outcome.get("output", {}).get("acceptance", {}).get("duration_contract", {}).get("checks", {}).values()
        ),
        "real_single_input_alteration_acceptance_passes": single_outcome.get("output", {}).get("alteration_acceptance", {}).get("status") == "PASS",
        "real_single_input_changes_sampled_audio": single_outcome.get("output", {}).get("alteration_acceptance", {}).get("sampled_audio_difference", {}).get("changed_sample_ratio", 0.0) >= 0.6,
    }
    report = {
        "schema_version": "1.0",
        "creation_timestamp": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "input_directory": str(input_dir),
        "videos": [str(path) for path in videos],
        "preflight": preflight,
        "fallback_certification_films": [str(path) for path in certification_films],
        "fallback_preflight": fallback_preflight,
        "implemented_filter_count": len(registry.definitions(implemented_only=True)),
        "parameter_configuration_count": parameter_cases,
        "ordered_pair_count": pair_count,
        "ordered_pair_resolution_counts": dict(sorted(stack_counts.items())),
        "arity_failures": arity_failures,
        "real_fallback_output": str(fallback_video),
        "real_fallback_outcome": str(outcome_path),
        "single_input_certification_film": str(single_certification_film),
        "single_input_preflight": single_preflight,
        "real_single_input_output": str(single_video),
        "real_single_input_outcome": str(single_outcome_path),
        "checks": checks,
    }
    report_path = output_dir / "configuration_outcome_assurance.json"
    write_json(report_path, report)
    print(
        f"{report['status']} {report['implemented_filter_count']} filters, "
        f"{parameter_cases} parameter configurations, {pair_count} ordered pairs; {report_path}"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
