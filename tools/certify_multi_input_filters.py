from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cinelingus.config import load_config
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.multi_input_guarantee import applicable_multi_input_filter_ids
from cinelingus.pipeline import Pipeline
from cinelingus.reliable_inputs import preflight_media_inputs
from cinelingus.util import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and certify every applicable filter for two or more complete inputs."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("films", type=Path, nargs="+")
    parser.add_argument("--filter", action="append", dest="filters")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len(args.films) < 2:
        parser.error("At least two films are required.")

    root = Path.cwd()
    output_root = args.output_dir.expanduser().resolve()
    preflight = preflight_media_inputs(args.films, output_dir=output_root)
    applicable = applicable_multi_input_filter_ids(len(args.films))
    selected = tuple(args.filters or applicable)
    unknown = sorted(set(selected) - set(applicable))
    if unknown:
        parser.error(
            "Filters not applicable to this input count: " + ", ".join(unknown)
        )

    summary_path = output_root / "multi_input_certification_matrix.json"
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "RUNNING",
        "film_count": len(args.films),
        "films": [str(path.expanduser().resolve()) for path in args.films],
        "preflight": preflight,
        "applicable_filter_ids": list(applicable),
        "selected_filter_ids": list(selected),
        "results": [],
    }
    write_json(summary_path, report)
    registry = default_filter_registry()
    for filter_id in selected:
        definition = registry.get(filter_id)
        filter_output = output_root / filter_id.replace(".", "_")
        config = load_config(root).with_films(args.films).with_overrides(output_dir=filter_output)
        parameters = dict(definition.parameter_defaults)
        if "intensity" in parameters:
            parameters["intensity"] = "Moderate"
        started = time.time()
        print(f"FILTER_START {filter_id}", flush=True)
        row: dict[str, Any] = {
            "filter_id": filter_id,
            "started_at_epoch": started,
            "output_dir": str(filter_output),
        }
        try:
            result = Pipeline(config).execute_transformation(
                filter_id,
                force=args.force,
                parameters=parameters,
            )
            guarantee = _artifact(result.artifacts.get("multi_input_guarantee"))
            certification = _artifact(result.artifacts.get("filter_certification"))
            render_acceptance = _artifact(result.artifacts.get("montage_render_acceptance"))
            filter_acceptance = _artifact(result.artifacts.get("filter_acceptance"))
            passed = bool(
                guarantee
                and guarantee.get("status") == "READY_TO_RENDER"
                and certification
                and certification.get("state") in {"CERTIFIED", "DEGRADED"}
                and render_acceptance
                and render_acceptance.get("acceptance_status") == "PASS"
                and filter_acceptance
                and filter_acceptance.get("status") == "pass"
            )
            row.update({
                "status": "PASS" if passed else "FAIL",
                "guarantee_status": guarantee.get("status") if guarantee else None,
                "certification_state": certification.get("state") if certification else None,
                "render_acceptance": render_acceptance.get("acceptance_status") if render_acceptance else None,
                "filter_acceptance": filter_acceptance.get("status") if filter_acceptance else None,
                "outputs": {key: str(value) for key, value in result.outputs.items()},
                "artifacts": {key: str(value) for key, value in result.artifacts.items()},
            })
        except Exception as exc:
            row.update({
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        row["elapsed_seconds"] = round(time.time() - started, 3)
        report["results"].append(row)
        write_json(summary_path, report)
        print(f"FILTER_END {filter_id} {row['status']} {row['elapsed_seconds']}", flush=True)

    report["status"] = (
        "PASS"
        if len(report["results"]) == len(selected)
        and all(row["status"] == "PASS" for row in report["results"])
        else "FAIL"
    )
    write_json(summary_path, report)
    print("MATRIX_RESULT " + json.dumps({
        "status": report["status"],
        "summary": str(summary_path),
        "passed": sum(row["status"] == "PASS" for row in report["results"]),
        "total": len(selected),
    }), flush=True)
    return 0 if report["status"] == "PASS" else 1


def _artifact(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    artifact_path = Path(path)
    return read_json(artifact_path) if artifact_path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
