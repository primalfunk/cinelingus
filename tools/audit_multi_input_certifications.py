from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cinelingus.multi_input_guarantee import applicable_multi_input_filter_ids
from cinelingus.util import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit existing evidence for every filter applicable to a multi-input experiment."
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_root = args.output_dir.expanduser().resolve()

    contract_paths = sorted(output_root.glob("multiworld_*/contracts/*/run_contract.json"))
    if not contract_paths:
        parser.error(f"No multi-input contracts found beneath {output_root}")
    first_contract = read_json(contract_paths[0])
    film_count = len(first_contract.get("media") or [])
    applicable = applicable_multi_input_filter_ids(film_count)
    rows = [_audit_filter(output_root, filter_id, first_contract) for filter_id in applicable]
    report = {
        "schema_version": "1.0",
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "film_count": film_count,
        "films": [row["path"] for row in first_contract["media"]],
        "media_hashes": [row["media_hash"] for row in first_contract["media"]],
        "applicable_filter_ids": list(applicable),
        "results": rows,
    }
    output_path = output_root / "multi_input_certification_audit.json"
    write_json(output_path, report)
    print(f"{report['status']} {sum(row['status'] == 'PASS' for row in rows)}/{len(rows)} {output_path}")
    return 0 if report["status"] == "PASS" else 1


def _audit_filter(output_root: Path, filter_id: str, reference_contract: dict[str, Any]) -> dict[str, Any]:
    folder = output_root / filter_id.replace(".", "_")
    contract_dir = folder / "contracts" / filter_id.replace(".", "_")
    paths = {
        "contract": contract_dir / "run_contract.json",
        "qualification": contract_dir / "schedule_qualification.json",
        "guarantee": contract_dir / "multi_input_guarantee.json",
        "certification": contract_dir / "filter_certification.json",
    }
    recursive = {
        "filter_acceptance": _one(folder, "filter_acceptance.json"),
        "render_acceptance": _one(folder, "montage_render_acceptance.json"),
        "schedule": _one(folder, "replacement_decisions.json"),
    }
    required = {**paths, "filter_acceptance": recursive["filter_acceptance"], "render_acceptance": recursive["render_acceptance"]}
    if filter_id == "multiworld.echo_chamber":
        required["schedule"] = recursive["schedule"]
    missing = [name for name, path in required.items() if path is None or not path.exists()]
    if missing:
        return {"filter_id": filter_id, "status": "FAIL", "reasons": ["Missing artifacts: " + ", ".join(missing)]}

    contract = read_json(paths["contract"])
    qualification = read_json(paths["qualification"])
    guarantee = read_json(paths["guarantee"])
    certification = read_json(paths["certification"])
    filter_acceptance = read_json(recursive["filter_acceptance"])
    render_acceptance = read_json(recursive["render_acceptance"])
    schedule = read_json(recursive["schedule"]) if recursive["schedule"] is not None else {}
    expected_media = [(row["path"], row["media_hash"]) for row in reference_contract["media"]]
    actual_media = [(row["path"], row["media_hash"]) for row in contract["media"]]
    invariant_rows = filter_acceptance.get("invariants") or []
    checks = {
        "same_input_world": actual_media == expected_media,
        "contract_filter_matches": contract.get("filter_id") == filter_id,
        "complete_media_scope": contract.get("input_scope") == "complete_media_files",
        "qualification_passed": qualification.get("status") in {"READY", "DEGRADED"} and _all_true(qualification.get("checks")),
        "guarantee_passed": guarantee.get("status") == "READY_TO_RENDER" and _all_true(guarantee.get("checks")),
        "certification_passed": certification.get("state") in {"CERTIFIED", "DEGRADED"} and _all_true(certification.get("checks")),
        "filter_acceptance_passed": filter_acceptance.get("status") == "pass",
        "filter_invariants_passed": bool(invariant_rows) and all(row.get("passed") is True for row in invariant_rows),
        "render_acceptance_passed": render_acceptance.get("acceptance_status") == "PASS" and _all_true(render_acceptance.get("checks")),
        "canonical_duration_matches": abs(float(render_acceptance.get("encoded_duration", -1)) - float(contract["timeline"]["output_duration"])) <= float(contract["acceptance"]["duration_tolerance_seconds"]),
        "final_video_exists": _final_video_exists(filter_acceptance),
        "post_qualification_law_holds": _post_qualification_law(filter_id, schedule, contract),
    }
    return {
        "filter_id": filter_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "certification_state": certification.get("state"),
        "qualification_status": qualification.get("status"),
        "canonical_output_duration": contract["timeline"]["output_duration"],
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "artifacts": {name: str(path) for name, path in {**paths, **recursive}.items()},
    }


def _post_qualification_law(filter_id: str, schedule: dict[str, Any], contract: dict[str, Any]) -> bool:
    if filter_id != "multiworld.echo_chamber":
        return True
    required = {row["media_hash"] for row in contract["media"]}
    groups: dict[str, set[str]] = {}
    for row in schedule.get("mappings") or []:
        if not row.get("enabled", True):
            continue
        group_id = str(row.get("echo_group_id") or "")
        source_hash = str(row.get("source_media_hash") or "")
        if group_id and source_hash:
            groups.setdefault(group_id, set()).add(source_hash)
    return bool(groups) and all(hashes == required for hashes in groups.values())


def _all_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def _final_video_exists(filter_acceptance: dict[str, Any]) -> bool:
    path = (filter_acceptance.get("outputs") or {}).get("final_video")
    return bool(path and Path(path).exists())


def _one(root: Path, filename: str) -> Path | None:
    rows = sorted(root.rglob(filename))
    return rows[0] if len(rows) == 1 else None


if __name__ == "__main__":
    raise SystemExit(main())
