from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .filter_lab.registry import default_filter_registry
from .util import read_json, utc_now, write_json


LOCK_FILENAME = ".cinelingus-active-run.json"


class RunInProgressError(RuntimeError):
    pass


class FilterExecutionMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class RunLease:
    output_dir: Path
    filter_id: str
    run_id: str
    pid: int
    started_at: str
    started_at_epoch: float
    lock_path: Path


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock(path: Path) -> dict:
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


@contextmanager
def exclusive_output_run(output_dir: Path, filter_id: str) -> Iterator[RunLease]:
    """Hold an atomic, cross-process lease for one Cinelingus output tree."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    canonical_filter_id = default_filter_registry().get(filter_id).id
    now = time.time()
    payload = {
        "schema_version": "1.0",
        "run_id": uuid.uuid4().hex,
        "filter_id": canonical_filter_id,
        "pid": os.getpid(),
        "started_at": utc_now(),
        "started_at_epoch": now,
    }

    acquired = False
    for _attempt in range(3):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_lock(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0)
            if existing and _pid_is_running(existing_pid):
                raise RunInProgressError(
                    "Another Cinelingus run is already using this output directory: "
                    f"filter={existing.get('filter_id', 'unknown')}, pid={existing_pid}, "
                    f"started_at={existing.get('started_at', 'unknown')}. Lock: {lock_path}"
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            acquired = True
            break
    if not acquired:
        raise RunInProgressError(f"Could not acquire the Cinelingus output lock: {lock_path}")

    lease = RunLease(
        output_dir=output_dir,
        filter_id=canonical_filter_id,
        run_id=str(payload["run_id"]),
        pid=int(payload["pid"]),
        started_at=str(payload["started_at"]),
        started_at_epoch=now,
        lock_path=lock_path,
    )
    try:
        yield lease
    finally:
        current = _read_lock(lock_path)
        if current.get("run_id") == lease.run_id:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def verify_filter_execution(
    lease: RunLease,
    *,
    requested_filter_id: str,
    evidence_paths: Iterable[Path],
    output: Path,
) -> Path:
    """Require current-run artifacts to identify the requested canonical filter."""
    registry = default_filter_registry()
    requested = registry.get(requested_filter_id).id
    evidence: list[dict[str, str]] = []
    observed: set[str] = set()
    for raw_path in evidence_paths:
        path = Path(raw_path)
        if not path.exists() or path.stat().st_mtime < lease.started_at_epoch:
            continue
        try:
            document = read_json(path)
            filter_id = registry.get(str(document.get("filter_id", ""))).id
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        observed.add(filter_id)
        evidence.append({"path": str(path), "filter_id": filter_id})

    status = "pass" if observed == {requested} and Path(output).exists() else "fail"
    receipt = {
        "schema_version": "1.0",
        "run_id": lease.run_id,
        "started_at": lease.started_at,
        "completed_at": utc_now(),
        "pid": lease.pid,
        "requested_filter_id": requested,
        "executed_filter_ids": sorted(observed),
        "status": status,
        "output": str(output),
        "evidence": evidence,
    }
    receipt_path = lease.output_dir / "run_receipts" / f"{lease.run_id}.json"
    write_json(receipt_path, receipt)
    if status != "pass":
        detail = ", ".join(sorted(observed)) or "no fresh filter identity evidence"
        raise FilterExecutionMismatch(
            f"Requested filter {requested}, but the completed run reported {detail}. "
            f"The output was not accepted. Receipt: {receipt_path}"
        )
    return receipt_path
