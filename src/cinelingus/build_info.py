from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .diarization_runtime import DIARIZATION_SCHEMA_VERSION

HEARTBEAT_IMPLEMENTATION_VERSION = "2.0"


def build_identification(root: Path) -> dict[str, str]:
    timestamp = datetime.fromtimestamp(Path(__file__).stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    return {
        "cinelingus_build_version": __version__,
        "git_commit_or_build_timestamp": os.environ.get("CINELINGUS_GIT_COMMIT") or timestamp,
        "diarization_schema_version": DIARIZATION_SCHEMA_VERSION,
        "heartbeat_implementation_version": HEARTBEAT_IMPLEMENTATION_VERSION,
    }


def format_build_identification(root: Path) -> list[str]:
    info = build_identification(root)
    return [
        f"Cinelingus build/version: {info['cinelingus_build_version']}",
        f"Git commit or build timestamp: {info['git_commit_or_build_timestamp']}",
        f"Diarization schema version: {info['diarization_schema_version']}",
        f"Heartbeat implementation version: {info['heartbeat_implementation_version']}",
    ]
