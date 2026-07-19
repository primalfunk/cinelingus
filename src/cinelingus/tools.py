from __future__ import annotations

import json
import subprocess
from pathlib import Path


class ToolError(RuntimeError):
    pass


def run(args: list[str], *, cwd: Path | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr or result.stdout or "no process output"
        raise ToolError(f"Command failed ({result.returncode}): {' '.join(args)}\n{detail}")
    return result


def ffprobe_json(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(result.stdout)
