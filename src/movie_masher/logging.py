from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunLogger:
    log_path: Path

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def error(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] ERROR {message}"
        print(line, file=sys.stderr)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
