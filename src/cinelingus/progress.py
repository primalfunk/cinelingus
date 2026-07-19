from __future__ import annotations

from dataclasses import dataclass, asdict
import time


@dataclass
class ProgressState:
    stage_id: str
    stage_label: str
    current: int = 0
    total: int = 0
    started_at: float = 0.0
    status_message: str = ""

    @classmethod
    def start(cls, stage_id: str, stage_label: str, *, total: int = 0, status_message: str = "") -> "ProgressState":
        return cls(stage_id=stage_id, stage_label=stage_label, total=max(0, int(total or 0)), started_at=time.time(), status_message=status_message)

    def update(
        self,
        *,
        current: int | None = None,
        total: int | None = None,
        stage_id: str | None = None,
        stage_label: str | None = None,
        status_message: str | None = None,
    ) -> "ProgressState":
        if current is not None:
            self.current = max(0, int(current))
        if total is not None:
            self.total = max(0, int(total))
        if stage_id is not None:
            self.stage_id = stage_id
        if stage_label is not None:
            self.stage_label = stage_label
        if status_message is not None:
            self.status_message = status_message
        return self

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at) if self.started_at else 0.0

    @property
    def percent(self) -> int | None:
        if self.total <= 0:
            return None
        return round(min(100.0, max(0.0, self.current / self.total * 100.0)))

    @property
    def estimated_remaining_seconds(self) -> float | None:
        if self.total <= 0 or self.current <= 0:
            return None
        rate = self.elapsed_seconds / self.current
        return max(0.0, rate * (self.total - self.current))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["percent"] = self.percent
        data["elapsed_seconds"] = round(self.elapsed_seconds, 3)
        remaining = self.estimated_remaining_seconds
        data["estimated_remaining_seconds"] = round(remaining, 3) if remaining is not None else None
        return data


def format_progress_status(state: ProgressState) -> str:
    percent = "Estimating" if state.percent is None else f"{state.percent}% complete"
    remaining = state.estimated_remaining_seconds
    eta = "Estimating" if remaining is None else _format_duration(remaining)
    total = f" {state.current}/{state.total}" if state.total else ""
    detail = f" {state.status_message}" if state.status_message else ""
    return f"{state.stage_label}{total} | {percent} | Elapsed: {_format_duration(state.elapsed_seconds)} | Remaining: {eta}{detail}"


def _format_duration(seconds: float | None) -> str:
    total = max(0, int(seconds or 0))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
