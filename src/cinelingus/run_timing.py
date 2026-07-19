from __future__ import annotations


def estimate_overall_remaining(elapsed_seconds: float, overall_percent: float) -> float | None:
    percent = max(0.0, min(100.0, float(overall_percent)))
    if percent < 1.0 or elapsed_seconds <= 0.0:
        return None
    return max(0.0, float(elapsed_seconds) * (100.0 - percent) / percent)


def completed_stage_text(label: str, elapsed_seconds: float) -> str:
    total = max(0, int(round(elapsed_seconds)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
    return f"[x] {label} - {duration}"
