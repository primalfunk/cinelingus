from __future__ import annotations

import re
from typing import Any


_TOKEN = re.compile(r"[a-z0-9']+")
_COMMON = {
    "a", "an", "and", "are", "as", "at", "be", "but", "for", "from", "he", "her",
    "his", "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "they", "this", "to", "was", "we",
    "were", "with", "you", "your",
}


def evaluate_voice_residue(*, schedule: dict[str, Any], rendered_timeline: dict[str, Any]) -> dict[str, Any]:
    """Compare rendered speech with intended donor and displaced destination words."""
    destination_regions = schedule.get("destination_speech_regions") or []
    rendered = rendered_timeline.get("windows") or rendered_timeline.get("events") or []
    mappings = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    rows = []
    for region in destination_regions:
        start, end = _bounds(region)
        if end <= start:
            continue
        overlapping_mappings = [mapping for mapping in mappings if _overlaps(start, end, *_bounds(mapping))]
        destination_text = str(region.get("transcript") or "")
        destination_tokens = _distinctive_tokens(destination_text)
        rendered_rows = [row for row in rendered if _overlaps(start, end, *_bounds(row))]
        rendered_text = " ".join(str(row.get("transcript") or row.get("text") or "") for row in rendered_rows).strip()
        donor_text = " ".join(
            str(row.get("source_transcript") or "")
            for row in overlapping_mappings
        ).strip()
        destination_similarity = _token_f1(destination_text, rendered_text)
        donor_similarity = _token_f1(donor_text, rendered_text)
        transcript_contrast = (
            len(destination_tokens) >= 3
            and len(_distinctive_tokens(rendered_text)) >= 2
            and destination_similarity >= 0.72
            and destination_similarity >= donor_similarity + 0.15
        )
        unattributed_speech = (
            not overlapping_mappings
            and bool(rendered_text)
            and (
                len(_distinctive_tokens(rendered_text)) >= 2
                or max((float(row.get("confidence", 0.0) or 0.0) for row in rendered_rows), default=0.0) >= 0.6
            )
        )
        unexpected_unmatched_speech = (
            unattributed_speech
            and len(destination_tokens) >= 2
            and destination_similarity >= 0.45
        )
        possible_residue = transcript_contrast or unexpected_unmatched_speech
        if not possible_residue and len(destination_tokens) < 3:
            continue
        rows.append({
            "destination_region_id": region.get("id"),
            "start": round(start, 3),
            "end": round(end, 3),
            "rendered_transcript": rendered_text,
            "destination_similarity": round(destination_similarity, 4),
            "donor_similarity": round(donor_similarity, 4),
            "possible_residue": possible_residue,
            "unattributed_speech_detected": unattributed_speech,
            "evidence_kind": (
                "unexpected_destination_like_speech_in_suppressed_region"
                if unexpected_unmatched_speech
                else "unattributed_speech_without_destination_match"
                if unattributed_speech and not transcript_contrast
                else "destination_transcript_contrast"
            ),
            "replacement_mapping_count": len(overlapping_mappings),
        })
    flagged = [row for row in rows if row["possible_residue"]]
    unattributed = [row for row in rows if row.get("unattributed_speech_detected")]
    status = "POSSIBLE_DESTINATION_SPEECH_DETECTED" if flagged else "NONE_DETECTED" if rows else "INCONCLUSIVE"
    return {
        "status": status,
        "method": "post_render_whisper_hybrid_residue_v2",
        "evaluated_region_count": len(rows),
        "flagged_region_count": len(flagged),
        "unattributed_speech_region_count": len(unattributed),
        "limitations": "Transcript contrast is not source-separation proof; unrelated or hallucinated speech is reported separately and is not treated as destination residue.",
        "regions": rows,
    }


def build_residue_correction_regions(
    verification: dict[str, Any],
    *,
    padding: float = 0.12,
    duration: float | None = None,
) -> list[dict[str, Any]]:
    """Turn verified residue evidence into bounded regions for one corrective render."""
    regions = []
    for index, row in enumerate(verification.get("regions", []), start=1):
        if not row.get("possible_residue"):
            continue
        start = max(0.0, float(row.get("start", 0.0) or 0.0) - max(0.0, padding))
        end = float(row.get("end", start) or start) + max(0.0, padding)
        if duration is not None:
            end = min(float(duration), end)
        if end <= start:
            continue
        regions.append({
            "id": f"residue_correction_{index:06d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "confidence": 0.95,
            "source_kind": "post_render_residue_correction",
            "recovered": True,
            "evidence_kind": row.get("evidence_kind"),
            "destination_region_id": row.get("destination_region_id"),
        })
    return regions


def unavailable_voice_residue(reason: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "method": "post_render_whisper_hybrid_residue_v2",
        "evaluated_region_count": 0,
        "flagged_region_count": 0,
        "limitations": reason,
        "regions": [],
    }


def _bounds(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row.get("start", row.get("destination_timestamp", 0.0)) or 0.0)
    end_value = row.get("end")
    if end_value is None:
        duration = row.get("planned_render_duration", row.get("clip_trim_duration", row.get("duration", 0.0)))
        end_value = start + float(duration or 0.0)
    return start, float(end_value)


def _overlaps(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return min(left_end, right_end) > max(left_start, right_start)


def _distinctive_tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _COMMON]


def _token_f1(reference: str, observed: str) -> float:
    expected = set(_distinctive_tokens(reference))
    actual = set(_distinctive_tokens(observed))
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
