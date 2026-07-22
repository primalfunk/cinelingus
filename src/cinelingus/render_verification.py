from __future__ import annotations

import re
from typing import Any


TOKEN = re.compile(r"[a-z0-9']+")
RENDER_VERIFICATION_VERSION = "rendered_dialogue_verification_v4_expressive_elongation"


def evaluate_rendered_dialogue(
    *,
    schedule: dict[str, Any],
    rendered_timeline: dict[str, Any],
) -> dict[str, Any]:
    rendered = rendered_timeline.get("windows") or rendered_timeline.get("events") or []
    mappings = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    rows = []
    for index, mapping in enumerate(mappings):
        start, end = _bounds(mapping)
        intended = str(mapping.get("source_transcript") or "").strip()
        expected = _tokens(intended)
        observed_rows = [row for row in rendered if _overlap(start, end, *_bounds(row))]
        observed_text = " ".join(str(row.get("transcript") or row.get("text") or "") for row in observed_rows).strip()
        observed = _tokens(observed_text)
        coverage = _ordered_coverage(expected, observed)
        beginning_present = _contains_ordered(expected[: min(2, len(expected))], observed)
        ending_present = _contains_ordered(expected[-min(2, len(expected)):], observed)
        aligned_start, aligned_end = _contiguous_span(expected, observed)
        adjacent_before = aligned_start is not None and aligned_start > 0
        adjacent_after = aligned_end is not None and aligned_end < len(observed)
        confidence = max((float(row.get("confidence", 0.0) or 0.0) for row in observed_rows), default=0.0)
        # Whisper windows often straddle neighboring edits. Only tokens attributable
        # to the intended line belong in its delivery-rate estimate.
        observed_rate = (coverage * len(expected)) / max(end - start, 0.001)
        labels = []
        _label(labels, "audio", "sentence_beginning", beginning_present, confidence)
        _label(labels, "audio", "sentence_ending", ending_present, confidence)
        _label(labels, "audio", "word_coverage", coverage >= 0.72, confidence, score=coverage)
        _label(labels, "audio", "speaking_rate", 0.45 <= observed_rate <= 4.2 if observed else False, confidence, score=_rate_score(observed_rate))
        _label(labels, "audio", "intelligibility", confidence >= 0.45 and coverage >= 0.6, confidence, score=min(confidence, coverage))
        if adjacent_before or adjacent_after:
            labels.append({
                "domain": "editing",
                "label": "adjacent_dialogue_overlap",
                "status": "warning",
                "score": round((int(adjacent_before) + int(adjacent_after)) / 2.0, 4),
                "confidence": round(confidence, 4),
            })
        missing_beginning = bool(expected) and not beginning_present
        missing_ending = bool(expected) and not ending_present
        mid_word_cut = missing_ending and bool(observed) and intended[-1:] not in {".", "!", "?", "…"}
        fade_masking = (missing_beginning or missing_ending) and any(row.get("operation") == "fade_in_out" for row in mapping.get("render_operations", []))
        audio_masking = bool(observed) and confidence < 0.45
        incomplete_phrase = bool(expected) and (coverage < 0.72 or missing_beginning or missing_ending)
        rows.append({
            "mapping_index": index,
            "mapping_id": mapping.get("id"),
            "editorial_placement_id": mapping.get("editorial_placement_id"),
            "window_id": mapping.get("window_id"),
            "clip_id": mapping.get("clip_id"),
            "destination_start": round(start, 3),
            "destination_end": round(end, 3),
            "intended_transcript": intended,
            "rendered_transcript": observed_text,
            "word_coverage_percentage": round(coverage * 100.0, 2),
            "missing_sentence_beginning": missing_beginning,
            "missing_sentence_ending": missing_ending,
            "unexpected_word_loss": round(max(0.0, 1.0 - coverage), 4),
            "mid_word_cut": mid_word_cut,
            "incomplete_phrase": incomplete_phrase,
            "speaking_rate_anomaly": bool(observed) and not 0.45 <= observed_rate <= 4.2,
            "observed_words_per_second": round(observed_rate, 3),
            "fade_masking_possible": fade_masking,
            "audio_masking_possible": audio_masking,
            "adjacent_dialogue_before": adjacent_before,
            "adjacent_dialogue_after": adjacent_after,
            "poor_intelligibility": confidence < 0.45 or coverage < 0.6,
            "confidence": round(confidence, 4),
            "labels": labels,
            "status": (
                "fail" if coverage < 0.6 or (missing_beginning and missing_ending)
                else "warning" if incomplete_phrase or audio_masking or adjacent_before or adjacent_after
                else "pass"
            ),
        })
    failures = [row for row in rows if row["status"] == "fail"]
    warnings = [row for row in rows if row["status"] == "warning"]
    measurable = [row for row in rows if row["intended_transcript"]]
    return {
        "schema_version": "1.0",
        "verification_version": RENDER_VERIFICATION_VERSION,
        "status": "FAIL" if failures else "WARN" if warnings else "PASS" if measurable else "INCONCLUSIVE",
        "mapping_count": len(rows),
        "measurable_mapping_count": len(measurable),
        "failed_mapping_count": len(failures),
        "warning_mapping_count": len(warnings),
        "average_word_coverage_percentage": round(sum(row["word_coverage_percentage"] for row in measurable) / max(len(measurable), 1), 2),
        "limitations": "Transcript alignment is conservative and cannot prove acoustic source separation or literal phoneme-boundary integrity. Neighboring words inside broad Whisper windows are labeled as adjacent overlap, not line truncation.",
        "mappings": rows,
    }


def merge_rendered_dialogue_verification(
    baseline: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Merge targeted re-verification rows into a complete-film report."""
    rows = [dict(row) for row in baseline.get("mappings", [])]
    replacement_by_key = {_row_key(row): dict(row) for row in replacement.get("mappings", [])}
    merged = []
    consumed: set[str] = set()
    for row in rows:
        key = _row_key(row)
        if key in replacement_by_key:
            merged.append(replacement_by_key[key])
            consumed.add(key)
        else:
            merged.append(row)
    merged.extend(row for key, row in replacement_by_key.items() if key not in consumed)
    measurable = [row for row in merged if row.get("intended_transcript")]
    failures = [row for row in merged if row.get("status") == "fail"]
    warnings = [row for row in merged if row.get("status") == "warning"]
    return {
        **{key: value for key, value in baseline.items() if key != "mappings"},
        "verification_version": RENDER_VERIFICATION_VERSION,
        "status": "FAIL" if failures else "WARN" if warnings else "PASS" if measurable else "INCONCLUSIVE",
        "mapping_count": len(merged),
        "measurable_mapping_count": len(measurable),
        "failed_mapping_count": len(failures),
        "warning_mapping_count": len(warnings),
        "average_word_coverage_percentage": round(
            sum(float(row.get("word_coverage_percentage", 0.0) or 0.0) for row in measurable)
            / max(len(measurable), 1),
            2,
        ),
        "mappings": merged,
    }


def unavailable_rendered_dialogue_verification(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "verification_version": RENDER_VERIFICATION_VERSION,
        "status": "UNAVAILABLE",
        "mapping_count": 0,
        "measurable_mapping_count": 0,
        "failed_mapping_count": 0,
        "warning_mapping_count": 0,
        "average_word_coverage_percentage": 0.0,
        "limitations": reason,
        "mappings": [],
    }


def _label(rows: list[dict[str, Any]], domain: str, label: str, passed: bool, confidence: float, score: float | None = None) -> None:
    rows.append({"domain": domain, "label": label, "status": "pass" if passed else "fail", "score": round(float(score if score is not None else passed), 4), "confidence": round(confidence, 4)})


def _ordered_coverage(expected: list[str], observed: list[str]) -> float:
    if not expected:
        return 0.0
    prior = [0] * (len(observed) + 1)
    for token in expected:
        current = [0]
        for index, actual in enumerate(observed, start=1):
            current.append(prior[index - 1] + 1 if token == actual else max(current[-1], prior[index]))
        prior = current
    return prior[-1] / len(expected)


def _contains_ordered(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False
    cursor = 0
    for token in haystack:
        if token == needle[cursor]:
            cursor += 1
            if cursor == len(needle):
                return True
    return False


def _contiguous_span(expected: list[str], observed: list[str]) -> tuple[int | None, int | None]:
    """Locate a complete intended line when Whisper retained it contiguously."""
    if not expected or len(expected) > len(observed):
        return None, None
    for start in range(len(observed) - len(expected) + 1):
        if observed[start:start + len(expected)] == expected:
            return start, start + len(expected)
    return None, None


def _rate_score(value: float) -> float:
    if 0.45 <= value <= 4.2:
        return 1.0
    return max(0.0, 1.0 - min(abs(value - 0.45), abs(value - 4.2)) / 4.2)


def _tokens(text: str) -> list[str]:
    return [_normalize_expressive_elongation(token) for token in TOKEN.findall(text.lower())]


def lexically_equivalent_transcript(intended: Any, observed: Any) -> bool:
    """Accept a transcript identity or one bounded breathy terminal suffix.

    This handles short-clip Whisper spellings such as ``Sorryhh`` while keeping
    substitutions and additional words distinct.
    """
    expected, actual = tuple(_tokens(str(intended or ""))), tuple(_tokens(str(observed or "")))
    if expected == actual:
        return bool(expected)
    if len(expected) == len(actual) and expected and all(
        _phonetic_spelling(token) == _phonetic_spelling(other)
        for token, other in zip(expected, actual)
    ):
        return True
    if len(expected) != 1 or len(actual) != 1:
        return False
    suffix = actual[0][len(expected[0]):] if actual[0].startswith(expected[0]) else ""
    return 2 <= len(suffix) <= 3 and len(set(suffix)) == 1 and suffix.isalpha()


def _phonetic_spelling(token: str) -> str:
    """Normalize one narrow family of common phonetic ASR spellings."""
    value = re.sub(r"^c(?=[eiy])", "s", token)
    return re.sub(r"(?<=^[cs])y(?=[a-z])", "i", value)


def _normalize_expressive_elongation(token: str) -> str:
    """Normalize emphatic ASR spellings without altering ordinary doubles.

    Whisper can render a sustained cinematic interjection such as ``Ah!`` as
    ``AAAAAAAH!``. Runs of three or more identical characters carry duration,
    not a different lexical token. Ordinary spellings such as ``book`` remain
    untouched.
    """
    return re.sub(r"(.)\1{2,}", r"\1", token)


def _bounds(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row.get("start", row.get("destination_timestamp", 0.0)) or 0.0)
    end = row.get("end")
    if end is None:
        end = start + float(row.get("planned_render_duration", row.get("duration", 0.0)) or 0.0)
    return start, float(end)


def _overlap(a: float, b: float, c: float, d: float) -> bool:
    return min(b, d) > max(a, c)


def _row_key(row: dict[str, Any]) -> str:
    # A repair changes the donor mapping id but not the destination placement.
    # Window identity therefore keeps targeted re-verification replace-in-place.
    return str(
        row.get("editorial_placement_id")
        or row.get("mapping_id")
        or row.get("window_id")
        or f"index:{row.get('mapping_index', -1)}"
    )
