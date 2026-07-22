from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..util import stable_hash
from .bundle import USABLE_CLASSIFICATION_STATES, validate_function_bundle


class FunctionMode(str, Enum):
    DISABLED = "FUNCTION_DISABLED"
    REPORT_ONLY = "FUNCTION_REPORT_ONLY"
    ASSISTED = "FUNCTION_ASSISTED"
    PRESERVING = "FUNCTION_PRESERVING"


@dataclass(frozen=True)
class FunctionScheduleContext:
    mode: FunctionMode
    weight: float
    source_by_reference: dict[str, dict[str, Any]]
    destination_by_reference: dict[str, dict[str, Any]]
    identity: dict[str, Any]
    minimum_confidence: float = 0.62
    source_by_start: dict[str, tuple[dict[str, Any], ...]] | None = None
    destination_by_start: dict[str, tuple[dict[str, Any], ...]] | None = None
    source_by_text: dict[str, tuple[dict[str, Any], ...]] | None = None
    destination_by_text: dict[str, tuple[dict[str, Any], ...]] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", FunctionMode(self.mode))
        if not 0.0 <= self.weight <= 1.0 or not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("Function scheduling weights and confidence must be unit interval values")
        if self.mode in {FunctionMode.DISABLED, FunctionMode.REPORT_ONLY} and self.weight != 0.0:
            raise ValueError("Disabled and report-only function modes cannot influence ranking")

    @property
    def active(self) -> bool:
        return self.mode is FunctionMode.REPORT_ONLY or (self.mode in {FunctionMode.ASSISTED, FunctionMode.PRESERVING} and self.weight > 0.0)

    @classmethod
    def from_bundles(
        cls, *, mode: FunctionMode, weight: float,
        source_model: dict[str, Any], source_bundle: dict[str, Any],
        destination_model: dict[str, Any], destination_bundle: dict[str, Any],
        minimum_confidence: float = 0.62,
    ) -> "FunctionScheduleContext":
        for label, model, bundle in (("source", source_model, source_bundle), ("destination", destination_model, destination_bundle)):
            validation = validate_function_bundle(bundle, model)
            if validation["status"] != "VALID":
                raise ValueError(f"{label} dialogue-function bundle is invalid: {validation['errors']}")
        source_identity = _bundle_identity(source_bundle)
        destination_identity = _bundle_identity(destination_bundle)
        if source_identity != destination_identity:
            raise ValueError("Source and destination dialogue-function bundles are incompatible")
        return cls(
            mode, weight,
            _index_by_reference(source_model, source_bundle), _index_by_reference(destination_model, destination_bundle),
            {**source_identity, "source_bundle_signature": stable_hash(source_bundle), "destination_bundle_signature": stable_hash(destination_bundle)},
            minimum_confidence,
            _index_by_start(source_model, source_bundle), _index_by_start(destination_model, destination_bundle),
            _index_by_text(source_model, source_bundle), _index_by_text(destination_model, destination_bundle),
        )

    def annotate_clips(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._annotate(row, self.source_by_reference, self.source_by_start or {}, self.source_by_text or {}) for row in rows]

    def annotate_windows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            item = dict(row)
            children = [self._annotate(child, self.destination_by_reference, self.destination_by_start or {}, self.destination_by_text or {}) for child in row.get("speech_windows") or []]
            if children:
                item["speech_windows"] = children
                item = inherit_aggregate_functions(item, children)
            else:
                item = self._annotate(item, self.destination_by_reference, self.destination_by_start or {}, self.destination_by_text or {})
            result.append(item)
        return result

    def annotate_source_performance_groups(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            item = dict(row)
            children = [self._annotate(child, self.source_by_reference, self.source_by_start or {}, self.source_by_text or {}) for child in row.get("clips") or []]
            item["clips"] = children
            result.append(inherit_aggregate_functions(item, children))
        return result

    def _annotate(
        self, row: dict[str, Any], by_reference: dict[str, dict[str, Any]],
        by_start: dict[str, tuple[dict[str, Any], ...]], by_text: dict[str, tuple[dict[str, Any], ...]],
    ) -> dict[str, Any]:
        item = dict(row)
        references = [str(row.get("id") or ""), str(row.get("event_id") or ""), *[str(value) for value in row.get("event_ids") or []]]
        evidence = [by_reference[value] for value in references if value in by_reference]
        scope = "direct_passage" if evidence else "unavailable"
        if not evidence:
            evidence = list(by_start.get(_start_key(row), ()))
            scope = "direct_passage_start_bridge" if evidence else scope
        if not evidence:
            text = _normalize_text(row.get("transcript") or row.get("original_transcript"))
            if len(text.split()) >= 2:
                evidence = list(by_text.get(text, ()))
                scope = "direct_passage_text_bridge" if evidence else scope
        if evidence:
            item["_function_classification"] = _aggregate_classifications(evidence)
            item["_function_entity_ids"] = [str(value["source_entity_id"]) for value in evidence]
            item["_function_evidence_scope"] = scope
        item["_function_mode"] = self.mode.value
        item["_function_weight"] = self.weight
        item["_function_minimum_confidence"] = self.minimum_confidence
        item["_function_identity"] = self.identity
        return item


def inherit_aggregate_functions(target: dict[str, Any], children: list[dict[str, Any]]) -> dict[str, Any]:
    item = dict(target)
    evidence = [child for child in children if child.get("_function_classification")]
    if not evidence:
        return item
    item["_function_classification"] = _aggregate_classifications([
        {"source_entity_id": entity_id, "classification": child["_function_classification"]}
        for child in evidence for entity_id in child.get("_function_entity_ids") or [str(child.get("id") or "aggregate")]
    ])
    item["_function_entity_ids"] = [entity_id for child in evidence for entity_id in child.get("_function_entity_ids") or []]
    item["_function_evidence_scope"] = "passage_aggregate"
    for key in ("_function_mode", "_function_weight", "_function_minimum_confidence", "_function_identity"):
        item[key] = evidence[0].get(key)
    return item


def dialogue_function_compatibility(source: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any] | None:
    mode_value = source.get("_function_mode") or destination.get("_function_mode")
    if not mode_value:
        return None
    mode = FunctionMode(mode_value)
    weight = float(source.get("_function_weight", destination.get("_function_weight", 0.0)) or 0.0)
    minimum_confidence = float(source.get("_function_minimum_confidence", destination.get("_function_minimum_confidence", 0.62)) or 0.62)
    source_evidence = source.get("_function_classification")
    destination_evidence = destination.get("_function_classification")
    available = bool(source_evidence and destination_evidence)
    warnings: list[str] = []
    axes: dict[str, Any] = {}
    if available:
        for axis in ("surface_form", "interaction_function", "sequence_position"):
            axes[axis] = _axis_compatibility(source_evidence, destination_evidence, axis, minimum_confidence)
        supported = [row for row in axes.values() if row["supported"]]
        normalized = sum(row["compatibility"] * row["axis_weight"] for row in supported) / sum(row["axis_weight"] for row in supported) if supported else 0.5
        confidence = min(float(source_evidence.get("confidence", 0.0)), float(destination_evidence.get("confidence", 0.0)))
        ambiguous = source_evidence.get("ambiguity_state") == "AMBIGUOUS" or destination_evidence.get("ambiguity_state") == "AMBIGUOUS"
        if confidence < minimum_confidence or ambiguous:
            normalized = 0.5 + (normalized - 0.5) * max(0.0, confidence / max(minimum_confidence, 1e-9)) * (0.5 if ambiguous else 1.0)
            warnings.append("Function evidence is ambiguous or below calibrated confidence; contribution is weakened toward neutral.")
    else:
        normalized, confidence, ambiguous = 0.5, 0.0, False
        warnings.append("Source or destination dialogue-function evidence is unavailable; existing score preserved.")
    effective = weight * normalized if available and mode in {FunctionMode.ASSISTED, FunctionMode.PRESERVING} else 0.0
    return {
        "source_function_entity_ids": list(source.get("_function_entity_ids") or []),
        "destination_function_entity_ids": list(destination.get("_function_entity_ids") or []),
        "source_distribution": source_evidence,
        "destination_distribution": destination_evidence,
        "per_axis_compatibility": axes,
        "normalized_function_contribution": round(float(normalized), 6),
        "confidence": round(float(confidence), 4), "ambiguity": ambiguous,
        "configured_weight": weight, "effective_weighted_contribution": round(effective, 6),
        "taxonomy_and_classifier_identity": source.get("_function_identity") or destination.get("_function_identity") or {},
        "source_evidence_scope": source.get("_function_evidence_scope", "unavailable"),
        "destination_evidence_scope": destination.get("_function_evidence_scope", "unavailable"),
        "fallback_state": "NONE" if available else "NEUTRAL_EXISTING_SCORE",
        "warnings": warnings, "mode": mode.value, "available": available,
        "scoring_policy": "dialogue_function_axis_compatibility_v1",
    }


def apply_function_contribution(base_score: float, compatibility: dict[str, Any] | None) -> float:
    if not compatibility or not compatibility["available"] or compatibility["mode"] not in {FunctionMode.ASSISTED.value, FunctionMode.PRESERVING.value}:
        return base_score
    weight = float(compatibility["configured_weight"])
    return base_score if weight <= 0.0 else base_score * (1.0 - weight) + float(compatibility["normalized_function_contribution"]) * weight


def _axis_compatibility(source: dict[str, Any], destination: dict[str, Any], axis: str, minimum_confidence: float) -> dict[str, Any]:
    source_axis = (source.get("axes") or {}).get(axis) or {}
    destination_axis = (destination.get("axes") or {}).get(axis) or {}
    unsupported = not source_axis.get("supported") or not destination_axis.get("supported")
    source_labels = _label_distribution(source_axis)
    destination_labels = _label_distribution(destination_axis)
    state_labels = {"unknown", "ambiguous", "not_applicable", "unavailable"}
    if unsupported or not source_labels or not destination_labels or set(source_labels) <= state_labels or set(destination_labels) <= state_labels:
        return {"supported": False, "compatibility": 0.5, "axis_weight": 0.0 if axis == "sequence_position" else (0.3 if axis == "surface_form" else 0.6), "reason": "UNAVAILABLE_OR_STATE_ONLY"}
    overlap = sum(min(source_labels.get(label, 0.0), destination_labels.get(label, 0.0)) for label in set(source_labels) | set(destination_labels) if label not in state_labels)
    union = sum(max(source_labels.get(label, 0.0), destination_labels.get(label, 0.0)) for label in set(source_labels) | set(destination_labels) if label not in state_labels)
    score = overlap / union if union else 0.5
    if axis == "surface_form" and score == 0.0:
        score = 0.2
    return {"supported": True, "compatibility": round(score, 6), "axis_weight": {"surface_form": 0.3, "interaction_function": 0.6, "sequence_position": 0.1}[axis], "reason": "DISTRIBUTION_OVERLAP", "source_labels": source_labels, "destination_labels": destination_labels}


def _label_distribution(axis: dict[str, Any]) -> dict[str, float]:
    return {str(row.get("label")): float(row.get("confidence", 0.0) or 0.0) for row in axis.get("labels") or []}


def _aggregate_classifications(records: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = [row.get("classification") or {} for row in records]
    axes = {}
    for axis in ("surface_form", "interaction_function", "sequence_position"):
        values: dict[str, list[float]] = {}
        supported = False
        for classification in classifications:
            axis_row = (classification.get("axes") or {}).get(axis) or {}
            supported = supported or bool(axis_row.get("supported"))
            for label in axis_row.get("labels") or []:
                values.setdefault(str(label.get("label")), []).append(float(label.get("confidence", 0.0) or 0.0))
        labels = [{"label": label, "label_id": f"{axis}.{label}", "confidence": round(sum(scores) / len(classifications), 4)} for label, scores in values.items()]
        labels.sort(key=lambda row: (-row["confidence"], row["label"]))
        axes[axis] = {"labels": labels, "supported": supported, "multi_label": axis == "interaction_function"}
    confidence = sum(float(value.get("confidence", 0.0) or 0.0) for value in classifications) / len(classifications)
    return {"axes": axes, "confidence": round(confidence, 4), "ambiguity_state": "AMBIGUOUS" if any(value.get("ambiguity_state") == "AMBIGUOUS" for value in classifications) else "UNAMBIGUOUS", "abstention": {"abstained": all((value.get("abstention") or {}).get("abstained") for value in classifications)}}


def _index_by_reference(model: dict[str, Any], bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = _usable_records(bundle)
    return {str(passage["source_transcript_reference"]): records[str(passage["speech_passage_id"])] for passage in model.get("speech_passages") or [] if passage.get("source_transcript_reference") and str(passage["speech_passage_id"]) in records}


def _index_by_start(model: dict[str, Any], bundle: dict[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    records = _usable_records(bundle); grouped: dict[str, list[dict[str, Any]]] = {}
    for passage in model.get("speech_passages") or []:
        record = records.get(str(passage.get("speech_passage_id")))
        if record: grouped.setdefault(_start_key(passage), []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def _index_by_text(model: dict[str, Any], bundle: dict[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    records = _usable_records(bundle); grouped: dict[str, list[dict[str, Any]]] = {}
    for passage in model.get("speech_passages") or []:
        record = records.get(str(passage.get("speech_passage_id"))); text = _normalize_text(passage.get("original_transcript"))
        if record and text: grouped.setdefault(text, []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def _usable_records(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["source_entity_id"]): row for row in bundle.get("entities") or [] if row.get("classification_state") in USABLE_CLASSIFICATION_STATES}


def _bundle_identity(bundle: dict[str, Any]) -> dict[str, Any]:
    return {"taxonomy_version": bundle.get("taxonomy_version"), "taxonomy_signature": bundle.get("taxonomy_signature"), "classifier_version": (bundle.get("classifier") or {}).get("classifier_version"), "configuration_signature": bundle.get("configuration_signature")}


def _start_key(row: dict[str, Any]) -> str:
    return f"{float(row.get('movie_timestamp', row.get('start', 0.0)) or 0.0):.3f}"


def _normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(value or "").lower()))
