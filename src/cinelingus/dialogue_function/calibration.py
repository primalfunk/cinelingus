from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..util import read_json, stable_hash, utc_now, write_json
from .taxonomy import load_taxonomy

CALIBRATION_VERSION = "dialogue_function_calibration_v1"
TARGET_INTERACTION_LABELS = (
    "request_information", "provide_information", "request_action", "command", "refusal",
    "acknowledgment", "agreement", "disagreement", "warning", "accusation", "defense",
    "explanation", "reassurance", "confession", "threat", "revelation", "interruption",
    "deflection", "narration", "greeting_or_address", "clarification_or_repair",
    "discourse_management", "evaluation_or_reaction",
)


def prepare_calibration_review(cases: list[dict[str, Any]], output_dir: Path, *, maximum_samples: int = 36) -> dict[str, Any]:
    if not cases or maximum_samples < 1:
        raise ValueError("Calibration preparation requires corpus cases and a positive sample limit")
    candidates: list[dict[str, Any]] = []
    for case in cases:
        model = read_json(Path(case["model_path"]))
        bundle = read_json(Path(case["function_bundle_path"]))
        passages = {str(row["speech_passage_id"]): row for row in model.get("speech_passages") or []}
        for record in bundle.get("entities") or []:
            entity_id = str(record.get("source_entity_id"))
            passage = passages.get(entity_id)
            if not passage:
                continue
            classification = record.get("classification") or {}
            interaction = ((classification.get("axes") or {}).get("interaction_function") or {}).get("labels") or []
            surface = ((classification.get("axes") or {}).get("surface_form") or {}).get("labels") or []
            proposed = [str(row.get("label")) for row in interaction if row.get("label") not in {"unknown", "ambiguous", "not_applicable"}]
            candidates.append({
                "case_id": str(case.get("case_id") or model["film_id"]), "media_class": str(case.get("media_class") or "unknown"),
                "film_id": model["film_id"], "speech_passage_id": entity_id,
                "source_time_range": {key: passage[key] for key in ("start", "end", "duration")},
                "transcript": passage.get("original_transcript") or "", "transcript_signature": record.get("transcript_signature"),
                "language_state": passage.get("language"), "source_provenance_id": passage.get("provenance_id"),
                "context_used": record.get("context_used") or {}, "context_signature": record.get("context_signature"),
                "classifier_proposal": classification, "proposed_interaction_labels": proposed,
                "proposed_surface_label": str(surface[0].get("label")) if surface else "unknown",
                "proposal_confidence": float(classification.get("confidence", 0.0) or 0.0),
                "proposal_is_ground_truth": False,
            })
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()

    def admit(row: dict[str, Any], reason: str) -> None:
        key = (row["film_id"], row["speech_passage_id"])
        if key in used or len(selected) >= maximum_samples:
            return
        used.add(key)
        selected.append({**row, "selection_reasons": [reason]})

    for label in TARGET_INTERACTION_LABELS:
        eligible = [row for row in candidates if label in row["proposed_interaction_labels"]]
        eligible.sort(key=lambda row: (-row["proposal_confidence"], 0 if row["media_class"] == "live_action" else 1, row["speech_passage_id"]))
        if eligible:
            admit(eligible[0], f"candidate_for_{label}")
    for special, predicate in (
        ("non_lexical", lambda row: row["proposed_surface_label"] == "non_lexical"),
        ("fragment", lambda row: row["proposed_surface_label"] == "fragment"),
        ("generic_short", lambda row: len(str(row["transcript"]).split()) <= 3),
        ("noisy_or_uncertain", lambda row: row["proposal_confidence"] < 0.62),
    ):
        eligible = sorted((row for row in candidates if predicate(row)), key=lambda row: (row["media_class"], row["speech_passage_id"]))
        if eligible:
            admit(eligible[0], special)
    selected.sort(key=lambda row: (row["case_id"], float(row["source_time_range"]["start"]), row["speech_passage_id"]))
    samples = [{"sample_id": f"function_sample_{index:03d}", **row} for index, row in enumerate(selected, start=1)]
    covered = sorted({label for row in samples for label in row["proposed_interaction_labels"]})
    package_signature = stable_hash({"version": CALIBRATION_VERSION, "samples": samples})
    manifest = {
        "schema_version": "1.0", "calibration_version": CALIBRATION_VERSION,
        "creation_timestamp": utc_now(), "review_state": "PENDING_HUMAN_ANNOTATION",
        "package_signature": package_signature, "sample_count": len(samples), "samples": samples,
        "target_interaction_labels": list(TARGET_INTERACTION_LABELS),
        "proposal_coverage_labels": covered,
        "proposal_coverage_gaps": sorted(set(TARGET_INTERACTION_LABELS) - set(covered)),
        "instructions": "Annotate from transcript and declared context only. Classifier proposals are selection aids, not ground truth. Preserve ambiguity and disagreement.",
    }
    annotations = {
        "schema_version": "1.0", "calibration_version": CALIBRATION_VERSION,
        "package_signature": package_signature,
        "samples": [{"sample_id": row["sample_id"], "annotations": []} for row in samples],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "calibration_manifest.json", manifest)
    write_json(output_dir / "calibration_annotations.json", annotations)
    (output_dir / "calibration_review.md").write_text(_review_worksheet(manifest), encoding="utf-8")
    return manifest


def finalize_calibration_review(
    package_dir: Path, *, annotations_path: Path | None = None, output_path: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(package_dir / "calibration_manifest.json")
    annotations = read_json(annotations_path or package_dir / "calibration_annotations.json")
    if manifest.get("package_signature") != annotations.get("package_signature"):
        raise ValueError("Calibration manifest and annotations do not describe the same package")
    manifest_samples = {str(row["sample_id"]): row for row in manifest.get("samples") or []}
    annotation_samples = {str(row["sample_id"]): row for row in annotations.get("samples") or []}
    if not manifest_samples or set(manifest_samples) != set(annotation_samples):
        raise ValueError("Calibration sample identities are missing or inconsistent")
    valid = _valid_labels()
    reviewed, incomplete = [], []
    for sample_id in sorted(manifest_samples):
        human = annotation_samples[sample_id].get("annotations") or []
        if not human:
            incomplete.append(sample_id)
        validated = []
        for annotation in human:
            axes = annotation.get("axes") or {}
            if set(axes) != {"surface_form", "interaction_function", "sequence_position"}:
                raise ValueError(f"Annotation {sample_id} must include all three axes")
            for axis, labels in axes.items():
                if not labels or not set(labels) <= valid[axis]:
                    raise ValueError(f"Annotation {sample_id} contains invalid {axis} labels")
            confidence = float(annotation.get("annotator_confidence", 0.0) or 0.0)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Annotation confidence must be a unit value for {sample_id}")
            validated.append({
                "annotator_id": str(annotation.get("annotator_id") or "anonymous"), "axes": axes,
                "annotator_confidence": confidence,
                "ambiguity_state": str(annotation.get("ambiguity_state") or "UNAMBIGUOUS"),
                "notes": str(annotation.get("notes") or ""),
            })
        reviewed.append({**manifest_samples[sample_id], "human_annotations": validated, "disagreement_state": _disagreement(validated)})
    state = "COMPLETE" if not incomplete else "INCOMPLETE"
    result = {
        "schema_version": "1.0", "calibration_version": CALIBRATION_VERSION,
        "creation_timestamp": utc_now(), "package_signature": manifest["package_signature"],
        "review_state": state, "sample_count": len(reviewed), "reviewed_sample_count": len(reviewed) - len(incomplete),
        "incomplete_sample_ids": incomplete, "samples": reviewed,
        "metrics": _metrics(reviewed),
        "claim_scope": "Human-reviewed calibration evidence; classifier proposals are not ground truth.",
    }
    write_json(output_path or package_dir / "reviewed_calibration_set.json", result)
    return result


def _valid_labels() -> dict[str, set[str]]:
    taxonomy = load_taxonomy()
    return {axis: {str(row["name"]) for row in data["labels"]} for axis, data in taxonomy["axes"].items()}


def _disagreement(annotations: list[dict[str, Any]]) -> str:
    if len(annotations) < 2:
        return "NOT_MEASURED"
    signatures = {stable_hash(row["axes"]) for row in annotations}
    return "DISAGREEMENT" if len(signatures) > 1 else "AGREEMENT"


def _metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [row for row in samples if row["human_annotations"]]
    ambiguity = sum(1 for row in reviewed if any(value["ambiguity_state"] == "AMBIGUOUS" for value in row["human_annotations"]))
    disagreements = sum(1 for row in reviewed if row["disagreement_state"] == "DISAGREEMENT")
    frequencies = Counter(
        label for row in reviewed for annotation in row["human_annotations"]
        for label in annotation["axes"]["interaction_function"]
    )
    axis_metrics = {axis: _axis_metrics(reviewed, axis) for axis in ("surface_form", "interaction_function", "sequence_position")}
    return {
        "metrics_version": "dialogue_function_calibration_metrics_v2",
        "reviewed_count": len(reviewed),
        "ambiguity_count": ambiguity, "ambiguity_rate": round(ambiguity / len(reviewed), 4) if reviewed else None,
        "multi_annotator_sample_count": sum(1 for row in reviewed if len(row["human_annotations"]) > 1),
        "disagreement_count": disagreements,
        "interaction_label_frequency": dict(sorted(frequencies.items())),
        "evaluation_unit": "classifier_proposal_against_each_preserved_human_annotation",
        "axis_metrics": axis_metrics,
        "confidence_calibration": _confidence_calibration(reviewed),
        "abstention_analysis": _abstention_analysis(reviewed),
    }


def _axis_metrics(samples: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    states = {"unknown", "ambiguous", "not_applicable", "unavailable"}
    counts: dict[str, Counter[str]] = {}
    exact, units = 0, 0
    confusion: Counter[tuple[str, str]] = Counter()
    for sample in samples:
        proposed_axis = (((sample.get("classifier_proposal") or {}).get("axes") or {}).get(axis) or {})
        proposed = {str(row.get("label")) for row in proposed_axis.get("labels") or []}
        for annotation in sample.get("human_annotations") or []:
            human = {str(value) for value in (annotation.get("axes") or {}).get(axis) or []}
            units += 1
            exact += proposed == human
            for label in proposed | human:
                bucket = counts.setdefault(label, Counter())
                bucket["tp" if label in proposed and label in human else "fp" if label in proposed else "fn"] += 1
            if len(proposed) == 1 and len(human) == 1:
                confusion[(next(iter(proposed)), next(iter(human)))] += 1
    per_label = {}
    for label, values in sorted(counts.items()):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        per_label[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "state_label": label in states,
        }
    tp = sum(row["tp"] for row in per_label.values())
    fp = sum(row["fp"] for row in per_label.values())
    fn = sum(row["fn"] for row in per_label.values())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "annotation_comparison_count": units,
        "exact_set_match_count": exact,
        "exact_set_match_rate": round(exact / units, 4) if units else None,
        "micro_precision": round(precision, 4) if precision is not None else None,
        "micro_recall": round(recall, 4) if recall is not None else None,
        "micro_f1": round(f1, 4) if f1 is not None else None,
        "per_label": per_label,
        "single_label_confusion": [
            {"classifier_label": proposed, "human_label": human, "count": count}
            for (proposed, human), count in sorted(confusion.items())
        ],
    }


def _confidence_calibration(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [{"lower": value / 5, "upper": (value + 1) / 5, "scores": [], "confidences": []} for value in range(5)]
    for sample in samples:
        proposed = {
            str(row.get("label"))
            for row in ((((sample.get("classifier_proposal") or {}).get("axes") or {}).get("interaction_function") or {}).get("labels") or [])
        }
        confidence = float(sample.get("proposal_confidence", 0.0) or 0.0)
        index = min(4, max(0, int(confidence * 5)))
        for annotation in sample.get("human_annotations") or []:
            human = {str(value) for value in annotation["axes"]["interaction_function"]}
            union = proposed | human
            score = len(proposed & human) / len(union) if union else 1.0
            bins[index]["scores"].append(score)
            bins[index]["confidences"].append(confidence)
    return [{
        "lower": row["lower"], "upper": row["upper"], "count": len(row["scores"]),
        "mean_confidence": round(sum(row["confidences"]) / len(row["confidences"]), 4) if row["scores"] else None,
        "mean_interaction_jaccard": round(sum(row["scores"]) / len(row["scores"]), 4) if row["scores"] else None,
    } for row in bins]


def _abstention_analysis(samples: list[dict[str, Any]]) -> dict[str, Any]:
    state_labels = {"unknown", "ambiguous", "not_applicable"}
    abstained = false_abstentions = supported_abstentions = missed_uncertainty = units = 0
    for sample in samples:
        proposal = sample.get("classifier_proposal") or {}
        did_abstain = bool((proposal.get("abstention") or {}).get("abstained"))
        for annotation in sample.get("human_annotations") or []:
            units += 1
            human = {str(value) for value in annotation["axes"]["interaction_function"]}
            human_uncertain = bool(human) and human <= state_labels
            if did_abstain:
                abstained += 1
                supported_abstentions += int(human_uncertain)
                false_abstentions += int(not human_uncertain)
            elif human_uncertain:
                missed_uncertainty += 1
    return {
        "annotation_comparison_count": units, "abstained_count": abstained,
        "supported_abstention_count": supported_abstentions, "false_abstention_count": false_abstentions,
        "missed_human_uncertainty_count": missed_uncertainty,
        "abstention_rate": round(abstained / units, 4) if units else None,
    }


def _review_worksheet(manifest: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 dialogue-function calibration review",
        "",
        "Classifier proposals below are sampling aids, not ground truth. For each sample, record surface form, one or more interaction functions, sequence position, confidence (0–1), ambiguity, and notes. Use `sequence_position: unavailable` unless the declared context supplies valid ordered-turn evidence.",
        "",
    ]
    for row in manifest.get("samples") or []:
        lines.extend([
            f"## {row['sample_id']} — {row['case_id']} ({row['media_class']})",
            "",
            f"Transcript: **{row['transcript']}**",
            "",
            f"Film / passage: `{row['film_id']}` / `{row['speech_passage_id']}`",
            "",
            f"Declared context: `{row['context_used']}`",
            "",
            f"Classifier proposal (advisory): surface `{row['proposed_surface_label']}`; interaction `{', '.join(row['proposed_interaction_labels']) or 'none'}`.",
            "",
            "- Surface form:",
            "- Interaction function(s):",
            "- Sequence position:",
            "- Confidence:",
            "- Ambiguous?",
            "- Notes:",
            "",
        ])
    return "\n".join(lines)
