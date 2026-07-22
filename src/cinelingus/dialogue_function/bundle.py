from __future__ import annotations

from pathlib import Path
from typing import Any

from ..util import read_json, stable_hash, utc_now, write_json
from .classifier import FunctionClassifierConfig, RuleDialogueFunctionClassifier
from .taxonomy import load_taxonomy, validate_taxonomy

BUNDLE_VERSION = "dialogue_function_bundle_v2"
USABLE_CLASSIFICATION_STATES = frozenset({"CLASSIFIED", "ABSTAINED"})


def build_function_bundle(
    model: dict[str, Any], output_dir: Path, classifier: RuleDialogueFunctionClassifier,
    *, resume: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dialogue_function_bundle.json"
    taxonomy = classifier.taxonomy
    taxonomy_report = validate_taxonomy(taxonomy)
    passages = sorted(model.get("speech_passages") or [], key=lambda row: (float(row.get("start", 0.0)), str(row.get("speech_passage_id"))))
    context_by_id = _classification_contexts(model, passages, classifier.config)
    prior = _compatible_prior(output_path, model, classifier) if resume else None
    reusable = {str(row.get("source_entity_id")): row for row in (prior or {}).get("entities", [])}
    bundle = {
        "schema_version": "1.0",
        "bundle_version": BUNDLE_VERSION,
        "construction_state": "BUILDING",
        "creation_timestamp": utc_now(),
        "film_id": model["film_id"],
        "film_model_signature": model["created_from_signature"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "taxonomy_signature": taxonomy_report["taxonomy_signature"],
        "classifier": classifier.describe(),
        "configuration_signature": classifier.config.signature,
        "entities": [],
        "turns": [],
        "sequences": [],
        "coverage": _coverage([], len(passages)),
        "validation_state": {"status": "NOT_VALIDATED", "errors": [], "warnings": []},
    }
    cache_hits = 0
    for index, passage in enumerate(passages, start=1):
        entity_id = str(passage["speech_passage_id"])
        transcript = str(passage.get("original_transcript") or "")
        context = context_by_id[entity_id]
        transcript_signature = stable_hash(transcript)
        context_signature = stable_hash(context)
        cache_signature = stable_hash({
            "film_model_signature": model["created_from_signature"],
            "source_entity_id": entity_id,
            "transcript_signature": transcript_signature,
            "context_signature": context_signature,
            "taxonomy_signature": taxonomy_report["taxonomy_signature"],
            "classifier_configuration_signature": classifier.config.signature,
        })
        cached = reusable.get(entity_id)
        if cached and cached.get("entity_cache_signature") == cache_signature:
            record = cached
            cache_hits += 1
        else:
            language = passage.get("language")
            if language and str(language).lower() not in {"en", "eng", "english"}:
                classification = _unsupported_language_classification(classifier, context, str(language))
                state = "UNAVAILABLE"
            else:
                classification = classifier.classify(transcript, context=context)
                state = "ABSTAINED" if classification["abstention"]["abstained"] else "CLASSIFIED"
            record = {
                "source_entity_type": "speech_passage",
                "source_entity_id": entity_id,
                "film_id": model["film_id"],
                "source_time_range": {key: passage[key] for key in ("start", "end", "duration")},
                "transcript_signature": transcript_signature,
                "language_state": passage.get("language"),
                "source_provenance_id": passage["provenance_id"],
                "source_transcript_reference": passage.get("source_transcript_reference"),
                "taxonomy_version": taxonomy["taxonomy_version"],
                "classifier_version": classifier.describe()["classifier_version"],
                "configuration_signature": classifier.config.signature,
                "context_signature": context_signature,
                "context_used": classification["input_context"],
                "classification_state": state,
                "classification": classification,
                "entity_cache_signature": cache_signature,
                "validation_state": "VALID",
            }
        bundle["entities"].append(record)
        bundle["coverage"] = _coverage(bundle["entities"], len(passages))
        if index % 32 == 0 or index == len(passages):
            write_json(output_path, bundle)
    bundle["turns"] = _aggregate_turns(model, bundle["entities"])
    bundle["sequences"] = _build_function_sequences(model, bundle["turns"])
    bundle["coverage"] = _coverage(bundle["entities"], len(passages), bundle["turns"], bundle["sequences"])
    validation = validate_function_bundle(bundle, model, classifier=classifier, require_ready=False)
    bundle["construction_state"] = "READY" if validation["status"] == "VALID" else "INVALID"
    bundle["validation_state"] = validation
    bundle["cache_report"] = {
        "entity_count": len(passages), "cache_hits": cache_hits,
        "entities_classified": len(passages) - cache_hits, "resume_used": prior is not None,
    }
    write_json(output_path, bundle)
    return bundle


def validate_function_bundle(
    bundle: dict[str, Any], model: dict[str, Any], *, classifier: RuleDialogueFunctionClassifier | None = None,
    require_ready: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    passages = {str(row.get("speech_passage_id")): row for row in model.get("speech_passages") or []}
    entities = bundle.get("entities") or []
    turns = bundle.get("turns") or []
    sequences = bundle.get("sequences") or []
    ids = [str(row.get("source_entity_id")) for row in entities]
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        errors.append("Unsupported bundle version")
    if bundle.get("film_id") != model.get("film_id") or bundle.get("film_model_signature") != model.get("created_from_signature"):
        errors.append("FilmModel identity or construction signature mismatch")
    if len(ids) != len(set(ids)) or set(ids) != set(passages):
        errors.append("Every SpeechPassage must have exactly one classification record")
    taxonomy_report = validate_taxonomy(load_taxonomy())
    if bundle.get("taxonomy_signature") != taxonomy_report["taxonomy_signature"]:
        errors.append("Taxonomy signature mismatch")
    if classifier and bundle.get("configuration_signature") != classifier.config.signature:
        errors.append("Classifier configuration signature mismatch")
    for row in entities:
        if row.get("classification_state") not in {"CLASSIFIED", "ABSTAINED", "UNAVAILABLE", "FAILED"}:
            errors.append(f"Invalid classification state for {row.get('source_entity_id')}")
        classification = row.get("classification") or {}
        if set((classification.get("axes") or {})) != {"surface_form", "interaction_function", "sequence_position"}:
            errors.append(f"Missing classification axes for {row.get('source_entity_id')}")
        if row.get("classification_state") in {"UNAVAILABLE", "FAILED"}:
            warnings.append(f"Unavailable classification: {row.get('source_entity_id')}")
    model_turns = {str(row.get("dialogue_turn_id")): row for row in model.get("dialogue_turns") or []}
    turn_ids = [str(row.get("source_entity_id")) for row in turns]
    if len(turn_ids) != len(set(turn_ids)) or set(turn_ids) != set(model_turns):
        errors.append("Every DialogueTurn must have exactly one aggregation record")
    entity_ids = set(ids)
    for row in turns:
        turn_id = str(row.get("source_entity_id"))
        state = row.get("aggregation_state")
        if state not in {"AVAILABLE", "UNAVAILABLE"}:
            errors.append(f"Invalid turn aggregation state for {turn_id}")
        references = [str(value) for value in row.get("ordered_speech_passage_references") or []]
        expected = [str(value) for value in (model_turns.get(turn_id) or {}).get("ordered_speech_passage_references") or []]
        if references != expected:
            errors.append(f"Turn passage order mismatch for {turn_id}")
        if any(value not in entity_ids for value in references):
            errors.append(f"Turn references unavailable passage classifications for {turn_id}")
        if state == "AVAILABLE" and set((row.get("axes") or {})) != {"surface_form", "interaction_function", "sequence_position"}:
            errors.append(f"Turn aggregate missing axes for {turn_id}")
    model_performances = {str(row.get("performance_id")): row for row in model.get("performances") or []}
    sequence_ids = [str(row.get("source_performance_id")) for row in sequences]
    if len(sequence_ids) != len(set(sequence_ids)) or set(sequence_ids) != set(model_performances):
        errors.append("Every Performance must have exactly one function-sequence record")
    turn_id_set = set(turn_ids)
    for row in sequences:
        performance_id = str(row.get("source_performance_id"))
        references = [str(value) for value in row.get("ordered_dialogue_turn_references") or []]
        expected = [str(value) for value in (model_performances.get(performance_id) or {}).get("dialogue_turn_references") or []]
        if references != expected:
            errors.append(f"Performance turn order mismatch for {performance_id}")
        if any(value not in turn_id_set for value in references):
            errors.append(f"Function sequence references unavailable turn for {performance_id}")
        if row.get("sequence_state") == "AVAILABLE" and len(row.get("function_sequence") or []) != len(references):
            errors.append(f"Function sequence length mismatch for {performance_id}")
    if require_ready and bundle.get("construction_state") != "READY":
        errors.append("Only a READY function bundle is admissible")
    return {
        "validator_version": "dialogue_function_bundle_validator_v1",
        "status": "INVALID" if errors else "VALID",
        "error_count": len(errors), "warning_count": len(warnings),
        "errors": errors, "warnings": warnings,
    }


def _classification_contexts(
    model: dict[str, Any], passages: list[dict[str, Any]], config: FunctionClassifierConfig,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    turns = {str(row.get("dialogue_turn_id")): row for row in model.get("dialogue_turns") or []}
    valid_turn_passages = {
        str(passage_id): turn
        for turn in turns.values()
        for passage_id in turn.get("ordered_speech_passage_references") or []
    }
    for index, passage in enumerate(passages):
        entity_id = str(passage["speech_passage_id"])
        context: dict[str, Any] = {"containing_performance_ids": list(passage.get("linked_performance_ids") or [])}
        if config.context_mode in {"adjacent_passages", "dialogue_turn"}:
            if index:
                previous = passages[index - 1]
                context.update(previous_speech_passage_id=previous["speech_passage_id"], previous_transcript_signature=stable_hash(str(previous.get("original_transcript") or "")))
            if index + 1 < len(passages):
                following = passages[index + 1]
                context.update(next_speech_passage_id=following["speech_passage_id"], next_transcript_signature=stable_hash(str(following.get("original_transcript") or "")))
        turn = valid_turn_passages.get(entity_id)
        if config.context_mode == "dialogue_turn" and turn is not None:
            context.update({
                "ordered_turn_evidence": True,
                "dialogue_turn_id": turn["dialogue_turn_id"],
                "preceding_turn_reference": turn.get("preceding_turn_reference"),
                "following_turn_reference": turn.get("following_turn_reference"),
            })
            if not turn.get("preceding_turn_reference") and turn.get("following_turn_reference"):
                context["sequence_position"] = "initiating"
        result[entity_id] = context
    return result


def _compatible_prior(path: Path, model: dict[str, Any], classifier: RuleDialogueFunctionClassifier) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    prior = read_json(path)
    if (
        prior.get("bundle_version") == BUNDLE_VERSION
        and prior.get("film_id") == model.get("film_id")
        and prior.get("film_model_signature") == model.get("created_from_signature")
        and prior.get("configuration_signature") == classifier.config.signature
    ):
        return prior
    return None


def _coverage(
    entities: list[dict[str, Any]], source_count: int, turns: list[dict[str, Any]] | None = None,
    sequences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = {state: 0 for state in ("CLASSIFIED", "ABSTAINED", "UNAVAILABLE", "FAILED")}
    sequence_available = 0
    for row in entities:
        state = str(row.get("classification_state"))
        if state in counts:
            counts[state] += 1
        sequence = ((row.get("classification") or {}).get("axes") or {}).get("sequence_position") or {}
        if sequence.get("supported"):
            sequence_available += 1
    turn_rows = turns or []
    sequence_rows = sequences or []
    return {
        "source_entity_count": source_count,
        "accounted_entity_count": len(entities),
        "status_counts": counts,
        "sequence_position_available_count": sequence_available,
        "source_turn_count": len(turn_rows),
        "turn_aggregate_available_count": sum(row.get("aggregation_state") == "AVAILABLE" for row in turn_rows),
        "turn_aggregate_unavailable_count": sum(row.get("aggregation_state") == "UNAVAILABLE" for row in turn_rows),
        "source_performance_count": len(sequence_rows),
        "function_sequence_available_count": sum(row.get("sequence_state") == "AVAILABLE" for row in sequence_rows),
        "function_sequence_unavailable_count": sum(row.get("sequence_state") == "UNAVAILABLE" for row in sequence_rows),
    }


def _aggregate_turns(model: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate passage evidence without inventing labels absent from the source classifications."""
    by_id = {str(row.get("source_entity_id")): row for row in entities}
    rows: list[dict[str, Any]] = []
    for turn in sorted(
        model.get("dialogue_turns") or [],
        key=lambda row: (float(row.get("start", 0.0) or 0.0), str(row.get("dialogue_turn_id"))),
    ):
        turn_id = str(turn["dialogue_turn_id"])
        references = [str(value) for value in turn.get("ordered_speech_passage_references") or []]
        records = [by_id[value] for value in references if value in by_id]
        usable = [row for row in records if row.get("classification_state") in USABLE_CLASSIFICATION_STATES]
        state = "AVAILABLE" if references and len(records) == len(references) and usable else "UNAVAILABLE"
        axes = _aggregate_axes(usable) if state == "AVAILABLE" else {}
        ambiguity = any(
            ((row.get("classification") or {}).get("ambiguity_state") == "AMBIGUOUS")
            for row in usable
        ) or any(bool(axis.get("ambiguous")) for axis in axes.values())
        confidences = [float((row.get("classification") or {}).get("confidence", 0.0) or 0.0) for row in usable]
        rows.append({
            "source_entity_type": "dialogue_turn",
            "source_entity_id": turn_id,
            "film_id": model["film_id"],
            "source_time_range": {key: turn.get(key) for key in ("start", "end", "duration")},
            "source_provenance_id": turn.get("provenance_id"),
            "ordered_speech_passage_references": references,
            "speaker_cluster_reference": turn.get("speaker_cluster_reference"),
            "preceding_turn_reference": turn.get("preceding_turn_reference"),
            "following_turn_reference": turn.get("following_turn_reference"),
            "aggregation_state": state,
            "aggregation_policy": "confidence_weighted_passage_distribution_v1",
            "axes": axes,
            "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "ambiguity_state": "AMBIGUOUS" if ambiguity else "UNAMBIGUOUS",
            "turn_cache_signature": stable_hash({
                "turn_id": turn_id,
                "ordered_passages": references,
                "entity_cache_signatures": [row.get("entity_cache_signature") for row in records],
                "structure": {
                    "preceding": turn.get("preceding_turn_reference"),
                    "following": turn.get("following_turn_reference"),
                    "speaker": turn.get("speaker_cluster_reference"),
                },
            }),
            "validation_state": "VALID",
        })
    return rows


def _aggregate_axes(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis_name in ("surface_form", "interaction_function", "sequence_position"):
        totals: dict[str, float] = {}
        support_count = 0
        for record in records:
            axis = (((record.get("classification") or {}).get("axes") or {}).get(axis_name) or {})
            if not axis.get("supported"):
                continue
            support_count += 1
            for label in axis.get("labels") or []:
                name = str(label.get("label"))
                totals[name] = totals.get(name, 0.0) + float(label.get("confidence", 0.0) or 0.0)
        total = sum(totals.values())
        labels = [
            {
                "label": name,
                "label_id": f"{axis_name}.{name}",
                "confidence": round(value / total, 4) if total else 0.0,
            }
            for name, value in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        ]
        gap = (labels[0]["confidence"] - labels[1]["confidence"]) if len(labels) > 1 else 1.0
        result[axis_name] = {
            "supported": bool(labels),
            "supporting_passage_count": support_count,
            "labels": labels,
            "primary_label": labels[0]["label"] if labels else None,
            "ambiguous": len(labels) > 1 and gap < 0.2,
        }
    return result


def _build_function_sequences(model: dict[str, Any], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("source_entity_id")): row for row in turns}
    rows: list[dict[str, Any]] = []
    for performance in sorted(
        model.get("performances") or [],
        key=lambda row: (float(row.get("start", 0.0) or 0.0), str(row.get("performance_id"))),
    ):
        performance_id = str(performance["performance_id"])
        references = [str(value) for value in performance.get("dialogue_turn_references") or []]
        ordered = [by_id[value] for value in references if value in by_id]
        available = bool(references) and len(ordered) == len(references) and all(row.get("aggregation_state") == "AVAILABLE" for row in ordered)
        sequence = [{
            "sequence_index": index,
            "dialogue_turn_id": turn["source_entity_id"],
            "speaker_cluster_reference": turn.get("speaker_cluster_reference"),
            "axes": turn.get("axes") or {},
            "confidence": turn.get("confidence", 0.0),
            "ambiguity_state": turn.get("ambiguity_state"),
        } for index, turn in enumerate(ordered)] if available else []
        rows.append({
            "source_performance_id": performance_id,
            "source_provenance_id": performance.get("provenance_id"),
            "source_time_range": {key: performance.get(key) for key in ("start", "end", "duration")},
            "ordered_dialogue_turn_references": references,
            "speaker_sequence": list(performance.get("speaker_sequence") or []),
            "sequence_state": "AVAILABLE" if available else "UNAVAILABLE",
            "representation_policy": "ordered_turn_function_distributions_v1_no_flattening",
            "function_sequence": sequence,
            "sequence_cache_signature": stable_hash({
                "performance_id": performance_id,
                "ordered_turns": references,
                "turn_signatures": [row.get("turn_cache_signature") for row in ordered],
                "speaker_sequence": performance.get("speaker_sequence") or [],
            }),
            "validation_state": "VALID",
        })
    return rows


def _unsupported_language_classification(
    classifier: RuleDialogueFunctionClassifier, context: dict[str, Any], language: str,
) -> dict[str, Any]:
    result = classifier.classify("", context=context)
    result["abstention"] = {"abstained": True, "reason": "LANGUAGE_OUTSIDE_CALIBRATED_SCOPE", "threshold": classifier.config.confidence_threshold}
    result["language_warning"] = f"Rules are calibrated only for English; observed {language}."
    return result
