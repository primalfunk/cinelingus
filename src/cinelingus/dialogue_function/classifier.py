from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ..util import stable_hash
from .taxonomy import TAXONOMY_VERSION, load_taxonomy

CLASSIFIER_VERSION = "dialogue_function_rules_v3_calibration_refinement"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
NON_LEXICAL = frozenset({"ah", "ahh", "oh", "ohh", "uh", "uhh", "um", "hmm", "hm", "mm", "huh", "ha"})
SURFACE_LABELS = frozenset({"declarative", "interrogative", "imperative", "exclamatory", "fragment", "non_lexical", "unknown"})
SEQUENCE_LABELS = frozenset({"initiating", "responding", "continuing", "closing", "standalone", "unavailable"})


@dataclass(frozen=True)
class FunctionClassifierConfig:
    confidence_threshold: float = 0.62
    ambiguity_margin: float = 0.08
    context_mode: str = "passage_alone"
    language: str = "en"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0 or not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("Function classifier confidence settings must be unit interval values")
        if self.context_mode not in {"passage_alone", "adjacent_passages", "dialogue_turn"}:
            raise ValueError(f"Unsupported function context mode: {self.context_mode}")

    @property
    def signature(self) -> str:
        return stable_hash({"classifier_version": CLASSIFIER_VERSION, "taxonomy_version": TAXONOMY_VERSION, **asdict(self)})


class RuleDialogueFunctionClassifier:
    """Local, deterministic, inspectable baseline classifier for English calibration."""

    def __init__(self, config: FunctionClassifierConfig | None = None):
        self.config = config or FunctionClassifierConfig()
        self.taxonomy = load_taxonomy()

    def describe(self) -> dict[str, Any]:
        return {
            "classifier_type": "deterministic_rules",
            "classifier_version": CLASSIFIER_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "configuration_signature": self.config.signature,
            "language_scope": self.config.language,
            "dependencies": [],
            "local_only": True,
            "general_purpose_llm_required": False,
            "approach": "ordered lexical and punctuation rules with explicit confidence, ambiguity, and abstention",
        }

    def classify(self, transcript: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = dict(context or {})
        text = " ".join(str(transcript or "").split())
        lower = text.lower()
        tokens = TOKEN_RE.findall(lower)
        evidence: list[dict[str, Any]] = []
        surface = self._surface(text, lower, tokens, evidence)
        interaction = self._interaction(text, lower, tokens, context, evidence, surface)
        sequence = self._sequence(lower, context, evidence)
        non_state = [row for row in interaction if row["label"] not in {"unknown", "ambiguous", "not_applicable"}]
        best = max((row["confidence"] for row in non_state), default=0.0)
        abstained = not non_state or best < self.config.confidence_threshold
        ambiguity = any(row["label"] == "ambiguous" for row in interaction)
        overall = round(min(float(surface[0]["confidence"]), max(best, 0.5 if not abstained else 0.0)), 4)
        return {
            "taxonomy_version": TAXONOMY_VERSION,
            "classifier_version": CLASSIFIER_VERSION,
            "configuration_signature": self.config.signature,
            "input_context": _public_context(context, self.config.context_mode),
            "context_signature": stable_hash(_public_context(context, self.config.context_mode)),
            "axes": {
                "surface_form": {"labels": surface, "supported": True},
                "interaction_function": {"labels": interaction, "supported": bool(text), "multi_label": True},
                "sequence_position": {"labels": sequence, "supported": sequence[0]["label"] != "unavailable"},
            },
            "confidence": overall,
            "abstention": {"abstained": abstained, "reason": "BELOW_THRESHOLD_OR_UNSUPPORTED" if abstained else None, "threshold": self.config.confidence_threshold},
            "ambiguity_state": "AMBIGUOUS" if ambiguity else "UNAMBIGUOUS",
            "evidence": evidence,
            "claim_scope": "Observable English conversational form/function from transcript and declared bounded context only.",
        }

    def _surface(self, text: str, lower: str, tokens: list[str], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tokens:
            return [_label("unknown", 1.0)]
        if len(tokens) <= 2 and all(token in NON_LEXICAL for token in tokens):
            evidence.append(_evidence("surface.non_lexical_lexicon", tokens))
            return [_label("non_lexical", 0.96)]
        interrogative_open = tokens[0] in {"who", "what", "when", "where", "why", "how", "which", "whose"}
        auxiliary_open = tokens[0] in {"am", "are", "is", "was", "were", "do", "does", "did", "can", "could", "would", "will", "shall", "should", "have", "has"}
        if text.endswith("?") or interrogative_open or auxiliary_open:
            evidence.append(_evidence("surface.interrogative_form", tokens[:2]))
            return [_label("interrogative", 0.94 if text.endswith("?") else 0.82)]
        directive = bool(re.match(r"^(please\s+)?(go|get|give|tell|show|look|listen|stop|wait|come|take|put|leave|let|keep|stay|run|move|open|close|hold|remember|don't|do not)\b", lower))
        if directive:
            evidence.append(_evidence("surface.imperative_opening", tokens[:2]))
            return [_label("imperative", 0.86)]
        if text.endswith("!"):
            return [_label("exclamatory", 0.84)]
        if text.endswith(("...", "—", "-")) or (len(tokens) <= 3 and not re.search(r"\b(am|is|are|was|were|be|been|have|has|do|does|did|will|would|can|could|should|shall|may|might)\b", lower)):
            return [_label("fragment", 0.72)]
        return [_label("declarative", 0.78)]

    def _interaction(
        self, text: str, lower: str, tokens: list[str], context: dict[str, Any], evidence: list[dict[str, Any]], surface: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not tokens:
            return [_label("unknown", 1.0)]
        if surface[0]["label"] == "non_lexical":
            return [_label("not_applicable", 0.92)]
        scored: dict[str, float] = {}

        def add(label: str, confidence: float, rule: str) -> None:
            scored[label] = max(scored.get(label, 0.0), confidence)
            evidence.append(_evidence(rule, label))

        if re.search(r"\b(hello|hi|goodbye|farewell|hey|good morning|good evening)\b", lower): add("greeting_or_address", 0.91, "interaction.greeting")
        if re.search(r"\b(okay|ok|understood|i see|got it|right)\b", lower): add("acknowledgment", 0.82, "interaction.acknowledgment")
        if re.search(r"^(yes|yeah|yep|indeed|exactly|agreed)\b|\byou('re| are) right\b", lower): add("agreement", 0.88, "interaction.agreement")
        if re.search(r"^(no|nope)\b|\b(that('s| is) not true|you('re| are) wrong|i disagree)\b", lower): add("disagreement", 0.87, "interaction.disagreement")
        if re.search(r"\b(i (won't|will not|refuse|can't)|no,? i won't|not going to)\b", lower): add("refusal", 0.9, "interaction.refusal")
        if re.search(r"\b(watch out|look out|beware|be careful|danger)\b", lower): add("warning", 0.94, "interaction.warning")
        if re.search(r"\b(you (stole|broke|lied|cheated|caused|did this)|this is your fault|you('re| are) to blame|what have you done)\b", lower): add("accusation", 0.9, "interaction.accusation")
        if re.search(r"\b(i (didn't|did not) (do|take|break|steal|lie|cause)|not my fault|i was only|i had to)\b", lower): add("defense", 0.87, "interaction.defense")
        if re.search(r"\b(because|that's why|that is why|the reason|so that)\b", lower): add("explanation", 0.85, "interaction.explanation")
        if re.search(r"\b(don't worry|do not worry|it('ll| will) be (all right|okay)|you('re| are) safe|everything('s| is) fine)\b", lower): add("reassurance", 0.92, "interaction.reassurance")
        if re.search(r"\b(i confess|i admit|it was me|i was the one)\b", lower): add("confession", 0.94, "interaction.confession")
        if re.search(r"\b(i('ll| will) (kill|hurt|destroy|punish)|you('ll| will) regret|or else)\b", lower): add("threat", 0.94, "interaction.threat")
        if re.search(r"\b(the truth is|(it )?turns out|what you don't know|i never told you)\b", lower): add("revelation", 0.88, "interaction.revelation")
        if re.search(r"^(wait|hold on|stop)\b|\b(let me finish|don't interrupt)\b", lower): add("interruption", 0.86, "interaction.interruption")
        if re.search(r"\b(never mind|anyway|let's not talk about|forget i said)\b", lower): add("deflection", 0.85, "interaction.deflection")
        if surface[0]["label"] != "interrogative" and re.search(r"\b(once upon a time|and then|after that|afterward|eventually)\b", lower): add("narration", 0.78, "interaction.narration")
        if re.search(r"\b(i mean|rather|what do you mean|pardon|let me rephrase|in other words)\b", lower): add("clarification_or_repair", 0.9, "interaction.clarification")
        if re.search(r"\b(let me finish|moving on|first of all|back to the point|your turn)\b", lower): add("discourse_management", 0.86, "interaction.discourse_management")
        if re.search(r"\b(that('s| is) (great|good|bad|terrible|wonderful|awful|amazing)|how (wonderful|awful|strange))\b", lower): add("evaluation_or_reaction", 0.86, "interaction.evaluation")

        interrogative = surface[0]["label"] == "interrogative"
        action_question = bool(re.match(r"^(can|could|would|will) you\s+(?!tell|say|explain|remember|know)\w+", lower))
        if interrogative and action_question:
            add("request_action", 0.74, "interaction.indirect_action_request")
            scored["ambiguous"] = 0.66
        elif interrogative:
            add("request_information", 0.9, "interaction.information_question")
        if surface[0]["label"] == "imperative":
            polite = lower.startswith("please ") or "would you" in lower
            specialized = {"warning", "reassurance", "interruption", "clarification_or_repair", "discourse_management", "deflection"}
            if not specialized.intersection(scored):
                add("request_action" if polite else "command", 0.86, "interaction.directive")
        if not scored and surface[0]["label"] in {"declarative", "fragment", "exclamatory"}:
            add("provide_information" if surface[0]["label"] == "declarative" else "evaluation_or_reaction", 0.66, "interaction.form_fallback")

        ranked = [_label(label, confidence) for label, confidence in sorted(scored.items(), key=lambda row: (-row[1], row[0]))]
        substantive = [row for row in ranked if row["label"] != "ambiguous"]
        if len(substantive) >= 2 and substantive[0]["confidence"] - substantive[1]["confidence"] <= self.config.ambiguity_margin:
            if not _allowed_combination({substantive[0]["label"], substantive[1]["label"]}):
                ranked.append(_label("ambiguous", max(0.62, substantive[1]["confidence"])))
        if not ranked or max((row["confidence"] for row in substantive), default=0.0) < self.config.confidence_threshold:
            if not any(row["label"] == "ambiguous" for row in ranked):
                ranked.append(_label("unknown", 1.0 - max((row["confidence"] for row in substantive), default=0.0)))
        return sorted(ranked, key=lambda row: (-row["confidence"], row["label"]))

    def _sequence(self, lower: str, context: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not context.get("ordered_turn_evidence"):
            return [_label("unavailable", 1.0)]
        explicit = context.get("sequence_position")
        if explicit in {"initiating", "responding", "continuing", "closing", "standalone"}:
            evidence.append(_evidence("sequence.explicit_structural_position", explicit))
            return [_label(str(explicit), float(context.get("sequence_confidence", 0.85) or 0.85))]
        preceding = context.get("preceding_turn_reference")
        following = context.get("following_turn_reference")
        if not preceding and following:
            return [_label("initiating", 0.82)]
        if preceding and not following and re.search(r"\b(goodbye|farewell|that's all|that is all)\b", lower):
            return [_label("closing", 0.82)]
        if preceding:
            return [_label("continuing", 0.64)]
        return [_label("standalone", 0.7)]


def _label(label: str, confidence: float) -> dict[str, Any]:
    axis = "surface_form" if label in SURFACE_LABELS else "sequence_position" if label in SEQUENCE_LABELS else "interaction_function"
    return {"label": label, "label_id": f"{axis}.{label}", "confidence": round(max(0.0, min(1.0, confidence)), 4)}


def _evidence(rule_id: str, observation: Any) -> dict[str, Any]:
    return {"rule_id": rule_id, "observation": observation}


def _public_context(context: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "previous_speech_passage_id": context.get("previous_speech_passage_id"),
        "previous_transcript_signature": context.get("previous_transcript_signature"),
        "next_speech_passage_id": context.get("next_speech_passage_id"),
        "next_transcript_signature": context.get("next_transcript_signature"),
        "dialogue_turn_id": context.get("dialogue_turn_id"),
        "containing_performance_ids": list(context.get("containing_performance_ids") or []),
        "ordered_turn_evidence": bool(context.get("ordered_turn_evidence")),
        "preceding_turn_reference": context.get("preceding_turn_reference"),
        "following_turn_reference": context.get("following_turn_reference"),
        "sequence_position": context.get("sequence_position"),
    }


def _allowed_combination(labels: set[str]) -> bool:
    allowed = (
        {"acknowledgment", "agreement"}, {"disagreement", "refusal"}, {"command", "warning"},
        {"accusation", "request_information"}, {"defense", "explanation"},
        {"greeting_or_address", "discourse_management"}, {"clarification_or_repair", "request_information"},
        {"evaluation_or_reaction", "provide_information"},
    )
    return labels in allowed
