from cinelingus.dialogue_function import FunctionClassifierConfig, RuleDialogueFunctionClassifier


def _labels(result, axis):
    return {row["label"] for row in result["axes"][axis]["labels"]}


def test_classifier_produces_inspectable_multi_axis_outputs() -> None:
    classifier = RuleDialogueFunctionClassifier()
    result = classifier.classify("Why did you leave?")

    assert _labels(result, "surface_form") == {"interrogative"}
    assert result["axes"]["surface_form"]["labels"][0]["label_id"] == "surface_form.interrogative"
    assert "request_information" in _labels(result, "interaction_function")
    assert _labels(result, "sequence_position") == {"unavailable"}
    assert result["abstention"]["abstained"] is False
    assert result["evidence"]
    assert classifier.describe()["general_purpose_llm_required"] is False


def test_classifier_supports_multi_label_and_preserves_ambiguity() -> None:
    classifier = RuleDialogueFunctionClassifier()
    defense = classifier.classify("I didn't do it because I was outside.")
    indirect = classifier.classify("Could you leave?")

    assert {"defense", "explanation"} <= _labels(defense, "interaction_function")
    assert {"request_action", "ambiguous"} <= _labels(indirect, "interaction_function")
    assert indirect["ambiguity_state"] == "AMBIGUOUS"


def test_classifier_abstains_on_unsupported_and_nonlexical_inputs() -> None:
    classifier = RuleDialogueFunctionClassifier()
    empty = classifier.classify("")
    vocal = classifier.classify("Ah!")

    assert empty["abstention"]["abstained"] is True
    assert _labels(empty, "interaction_function") == {"unknown"}
    assert _labels(vocal, "surface_form") == {"non_lexical"}
    assert _labels(vocal, "interaction_function") == {"not_applicable"}


def test_sequence_position_requires_explicit_ordered_structure() -> None:
    classifier = RuleDialogueFunctionClassifier(FunctionClassifierConfig(context_mode="dialogue_turn"))
    unavailable = classifier.classify("Yes.", context={"preceding_turn_reference": "turn_1"})
    responding = classifier.classify("Yes.", context={
        "ordered_turn_evidence": True, "sequence_position": "responding",
        "preceding_turn_reference": "turn_1", "dialogue_turn_id": "turn_2",
    })

    assert _labels(unavailable, "sequence_position") == {"unavailable"}
    assert _labels(responding, "sequence_position") == {"responding"}


def test_classifier_is_deterministic() -> None:
    classifier = RuleDialogueFunctionClassifier(FunctionClassifierConfig(context_mode="adjacent_passages"))
    context = {"previous_speech_passage_id": "p1", "next_speech_passage_id": "p3"}
    assert classifier.classify("Wait—listen to me.", context=context) == classifier.classify("Wait—listen to me.", context=context)


def test_calibration_refinements_avoid_broad_false_rules() -> None:
    classifier = RuleDialogueFunctionClassifier()

    assert "defense" not in _labels(classifier.classify("I didn't recognise him."), "interaction_function")
    assert "narration" not in _labels(classifier.classify("And then what?"), "interaction_function")
    assert "command" not in _labels(classifier.classify("Don't worry. He looks great."), "interaction_function")
    assert "revelation" in _labels(classifier.classify("Turns out Mert needs someone."), "interaction_function")
    assert "accusation" in _labels(classifier.classify("What have you done with the controls?"), "interaction_function")
