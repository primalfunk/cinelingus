from cinelingus.cinematic_compatibility import score_cinematic_compatibility


def _destination(*, action=0.1, reaction=0.8):
    return {
        "duration": 3.0,
        "words_per_second": 2.0,
        "signature": {"speaker_count": 1, "turn_count": 1, "estimated_energy": 0.5, "words_per_second": 2.0},
        "visual": {"mouth_activity": 0.7, "faces": 1.0, "confidence": 0.8, "cinematic_intent": {"dialogue": 0.75, "reaction": reaction, "listening": 0.2}},
        "conversation": {"participant_count": 1, "turn_density": 0.33},
        "editing": {"continuity": 0.9, "reaction_alignment": reaction},
        "movement": {"action_intensity": action},
        "emotion": {"energy": 0.5},
        "metadata": {"confidence": 0.8},
    }


def test_cinematic_compatibility_is_explainable_and_multi_axis() -> None:
    result = score_cinematic_compatibility(
        source={
            "duration": 3.0, "transcript": "This line fits the moment.", "confidence": 0.9,
            "source_performance_signature": {"speaker_count": 1, "turn_count": 1, "estimated_energy": 0.5, "words_per_second": 2.0},
        },
        destination=_destination(),
    )

    assert result["score"] > 0.7
    assert set(result["axes"]) == {"realism", "comedy", "surprise", "novelty", "compatibility", "confidence"}
    assert {row["domain"] for row in result["observations"]} == {"audio", "visual", "conversation", "editing", "novelty"}
    assert result["explanation"]


def test_action_conflict_reduces_realism_without_erasing_comedy_axis() -> None:
    source = {"duration": 3.0, "transcript": "An absurdly calm instruction.", "confidence": 0.9, "source_performance_signature": {"speaker_count": 1, "turn_count": 1, "estimated_energy": 0.1}}
    calm = score_cinematic_compatibility(source=source, destination=_destination(action=0.1))
    action = score_cinematic_compatibility(source=source, destination=_destination(action=0.95))

    assert action["axes"]["realism"] < calm["axes"]["realism"]
    assert action["axes"]["comedy"] >= 0.0
