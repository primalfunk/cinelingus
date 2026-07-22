import math

import pytest

from cinelingus.semantic import (
    DeterministicFakeProvider, SemanticConfig, SemanticEntity, SemanticMode,
    SemanticProviderUnavailable, SemanticTextRole, UnavailableProvider,
    compare_entities, top_k,
)


def test_semantic_config_is_disabled_and_zero_weight_by_default() -> None:
    config = SemanticConfig()
    assert config.mode is SemanticMode.DISABLED
    assert config.weight == 0.0
    assert len(config.configuration_signature) == 64
    with pytest.raises(ValueError):
        SemanticConfig(mode=SemanticMode.REPORT_ONLY, weight=0.1)


def test_fake_provider_is_deterministic_normalized_and_reports_truncation() -> None:
    config = SemanticConfig(dimensions=8, token_limit=3)
    provider = DeterministicFakeProvider(config)
    first = provider.encode(["one two three four"], role=SemanticTextRole.PASSAGE)
    second = provider.encode(["one two three four"], role=SemanticTextRole.PASSAGE)
    assert first == second
    assert first.truncated == (True,)
    assert len(first.vectors[0]) == 8
    assert math.isclose(sum(value * value for value in first.vectors[0]), 1.0)


def test_unavailable_provider_fails_explicitly() -> None:
    provider = UnavailableProvider(SemanticConfig(), state="DOWNLOAD_REQUIRED", reason="fixture")
    with pytest.raises(SemanticProviderUnavailable) as caught:
        provider.encode(["text"], role=SemanticTextRole.PASSAGE)
    assert caught.value.state == "DOWNLOAD_REQUIRED"


def test_exact_similarity_filters_and_breaks_ties_by_stable_id() -> None:
    query = SemanticEntity("speech_q", "film_q", "speech_passage", "en", (1.0, 0.0), {"p": "q"})
    a = SemanticEntity("speech_a", "film_d", "speech_passage", "en", (1.0, 0.0), {"p": "a"})
    b = SemanticEntity("speech_b", "film_d", "speech_passage", "fr", (1.0, 0.0), {"p": "b"})
    other = SemanticEntity("turn_a", "film_d", "dialogue_turn", "en", (1.0, 0.0), {})
    match = compare_entities(query, a)
    assert match.raw_cosine_similarity == 1.0
    assert match.normalized_scheduling_contribution == 1.0
    assert [row.candidate_entity_id for row in top_k(query, [b, other, a], limit=3, source_film_id="film_d", entity_type="speech_passage")] == ["speech_a", "speech_b"]
    assert [row.candidate_entity_id for row in top_k(query, [b, a], limit=3, language="en")] == ["speech_a"]
