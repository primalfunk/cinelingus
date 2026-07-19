from __future__ import annotations

from json import loads
from pathlib import Path
from types import SimpleNamespace

import pytest

from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.reliable_experiment import (
    OutputDurationError,
    execute_reliably,
    normalize_parameters_tolerantly,
    render_passthrough,
    repair_output_duration,
    resolve_configuration,
    validate_usable_output,
)
from cinelingus.transformations.base import TransformationResult
from cinelingus.validation import validate_artifact


def _probe(_path: Path) -> dict:
    return {
        "format": {"duration": "42.0"},
        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
    }


def _pipeline(tmp_path: Path, *, cancelled: bool = False):
    anchor = tmp_path / "anchor.mp4"
    anchor.write_bytes(b"anchor")
    config = SimpleNamespace(
        anchor_film=anchor,
        output_dir=tmp_path / "output",
        films=(anchor,),
    )
    return SimpleNamespace(config=config, cancel_check=lambda: cancelled)


def _video_result(tmp_path: Path, filter_id: str) -> TransformationResult:
    video = tmp_path / f"{filter_id.replace('.', '_')}.mp4"
    video.write_bytes(b"video")
    return TransformationResult(transformation_id=filter_id, outputs={"video": video})


def _fallback(tmp_path: Path):
    def render(_anchor: Path, _output_dir: Path):
        video = tmp_path / "safe_result.mp4"
        video.write_bytes(b"safe")
        return video, "TEST_COPY"

    return render


def _accepted_requested(**_kwargs):
    return {"status": "PASS", "artifact_path": None, "failure_summary": None}


def _altered_fallback(tmp_path: Path):
    def render(_films, _output_dir: Path, expected_duration: float):
        video = tmp_path / "altered_result.mp4"
        video.write_bytes(b"altered")
        return video, "TEST_ALTERED_RENDER", {
            "strategy": "TEST_ALTERATION",
            "expected_duration": expected_duration,
            "output": str(video),
        }

    return render


def _accepted_universal(**_kwargs):
    return {"status": "PASS", "artifact_path": None, "failure_summary": None}


def _outcome(result: TransformationResult) -> dict:
    return loads(result.artifacts["configuration_outcome"].read_text(encoding="utf-8"))


def test_requested_success_is_returned_and_schema_valid(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    result = execute_reliably(
        pipeline,
        "memory.dream",
        run_filter=lambda filter_id, **_kwargs: _video_result(tmp_path, filter_id),
        fallback_renderer=_fallback(tmp_path),
        requested_alteration_evaluator=_accepted_requested,
        probe=_probe,
    )
    assert _outcome(result)["status"] == "REQUESTED_SUCCESS"
    validate_artifact("configuration_outcome", result.artifacts["configuration_outcome"], Path.cwd() / "schemas")


def test_invalid_parameters_are_replaced_with_defaults_without_error(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    observed = {}

    def run_filter(filter_id, **kwargs):
        observed.update(kwargs["parameters"])
        return _video_result(tmp_path, filter_id)

    result = execute_reliably(
        pipeline,
        "memory.dream",
        parameters={"intensity": "impossible", "unknown": 3},
        run_filter=run_filter,
        fallback_renderer=_fallback(tmp_path),
        requested_alteration_evaluator=_accepted_requested,
        probe=_probe,
    )
    outcome = _outcome(result)
    assert outcome["status"] == "NORMALIZED_SUCCESS"
    assert observed["intensity"] == default_filter_registry().get("memory.dream").parameter_defaults["intensity"]
    assert {row["reason"] for row in outcome["resolution"]["parameter_adjustments"]} == {
        "invalid_value_replaced_with_default", "unknown_parameter_ignored"
    }


def test_unproven_stack_executes_primary_only(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    called = []
    result = execute_reliably(
        pipeline,
        "multiworld.translation",
        selected_filter_stack=["multiworld.translation", "experimental.bloom"],
        run_filter=lambda filter_id, **_kwargs: called.append(filter_id) or _video_result(tmp_path, filter_id),
        fallback_renderer=_fallback(tmp_path),
        requested_alteration_evaluator=_accepted_requested,
        probe=_probe,
    )
    assert called == ["multiworld.translation"]
    assert _outcome(result)["status"] == "PRIMARY_ONLY_SUCCESS"


def test_default_passthrough_applies_explicit_audio_support_duration(monkeypatch, tmp_path: Path) -> None:
    commands = []

    def fake_run(args):
        commands.append(args)
        Path(args[-1]).write_bytes(b"safe")

    monkeypatch.setattr("cinelingus.reliable_experiment.run", fake_run)
    video, method = render_passthrough(
        tmp_path / "long_anchor.mp4",
        tmp_path / "output",
        maximum_duration=42.25,
    )

    assert video.exists()
    assert method == "STREAM_COPY_MP4_CURTAILED"
    assert commands[0][commands[0].index("-t") + 1] == "42.250"


def test_overlong_requested_output_is_repaired_and_revalidated(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)

    def probe(path: Path) -> dict:
        duration = 42.0 if path.name in {"anchor.mp4", "repaired.mp4"} else 50.0
        return {
            "format": {"duration": str(duration)},
            "streams": [
                {"codec_type": "video", "duration": str(duration)},
                {"codec_type": "audio", "duration": str(duration)},
            ],
        }

    def repair(_source: Path, _output_dir: Path, expected_duration: float):
        assert expected_duration == 42.0
        repaired = tmp_path / "repaired.mp4"
        repaired.write_bytes(b"repaired")
        return repaired, "TEST_DURATION_CAP"

    result = execute_reliably(
        pipeline,
        "memory.dream",
        run_filter=lambda filter_id, **_kwargs: _video_result(tmp_path, filter_id),
        fallback_renderer=_fallback(tmp_path),
        duration_repairer=repair,
        requested_alteration_evaluator=_accepted_requested,
        probe=probe,
    )
    outcome = _outcome(result)
    assert outcome["status"] == "DURATION_REPAIRED_SUCCESS"
    assert outcome["duration_contract"]["expected_duration"] == 42.0
    assert outcome["execution"]["duration_repair_method"] == "TEST_DURATION_CAP"
    assert outcome["output"]["acceptance"]["duration_contract"]["checks"] == {
        "container_duration_matches_expected": True,
        "video_duration_matches_expected": True,
        "audio_duration_matches_expected": True,
    }
    validate_artifact("configuration_outcome", result.artifacts["configuration_outcome"], Path.cwd() / "schemas")


def test_short_requested_output_uses_fallback_instead_of_claiming_success(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)

    def probe(path: Path) -> dict:
        duration = 5.0 if path.name == "memory_dream.mp4" else 42.0
        return {
            "format": {"duration": str(duration)},
            "streams": [
                {"codec_type": "video", "duration": str(duration)},
                {"codec_type": "audio", "duration": str(duration)},
            ],
        }

    def unexpected_repair(*_args):
        raise AssertionError("A short output cannot be repaired by trimming.")

    result = execute_reliably(
        pipeline,
        "memory.dream",
        run_filter=lambda filter_id, **_kwargs: _video_result(tmp_path, filter_id),
        fallback_renderer=_fallback(tmp_path),
        duration_repairer=unexpected_repair,
        altered_renderer=_altered_fallback(tmp_path),
        universal_alteration_evaluator=_accepted_universal,
        probe=probe,
    )
    outcome = _outcome(result)
    assert outcome["status"] == "ALTERED_FALLBACK_SUCCESS"
    assert outcome["execution"]["duration_repair_method"] is None
    assert outcome["execution"]["execution_error"].startswith("OutputDurationError:")
    assert outcome["output"]["acceptance"]["duration"] == 42.0


def test_audio_stream_cannot_be_shorter_than_duration_contract(tmp_path: Path) -> None:
    video = tmp_path / "short_audio.mp4"
    video.write_bytes(b"media")

    def probe(_path: Path) -> dict:
        return {
            "format": {"duration": "42.0"},
            "streams": [
                {"codec_type": "video", "duration": "42.0"},
                {"codec_type": "audio", "duration": "5.0"},
            ],
        }

    with pytest.raises(OutputDurationError) as raised:
        validate_usable_output(video, expected_duration=42.0, probe=probe)
    evidence = raised.value.acceptance["duration_contract"]
    assert evidence["checks"]["container_duration_matches_expected"] is True
    assert evidence["checks"]["video_duration_matches_expected"] is True
    assert evidence["checks"]["audio_duration_matches_expected"] is False


def test_duration_repair_command_caps_the_output(monkeypatch, tmp_path: Path) -> None:
    commands = []

    def fake_run(args):
        commands.append(args)
        Path(args[-1]).write_bytes(b"repaired")

    monkeypatch.setattr("cinelingus.reliable_experiment.run", fake_run)
    output, method = repair_output_duration(tmp_path / "overlong.mp4", tmp_path / "output", 42.25)

    assert output.exists()
    assert method == "STREAM_COPY_DURATION_CAP"
    assert commands[0][commands[0].index("-t") + 1] == "42.250"


def test_filter_failure_returns_validated_altered_fallback(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("analysis failed")

    result = execute_reliably(
        pipeline,
        "memory.dream",
        run_filter=fail,
        fallback_renderer=_fallback(tmp_path),
        altered_renderer=_altered_fallback(tmp_path),
        universal_alteration_evaluator=_accepted_universal,
        probe=_probe,
    )
    outcome = _outcome(result)
    assert outcome["status"] == "ALTERED_FALLBACK_SUCCESS"
    assert outcome["execution"]["execution_error"] == "RuntimeError: analysis failed"
    assert result.outputs["video"].exists()


def test_unknown_configuration_returns_altered_fallback(tmp_path: Path) -> None:
    result = execute_reliably(
        _pipeline(tmp_path),
        "future.unknown",
        fallback_renderer=_fallback(tmp_path),
        altered_renderer=_altered_fallback(tmp_path),
        universal_alteration_evaluator=_accepted_universal,
        probe=_probe,
    )
    assert _outcome(result)["status"] == "ALTERED_FALLBACK_SUCCESS"


def test_universal_alteration_failure_returns_explicit_unaltered_recovery(tmp_path: Path) -> None:
    def fail_alteration(*_args, **_kwargs):
        raise RuntimeError("altered renderer failed")

    result = execute_reliably(
        _pipeline(tmp_path),
        "future.unknown",
        fallback_renderer=_fallback(tmp_path),
        altered_renderer=fail_alteration,
        probe=_probe,
    )
    outcome = _outcome(result)
    assert outcome["status"] == "UNALTERED_RECOVERY"
    assert outcome["output"]["alteration_acceptance"]["status"] == "FAIL"
    assert outcome["execution"]["altered_fallback_error"] == "RuntimeError: altered renderer failed"


def test_user_cancellation_is_not_converted_to_a_result(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cancelled"):
        execute_reliably(
            _pipeline(tmp_path, cancelled=True),
            "memory.dream",
            run_filter=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cancelled")),
            fallback_renderer=_fallback(tmp_path),
            probe=_probe,
        )


def test_every_declared_parameter_value_normalizes_for_every_implemented_filter() -> None:
    for definition in default_filter_registry().definitions(implemented_only=True):
        defaults, adjustments = normalize_parameters_tolerantly(definition, definition.parameter_defaults)
        assert not adjustments, definition.id
        assert defaults == definition.parameter_defaults, definition.id
        for parameter in definition.parameters:
            values = list(parameter.choices) if parameter.kind == "choice" else [parameter.default]
            if parameter.kind == "boolean":
                values = [False, True]
            if parameter.kind in {"integer", "float"}:
                values = [value for value in (parameter.minimum, parameter.default, parameter.maximum) if value is not None]
            for value in values:
                normalized, adjustments = normalize_parameters_tolerantly(definition, {parameter.id: value})
                assert not adjustments, f"{definition.id}: {parameter.id}={value!r}"
                assert normalized[parameter.id] == parameter.validate(value)


def test_every_ordered_pair_resolves_to_execution_or_passthrough_without_error() -> None:
    definitions = default_filter_registry().definitions()
    for first in definitions:
        for second in definitions:
            if first.id == second.id:
                continue
            resolution = resolve_configuration(
                first.id,
                selected_filter_stack=[first.id, second.id],
            )
            assert resolution["stack_status"] in {"PRIMARY_ONLY", "PASSTHROUGH_REQUIRED", "CERTIFIED_STACK"}
            if resolution["stack_status"] == "PRIMARY_ONLY":
                assert resolution["executable_filter_id"] is not None
