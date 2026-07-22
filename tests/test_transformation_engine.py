from pathlib import Path

from cinelingus.transformations import (
    TranslationTransformation,
    SelfShuffleTransformation,
    TransformationContext,
    TransformationResult,
    default_registry,
)
from cinelingus.transformations.report import write_transformation_report
from cinelingus.transformations.translation import _primary_stream_duration
from cinelingus.validation import validate_artifact


def test_translation_duration_uses_default_stream_instead_of_container_padding() -> None:
    movie = {
        "duration": 12.0,
        "streams": [
            {"codec_type": "audio", "duration": "11.2", "disposition": {"default": 1}},
            {"codec_type": "audio", "duration": "11.8", "disposition": {"default": 0}},
        ],
    }

    assert _primary_stream_duration(movie, "audio") == 11.2
    assert _primary_stream_duration(movie, "video") == 12.0


def test_default_registry_contains_transformations() -> None:
    registry = default_registry()

    assert registry.get("translation") is TranslationTransformation
    assert registry.get("self_shuffle") is SelfShuffleTransformation
    assert "translation" in registry.ids()
    assert "self_shuffle" in registry.ids()


def test_translation_metadata_declares_contract() -> None:
    metadata = TranslationTransformation.metadata

    assert metadata.id == "translation"
    assert metadata.required_inputs == ("films",)
    assert "translation_output.mp4" in metadata.generated_outputs


def test_self_shuffle_metadata_declares_contract() -> None:
    metadata = SelfShuffleTransformation.metadata

    assert metadata.id == "self_shuffle"
    assert metadata.required_inputs == ("destination_video",)
    assert "self_shuffle_output.mp4" in metadata.generated_outputs
    assert "transformation_report.json" in metadata.generated_outputs


def test_write_transformation_report(tmp_path: Path) -> None:
    config = type(
        "Config",
        (),
        {
            "root": tmp_path,
            "output_dir": tmp_path / "output",
            "destination_video": tmp_path / "dest.mp4",
            "source_dialogue": tmp_path / "source.mp4",
        },
    )()
    pipeline = type(
        "Pipeline",
        (),
        {
            "config": config,
            "destination": type("Dest", (), {"media_hash": "desthash"})(),
            "source": type("Source", (), {"media_hash": "sourcehash"})(),
        },
    )()
    result = TransformationResult(
        transformation_id="translation",
        outputs={"video": tmp_path / "output" / "translation_output.mp4"},
    )

    path = write_transformation_report(metadata=TranslationTransformation.metadata, pipeline=pipeline, result=result)

    latest_path = tmp_path / "output" / "transformation_report.json"

    assert path == tmp_path / "output" / "translation" / "transformation_report.json"
    assert path.exists()
    assert latest_path.exists()
    assert "translation" in path.read_text(encoding="utf-8")
    validate_artifact("transformation_report", path, Path.cwd() / "schemas")
    validate_artifact("transformation_report", latest_path, Path.cwd() / "schemas")



def test_translation_transformation_passes_artifacts_forward(tmp_path: Path, monkeypatch) -> None:
    import cinelingus.transformations.translation as translation_module

    class Config:
        root = tmp_path
        output_dir = tmp_path / "output"
        destination_video = tmp_path / "destination.mp4"
        source_dialogue = tmp_path / "source.mp4"
        max_time_stretch = 1.1
        target_duration_seconds = 8.0
        minimum_duration_seconds = 7.0
        render_sample_rate = 48000
        render_channels = 2
        target_lufs = -16.0
        audio_fade_duration = 0.04
        films = (destination_video, source_dialogue)

    class FakePipeline:
        def __init__(self) -> None:
            Config.output_dir.mkdir()
            Config.destination_video.write_text("dest", encoding="utf-8")
            Config.source_dialogue.write_text("source", encoding="utf-8")
            self.config = Config
            self.destination = type("Dest", (), {"media_hash": "desthash", "media_path": Config.destination_video})()
            self.source = type("Source", (), {"media_hash": "sourcehash"})()
            self.schemas_dir = Path.cwd() / "schemas"
            self.calls: dict[str, int] = {}
            self.logger = type("Logger", (), {"info": lambda self, _message: None})()

        def _called(self, name: str) -> None:
            self.calls[name] = self.calls.get(name, 0) + 1

        def inspect(self, *, force: bool = False):
            self._called("inspect")
            return {"duration": 10.0}, {"duration": 8.0}

        def extract_source_dialogue(self, *, force: bool = False, source_movie=None):
            self._called("extract_source_dialogue")
            assert source_movie == {"duration": 8.0}
            return {"events": [{"id": "e1"}], "config_signature": "raw_source"}

        def filter_source_dialogue_from_events(self, raw, *, force: bool = False):
            self._called("filter_source_dialogue_from_events")
            assert raw["config_signature"] == "raw_source"
            return {"events": [{"id": "e1"}], "config_signature": "filtered_source"}

        def build_clip_library_from_events(self, events, *, force: bool = False):
            self._called("build_clip_library_from_events")
            assert events["config_signature"] == "filtered_source"
            return {"clips": [{"id": "c1"}], "config_signature": "clips"}

        def detect_destination_timeline(self, *, force: bool = False, dest_movie=None):
            self._called("detect_destination_timeline")
            assert dest_movie == {"duration": 10.0}
            return {"windows": [{"id": "w1"}], "config_signature": "raw_timeline"}

        def filter_destination_timeline_from_timeline(self, raw, *, force: bool = False):
            self._called("filter_destination_timeline_from_timeline")
            assert raw["config_signature"] == "raw_timeline"
            return {"windows": [{"id": "w1"}], "config_signature": "timeline"}

        def analyze_visual(self, *, force: bool = False, dest_movie=None):
            self._called("analyze_visual")
            assert dest_movie == {"duration": 10.0}
            return {"shots": {"shots": [], "config_signature": "shots"}, "visual_report": {}}

        def build_source_performances(self, *, source_events=None, force: bool = False):
            self._called("build_source_performances")
            assert source_events["config_signature"] == "filtered_source"
            return {"performances": [{"id": "sp1"}], "config_signature": "source_perf"}

        def build_destination_performances(self, *, timeline=None, visual=None, force: bool = False):
            self._called("build_destination_performances")
            assert timeline["config_signature"] == "timeline"
            return {"performances": [{"id": "dp1"}], "config_signature": "dest_perf"}

        def schedule_from_artifacts(self, **kwargs):
            self._called("schedule_from_artifacts")
            assert kwargs["library"]["config_signature"] == "clips"
            assert kwargs["timeline"]["config_signature"] == "timeline"
            assert kwargs["destination_performances"]["config_signature"] == "dest_perf"
            return {
                "mappings": [
                    {"id": "m1", "clip_id": "c1", "destination_timestamp": 0.5, "planned_render_duration": 1.0},
                    {"id": "m2", "clip_id": "c2", "destination_timestamp": 3.0, "planned_render_duration": 1.0},
                    {"id": "m3", "clip_id": "c3", "destination_timestamp": 6.0, "planned_render_duration": 1.0},
                ]
            }

        def build_cinematic_moments(self, **kwargs):
            self._called("build_cinematic_moments")
            return {
                "source_id": "film_1", "source_media_hash": "desthash", "moment_count": 3,
                "moments": [
                    {"id": "moment_1", "source_id": "film_1", "source_media_hash": "desthash", "scene_id": "scene_1", "shot_ids": ["shot_1"], "start": 0.0, "end": 2.0, "duration": 2.0, "visual_boundary": {"start": 0.0, "end": 2.0}, "audio_boundary": {"start": 0.0, "end": 2.0}, "assertions": [], "fallback_status": "none"},
                    {"id": "moment_2", "source_id": "film_1", "source_media_hash": "desthash", "scene_id": "scene_2", "shot_ids": ["shot_2"], "start": 2.5, "end": 4.5, "duration": 2.0, "visual_boundary": {"start": 2.5, "end": 4.5}, "audio_boundary": {"start": 2.5, "end": 4.5}, "assertions": [], "fallback_status": "none"},
                    {"id": "moment_3", "source_id": "film_1", "source_media_hash": "desthash", "scene_id": "scene_3", "shot_ids": ["shot_3"], "start": 5.5, "end": 7.5, "duration": 2.0, "visual_boundary": {"start": 5.5, "end": 7.5}, "audio_boundary": {"start": 5.5, "end": 7.5}, "assertions": [], "fallback_status": "none"},
                ],
            }

        def _write_and_validate(self, _kind, path, data):
            from cinelingus.util import write_json
            write_json(path, data)

        def render_audio_from_schedule(self, **kwargs):
            self._called("render_audio_from_schedule")
            assert kwargs["schedule"]["mappings"][0]["id"] == "m1"
            return self.config.output_dir / "replacement_dialogue.wav"

        def render_video_from_audio(self, **kwargs):
            self._called("render_video_from_audio")
            return self.config.output_dir / "translation_output.mp4"

        def filter_source_dialogue(self, *, force: bool = False):
            raise AssertionError("TranslationTransformation should use filter_source_dialogue_from_events")

        def build_clip_library(self, *, force: bool = False):
            raise AssertionError("TranslationTransformation should use build_clip_library_from_events")

        def filter_destination_timeline(self, *, force: bool = False):
            raise AssertionError("TranslationTransformation should use filter_destination_timeline_from_timeline")

        def schedule(self, *, force: bool = False):
            raise AssertionError("TranslationTransformation should use schedule_from_artifacts")

        def render_audio(self, *, force: bool = False):
            raise AssertionError("TranslationTransformation should use render_audio_from_schedule")

        def render_video(self, *, force: bool = False):
            raise AssertionError("TranslationTransformation should use render_video_from_audio")

    pipeline = FakePipeline()
    monkeypatch.setattr(translation_module, "render_mutation_media", lambda **kwargs: (kwargs["audio_output"].write_bytes(b"wav"), kwargs["video_output"].write_bytes(b"mp4")))
    monkeypatch.setattr(translation_module, "validate_filter_output", lambda **_kwargs: {})
    monkeypatch.setattr(translation_module, "ffprobe_json", lambda _path: {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "8.0"}})
    transformation = TranslationTransformation(TransformationContext(pipeline=pipeline))

    transformation.validate_inputs()
    selections = transformation.select()
    transformed = transformation.transform(selections)
    transformation.validate(transformed)
    outputs = transformation.render(transformed)

    assert outputs["audio"].name == "replacement_dialogue.wav"
    assert outputs["video"].name == "translation_output.mp4"
    assert (tmp_path / "output" / "translation" / "transformation_plan.json").exists()
    assert pipeline.calls["inspect"] == 1
    assert pipeline.calls["extract_source_dialogue"] == 1
    assert pipeline.calls["filter_source_dialogue_from_events"] == 1
    assert pipeline.calls["build_clip_library_from_events"] == 1
    assert pipeline.calls["detect_destination_timeline"] == 1
    assert pipeline.calls["filter_destination_timeline_from_timeline"] == 1
    assert pipeline.calls["analyze_visual"] == 1
    assert pipeline.calls["build_source_performances"] == 1
    assert pipeline.calls["build_destination_performances"] == 1
    assert pipeline.calls["schedule_from_artifacts"] == 1
    assert transformed["schedule"]["montage_native"] is True
    assert transformed["schedule"]["full_timeline_native"] is True
    assert transformed["montage_plan"]["actual_duration"] == 8.0
    assert transformed["montage_plan"]["duration_resolution"]["shortened"] is True
    assert transformation._montage_plan_path.exists()
    assert transformation._montage_acceptance_path.exists()
