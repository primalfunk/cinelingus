from pathlib import Path

from movie_masher.transformations import (
    MovieMasherTransformation,
    SelfShuffleTransformation,
    TransformationContext,
    TransformationResult,
    default_registry,
)
from movie_masher.transformations.report import write_transformation_report
from movie_masher.validation import validate_artifact


def test_default_registry_contains_transformations() -> None:
    registry = default_registry()

    assert registry.get("movie_masher") is MovieMasherTransformation
    assert registry.get("self_shuffle") is SelfShuffleTransformation
    assert "movie_masher" in registry.ids()
    assert "self_shuffle" in registry.ids()


def test_movie_masher_metadata_declares_contract() -> None:
    metadata = MovieMasherTransformation.metadata

    assert metadata.id == "movie_masher"
    assert metadata.required_inputs == ("films",)
    assert "movie_masher_output.mp4" in metadata.generated_outputs


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
        transformation_id="movie_masher",
        outputs={"video": tmp_path / "output" / "movie_masher_output.mp4"},
    )

    path = write_transformation_report(metadata=MovieMasherTransformation.metadata, pipeline=pipeline, result=result)

    latest_path = tmp_path / "output" / "transformation_report.json"

    assert path == tmp_path / "output" / "movie_masher" / "transformation_report.json"
    assert path.exists()
    assert latest_path.exists()
    assert "movie_masher" in path.read_text(encoding="utf-8")
    validate_artifact("transformation_report", path, Path.cwd() / "schemas")
    validate_artifact("transformation_report", latest_path, Path.cwd() / "schemas")



def test_movie_masher_transformation_passes_artifacts_forward(tmp_path: Path) -> None:
    class Config:
        root = tmp_path
        output_dir = tmp_path / "output"
        destination_video = tmp_path / "destination.mp4"
        source_dialogue = tmp_path / "source.mp4"
        max_time_stretch = 1.1
        films = (destination_video, source_dialogue)

    class FakePipeline:
        def __init__(self) -> None:
            Config.output_dir.mkdir()
            Config.destination_video.write_text("dest", encoding="utf-8")
            Config.source_dialogue.write_text("source", encoding="utf-8")
            self.config = Config
            self.destination = type("Dest", (), {"media_hash": "desthash"})()
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

        def render_audio_from_schedule(self, **kwargs):
            self._called("render_audio_from_schedule")
            assert kwargs["schedule"]["mappings"][0]["id"] == "m1"
            return self.config.output_dir / "replacement_dialogue.wav"

        def render_video_from_audio(self, **kwargs):
            self._called("render_video_from_audio")
            return self.config.output_dir / "movie_masher_output.mp4"

        def filter_source_dialogue(self, *, force: bool = False):
            raise AssertionError("MovieMasherTransformation should use filter_source_dialogue_from_events")

        def build_clip_library(self, *, force: bool = False):
            raise AssertionError("MovieMasherTransformation should use build_clip_library_from_events")

        def filter_destination_timeline(self, *, force: bool = False):
            raise AssertionError("MovieMasherTransformation should use filter_destination_timeline_from_timeline")

        def schedule(self, *, force: bool = False):
            raise AssertionError("MovieMasherTransformation should use schedule_from_artifacts")

        def render_audio(self, *, force: bool = False):
            raise AssertionError("MovieMasherTransformation should use render_audio_from_schedule")

        def render_video(self, *, force: bool = False):
            raise AssertionError("MovieMasherTransformation should use render_video_from_audio")

    pipeline = FakePipeline()
    transformation = MovieMasherTransformation(TransformationContext(pipeline=pipeline))

    transformation.validate_inputs()
    selections = transformation.select()
    transformed = transformation.transform(selections)
    transformation.validate(transformed)
    outputs = transformation.render(transformed)

    assert outputs["audio"].name == "replacement_dialogue.wav"
    assert outputs["video"].name == "movie_masher_output.mp4"
    assert (tmp_path / "output" / "movie_masher" / "transformation_plan.json").exists()
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
    assert pipeline.calls["render_audio_from_schedule"] == 1
    assert pipeline.calls["render_video_from_audio"] == 1
