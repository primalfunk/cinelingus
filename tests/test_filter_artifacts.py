from pathlib import Path

from cinelingus.filter_lab.artifacts import materialize_required_artifacts
from cinelingus.util import read_json, write_json


def test_required_artifacts_are_materialized_on_demand(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    output = tmp_path / "output"
    write_json(source / "dialogue_events.json", {"events": []})
    write_json(source / "performance.json", {"media_hash": "source", "performances": []})
    write_json(destination / "performance.json", {
        "media_hash": "destination",
        "performances": [{"id": "scene1", "start": 0.0, "end": 4.0, "duration": 4.0, "speaker_sequence": ["s1", "s2"], "dialogue_density": 0.8}],
    })
    write_json(destination / "speaker_map.json", {
        "media_hash": "destination", "requested_backend": "pyannote", "actual_backend": "pyannote_partial", "speaker_count": 2,
        "speakers": [{"speaker_id": "s1", "total_duration": 3.0, "event_count": 2, "confidence": 0.9}],
    })
    pipeline = type("Pipeline", (), {
        "source": type("Source", (), {"cache_dir": source})(),
        "destination": type("Destination", (), {"cache_dir": destination})(),
    })()

    paths = materialize_required_artifacts(
        pipeline=pipeline,
        required_artifacts=("dialogue_events", "performances", "speakers", "scenes", "speaker_graph"),
        output_dir=output,
        schedule={"speaker_graph": {"s1": {"s2": 2.5}}},
    )

    assert set(paths) == {"dialogue_events", "performances", "speakers", "scenes", "speaker_graph"}
    assert read_json(paths["speakers"])["speakers"][0]["speaker_id"] == "s1"
    assert read_json(paths["scenes"])["scenes"][0]["participating_speakers"] == ["s1", "s2"]
    assert not (output / "shots.json").exists()
