from pathlib import Path

from cinelingus.cinematic_model.corpus_validation import CorpusCase, validate_corpus_cases
from cinelingus.util import write_json


def test_bounded_corpus_measurement_is_deterministic_and_non_destructive(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"read-only-source-sentinel")
    write_json(artifact_dir / "movie.json", {
        "schema_version": "1.0", "tool_version": "fixture", "media_hash": "f" * 64,
        "path": str(source), "duration": 2.0, "resolution": "640x360", "frame_rate": 24.0,
        "sample_rate": 48000, "channels": 2, "codec": "h264", "streams": [],
    })
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    report = validate_corpus_cases(
        [CorpusCase("fixture", "short_form", artifact_dir, "partial")],
        schemas_dir=Path("schemas"), output_root=tmp_path / "models",
    )
    after = (source.stat().st_size, source.stat().st_mtime_ns)
    assert report["all_valid"] is True
    assert report["all_deterministic"] is True
    assert report["all_cache_hits"] is True
    assert report["source_media_unchanged"] is True
    assert before == after
    assert report["cases"][0]["capability_status_counts"]["AVAILABLE"] == 1
