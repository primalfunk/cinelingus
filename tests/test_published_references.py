from pathlib import Path

from movie_masher.pipeline import _rewrite_published_video_references
from movie_masher.util import read_json, write_json


def test_published_video_references_follow_the_surviving_deliverable(tmp_path: Path) -> None:
    rendered = tmp_path / "output" / "runs" / "final.mp4"
    published = tmp_path / "output" / "cinelingus_possession-short_2026-07-13_13-17-30.mp4"
    report = tmp_path / "output" / "runs" / "report.json"
    report.parent.mkdir(parents=True)
    write_json(
        report,
        {
            "outputs": {"video": str(rendered)},
            "relative_outputs": {"video": rendered.relative_to(tmp_path).as_posix()},
        },
    )

    _rewrite_published_video_references(
        artifact_paths=[report],
        rendered_video=rendered,
        published_video=published,
        root=tmp_path,
    )

    updated = read_json(report)
    assert updated["outputs"]["video"] == str(published.resolve())
    assert updated["relative_outputs"]["video"] == published.relative_to(tmp_path).as_posix()
