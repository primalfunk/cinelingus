import json
import os

from cinelingus.util import write_json


def test_write_json_atomically_replaces_target_without_leaving_temporary_files(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"old": true}', encoding="utf-8")

    write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.glob(".artifact.json.*.tmp")) == []


def test_write_json_retries_transient_windows_reader_contention(monkeypatch, tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    real_replace = os.replace
    attempts = []

    def contested_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError(5, "simulated checkpoint reader contention")
        return real_replace(source, destination)

    monkeypatch.setattr("cinelingus.util.os.replace", contested_replace)
    monkeypatch.setattr("cinelingus.util.time.sleep", lambda _seconds: None)

    write_json(path, {"stage": "editorial_candidate_prepared"})

    assert len(attempts) == 3
    assert json.loads(path.read_text(encoding="utf-8"))["stage"] == "editorial_candidate_prepared"
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []
