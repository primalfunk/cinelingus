import subprocess

from movie_masher import tools


def test_run_uses_utf8_replacement_decoding(monkeypatch) -> None:
    captured = {}

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools.run(["ffprobe"])

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
