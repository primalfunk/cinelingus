import importlib.util

from movie_masher.module_probe import resilient_find_spec


def test_module_probe_retries_transient_windows_oserror(monkeypatch) -> None:
    expected = object()
    outcomes = [OSError(6714, "invalid transaction handle"), expected]

    def fake_find_spec(_name):
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    spec, error = resilient_find_spec("pyannote.audio")
    assert spec is expected
    assert error is None


def test_module_probe_returns_actionable_error_after_retries(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: (_ for _ in ()).throw(OSError(6714, "invalid transaction handle")))
    spec, error = resilient_find_spec("pyannote.audio", attempts=2)
    assert spec is None
    assert "6714" in error
