import importlib.util
from pathlib import Path

def load_runner(script_name: str = "run_cinelingus.py"):
    path = Path.cwd() / script_name
    spec = importlib.util.spec_from_file_location(Path(script_name).stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_orchestrator_parser_defaults_to_gui() -> None:
    runner = load_runner()

    args = runner.build_parser().parse_args([])

    assert args.action == "gui"


def test_root_launcher_dispatches_default_to_qt_gui_without_leaking_cli_arguments(monkeypatch) -> None:
    runner = load_runner()
    calls = []
    monkeypatch.setattr("run_translation.gui_main", lambda args: calls.append(args) or 0)

    assert runner.main([]) == 0
    assert calls == [[]]


def test_root_launcher_forwards_qt_window_and_explicit_legacy_options(monkeypatch) -> None:
    runner = load_runner()
    calls = []
    monkeypatch.setattr("run_translation.gui_main", lambda args: calls.append(args) or 0)

    assert runner.main(["gui", "--windowed"]) == 0
    assert runner.main(["gui", "--legacy-tk"]) == 0
    assert calls == [["--windowed"], ["--legacy-tk"]]


def test_legacy_translation_launcher_still_imports() -> None:
    runner = load_runner("run_cinelingus.py")

    args = runner.build_parser().parse_args([])

    assert args.action == "gui"


def test_orchestrator_parser_rejects_removed_preview_action(tmp_path: Path) -> None:
    runner = load_runner()

    import pytest
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["preview", "--output-dir", str(tmp_path)])


def test_orchestrator_parser_accepts_mutation_action(tmp_path: Path) -> None:
    runner = load_runner()

    args = runner.build_parser().parse_args([
        "mutate",
        "--input-video",
        str(tmp_path / "film.mp4"),
        "--mutation",
        "drift",
    ])

    assert args.action == "mutate"
    assert args.input_video == tmp_path / "film.mp4"
    assert args.mutation == "drift"


def test_orchestrator_parser_accepts_problem_previews(tmp_path: Path) -> None:
    import run_translation as runner

    args = runner.build_parser().parse_args(["problem-previews", "--max-regions", "2", "--output-dir", str(tmp_path)])

    assert args.action == "problem-previews"
    assert args.max_regions == 2


def test_orchestrator_parser_accepts_performance_previews(tmp_path: Path) -> None:
    import run_translation as runner

    args = runner.build_parser().parse_args(["performance-previews", "--max-regions", "2", "--output-dir", str(tmp_path)])

    assert args.action == "performance-previews"
    assert args.max_regions == 2


def test_orchestrator_parser_accepts_quality_corpus(tmp_path: Path) -> None:
    runner = load_runner()

    args = runner.build_parser().parse_args([
        "quality-corpus",
        "--corpus-manifest", str(tmp_path / "corpus.json"),
        "--runs-root", str(tmp_path / "runs"),
    ])

    assert args.action == "quality-corpus"
    assert args.corpus_manifest == tmp_path / "corpus.json"




def test_mutate_requires_explicit_input_video(capsys) -> None:
    runner = load_runner()

    result = runner.main(["mutate", "--mutation", "echo"])

    captured = capsys.readouterr()
    assert result == 1
    assert "requires --input-video" in captured.err
