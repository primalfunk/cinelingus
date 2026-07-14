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


def test_legacy_movie_masher_launcher_still_imports() -> None:
    runner = load_runner("run_movie_masher.py")

    args = runner.build_parser().parse_args([])

    assert args.action == "gui"


def test_orchestrator_parser_accepts_preview_mappings(tmp_path: Path) -> None:
    runner = load_runner()

    args = runner.build_parser().parse_args(["preview", "--mapping", "2", "--mapping", "4", "--output-dir", str(tmp_path)])

    assert args.mapping == [2, 4]
    assert args.output_dir == tmp_path


def test_default_preview_indices_prefers_cross_shot_mapping() -> None:
    runner = load_runner()

    class FakePipeline:
        def schedule(self, force=False):
            return {
                "mappings": [
                    {"mapping_crosses_shot_boundary": False},
                    {"mapping_crosses_shot_boundary": True},
                ]
            }

    assert runner.default_preview_indices(FakePipeline()) == [1]


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
    import run_movie_masher as runner

    args = runner.build_parser().parse_args(["problem-previews", "--max-regions", "2", "--output-dir", str(tmp_path)])

    assert args.action == "problem-previews"
    assert args.max_regions == 2




def test_mutate_requires_explicit_input_video(capsys) -> None:
    runner = load_runner()

    result = runner.main(["mutate", "--mutation", "echo"])

    captured = capsys.readouterr()
    assert result == 1
    assert "requires --input-video" in captured.err
