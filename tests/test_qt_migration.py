from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QWidget

from cinelingus.qt_controller import QtEngineController, _quality_summary_from_problem_report
from cinelingus.qt_faceplate import ConfigurationDialog, FaceplateWidget


ROOT = Path.cwd()


def test_problem_report_summary_normalization_tolerates_legacy_and_invalid_shapes() -> None:
    assert _quality_summary_from_problem_report({
        "problem_count": "3",
        "summary": {"possible_residue_count": 1},
    }) == {"problem_count": 3, "possible_residue_count": 1}
    assert _quality_summary_from_problem_report({"problem_count": 4, "summary": 4}) == {"problem_count": 4}
    assert _quality_summary_from_problem_report(["partial", "report"]) == {}


def test_production_gui_module_does_not_import_tk_until_legacy_surface_is_requested() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import cinelingus.gui; assert 'tkinter' not in sys.modules",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_production_launcher_routes_to_qt_screenshot(tmp_path: Path) -> None:
    from cinelingus.gui import main

    output = tmp_path / "production-qt.png"
    assert main(["--screenshot", str(output), "--state", "ready"]) == 0
    assert Image.open(output).size == (1536, 1024)


def test_controller_builds_medium_whisper_configuration_from_default_scrutiny(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    controller = QtEngineController(ROOT)
    films = [tmp_path / "anchor.mp4", tmp_path / "donor.mp4"]
    for film in films:
        film.write_bytes(b"media")
    controller.configure(
        reality=controller.state.reality,
        discipline=controller.state.discipline,
        apparatus=controller.state.apparatus,
        films=films,
        output_dir=tmp_path / "output",
        quality="Divination",
        matching="Balanced",
        parameters=controller.definition.parameter_defaults,
    )

    config = controller.selected_config()

    assert config.whisper_model == "medium"
    assert controller.whisper_model == "medium"
    assert config.transcription_mode == "quality"
    assert config.films == tuple(path.resolve() for path in films)

    recipe_path = controller.save_recipe(tmp_path / "recipe.json")
    controller.cycle_matching()
    controller.load_recipe(recipe_path)
    assert controller.state.apparatus == "Transposition"
    assert controller.state.matching == "Balanced"
    assert controller.state.films == films


def test_configuration_dialog_exposes_all_required_production_controls() -> None:
    _app = QApplication.instance() or QApplication([])
    controller = QtEngineController(ROOT)
    dialog = ConfigurationDialog(controller)

    assert dialog.reality.currentText() == controller.state.reality
    assert dialog.discipline.currentText() == controller.state.discipline
    assert dialog.apparatus.currentText().startswith(controller.state.apparatus)
    assert dialog.films.count() == controller.definition.minimum_films
    assert set(dialog.parameter_widgets) == {parameter.id for parameter in controller.definition.parameters}
    dialog.close()


def test_live_controller_state_drives_faceplate_without_rectangular_child_widgets() -> None:
    _app = QApplication.instance() or QApplication([])
    controller = QtEngineController(ROOT)
    widget = FaceplateWidget(ROOT, controller)

    assert widget.findChildren(QWidget) == []
    assert widget.keyboard_focus_visible is False
    assert widget.focusNextPrevChild(True) is True
    assert widget.keyboard_focus_visible is True
    controller._events.put(("stage", "inspect"))
    controller._drain_events()
    assert controller.state.active_stage_index == 0
    assert controller.state.operation == "CATALOG"
    assert controller.state.overall_progress == 0.06

    controller._events.put(("stage", "multiworld:apply_cinematic_law"))
    controller._drain_events()
    assert controller.state.active_stage_index == 4
    assert controller.state.operation == "APPLYING THE CINEMATIC LAW"


def test_completed_performance_summary_is_exposed_to_the_ledger(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    controller = QtEngineController(ROOT)
    output = tmp_path / "translation_output.mp4"
    output.write_bytes(b"video")
    controller._events.put(("complete", {
        "output": output,
        "performance_summary": {
            "destination_performance_count": 24,
            "performance_couplings": 18,
            "adapted_performances": 4,
            "turn_sequence_matches": 1,
            "linewise_fallbacks": 1,
            "preserved_original_regions": 2,
        },
        "quality_summary": {"problem_count": 3},
    }))

    controller._drain_events()

    assert controller.state.completed is True
    assert controller.state.performance_summary["performance_couplings"] == 18
    assert "COUPLINGS 18/24" in controller.state.summary
    assert "PRESERVED 2" in controller.state.summary
    assert "REVIEW 3" in controller.state.summary
