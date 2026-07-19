from pathlib import Path

from cinelingus.gui import CinelingusInstrumentApp
from cinelingus.util import write_json


def test_instrument_reports_experimental_without_evidence(tmp_path: Path) -> None:
    state = CinelingusInstrumentApp._certification_state(tmp_path, "time.foreshadow")
    assert state.startswith("EXPERIMENTAL")


def test_instrument_reads_evidence_derived_state(tmp_path: Path) -> None:
    path = tmp_path / "contracts" / "time_foreshadow" / "filter_certification.json"
    write_json(path, {"state": "CERTIFIED"})
    assert CinelingusInstrumentApp._certification_state(tmp_path, "time.foreshadow") == "CERTIFIED"
