from pathlib import Path

import pytest

from movie_masher.presets import list_presets, load_preset


def test_list_presets_loads_default_preset_contracts() -> None:
    presets = {preset.id: preset for preset in list_presets(Path.cwd())}

    assert "movie_masher" in presets
    assert "self_shuffle" in presets
    assert presets["movie_masher"].transformation_strategy == "movie_masher"
    assert presets["self_shuffle"].parameters["seed"]["default"] == 1


def test_load_preset_reports_available_choices() -> None:
    with pytest.raises(ValueError, match="Available presets"):
        load_preset(Path.cwd(), "missing")
