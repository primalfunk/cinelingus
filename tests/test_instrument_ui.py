from pathlib import Path

from PIL import Image

from cinelingus.gui import CURATOR_SELECTIONS, CinelingusInstrumentApp
from cinelingus.instrument_ui import (
    INSTRUMENT_COLORS,
    INSTRUMENT_OVERLAY_BOXES,
    fit_plate_bounds,
    instrument_plate_path,
    meter_fraction,
    selector_angle,
)


def test_instrument_plate_is_project_bound_and_has_reference_proportions() -> None:
    path = instrument_plate_path(Path.cwd())

    assert path == Path.cwd() / "assets" / "instrument_plate.png"
    assert path.exists()
    with Image.open(path) as image:
        assert image.size == (1536, 1024)


def test_plate_bounds_preserve_aspect_ratio_and_center_margins() -> None:
    assert fit_plate_bounds(1200, 1000) == (0, 100, 1200, 800)
    assert fit_plate_bounds(1800, 900) == (225, 0, 1350, 900)


def test_every_live_overlay_stays_inside_the_faceplate() -> None:
    assert set(INSTRUMENT_OVERLAY_BOXES) == {
        "transformation", "material", "quality", "filter", "status",
        "activate", "progress", "stages", "curator", "notes",
    }
    for box in INSTRUMENT_OVERLAY_BOXES.values():
        assert 0 <= box.x < 1
        assert 0 <= box.y < 1
        assert box.width > 0 and box.height > 0
        assert box.x + box.width <= 1
        assert box.y + box.height <= 1


def test_instrument_exposes_required_curator_observations() -> None:
    assert tuple(CURATOR_SELECTIONS) == (
        "Most Convincing",
        "Beautiful Accident",
        "Unstable",
        "Rare Alignment",
        "Worth Revisiting",
        "Needs Attention",
    )
    assert CinelingusInstrumentApp.__name__ == "CinelingusInstrumentApp"

def test_rotary_calibration_spans_the_instrument_arc() -> None:
    assert selector_angle(0, 5) == 225.0
    assert selector_angle(2, 5) == 90.0
    assert selector_angle(4, 5) == -45.0
    assert selector_angle(-8, 5) == 225.0
    assert selector_angle(99, 5) == -45.0


def test_meter_fraction_is_bounded_for_runtime_progress() -> None:
    assert meter_fraction(-10) == 0.0
    assert meter_fraction(25) == 0.25
    assert meter_fraction(250) == 1.0
    assert meter_fraction(20, maximum=0) == 0.0


def test_instrument_color_grammar_has_distinct_material_and_emitted_light() -> None:
    assert INSTRUMENT_COLORS["surface"] != INSTRUMENT_COLORS["surface_deep"]
    assert INSTRUMENT_COLORS["brass"] != INSTRUMENT_COLORS["cyan"]
    assert INSTRUMENT_COLORS["text"] != INSTRUMENT_COLORS["muted"]
