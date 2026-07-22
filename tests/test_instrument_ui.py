from pathlib import Path

import pytest
from PIL import Image

from cinelingus.gui import CURATOR_SELECTIONS, CinelingusInstrumentApp
from cinelingus.instrument_ui import (
    CONTROL_STATES,
    INSTRUMENT_COLORS,
    INSTRUMENT_DIMENSIONS,
    INSTRUMENT_FONTS,
    INSTRUMENT_OVERLAY_BOXES,
    INSTRUMENT_RECESS_BOXES,
    INSTRUMENT_SPACING,
    OverlayBox,
    concise_material_name,
    control_palette,
    ellipsize_text,
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


def test_every_live_overlay_leaves_the_plate_recess_bezel_visible() -> None:
    assert set(INSTRUMENT_OVERLAY_BOXES) == set(INSTRUMENT_RECESS_BOXES)
    plate_bounds = (0, 0, 1536, 1024)

    for name, overlay in INSTRUMENT_OVERLAY_BOXES.items():
        recess = INSTRUMENT_RECESS_BOXES[name]
        ox, oy, ow, oh = overlay.pixels(plate_bounds)
        rx, ry, rw, rh = recess.pixels(plate_bounds)

        assert ox > rx
        assert oy > ry
        assert ox + ow < rx + rw
        assert oy + oh < ry + rh


def test_overlay_box_inset_rejects_invalid_gutters() -> None:
    box = OverlayBox(0.1, 0.2, 0.3, 0.4)
    inset = box.inset(0.01, 0.02)
    assert (inset.x, inset.y, inset.width, inset.height) == pytest.approx((0.11, 0.22, 0.28, 0.36))

    with pytest.raises(ValueError):
        box.inset(-0.01, 0.0)
    with pytest.raises(ValueError):
        box.inset(0.15, 0.0)


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


def test_machine_control_system_centralizes_geometry_typography_and_states() -> None:
    assert INSTRUMENT_SPACING["unit"] == 8
    assert INSTRUMENT_DIMENSIONS["standard_height"] == 32
    assert set(INSTRUMENT_FONTS) == {"display", "display_large", "technical", "caps"}
    assert set(CONTROL_STATES) == {"normal", "hover", "focused", "pressed", "selected", "disabled", "active", "warning", "failed"}
    assert control_palette("active")["border"] == INSTRUMENT_COLORS["cyan"]
    assert control_palette("warning")["border"] == INSTRUMENT_COLORS["amber"]
    assert control_palette("failed")["border"] == INSTRUMENT_COLORS["red"]


def test_material_trough_names_are_concise_but_paths_remain_external() -> None:
    assert concise_material_name("") == "EMPTY CHAMBER"
    assert concise_material_name(r"C:\\films\\Robocop.mp4") == "Robocop.mp4"
    shortened = ellipsize_text("an_extremely_long_specimen_filename_that_needs_truncation.mp4", 24)
    assert "…" in shortened
    assert len(shortened) <= 24
    concise_material_name,
    control_palette,
    ellipsize_text,
