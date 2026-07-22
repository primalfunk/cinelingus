from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from cinelingus.qt_faceplate import DESIGN_HEIGHT, DESIGN_WIDTH, load_manifest, marquee_offset, render_screenshot


ROOT = Path.cwd()


def test_faceplate_manifest_uses_requested_typography_and_distinct_apertures() -> None:
    manifest = load_manifest(ROOT)
    ids = [row["id"] for row in manifest["apertures"]]
    apertures = {row["id"]: row for row in manifest["apertures"]}

    assert manifest["design_size"] == [1536, 1024]
    assert manifest["title"]["text"] == "CINELINGUS ENGINE"
    assert manifest["title"]["center"] == [768, 48]
    assert manifest["title"]["font_role"] == "machine_name"
    assert manifest["title"]["tracking_em"] == 0.10
    assert manifest["title"]["plate_width"] == 458
    assert len(ids) == len(set(ids))
    assert {"apparatus", "materials", "materials_secondary", "scrutiny", "calibration", "observation", "actuation", "stations", "curator", "ledger"}.issubset(ids)
    assert {row["text"] for row in manifest["panel_labels"]} >= {
        "APPARATUS", "MATERIALS", "SCRUTINY", "CALIBRATION", "OBSERVATION", "ACTUATION", "PROCESSION", "CURATOR", "LEDGER", "SERVICE"
    }
    assert {center[1] for center in apertures["stations"]["centers"]} == {774}
    assert apertures["actuation"]["bounds"] == [1190, 350, 1392, 552]
    assert apertures["overall_progress"]["bounds"] == [400, 611, 1136, 634]
    assert apertures["stage_progress"]["bounds"] == [400, 682, 1136, 705]
    panel_labels = {row["text"]: row for row in manifest["panel_labels"]}
    assert panel_labels["CURATOR"]["center"] == [410, 829]
    assert panel_labels["LEDGER"]["center"] == [1125, 829]
    assert panel_labels["SERVICE"]["center"] == [1450, 974]
    viewport_labels = {row["text"] for row in manifest["viewport_labels"]}
    assert {"REALITY", "DISCIPLINE", "FILM A", "FILM B", "FILM C", "OVERALL", "STAGE", "CATALOG", "REVIEW"}.issubset(viewport_labels)
    metric_labels = {
        row["text"]: row.get("center")
        for row in manifest["viewport_labels"]
        if row["text"] in {"ELAPSED", "REMAINING", "COMPLETION"}
    }
    assert metric_labels == {
        "ELAPSED": [590, 497],
        "REMAINING": [768, 497],
        "COMPLETION": [946, 497],
    }
    operation = next(row for row in manifest["viewport_labels"] if row["text"] == "OPERATION")
    assert operation["center"] == [838, 366]


def test_bundled_fonts_and_open_font_licenses_are_present() -> None:
    required = (
        ROOT / "assets/fonts/cinzel/Cinzel-Variable.ttf",
        ROOT / "assets/fonts/ibm-plex-sans-condensed/IBMPlexSansCondensed-Medium.ttf",
        ROOT / "assets/fonts/share-tech-mono/ShareTechMono-Regular.ttf",
    )
    assert all(path.exists() and path.stat().st_size > 10_000 for path in required)
    assert all((path.parent / "OFL.txt").exists() for path in required)


def test_faceplate_overlay_is_rgba_and_exposes_shaped_viewports() -> None:
    manifest = load_manifest(ROOT)
    overlay = Image.open(ROOT / "assets" / manifest["plate_overlay"])

    assert overlay.mode == "RGBA"
    assert overlay.size == (DESIGN_WIDTH, DESIGN_HEIGHT)
    assert overlay.getchannel("A").getextrema() == (0, 255)
    assert overlay.getpixel((768, 440))[3] == 0
    assert overlay.getpixel((1290, 450))[3] == 0
    assert overlay.getpixel((20, 20))[3] == 255
    # The narrowed apertures preserve the brass collars and meter end caps.
    assert overlay.getpixel((1170, 450))[3] == 255
    assert overlay.getpixel((426, 774))[3] == 255
    assert overlay.getpixel((402, 774))[3] == 0
    assert overlay.getpixel((390, 617))[3] == 255
    assert overlay.getpixel((500, 617))[3] == 0
    assert overlay.getpixel((500, 694))[3] == 0


def test_marquee_holds_scrolls_and_resets_only_for_overflowing_values() -> None:
    assert marquee_offset(90, 100, 20) == 0
    assert marquee_offset(180, 100, 1.0) == 0
    assert marquee_offset(180, 100, 2.5) == -22
    cycle = 1.5 + (180 + 44) / 22
    assert marquee_offset(180, 100, cycle) == 0


def test_qt_faceplate_renders_canonical_and_dpi_review_sizes(tmp_path: Path) -> None:
    _app = QApplication.instance() or QApplication([])
    for scale in (1.0, 1.25, 1.5):
        output = render_screenshot(tmp_path / f"faceplate-{scale}.png", state="running", scale=scale)
        image = Image.open(output)
        assert image.size == (round(DESIGN_WIDTH * scale), round(DESIGN_HEIGHT * scale))
        assert output.stat().st_size > 100_000
