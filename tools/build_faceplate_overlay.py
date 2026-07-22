from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MANIFEST_PATH = ASSETS / "instrument_apertures.json"
FONT_PATHS = {
    "machine_name": ASSETS / "fonts" / "cinzel" / "Cinzel-Variable.ttf",
    "panel_label": ASSETS / "fonts" / "ibm-plex-sans-condensed" / "IBMPlexSansCondensed-Medium.ttf",
}


def tracked_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
    widths = [draw.textlength(character, font=font) for character in text]
    return sum(widths) + max(0, len(text) - 1) * tracking


def draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    tracking: float,
    fill: tuple[int, int, int, int],
) -> None:
    x = center[0] - tracked_width(draw, text, font, tracking) / 2
    bounds = draw.textbbox((0, 0), text, font=font)
    y = center[1] - (bounds[3] - bounds[1]) / 2 - bounds[1]
    for character in text:
        draw.text((x, y), character, font=font, fill=fill)
        x += draw.textlength(character, font=font) + tracking


def aperture_mask(size: tuple[int, int], apertures: list[dict]) -> Image.Image:
    mask = Image.new("L", size, 255)
    draw = ImageDraw.Draw(mask)
    for aperture in apertures:
        shape = aperture["shape"]
        if shape == "rounded_rect":
            draw.rounded_rectangle(aperture["bounds"], radius=aperture["radius"], fill=0)
        elif shape == "ellipse":
            draw.ellipse(aperture["bounds"], fill=0)
        elif shape == "polygon":
            draw.polygon([tuple(point) for point in aperture["points"]], fill=0)
        elif shape == "circles":
            radius = aperture["radius"]
            for x, y in aperture["centers"]:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)
        else:
            raise ValueError(f"Unknown aperture shape: {shape}")
    return mask


def chamfered_box(center: tuple[float, float], width: float, height: float, chamfer: float) -> list[tuple[float, float]]:
    left = center[0] - width / 2
    right = center[0] + width / 2
    top = center[1] - height / 2
    bottom = center[1] + height / 2
    return [
        (left + chamfer, top), (right - chamfer, top), (right, top + chamfer),
        (right, bottom - chamfer), (right - chamfer, bottom), (left + chamfer, bottom),
        (left, bottom - chamfer), (left, top + chamfer),
    ]


def draw_nameplate(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    chamfer: float,
) -> None:
    """Lay an engraved dark-metal plaque over the plate's ornamental rule."""
    draw.polygon(
        chamfered_box((center[0], center[1] + 2), width, height, chamfer),
        fill=(5, 6, 6, 220),
    )
    draw.polygon(
        chamfered_box(center, width, height, chamfer),
        fill=(23, 24, 21, 255),
        outline=(104, 83, 48, 255),
        width=1,
    )
    inset = 2
    draw.line(
        chamfered_box(center, width - inset * 2, height - inset * 2, max(2, chamfer - inset))[:4],
        fill=(158, 126, 72, 150),
        width=1,
    )


def build_overlay() -> Path:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    size = tuple(manifest["design_size"])
    plate = Image.open(ASSETS / manifest["plate_source"]).convert("RGBA")
    if plate.size != size:
        raise ValueError(f"Plate size {plate.size} does not match manifest size {size}.")
    plate.putalpha(aperture_mask(size, manifest["apertures"]))
    draw = ImageDraw.Draw(plate)

    title = manifest["title"]
    draw_nameplate(
        draw,
        center=tuple(title["center"]),
        width=float(title.get("plate_width", 458)),
        height=float(title.get("plate_height", 36)),
        chamfer=float(title.get("plate_chamfer", 8)),
    )
    for label in manifest["panel_labels"]:
        draw_nameplate(
            draw,
            center=tuple(label["center"]),
            width=float(label.get("plate_width", 100)),
            height=float(label.get("plate_height", 20)),
            chamfer=float(label.get("plate_chamfer", 4)),
        )
    for label in manifest.get("viewport_labels", []):
        if "plate_width" not in label:
            continue
        draw_nameplate(
            draw,
            center=tuple(label["center"]),
            width=float(label["plate_width"]),
            height=float(label.get("plate_height", 18)),
            chamfer=float(label.get("plate_chamfer", 3)),
        )

    title_font = ImageFont.truetype(FONT_PATHS["machine_name"], 31)
    try:
        title_font.set_variation_by_name("Bold")
    except (AttributeError, OSError):
        pass
    tracking = 31 * float(title["tracking_em"])
    for offset, color in (((0, 2), (18, 13, 8, 230)), ((0, 0), (214, 190, 132, 255))):
        draw_tracked_text(
            draw,
            center=(title["center"][0] + offset[0], title["center"][1] + offset[1]),
            text=title["text"],
            font=title_font,
            tracking=tracking,
            fill=color,
        )

    label_font = ImageFont.truetype(FONT_PATHS["panel_label"], 14)
    for label in manifest["panel_labels"]:
        for offset, color in (((0, 1), (20, 15, 9, 235)), ((0, 0), (196, 173, 119, 255))):
            draw_tracked_text(
                draw,
                center=(label["center"][0] + offset[0], label["center"][1] + offset[1]),
                text=label["text"].upper(),
                font=label_font,
                tracking=-0.4,
                fill=color,
            )

    for label in manifest.get("viewport_labels", []):
        font = ImageFont.truetype(FONT_PATHS["panel_label"], int(label.get("size", 11)))
        position = label.get("position")
        if position is not None:
            draw.text(tuple(position), label["text"].upper(), font=font, fill=(198, 177, 126, 255))
        else:
            draw_tracked_text(
                draw,
                center=tuple(label["center"]),
                text=label["text"].upper(),
                font=font,
                tracking=-0.25,
                fill=(198, 177, 126, 255),
            )

    output = ASSETS / manifest["plate_overlay"]
    plate.save(output, optimize=True)
    return output


if __name__ == "__main__":
    print(build_overlay())
