from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter


VARIANTS = {
    "cinelingus_emblem_header.png": 68,
    "cinelingus_emblem_hero.png": 240,
}


def resize_emblem(source: Image.Image, size: int) -> Image.Image:
    resized = source.resize((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
    alpha = resized.getchannel("A")
    rgb = resized.convert("RGB").filter(ImageFilter.UnsharpMask(radius=0.65, percent=75, threshold=2))
    red, green, blue = rgb.split()
    return Image.merge("RGBA", (red, green, blue, alpha))


def build_variants(source_path: Path, output_dir: Path) -> list[Path]:
    source = Image.open(source_path).convert("RGBA")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, size in VARIANTS.items():
        output = output_dir / name
        resize_emblem(source, size).save(output, optimize=True)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-quality Cinelingus GUI emblem sizes.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for output in build_variants(args.source, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
