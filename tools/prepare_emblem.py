from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


def ellipse_mask(size: tuple[int, int], inset: tuple[int, int, int, int], *, blur: float = 0.0) -> Image.Image:
    width, height = size
    left, top, right, bottom = inset
    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).ellipse((left, top, width - right, height - bottom), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(blur)) if blur else mask


def prepare_emblem(source: Path, keyed_matte: Path, output: Path) -> None:
    original = Image.open(source).convert('RGBA')
    keyed_alpha = Image.open(keyed_matte).convert('RGBA').getchannel('A')
    if keyed_alpha.size != original.size:
        raise ValueError('The keyed matte must match the source dimensions.')

    # The supplied artwork is a circular opaque coin. Preserve its complete
    # interior and use the color-derived matte only on the narrow outside rim.
    inner_opaque = ellipse_mask(original.size, (20, 16, 20, 20))
    outer_cutoff = ellipse_mask(original.size, (9, 5, 9, 9), blur=1.0)
    rim_alpha = ImageChops.darker(keyed_alpha, outer_cutoff)
    final_alpha = ImageChops.lighter(inner_opaque, rim_alpha)

    original.putalpha(final_alpha)
    output.parent.mkdir(parents=True, exist_ok=True)
    original.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare the supplied Cinelingus coin for transparent GUI use.')
    parser.add_argument('source', type=Path)
    parser.add_argument('keyed_matte', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare_emblem(args.source, args.keyed_matte, args.output)


if __name__ == '__main__':
    main()
