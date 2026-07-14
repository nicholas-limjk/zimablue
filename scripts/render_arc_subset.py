from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("Category 1", ROOT / "examples/arc/category_1_good_fit/007bbfb7_grid_expansion.json"),
    ("Category 1", ROOT / "examples/arc/category_1_good_fit/1cf80156_object_extraction.json"),
    ("Category 1", ROOT / "examples/arc/category_1_good_fit/0520fde7_spatial_mapping.json"),
    ("Category 2", ROOT / "examples/arc/category_2_reflection_helps/0ca9ddb6_marker_to_shape_revision.json"),
    ("Category 2", ROOT / "examples/arc/category_2_reflection_helps/28e73c20_boundary_fill_revision.json"),
    ("Category 2", ROOT / "examples/arc/category_2_reflection_helps/3de23699_selector_revision.json"),
]
PALETTE = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
]


def main() -> int:
    img = Image.new("RGB", (1280, 900), (250, 248, 242))
    draw = ImageDraw.Draw(img)
    title = load_font(26)
    font = load_font(18)
    small = load_font(14)
    draw.text((36, 24), "Selected ARC Tasks For Skill Reflection", fill=(20, 30, 40), font=title)

    for i, (category, path) in enumerate(TASKS):
        task = json.loads(path.read_text(encoding="utf-8"))
        x = 42 + (i % 3) * 410
        y = 88 + (i // 3) * 380
        draw.text((x, y), category, fill=(88, 101, 114), font=small)
        draw.text((x, y + 24), path.stem, fill=(20, 30, 40), font=font)
        pair = task["train"][0]
        draw.text((x, y + 64), "train input", fill=(88, 101, 114), font=small)
        draw_grid(draw, pair["input"], x, y + 88)
        draw.text((x + 170, y + 154), "->", fill=(40, 40, 40), font=font)
        draw.text((x + 218, y + 64), "train output", fill=(88, 101, 114), font=small)
        draw_grid(draw, pair["output"], x + 218, y + 88)
    out = ROOT / "artifacts" / "arc_selected_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)
    return 0


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, max_size: int = 140) -> None:
    height = len(grid)
    width = len(grid[0])
    cell = max(3, min(max_size // max(width, height), 18))
    draw.rectangle((x - 2, y - 2, x + width * cell + 2, y + height * cell + 2), fill=(30, 38, 46))
    for yy, row in enumerate(grid):
        for xx, value in enumerate(row):
            draw.rectangle(
                (x + xx * cell, y + yy * cell, x + (xx + 1) * cell, y + (yy + 1) * cell),
                fill=PALETTE[value],
                outline=(220, 220, 220),
            )


def load_font(size: int):
    for path in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
