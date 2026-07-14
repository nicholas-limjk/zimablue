from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "arc_agi_source" / "data" / "training"
OUT_DIR = ROOT / "artifacts"
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
    files = sorted(TASK_DIR.glob("*.json"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(files), 50)):
        render_page(files[start : start + 50], page)
    return 0


def render_page(files: list[Path], page: int) -> None:
    font = load_font(12)
    img = Image.new("RGB", (1000, 1200), (250, 248, 242))
    draw = ImageDraw.Draw(img)
    for i, path in enumerate(files):
        task = json.loads(path.read_text(encoding="utf-8"))
        col = i % 5
        row = i // 5
        x = col * 200 + 8
        y = row * 118 + 8
        draw.text((x, y), path.stem, fill=(20, 30, 40), font=font)
        pair = task["train"][0]
        draw_grid(draw, pair["input"], x, y + 18)
        draw.text((x + 72, y + 36), "->", fill=(40, 40, 40), font=font)
        draw_grid(draw, pair["output"], x + 96, y + 18)
    out = OUT_DIR / f"arc_training_sheet_{page:02d}.png"
    img.save(out)
    print(out)


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, max_size: int = 58) -> None:
    height = len(grid)
    width = len(grid[0])
    cell = max(2, min(max_size // max(width, height), 12))
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
