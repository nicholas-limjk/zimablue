from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    0: (24, 27, 31),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}


EXAMPLES = [
    ("62c24649", "Birth task", "learned reflect_2d_tile", "scale_2x / tile-grid"),
    ("67e8384a", "Transfer", "reflect_2d_tile reused, no LLM", "scale_2x / tile-grid"),
    ("6fa7a44f", "Held-out transfer", "flip_v -> vertical_reflect_and_stack", "expand cols / tile-grid"),
    ("8be77c9e", "Held-out transfer", "flip_v -> vertical_reflect_and_stack", "expand rows / tile-grid"),
    ("c9e6f938", "Held-out transfer", "horizontal_reflect_and_concatenate", "expand cols"),
    ("ac0a08a4", "Adjacent family", "scale_by_num_nonzero_colors", "expand both axes"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ARC curriculum transfer examples.")
    parser.add_argument("--out", default="artifacts/arc_curriculum/curriculum_transfer_examples.png")
    args = parser.parse_args()

    font = ImageFont.load_default()
    title_font = load_font(24)
    h2_font = load_font(17)
    w = 1450
    card_h = 180
    h = 110 + len(EXAMPLES) * card_h
    img = Image.new("RGB", (w, h), (246, 247, 249))
    draw = ImageDraw.Draw(img)

    draw.text((32, 24), "Examples: Tile-Grid Operators Transferring Across ARC Tasks", fill=(22, 26, 33), font=title_font)
    draw.text((32, 60), "Each row shows one train input/output pair from a task solved by the frozen grown body.", fill=(83, 91, 104), font=font)

    for i, (task_id, label, sequence, family) in enumerate(EXAMPLES):
        y = 96 + i * card_h
        task = load_task(task_id)
        pair = task["train"][0]
        draw.rounded_rectangle((24, y, w - 24, y + card_h - 16), radius=9, fill=(255, 255, 255), outline=(216, 222, 230))
        draw.text((44, y + 20), task_id, fill=(20, 24, 31), font=h2_font)
        draw.text((44, y + 48), label, fill=(35, 126, 88) if "Transfer" in label or "Adjacent" in label else (75, 83, 96), font=font)
        draw.text((44, y + 74), family, fill=(105, 113, 126), font=font)

        draw.text((230, y + 18), "train input", fill=(83, 91, 104), font=font)
        draw_grid(draw, pair["input"], 230, y + 42, 105)
        draw.text((390, y + 18), "train output", fill=(83, 91, 104), font=font)
        draw_grid(draw, pair["output"], 390, y + 42, 105)

        draw.text((570, y + 20), "operator sequence", fill=(83, 91, 104), font=font)
        draw_wrapped(draw, sequence, 570, y + 48, 760, fill=(20, 24, 31), font=h2_font)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)
    return 0


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, box: int) -> None:
    rows = len(grid)
    cols = len(grid[0])
    cell = max(2, min(box // max(rows, cols), 18))
    ox = x + (box - cols * cell) // 2
    oy = y + (box - rows * cell) // 2
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x0 = ox + c * cell
            y0 = oy + r * cell
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=COLORS.get(int(value), (60, 60, 60)))


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fill, font) -> None:
    words = text.split()
    line = ""
    lines = []
    for word in words:
        probe = f"{line} {word}".strip()
        if draw.textlength(probe, font=font) <= width:
            line = probe
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    for i, item in enumerate(lines[:4]):
        draw.text((x, y + i * 22), item, fill=fill, font=font)


def load_task(task_id: str):
    return json.loads((ROOT / "arc_agi_source" / "data" / "training" / f"{task_id}.json").read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
