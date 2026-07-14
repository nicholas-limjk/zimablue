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
    (
        "46f33fce",
        "Birth task, solved by composition",
        "rotate180 -> paste_4x4_block_for_each_nonzero_cell_on_2x_scaled_canvas -> rotate180",
        "The LLM wrote the block-paste operator; search discovered it needed orientation wrappers.",
    ),
    (
        "6fa7a44f",
        "Held-out transfer",
        "flip_v -> vertical_reflect_and_stack",
        "A learned vertical stack operator transfers when paired with a base flip.",
    ),
    (
        "8be77c9e",
        "Held-out transfer",
        "flip_v -> vertical_reflect_and_stack",
        "Same learned operator reused on a different tile-grid task.",
    ),
    (
        "178fcbfb",
        "Growth task, operator + base primitives",
        "expand_seed_to_row_or_column -> complete_rows_color_3 -> complete_rows_color_1",
        "The new operator alone was partial; base line-completion primitives closed the task.",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render examples of learned ARC skills composed with base primitives.")
    parser.add_argument("--out", default="artifacts/arc_curriculum/skill_composition_examples.png")
    args = parser.parse_args()

    w, h = 1500, 820
    img = Image.new("RGB", (w, h), (246, 247, 249))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title_font = load_font(25)
    h2_font = load_font(17)

    draw.text((34, 26), "How The Learned Skills Combine", fill=(20, 24, 31), font=title_font)
    draw.text((34, 62), "The LLM writes operators; deterministic composition search chooses the sequence that passes train examples.", fill=(83, 91, 104), font=font)

    card_h = 168
    for i, (task_id, label, seq, note) in enumerate(EXAMPLES):
        y = 104 + i * card_h
        task = load_task(task_id)
        pair = task["train"][0]
        draw.rounded_rectangle((26, y, w - 26, y + card_h - 14), radius=9, fill=(255, 255, 255), outline=(216, 222, 230))
        draw.text((48, y + 18), task_id, fill=(20, 24, 31), font=h2_font)
        draw.text((48, y + 46), label, fill=(34, 132, 91), font=font)
        draw.text((218, y + 16), "input", fill=(83, 91, 104), font=font)
        draw_grid(draw, pair["input"], 218, y + 38, 98)
        draw.text((360, y + 16), "output", fill=(83, 91, 104), font=font)
        draw_grid(draw, pair["output"], 360, y + 38, 98)
        draw.text((525, y + 16), "winning sequence", fill=(83, 91, 104), font=font)
        draw_wrapped(draw, seq, 525, y + 42, 880, fill=(20, 24, 31), font=h2_font, line_h=22)
        draw_wrapped(draw, note, 525, y + 98, 880, fill=(83, 91, 104), font=font, line_h=17)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)
    return 0


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, box: int) -> None:
    rows = len(grid)
    cols = len(grid[0])
    cell = max(2, min(box // max(rows, cols), 17))
    ox = x + (box - cols * cell) // 2
    oy = y + (box - rows * cell) // 2
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            draw.rectangle((ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 1, oy + (r + 1) * cell - 1), fill=COLORS.get(int(value), (60, 60, 60)))


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fill, font, line_h: int) -> None:
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
        draw.text((x, y + i * line_h), item, fill=fill, font=font)


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
