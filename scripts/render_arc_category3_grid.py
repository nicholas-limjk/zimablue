from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IDS = ["045e512c", "025d127b", "a8c38be5", "b8825c91", "e5062a87", "f8a8fe49"]
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
GREEN = (54, 143, 93)
RED = (182, 68, 56)
AMBER = (183, 128, 36)
MUTED = (88, 101, 114)
INK = (23, 32, 42)
PAPER = (250, 248, 242)
CARD = (255, 253, 248)


def main() -> int:
    vanilla = json.loads((ROOT / "artifacts/arc_vanilla_hard_candidates_gpt5mini_low.json").read_text(encoding="utf-8"))
    vanilla_by_id = {Path(row["path"]).stem: row for row in vanilla["tasks"]}
    summary = json.loads((ROOT / "artifacts/arc_gpt5mini_low_hard_comparison.json").read_text(encoding="utf-8"))
    summary_by_id = {row["task"]: row for row in summary["rows"]}

    img = Image.new("RGB", (1800, 1300), PAPER)
    draw = ImageDraw.Draw(img)
    title = load_font(38)
    subtitle = load_font(20)
    font = load_font(18)
    small = load_font(13)
    badge = load_font(14)

    draw.text((42, 30), "Category 3 ARC Tasks", fill=INK, font=title)
    draw.text(
        (44, 78),
        "Hard cases: vanilla GPT-5 Mini low failed, and reflection did not solve the held-out test.",
        fill=MUTED,
        font=subtitle,
    )
    draw_metric(draw, 44, 116, "VANILLA", "0/6", RED)
    draw_metric(draw, 208, 116, "AFTER REFLECTION", "0/6", RED)
    draw_metric(draw, 454, 116, "TRAIN PASS TEST FAIL", "1", AMBER)

    for i, task_id in enumerate(IDS):
        task = json.loads((ROOT / "examples/arc/category_3_hard_failures" / f"{task_id}.json").read_text(encoding="utf-8"))
        vanilla_row = vanilla_by_id[task_id]
        row = summary_by_id[task_id]
        reflection = read_reflection(task_id)
        x = 42 + (i % 2) * 872
        y = 210 + (i // 2) * 348
        render_card(draw, x, y, task_id, task, vanilla_row, row, reflection, font, small, badge)

    out = ROOT / "artifacts/arc_category3_hard_failures_grid.png"
    img.save(out)
    print(out)
    return 0


def read_reflection(task_id: str) -> dict | None:
    deep = ROOT / "artifacts/arc_gpt5mini_low_hard_deep" / f"{task_id}_reflection.json"
    regular = ROOT / "artifacts/arc_gpt5mini_low_hard" / f"{task_id}_reflection.json"
    path = deep if deep.exists() else regular
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def render_card(draw, x, y, task_id, task, vanilla_row, row, reflection, font, small, badge_font) -> None:
    draw.rounded_rectangle((x, y, x + 820, y + 300), radius=8, fill=CARD, outline=RED, width=3)
    draw.text((x + 16, y + 14), task_id, fill=INK, font=font)
    draw_badge(draw, x + 16, y + 46, "vanilla FAIL", RED, badge_font)
    refl_label = f"reflection {row['reflection_train']} FAIL"
    if row["reflection_train"].split("/")[0] == row["reflection_train"].split("/")[-1]:
        refl_label = f"reflection {row['reflection_train']} train OK, test FAIL"
    draw_badge(draw, x + 140, y + 46, refl_label, AMBER if "train OK" in refl_label else RED, badge_font)

    input_grid = task["test"][0]["input"]
    expected = task["test"][0]["output"]
    vanilla_pred = vanilla_row["tests"][0]["prediction"]
    reflection_pred = None
    if reflection and reflection.get("test_predictions"):
        reflection_pred = reflection["test_predictions"][0]["prediction"]

    cells = [("input", input_grid), ("vanilla", vanilla_pred), ("reflect", reflection_pred), ("expected", expected)]
    for idx, (label, grid) in enumerate(cells):
        gx = x + 18 + idx * 196
        gy = y + 106
        draw.text((gx, gy - 22), label, fill=MUTED, font=small)
        if grid is None:
            draw_missing(draw, gx, gy, "train failed", small)
        else:
            draw_grid(draw, grid, gx, gy)


def draw_metric(draw, x, y, label, value, color):
    label_font = load_font(13)
    value_font = load_font(28)
    draw.rounded_rectangle((x, y, x + 150, y + 62), radius=8, fill=CARD, outline=(220, 214, 200))
    draw.text((x + 12, y + 9), label, fill=MUTED, font=label_font)
    draw.text((x + 12, y + 28), value, fill=color, font=value_font)


def draw_badge(draw, x, y, text, color, font):
    width = max(96, text_width(font, text) + 22)
    draw.rounded_rectangle((x, y, x + width, y + 28), radius=6, fill=color)
    draw.text((x + 10, y + 6), text, fill=(255, 255, 255), font=font)


def draw_missing(draw, x, y, label, font):
    draw.rounded_rectangle((x, y, x + 142, y + 142), radius=6, fill=(238, 235, 226), outline=(220, 214, 200))
    draw.text((x + 34, y + 63), label, fill=MUTED, font=font)


def draw_grid(draw, grid, x, y, max_size: int = 142):
    if not grid:
        return
    h = len(grid)
    w = len(grid[0])
    cell = max(2, min(max_size // max(w, h), 14))
    draw.rectangle((x - 2, y - 2, x + w * cell + 2, y + h * cell + 2), fill=(30, 38, 46))
    for yy, row in enumerate(grid):
        for xx, value in enumerate(row):
            draw.rectangle(
                (x + xx * cell, y + yy * cell, x + (xx + 1) * cell, y + (yy + 1) * cell),
                fill=PALETTE[int(value)],
                outline=(220, 220, 220),
            )


def load_font(size: int):
    for path in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(font, text: str) -> int:
    try:
        return int(font.getlength(text))
    except AttributeError:
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0])


if __name__ == "__main__":
    raise SystemExit(main())
