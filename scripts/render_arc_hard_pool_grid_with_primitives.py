from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IDS = [
    "00d62c1b",
    "045e512c",
    "025d127b",
    "1f85a75f",
    "91714a58",
    "9ecd008a",
    "a8c38be5",
    "b8825c91",
    "c8f0f002",
    "d511f180",
    "e5062a87",
    "f8a8fe49",
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
GREEN = (54, 143, 93)
RED = (182, 68, 56)
BLUE = (50, 112, 178)
MUTED = (88, 101, 114)
INK = (23, 32, 42)
PAPER = (250, 248, 242)
CARD = (255, 253, 248)


def main() -> int:
    vanilla = json.loads((ROOT / "artifacts/arc_vanilla_hard_candidates_gpt5mini_low.json").read_text(encoding="utf-8"))
    reflection = json.loads((ROOT / "artifacts/arc_gpt5mini_low_hard_comparison.json").read_text(encoding="utf-8"))
    primitive = json.loads((ROOT / "artifacts/arc_primitive_search_hard12.json").read_text(encoding="utf-8"))
    vanilla_by_id = {Path(row["path"]).stem: row for row in vanilla["tasks"]}
    reflection_by_id = {row["task"]: row for row in reflection["rows"]}
    primitive_by_id = {row["task"]: row for row in primitive["rows"]}

    img = Image.new("RGB", (2250, 1900), PAPER)
    draw = ImageDraw.Draw(img)
    title = load_font(38)
    subtitle = load_font(20)
    font = load_font(18)
    small = load_font(12)
    badge_font = load_font(13)

    draw.text((42, 28), "Hard ARC-AGI-1: From Direct Answers To Reflected Skills", fill=INK, font=title)
    draw.text(
        (44, 78),
        "GPT-5 Mini low improves from 4/12 direct answers to 6/12 with train-verified reflection. Primitive search currently solves 3/12.",
        fill=MUTED,
        font=subtitle,
    )
    draw_metric(draw, 44, 116, "VANILLA", "4/12", RED)
    draw_metric(draw, 210, 116, "BODY PRIMITIVES", "3/12", BLUE)
    draw_metric(draw, 444, 116, "REFLECTION", "6/12", GREEN)
    draw_metric(draw, 622, 116, "REPAIRED", "+2", BLUE)

    for i, task_id in enumerate(IDS):
        task = json.loads((ROOT / "arc_agi_source/data/training" / f"{task_id}.json").read_text(encoding="utf-8"))
        x = 42 + (i % 3) * 730
        y = 212 + (i // 3) * 392
        render_card(
            draw,
            x,
            y,
            task_id,
            task,
            vanilla_by_id[task_id],
            primitive_by_id[task_id],
            reflection_by_id[task_id],
            font,
            small,
            badge_font,
        )

    out = ROOT / "artifacts/arc_hard_pool_improved_grid.png"
    img.save(out)
    print(out)
    return 0


def render_card(draw, x, y, task_id, task, vanilla_row, primitive_row, reflection_row, font, small, badge_font) -> None:
    vanilla_ok = bool(vanilla_row["correct"])
    primitive_ok = bool(primitive_row["test_ok"])
    reflection_ok = bool(reflection_row["combined_ok"])
    repaired = not vanilla_ok and bool(reflection_row["reflection_test_ok"])
    border = GREEN if reflection_ok else RED
    if repaired:
        border = BLUE
    draw.rounded_rectangle((x, y, x + 680, y + 344), radius=8, fill=CARD, outline=border, width=3)
    draw.text((x + 16, y + 14), task_id, fill=INK, font=font)
    draw_badge(draw, x + 16, y + 46, "vanilla OK" if vanilla_ok else "vanilla FAIL", GREEN if vanilla_ok else RED, badge_font)
    draw_badge(draw, x + 132, y + 46, "primitive OK" if primitive_ok else "primitive FAIL", BLUE if primitive_ok else MUTED, badge_font)
    draw_badge(draw, x + 270, y + 46, "reflection OK" if reflection_ok else "reflection FAIL", GREEN if reflection_ok else RED, badge_font)

    expected = task["test"][0]["output"]
    primitive_pred = primitive_row.get("test_prediction")
    reflect_pred = read_reflection_prediction(task_id, reflection_row, vanilla_row)
    cells = [
        ("input", task["test"][0]["input"]),
        ("vanilla", vanilla_row["tests"][0]["prediction"]),
        ("primitive", primitive_pred),
        ("reflect", reflect_pred),
        ("expected", expected),
    ]
    for idx, (label, grid) in enumerate(cells):
        gx = x + 16 + idx * 130
        gy = y + 108
        draw.text((gx, gy - 20), label, fill=MUTED, font=small)
        if grid is None:
            draw_missing(draw, gx, gy, small)
        else:
            draw_grid(draw, grid, gx, gy)


def read_reflection_prediction(task_id: str, reflection_row: dict, vanilla_row: dict):
    if reflection_row["vanilla_test_ok"]:
        return vanilla_row["tests"][0]["prediction"]
    path = ROOT / "artifacts/arc_gpt5mini_low_hard" / f"{task_id}_reflection.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    preds = report.get("test_predictions") or []
    if preds:
        return preds[0]["prediction"]
    return None


def draw_metric(draw, x, y, label, value, color):
    label_font = load_font(13)
    value_font = load_font(28)
    draw.rounded_rectangle((x, y, x + 150, y + 62), radius=8, fill=CARD, outline=(220, 214, 200))
    draw.text((x + 12, y + 9), label, fill=MUTED, font=label_font)
    draw.text((x + 12, y + 28), value, fill=color, font=value_font)


def draw_badge(draw, x, y, text, color, font):
    width = max(98, text_width(font, text) + 20)
    draw.rounded_rectangle((x, y, x + width, y + 28), radius=6, fill=color)
    draw.text((x + 9, y + 6), text, fill=(255, 255, 255), font=font)


def draw_missing(draw, x, y, font):
    draw.rounded_rectangle((x, y, x + 112, y + 112), radius=6, fill=(238, 235, 226), outline=(220, 214, 200))
    draw.text((x + 25, y + 49), "no test", fill=MUTED, font=font)


def draw_grid(draw, grid, x, y, max_size: int = 112):
    if not grid:
        return
    h = len(grid)
    w = len(grid[0])
    cell = max(2, min(max_size // max(w, h), 12))
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
