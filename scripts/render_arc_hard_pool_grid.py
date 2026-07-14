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
AMBER = (183, 128, 36)
MUTED = (88, 101, 114)
INK = (23, 32, 42)
PAPER = (250, 248, 242)
CARD = (255, 253, 248)


def main() -> int:
    vanilla = json.loads((ROOT / "artifacts/arc_vanilla_hard_candidates_gpt5mini_low.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "artifacts/arc_gpt5mini_low_hard_comparison.json").read_text(encoding="utf-8"))
    vanilla_by_id = {Path(row["path"]).stem: row for row in vanilla["tasks"]}
    summary_by_id = {row["task"]: row for row in summary["rows"]}

    img = Image.new("RGB", (2100, 1800), PAPER)
    draw = ImageDraw.Draw(img)
    title = load_font(38)
    subtitle = load_font(21)
    font = load_font(18)
    small = load_font(13)
    badge_font = load_font(14)

    draw.text((42, 28), "Hard ARC-AGI-1 Task Grid", fill=INK, font=title)
    draw.text(
        (44, 78),
        "GPT-5 Mini low direct answers vs executable reflection. Green reflection badges are repaired vanilla misses.",
        fill=MUTED,
        font=subtitle,
    )
    draw_metric(draw, 44, 116, "VANILLA", "4/12", RED)
    draw_metric(draw, 210, 116, "AFTER REFLECTION", "6/12", GREEN)
    draw_metric(draw, 456, 116, "REPAIRED", "+2", BLUE)

    for i, task_id in enumerate(IDS):
        task = json.loads((ROOT / "arc_agi_source/data/training" / f"{task_id}.json").read_text(encoding="utf-8"))
        vanilla_row = vanilla_by_id[task_id]
        row = summary_by_id[task_id]
        reflection = read_reflection(task_id)
        x = 42 + (i % 3) * 680
        y = 206 + (i // 3) * 382
        render_card(draw, x, y, task_id, task, vanilla_row, row, reflection, font, small, badge_font)

    out = ROOT / "artifacts/arc_gpt5mini_low_hard_task_grid.png"
    img.save(out)
    print(out)
    return 0


def read_reflection(task_id: str) -> dict | None:
    path = ROOT / "artifacts/arc_gpt5mini_low_hard" / f"{task_id}_reflection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def render_card(draw, x, y, task_id, task, vanilla_row, row, reflection, font, small, badge_font) -> None:
    vanilla_ok = bool(vanilla_row["correct"])
    refl_ok = row["reflection_test_ok"]
    repaired = not vanilla_ok and bool(refl_ok)
    border = BLUE if repaired else GREEN if vanilla_ok else RED
    draw.rounded_rectangle((x, y, x + 630, y + 334), radius=8, fill=CARD, outline=border, width=3)
    draw.text((x + 16, y + 14), task_id, fill=INK, font=font)
    draw_badge(draw, x + 16, y + 46, "vanilla OK" if vanilla_ok else "vanilla FAIL", GREEN if vanilla_ok else RED, badge_font)
    if row["reflection_train"] == "not run":
        draw_badge(draw, x + 142, y + 46, "reflection PASS", GREEN, badge_font)
    else:
        label = f"reflection {row['reflection_train']} {'OK' if refl_ok else 'FAIL'}"
        color = BLUE if repaired else GREEN if refl_ok else RED
        draw_badge(draw, x + 142, y + 46, label, color, badge_font)

    input_grid = task["test"][0]["input"]
    expected = task["test"][0]["output"]
    vanilla_pred = vanilla_row["tests"][0]["prediction"]
    reflection_pred = None
    reflection_missing_label = "train failed"
    if reflection and reflection.get("test_predictions"):
        reflection_pred = reflection["test_predictions"][0]["prediction"]
    elif vanilla_ok and row["reflection_train"] == "not run":
        reflection_pred = vanilla_pred

    cells = [
        ("input", input_grid),
        ("vanilla", vanilla_pred),
        ("reflect", reflection_pred),
        ("expected", expected),
    ]
    for idx, (label, grid) in enumerate(cells):
        gx = x + 16 + idx * 152
        gy = y + 100
        draw.text((gx, gy - 22), label, fill=MUTED, font=small)
        if grid is None:
            draw_missing(draw, gx, gy, reflection_missing_label if label == "reflect" else "missing", small)
        else:
            draw_grid(draw, grid, gx, gy)


def draw_metric(draw, x, y, label, value, color):
    label_font = load_font(13)
    value_font = load_font(28)
    draw.rounded_rectangle((x, y, x + 140, y + 60), radius=8, fill=CARD, outline=(220, 214, 200))
    draw.text((x + 12, y + 8), label, fill=MUTED, font=label_font)
    draw.text((x + 12, y + 27), value, fill=color, font=value_font)


def draw_badge(draw, x, y, text, color, font) -> None:
    width = max(92, text_width(font, text) + 22)
    draw.rounded_rectangle((x, y, x + width, y + 28), radius=6, fill=color)
    draw.text((x + 10, y + 6), text, fill=(255, 255, 255), font=font)


def draw_missing(draw, x, y, label, font) -> None:
    draw.rounded_rectangle((x, y, x + 130, y + 130), radius=6, fill=(238, 235, 226), outline=(220, 214, 200))
    draw.text((x + 31, y + 57), label, fill=MUTED, font=font)


def draw_grid(draw, grid, x, y, max_size: int = 130) -> None:
    if not grid:
        return
    h = len(grid)
    w = len(grid[0])
    cell = max(2, min(max_size // max(w, h), 13))
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
