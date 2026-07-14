from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("007bbfb7_grid_expansion", ROOT / "examples/arc/category_1_good_fit/007bbfb7_grid_expansion.json"),
    ("1cf80156_object_extraction", ROOT / "examples/arc/category_1_good_fit/1cf80156_object_extraction.json"),
    ("0520fde7_spatial_mapping", ROOT / "examples/arc/category_1_good_fit/0520fde7_spatial_mapping.json"),
    ("0ca9ddb6_marker_to_shape_revision", ROOT / "examples/arc/category_2_reflection_helps/0ca9ddb6_marker_to_shape_revision.json"),
    ("28e73c20_boundary_fill_revision", ROOT / "examples/arc/category_2_reflection_helps/28e73c20_boundary_fill_revision.json"),
    ("3de23699_selector_revision", ROOT / "examples/arc/category_2_reflection_helps/3de23699_selector_revision.json"),
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
INK = (23, 32, 42)
MUTED = (88, 101, 114)
PAPER = (250, 248, 242)
CARD = (255, 253, 248)


def main() -> int:
    vanilla = json.loads((ROOT / "artifacts/arc_vanilla_selected6_gpt5mini_low.json").read_text(encoding="utf-8"))
    vanilla_by_task = {Path(row["path"]).stem: row for row in vanilla["tasks"]}
    comparison = json.loads((ROOT / "artifacts/arc_gpt5mini_low_comparison.json").read_text(encoding="utf-8"))
    comparison_by_task = {row["task"]: row for row in comparison}

    img = Image.new("RGB", (1800, 1320), PAPER)
    draw = ImageDraw.Draw(img)
    title = load_font(36)
    subtitle = load_font(20)
    font = load_font(18)
    small = load_font(14)
    badge_font = load_font(16)

    draw.text((46, 30), "ARC-AGI-1: GPT-5 Mini Low vs Reflection", fill=INK, font=title)
    draw.text(
        (48, 78),
        "Vanilla direct answers solve 3/6. Executable skill reflection repairs 1 of the 3 misses.",
        fill=MUTED,
        font=subtitle,
    )
    draw_metric(draw, 48, 112, "VANILLA", "3/6", RED)
    draw_metric(draw, 210, 112, "AFTER REFLECTION", "4/6", GREEN)
    draw_metric(draw, 442, 112, "REPAIRED", "+1", BLUE)

    for i, (task_name, task_path) in enumerate(TASKS):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        vanilla_row = vanilla_by_task[task_name]
        compare = comparison_by_task[task_name]
        reflection_path = ROOT / "artifacts/arc_gpt5mini_low" / f"{task_name}_reflection.json"
        reflection = json.loads(reflection_path.read_text(encoding="utf-8")) if reflection_path.exists() else None

        x = 46 + (i % 2) * 868
        y = 194 + (i // 2) * 356
        draw.rounded_rectangle((x, y, x + 820, y + 316), radius=8, fill=CARD, outline=(220, 214, 200))
        draw.text((x + 18, y + 18), short_name(task_name), fill=INK, font=font)
        draw_status(draw, x + 18, y + 52, "GPT-5 Mini low", bool(vanilla_row["correct"]), badge_font)
        if compare["reflection_train"] == "not run":
            draw_neutral(draw, x + 204, y + 52, "reflection not run", badge_font)
        else:
            reflection_ok = bool(compare["reflection_test_ok"])
            label = f"reflection {compare['reflection_train']}"
            draw_status(draw, x + 204, y + 52, label, reflection_ok, badge_font)

        test_input = task["test"][0]["input"]
        expected = task["test"][0]["output"]
        vanilla_pred = vanilla_row["tests"][0]["prediction"]
        reflection_pred = None
        if reflection and reflection.get("test_predictions"):
            reflection_pred = reflection["test_predictions"][0]["prediction"]

        labels = ["test input", "vanilla", "reflection", "expected"]
        grids = [test_input, vanilla_pred, reflection_pred, expected]
        for j, (label, grid) in enumerate(zip(labels, grids)):
            gx = x + 18 + j * 198
            gy = y + 104
            draw.text((gx, gy - 24), label, fill=MUTED, font=small)
            if grid is None:
                draw_missing(draw, gx, gy, 150, small)
            else:
                draw_grid(draw, grid, gx, gy)

    out = ROOT / "artifacts/arc_gpt5mini_low_reflection_comparison.png"
    img.save(out)
    print(out)
    return 0


def draw_metric(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, color: tuple[int, int, int]) -> None:
    label_font = load_font(13)
    value_font = load_font(28)
    draw.rounded_rectangle((x, y, x + 136, y + 58), radius=8, fill=(255, 253, 248), outline=(220, 214, 200))
    draw.text((x + 12, y + 8), label, fill=MUTED, font=label_font)
    draw.text((x + 12, y + 25), value, fill=color, font=value_font)


def draw_status(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, ok: bool, font) -> None:
    color = GREEN if ok else RED
    text = f"{label}: {'OK' if ok else 'FAIL'}"
    width = min(350, max(110, text_width(font, text) + 24))
    draw.rounded_rectangle((x, y, x + width, y + 30), radius=6, fill=color)
    draw.text((x + 12, y + 7), text, fill=(255, 255, 255), font=font)


def draw_neutral(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, font) -> None:
    width = text_width(font, label) + 24
    draw.rounded_rectangle((x, y, x + width, y + 30), radius=6, fill=(100, 114, 127))
    draw.text((x + 12, y + 7), label, fill=(255, 255, 255), font=font)


def draw_missing(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, font) -> None:
    draw.rounded_rectangle((x, y, x + size, y + size), radius=6, fill=(238, 235, 226), outline=(220, 214, 200))
    draw.text((x + 38, y + 64), "not run", fill=MUTED, font=font)


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, max_size: int = 150) -> None:
    height = len(grid)
    width = len(grid[0]) if height else 0
    if not height or not width:
        return
    cell = max(2, min(max_size // max(width, height), 15))
    draw.rectangle((x - 2, y - 2, x + width * cell + 2, y + height * cell + 2), fill=(30, 38, 46))
    for yy, row in enumerate(grid):
        for xx, value in enumerate(row):
            draw.rectangle(
                (x + xx * cell, y + yy * cell, x + (xx + 1) * cell, y + (yy + 1) * cell),
                fill=PALETTE[int(value)],
                outline=(220, 220, 220),
            )


def short_name(name: str) -> str:
    return name.replace("_", " ")


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
