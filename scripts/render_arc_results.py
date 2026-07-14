from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ROOT / "examples/arc/category_1_good_fit/007bbfb7_grid_expansion.json",
    ROOT / "examples/arc/category_1_good_fit/1cf80156_object_extraction.json",
    ROOT / "examples/arc/category_1_good_fit/0520fde7_spatial_mapping.json",
    ROOT / "examples/arc/category_2_reflection_helps/0ca9ddb6_marker_to_shape_revision.json",
    ROOT / "examples/arc/category_2_reflection_helps/28e73c20_boundary_fill_revision.json",
    ROOT / "examples/arc/category_2_reflection_helps/3de23699_selector_revision.json",
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
    img = Image.new("RGB", (1600, 1120), (250, 248, 242))
    draw = ImageDraw.Draw(img)
    title = load_font(28)
    font = load_font(18)
    small = load_font(14)
    draw.text((36, 24), "ARC Skill Synthesis Results", fill=(20, 30, 40), font=title)
    draw.text((38, 60), "Test output is from the public dataset for evaluation only; it was not included in the prompt.", fill=(88, 101, 114), font=small)

    summary = []
    for i, task_path in enumerate(TASKS):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        report_path = ROOT / "artifacts/arc" / f"{task_path.stem}_reflection.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        prediction = report["test_predictions"][0]["prediction"] if report.get("test_predictions") else []
        expected = task["test"][0].get("output")
        test_ok = bool(expected is not None and prediction == expected)
        train_eval = report["best_train_evaluation"]
        summary.append({"task": task_path.stem, "train": f"{train_eval['correct']}/{train_eval['total']}", "test_ok": test_ok})

        x = 38 + (i % 2) * 780
        y = 108 + (i // 2) * 330
        badge = (54, 143, 93) if test_ok else (182, 68, 56)
        draw.rounded_rectangle((x, y, x + 730, y + 292), radius=8, fill=(255, 253, 248), outline=(220, 214, 200))
        draw.text((x + 18, y + 16), task_path.stem, fill=(20, 30, 40), font=font)
        draw.rounded_rectangle((x + 585, y + 16, x + 700, y + 48), radius=6, fill=badge)
        draw.text((x + 608, y + 22), "TEST OK" if test_ok else "TEST FAIL", fill=(255, 255, 255), font=small)
        draw.text((x + 18, y + 52), f"train {train_eval['correct']}/{train_eval['total']}", fill=(88, 101, 114), font=small)

        draw.text((x + 18, y + 82), "test input", fill=(88, 101, 114), font=small)
        draw_grid(draw, task["test"][0]["input"], x + 18, y + 106)
        draw.text((x + 240, y + 82), "prediction", fill=(88, 101, 114), font=small)
        draw_grid(draw, prediction, x + 240, y + 106)
        if expected is not None:
            draw.text((x + 462, y + 82), "expected", fill=(88, 101, 114), font=small)
            draw_grid(draw, expected, x + 462, y + 106)

    out = ROOT / "artifacts/arc_results_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    (ROOT / "artifacts/arc_results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(summary, indent=2))
    return 0


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, max_size: int = 190) -> None:
    if not grid:
        return
    height = len(grid)
    width = len(grid[0])
    cell = max(2, min(max_size // max(width, height), 16))
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
