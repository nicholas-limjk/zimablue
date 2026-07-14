from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
GREEN = (54, 143, 93)
RED = (182, 68, 56)
BLUE = (50, 112, 178)
AMBER = (183, 128, 36)
INK = (23, 32, 42)
MUTED = (88, 101, 114)
PAPER = (250, 248, 242)
CARD = (255, 253, 248)


def main() -> int:
    summary = json.loads((ROOT / "artifacts/arc_gpt5mini_low_hard_comparison.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    img = Image.new("RGB", (1500, 1040), PAPER)
    draw = ImageDraw.Draw(img)
    title = load_font(36)
    subtitle = load_font(20)
    header = load_font(16)
    font = load_font(18)
    small = load_font(14)

    draw.text((46, 30), "Hard ARC-AGI-1 Pool: GPT-5 Mini Low vs Reflection", fill=INK, font=title)
    draw.text(
        (48, 78),
        "Reflection runs only on vanilla misses. Test outputs are used for evaluation only, not prompting.",
        fill=MUTED,
        font=subtitle,
    )
    draw_metric(draw, 48, 116, "VANILLA", f"{summary['vanilla_correct']}/{summary['total']}", RED)
    draw_metric(draw, 224, 116, "AFTER REFLECTION", f"{summary['combined_correct']}/{summary['total']}", GREEN)
    draw_metric(draw, 472, 116, "REPAIRED", f"+{summary['repaired']}", BLUE)
    draw_metric(draw, 638, 116, "TRAIN-PASS TEST-FAIL", str(count_overfit(rows)), AMBER)

    x = 48
    y = 208
    col_w = [150, 180, 220, 180, 620]
    headers = ["task", "vanilla test", "reflection train", "reflection test", "note"]
    cx = x
    for w, label in zip(col_w, headers):
        draw.text((cx + 10, y), label.upper(), fill=MUTED, font=header)
        cx += w
    y += 34

    for index, row in enumerate(rows):
        fill = CARD if index % 2 == 0 else (247, 244, 236)
        draw.rounded_rectangle((x, y, x + sum(col_w), y + 54), radius=6, fill=fill, outline=(226, 220, 207))
        cx = x
        draw.text((cx + 10, y + 17), row["task"], fill=INK, font=font)
        cx += col_w[0]
        draw_badge(draw, cx + 10, y + 12, "OK" if row["vanilla_test_ok"] else "FAIL", GREEN if row["vanilla_test_ok"] else RED, small)
        cx += col_w[1]
        train = row["reflection_train"]
        train_color = MUTED if train == "not run" else GREEN if train.split("/")[0] == train.split("/")[-1] else RED
        draw_badge(draw, cx + 10, y + 12, train, train_color, small)
        cx += col_w[2]
        refl = row["reflection_test_ok"]
        if refl is None:
            draw_badge(draw, cx + 10, y + 12, "PASS", GREEN, small)
        else:
            draw_badge(draw, cx + 10, y + 12, "OK" if refl else "FAIL", GREEN if refl else RED, small)
        cx += col_w[3]
        draw.text((cx + 10, y + 17), note_for(row), fill=note_color(row), font=small)
        y += 62

    out = ROOT / "artifacts/arc_gpt5mini_low_hard_reflection_dashboard.png"
    img.save(out)
    print(out)
    return 0


def note_for(row: dict) -> str:
    if row["vanilla_test_ok"]:
        return "reflection passed"
    if row["reflection_test_ok"]:
        return "reflection repaired the miss"
    if row["reflection_train"] == "not run":
        return "not attempted"
    if is_train_pass(row) and not row["reflection_test_ok"]:
        return "passed train but failed held-out test"
    if row["reflection_train"].startswith("0/"):
        return "reflection did not find a working train rule"
    return "partial train improvement, not solved"


def note_color(row: dict) -> tuple[int, int, int]:
    if row["reflection_test_ok"]:
        return GREEN
    if is_train_pass(row) and not row["reflection_test_ok"]:
        return AMBER
    if row["vanilla_test_ok"]:
        return MUTED
    return RED


def is_train_pass(row: dict) -> bool:
    train = row["reflection_train"]
    if "/" not in train:
        return False
    left, right = train.split("/", 1)
    return left == right


def count_overfit(rows: list[dict]) -> int:
    return sum(1 for row in rows if is_train_pass(row) and row["reflection_test_ok"] is False)


def draw_metric(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, color: tuple[int, int, int]) -> None:
    label_font = load_font(13)
    value_font = load_font(28)
    draw.rounded_rectangle((x, y, x + 150, y + 62), radius=8, fill=CARD, outline=(220, 214, 200))
    draw.text((x + 12, y + 9), label, fill=MUTED, font=label_font)
    draw.text((x + 12, y + 28), value, fill=color, font=value_font)


def draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int], font) -> None:
    width = max(72, text_width(font, text) + 24)
    draw.rounded_rectangle((x, y, x + width, y + 30), radius=6, fill=color)
    draw.text((x + 12, y + 7), text, fill=(255, 255, 255), font=font)


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
