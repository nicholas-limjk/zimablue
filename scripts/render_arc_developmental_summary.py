from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
GREEN = (54, 143, 93)
RED = (182, 68, 56)
BLUE = (50, 112, 178)
MUTED = (88, 101, 114)
INK = (23, 32, 42)
PAPER = (250, 248, 242)
CARD = (255, 253, 248)


def main() -> int:
    report = json.loads((ROOT / "artifacts/arc_developmental/gpt5mini_low_report.json").read_text(encoding="utf-8"))
    runs = report["runs"]
    img = Image.new("RGB", (1400, 760), PAPER)
    draw = ImageDraw.Draw(img)
    title = load_font(34)
    subtitle = load_font(19)
    font = load_font(18)
    small = load_font(14)

    draw.text((42, 30), "Developmental ARC Agent", fill=INK, font=title)
    draw.text(
        (44, 76),
        "Skills persist across tasks. The agent first tries its library, then writes/mutates a new skill after failure.",
        fill=MUTED,
        font=subtitle,
    )
    draw_metric(draw, 44, 114, "TASKS", f"{report['test_correct']}/{report['total']}", GREEN if report["test_correct"] else RED)
    draw_metric(draw, 206, 114, "LIBRARY SIZE", str(report["library_size"]), BLUE)

    x = 44
    y = 210
    widths = [150, 120, 140, 160, 650]
    headers = ["task", "mode", "train", "test", "library effect"]
    cx = x
    for label, width in zip(headers, widths):
        draw.text((cx + 10, y), label.upper(), fill=MUTED, font=small)
        cx += width
    y += 32

    for index, run in enumerate(runs):
        fill = CARD if index % 2 == 0 else (247, 244, 236)
        draw.rounded_rectangle((x, y, x + sum(widths), y + 54), radius=6, fill=fill, outline=(226, 220, 207))
        cx = x
        draw.text((cx + 10, y + 17), run["task"], fill=INK, font=font)
        cx += widths[0]
        draw_badge(draw, cx + 10, y + 12, run["mode"], BLUE if run["mode"] == "reuse" else MUTED, small)
        cx += widths[1]
        draw_badge(draw, cx + 10, y + 12, run["train_score"], GREEN if run["train_passed"] else RED, small)
        cx += widths[2]
        draw_badge(draw, cx + 10, y + 12, "OK" if run["test_ok"] else "FAIL", GREEN if run["test_ok"] else RED, small)
        cx += widths[3]
        draw.text((cx + 10, y + 17), note(run), fill=GREEN if run["test_ok"] else MUTED, font=small)
        y += 62

    out = ROOT / "artifacts/arc_developmental/gpt5mini_low_summary.png"
    img.save(out)
    print(out)
    return 0


def note(run: dict) -> str:
    if run["test_ok"] and run["mode"] == "write":
        return "new skill survived and was added to persistent library"
    if run["mode"] == "reuse":
        return f"reused skill from {run.get('source_task')}"
    return "library did not contain a useful prior; new proposal did not solve"


def draw_metric(draw, x, y, label, value, color):
    label_font = load_font(13)
    value_font = load_font(28)
    draw.rounded_rectangle((x, y, x + 140, y + 60), radius=8, fill=CARD, outline=(220, 214, 200))
    draw.text((x + 12, y + 8), label, fill=MUTED, font=label_font)
    draw.text((x + 12, y + 27), value, fill=color, font=value_font)


def draw_badge(draw, x, y, text, color, font):
    width = max(70, text_width(font, text) + 22)
    draw.rounded_rectangle((x, y, x + width, y + 30), radius=6, fill=color)
    draw.text((x + 10, y + 7), text, fill=(255, 255, 255), font=font)


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
