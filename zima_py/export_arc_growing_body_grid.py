from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ARC growing-body run summary.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report_path = resolve(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["runs"]

    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=20) if hasattr(ImageFont.load_default(), "size") else font
    w = 1480
    row_h = 190
    h = 120 + row_h * len(rows)
    img = Image.new("RGB", (w, h), (245, 246, 248))
    draw = ImageDraw.Draw(img)

    draw.text((28, 24), "ARC Growing Body: search base primitives, write only when search fails", fill=(24, 27, 31), font=title_font)
    draw.text((28, 56), f"{report['model']} / {report.get('reasoning_effort')}   solved {report['test_correct']}/{report['total']}   persisted skills {report['library_size']}", fill=(82, 91, 105), font=font)

    for idx, run in enumerate(rows):
        y = 96 + idx * row_h
        task = load_task(run["path"])
        first = task["train"][0]
        status = "SOLVED" if run["test_ok"] else "FAILED"
        if run["mode"] == "search" and not str(run["best_name"]).startswith("learned::"):
            mode = "BASE SEARCH"
        elif run.get("train_passed"):
            mode = "LEARNED + PERSISTED"
        else:
            mode = "WRITE ATTEMPT FAILED"
        color = (22, 132, 90) if run["test_ok"] else (190, 64, 64)
        draw.rounded_rectangle((20, y, w - 20, y + row_h - 14), radius=8, fill=(255, 255, 255), outline=(215, 219, 226))
        draw.text((40, y + 18), run["task"], fill=(20, 25, 31), font=title_font)
        draw.text((40, y + 48), mode, fill=(82, 91, 105), font=font)
        draw.rectangle((40, y + 80, 142, y + 110), fill=color)
        draw.text((50, y + 88), status, fill=(255, 255, 255), font=font)
        draw.text((40, y + 126), f"train {run['train_score']}", fill=(82, 91, 105), font=font)
        draw.text((210, y + 18), "train input", fill=(82, 91, 105), font=font)
        draw_grid(draw, first["input"], 210, y + 44, 104)
        draw.text((350, y + 18), "train output", fill=(82, 91, 105), font=font)
        draw_grid(draw, first["output"], 350, y + 44, 104)
        draw.text((500, y + 18), "best primitive", fill=(82, 91, 105), font=font)
        draw_wrapped(draw, run["best_name"], 500, y + 44, 360, fill=(24, 27, 31), font=font)
        design = ""
        if run.get("write_history"):
            design = run["write_history"][-1].get("design", "")
        elif run.get("search", {}).get("best_name"):
            design = "Selected by deterministic verifier over base and learned body primitives."
        draw.text((900, y + 18), "what happened", fill=(82, 91, 105), font=font)
        draw_wrapped(draw, design or "No LLM write required.", 900, y + 44, 510, fill=(24, 27, 31), font=font, max_lines=6)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)
    return 0


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def load_task(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, box: int) -> None:
    h = len(grid)
    w = len(grid[0])
    cell = max(2, min(box // max(h, w), 14))
    ox = x + (box - cell * w) // 2
    oy = y + (box - cell * h) // 2
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x0 = ox + c * cell
            y0 = oy + r * cell
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=COLORS.get(int(value), (60, 60, 60)))


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fill: tuple[int, int, int], font: ImageFont.ImageFont, max_lines: int = 5) -> None:
    words = text.replace("\n", " ").split()
    lines: list[str] = []
    line = ""
    for word in words:
        probe = f"{line} {word}".strip()
        if draw.textlength(probe, font=font) <= width:
            line = probe
            continue
        if line:
            lines.append(line)
        line = word
        if len(lines) >= max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    if len(lines) == max_lines and words:
        lines[-1] = lines[-1].rstrip(".") + "..."
    for i, line_text in enumerate(lines):
        draw.text((x, y + i * 18), line_text, fill=fill, font=font)


if __name__ == "__main__":
    raise SystemExit(main())
