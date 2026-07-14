from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the LLM skill discovery and transfer story.")
    parser.add_argument("--out", default="artifacts/arc_curriculum/llm_skill_growth_story.png")
    args = parser.parse_args()

    w, h = 1500, 1000
    img = Image.new("RGB", (w, h), (247, 248, 250))
    draw = ImageDraw.Draw(img)
    title = font(28)
    h2 = font(18)
    mono = mono_font(15)
    body = font(14)

    draw.text((36, 28), "LLM-Written Skills Become A Growing Body", fill=(22, 26, 33), font=title)
    draw.text(
        (36, 66),
        "The LLM proposes executable operators after failures; verifier/search persists and composes them on later tasks.",
        fill=(78, 87, 100),
        font=body,
    )

    # Pipeline
    y = 118
    steps = [
        ("1. Search Fails", "Base primitives cannot solve the train pairs."),
        ("2. LLM Writes Skill", "The LLM proposes a small Python operator."),
        ("3. Verifier Gates", "Only train-verified operators persist."),
        ("4. Body Reuses", "Future tasks search over base + learned operators."),
    ]
    x = 42
    for i, (head, text) in enumerate(steps):
        bx = x + i * 355
        draw.rounded_rectangle((bx, y, bx + 310, y + 110), radius=10, fill=(255, 255, 255), outline=(215, 222, 230))
        draw.text((bx + 18, y + 18), head, fill=(20, 24, 31), font=h2)
        wrapped(draw, text, bx + 18, y + 52, 270, fill=(81, 90, 104), font=body, line_h=18)
        if i < len(steps) - 1:
            draw.line((bx + 318, y + 55, bx + 352, y + 55), fill=(48, 118, 198), width=3)
            draw.polygon([(bx + 352, y + 55), (bx + 342, y + 48), (bx + 342, y + 62)], fill=(48, 118, 198))

    # Main examples
    panel_y = 270
    draw.rounded_rectangle((36, panel_y, w - 36, h - 36), radius=12, fill=(255, 255, 255), outline=(215, 222, 230))
    draw.text((64, panel_y + 28), "What the LLM discovered", fill=(20, 24, 31), font=h2)

    code1 = """def solve(grid):
    doubled_rows = []
    for row in grid:
        doubled_rows.append(list(row) + list(reversed(row)))
    out = [list(r) for r in doubled_rows]
    for r in reversed(doubled_rows):
        out.append(list(r))
    return out"""
    code2 = """def solve(grid):
    out = [list(row) for row in grid]
    for row in reversed(grid):
        out.append(list(row))
    return out"""
    code3 = """def solve(grid):
    out = copy(grid)
    for each nonzero seed:
        fill its row or column
    return out"""

    columns = [
        (
            "reflect_2d_tile",
            "Learned on 62c24649; reused on 67e8384a with no new LLM call.",
            code1,
            "birth: 62c24649 -> transfer: 67e8384a",
        ),
        (
            "vertical_reflect_and_stack",
            "Learned in tile-grid curriculum; solved held-out tasks when composed with flip_v.",
            code2,
            "flip_v -> vertical_reflect_and_stack",
        ),
        (
            "expand_seed_to_row_or_column",
            "Alone it was partial; composed with row-completion primitives it solved the task.",
            code3,
            "operator -> complete_rows_color_3 -> complete_rows_color_1",
        ),
    ]
    col_w = 435
    for i, (name, desc, code, seq) in enumerate(columns):
        cx = 64 + i * 462
        cy = panel_y + 74
        draw.rounded_rectangle((cx, cy, cx + col_w, cy + 500), radius=10, fill=(248, 250, 252), outline=(222, 227, 234))
        draw.text((cx + 18, cy + 18), name, fill=(22, 26, 33), font=h2)
        wrapped(draw, desc, cx + 18, cy + 50, col_w - 36, fill=(78, 87, 100), font=body, line_h=18)
        draw.rounded_rectangle((cx + 18, cy + 120, cx + col_w - 18, cy + 315), radius=7, fill=(30, 34, 40))
        draw_code(draw, code, cx + 34, cy + 136, fill=(232, 236, 241), font=mono, line_h=20)
        draw.text((cx + 18, cy + 346), "later composition", fill=(78, 87, 100), font=body)
        wrapped(draw, seq, cx + 18, cy + 374, col_w - 36, fill=(20, 116, 81), font=h2, line_h=24)

    draw.text((64, h - 100), "Key point:", fill=(20, 24, 31), font=h2)
    wrapped(
        draw,
        "The impressive behavior is not a single generated answer. It is the loop: failure -> LLM-written operator -> verification -> persistence -> reuse/composition on later tasks, including held-out tile-grid problems.",
        160,
        h - 99,
        1250,
        fill=(65, 74, 88),
        font=body,
        line_h=19,
    )

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)
    return 0


def wrapped(draw, text: str, x: int, y: int, width: int, fill, font, line_h: int) -> None:
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
    for i, item in enumerate(lines):
        draw.text((x, y + i * line_h), item, fill=fill, font=font)


def draw_code(draw, code: str, x: int, y: int, fill, font, line_h: int) -> None:
    for i, line in enumerate(code.splitlines()):
        draw.text((x, y + i * line_h), line, fill=fill, font=font)


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def mono_font(size: int):
    for name in ("consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


if __name__ == "__main__":
    raise SystemExit(main())
