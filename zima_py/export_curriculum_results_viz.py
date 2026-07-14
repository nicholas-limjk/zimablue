from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ARC curriculum probe summary.")
    parser.add_argument("--out", default="artifacts/arc_curriculum/curriculum_probe_summary.png")
    args = parser.parse_args()

    families = [
        ("same_shape_recolor", "recolor", "Recolor / Mask"),
        ("crop_both_axes", "crop", "Crop / Extract"),
        ("output_grid_of_input_tiles", "tilegrid", "Tile-Grid"),
    ]
    rows = []
    for family_id, prefix, label in families:
        grow = load_json(f"artifacts/arc_curriculum/{prefix}_grow_5_report.json")
        unseen = load_json(f"artifacts/arc_curriculum/{prefix}_unseen_5_after_grow.json")
        rows.append(
            {
                "family": family_id,
                "label": label,
                "grow_correct": grow["test_correct"],
                "grow_total": grow["total"],
                "unseen_correct": unseen["test_correct"],
                "unseen_total": unseen["total"],
                "grow_wins": [run for run in grow["runs"] if run["test_ok"]],
                "unseen_wins": [run for run in unseen["rows"] if run["test_ok"]],
            }
        )

    w, h = 1400, 940
    img = Image.new("RGB", (w, h), (247, 248, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title_font = ImageFont.truetype("arial.ttf", 28) if font_exists("arial.ttf") else font
    h2_font = ImageFont.truetype("arial.ttf", 18) if font_exists("arial.ttf") else font

    draw.text((36, 28), "ARC Growing Body: Curriculum Probe", fill=(20, 24, 31), font=title_font)
    draw.text(
        (36, 66),
        "Grow operators on 5 same-family tasks, freeze the library, then test 5 unseen tasks from the same family.",
        fill=(84, 92, 106),
        font=font,
    )

    x0, y0 = 80, 132
    chart_w, chart_h = 980, 360
    draw.text((x0, y0 - 34), "Growth vs Frozen Transfer", fill=(20, 24, 31), font=h2_font)
    draw.line((x0, y0 + chart_h, x0 + chart_w, y0 + chart_h), fill=(190, 196, 205), width=2)
    for i in range(6):
        y = y0 + chart_h - i * chart_h / 5
        draw.line((x0, y, x0 + chart_w, y), fill=(229, 232, 236), width=1)
        draw.text((x0 - 36, y - 6), f"{i}/5", fill=(120, 128, 140), font=font)

    group_w = chart_w / len(rows)
    colors = {"grow": (48, 118, 198), "unseen": (32, 148, 102)}
    for i, row in enumerate(rows):
        gx = x0 + i * group_w + 80
        bar_w = 72
        grow_h = chart_h * row["grow_correct"] / 5
        unseen_h = chart_h * row["unseen_correct"] / 5
        draw.rectangle((gx, y0 + chart_h - grow_h, gx + bar_w, y0 + chart_h), fill=colors["grow"])
        draw.rectangle((gx + 94, y0 + chart_h - unseen_h, gx + 94 + bar_w, y0 + chart_h), fill=colors["unseen"])
        draw.text((gx + 18, y0 + chart_h - grow_h - 22), f"{row['grow_correct']}/5", fill=(20, 24, 31), font=font)
        draw.text((gx + 112, y0 + chart_h - unseen_h - 22), f"{row['unseen_correct']}/5", fill=(20, 24, 31), font=font)
        draw.text((gx - 8, y0 + chart_h + 18), row["label"], fill=(20, 24, 31), font=font)

    legend_x = x0 + chart_w + 30
    draw.rectangle((legend_x, y0 + 24, legend_x + 18, y0 + 42), fill=colors["grow"])
    draw.text((legend_x + 28, y0 + 26), "Grow-on-family", fill=(45, 52, 62), font=font)
    draw.rectangle((legend_x, y0 + 56, legend_x + 18, y0 + 74), fill=colors["unseen"])
    draw.text((legend_x + 28, y0 + 58), "Frozen same-family unseen", fill=(45, 52, 62), font=font)

    panel_y = 560
    draw.rounded_rectangle((36, panel_y, w - 36, h - 36), radius=10, fill=(255, 255, 255), outline=(218, 223, 230))
    draw.text((64, panel_y + 28), "What transferred?", fill=(20, 24, 31), font=h2_font)
    bullets = [
        ("Tile-grid curriculum", "4/5 solved during growth; 2/5 solved on frozen same-family held-out tasks."),
        ("Reusable operator", "vertical_reflect_and_stack solved two unseen tasks when composed with flip_v."),
        ("Compositional signal", "reflect_2d_tile solved a later task with no new LLM call."),
        ("Negative result", "Recolor and crop/extract curricula did not produce held-out transfer in this small probe."),
    ]
    y = panel_y + 72
    for title, body in bullets:
        draw.ellipse((64, y + 4, 74, y + 14), fill=(32, 148, 102) if "Tile" in title or "Reusable" in title else (190, 96, 70))
        draw.text((88, y), title, fill=(20, 24, 31), font=h2_font)
        draw_wrapped(draw, body, 88, y + 24, 1180, fill=(75, 83, 96), font=font, line_h=18)
        y += 74

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)
    return 0


def draw_wrapped(draw, text: str, x: int, y: int, width: int, fill, font, line_h: int = 16) -> None:
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


def load_json(path: str):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def font_exists(name: str) -> bool:
    try:
        ImageFont.truetype(name, 12)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
