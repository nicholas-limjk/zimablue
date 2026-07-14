from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    "unseen": (17, 24, 32),
    "empty": (247, 245, 238),
    "wall": (89, 99, 108),
    "floor": (247, 245, 238),
    "door": (165, 107, 52),
    "key": (242, 201, 76),
    "ball": (78, 163, 241),
    "box": (169, 112, 214),
    "goal": (72, 168, 104),
    "lava": (217, 88, 50),
    "agent": (232, 75, 75),
}

COLOR_BY_IDX = {
    0: (217, 88, 50),
    1: (72, 168, 104),
    2: (78, 163, 241),
    3: (142, 68, 173),
    4: (242, 201, 76),
    5: (154, 163, 170),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MiniGrid visualization JSON as an MP4.")
    parser.add_argument("--data", default="public/minigrid-viz-data.json")
    parser.add_argument("--out", default="artifacts/minigrid-demo.mp4")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--hold-final", type=float, default=1.5)
    parser.add_argument("--show-prompt", action="store_true", help="Render the environment rules and LLM prompt summary.")
    parser.add_argument("--title", default="MiniGrid Skill Run")
    args = parser.parse_args()

    data_path = (ROOT / args.data).resolve()
    out_path = (ROOT / args.out).resolve()
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    frames = render_frames(payload, fps=args.fps, hold_final=args.hold_final, show_prompt=args.show_prompt, title=args.title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
    print(f"wrote {out_path} ({len(frames)} frames at {args.fps} fps)")
    return 0


def render_frames(payload: dict[str, Any], fps: int, hold_final: float, show_prompt: bool, title: str):
    frames = [render_frame(payload, frame, show_prompt=show_prompt, title=title) for frame in payload["frames"]]
    if frames:
        frames.extend([frames[-1]] * max(0, int(round(hold_final * fps))))
    return frames


def render_frame(payload: dict[str, Any], frame: dict[str, Any], show_prompt: bool, title: str):
    width, height = (1920, 1088) if show_prompt else (1280, 720)
    stage_x, stage_y = 42, 64 if show_prompt else 38
    board_size = 920 if show_prompt else 640
    panel_x, panel_y = stage_x + board_size + 48, 64 if show_prompt else 38
    panel_width = width - panel_x - 42
    img = Image.new("RGB", (width, height), (244, 241, 232))
    draw = ImageDraw.Draw(img)
    font = load_font(22 if show_prompt else 12)
    small_font = load_font(18 if show_prompt else 12)
    heading_font = load_font(30 if show_prompt else 12)

    draw.text((stage_x, 22), title, fill=(23, 32, 42), font=heading_font)
    draw.rounded_rectangle((stage_x, stage_y, stage_x + board_size + 40, stage_y + board_size + 40), radius=8, fill=(255, 253, 248), outline=(217, 210, 195))
    draw.rounded_rectangle((panel_x, panel_y, width - 42, height - 42), radius=8, fill=(255, 253, 248), outline=(217, 210, 195))

    image = frame.get("full_image") or frame["image"]
    visible_mask = frame.get("visible_mask")
    object_by_idx = {value: key for key, value in payload["object_to_idx"].items()}
    cols = len(image)
    rows = len(image[0])
    cell = board_size // max(cols, rows)
    ox = stage_x + 20 + (board_size - cols * cell) // 2
    oy = stage_y + 20 + (board_size - rows * cell) // 2

    draw.rectangle((ox, oy, ox + cols * cell, oy + rows * cell), fill=(17, 24, 32))
    for y in range(rows):
        for x in range(cols):
            object_idx, color_idx, state = image[x][y]
            obj = object_by_idx.get(object_idx, "empty")
            visible = True if visible_mask is None else bool(visible_mask[x][y])
            draw_cell(draw, ox + x * cell, oy + y * cell, cell, obj, color_idx, state, visible)

    if frame.get("agent_pos"):
        ax, ay = frame["agent_pos"]
        draw_cell(draw, ox + ax * cell, oy + ay * cell, cell, "agent", 0, frame.get("agent_dir", frame.get("direction", 0)), True)

    draw.text((panel_x + 28, panel_y + 26), payload["env_id"], fill=(23, 32, 42), font=heading_font)
    wrapped_text(draw, frame.get("mission", ""), panel_x + 28, panel_y + 68, panel_width - 56, small_font, (88, 101, 114), line_gap=6)
    metric(draw, panel_x + 28, panel_y + 128, "STEP", f"{frame['step']}/{len(payload['frames']) - 1}", font, width=180 if show_prompt else 148)
    metric(draw, panel_x + 226, panel_y + 128, "STATUS", frame["status"], font, width=180 if show_prompt else 148)
    metric(draw, panel_x + 28, panel_y + 208, "SOURCE", frame["source"], font, width=180 if show_prompt else 148)
    metric(draw, panel_x + 226, panel_y + 208, "ACTION", str(frame.get("action") or "-"), font, width=180 if show_prompt else 148)

    trace = payload["frames"][max(0, frame["step"] - 5) : frame["step"] + 1]
    y = panel_y + 306
    draw.text((panel_x + 28, y), "Recent actions", fill=(23, 32, 42), font=font)
    y += 34
    for item in trace[-5:]:
        note = f" {item['note']}" if item.get("note") else ""
        line = f"t={item['step']} {item['source']} {item.get('action') or ''} r={item['reward']:.2f} {item['status']}{note}"
        draw.text((panel_x + 28, y), line, fill=(23, 32, 42) if item is frame else (100, 114, 127), font=small_font)
        y += 28

    if show_prompt:
        y += 24
        draw_section(
            draw,
            panel_x + 28,
            y,
            panel_width - 56,
            "World / Body Rules",
            [
                "Goal: use the key, open the door, reach the green goal.",
                "The body acts one primitive MiniGrid action per turn.",
                "Primitive actions: left, right, forward, pickup, toggle.",
                "The body only observes a partial egocentric 7x7 view.",
                "Dimmed cells are shown to us only; the agent cannot see them.",
            ],
            font,
            small_font,
        )
        y += 250
        draw_section(
            draw,
            panel_x + 28,
            y,
            panel_width - 56,
            "Lightweight LLM Prompt",
            [
                "Base impulse: reach_goal.",
                "Inspect only partial observation, learned memory, and recent failures.",
                "No full grid, hidden map, oracle position, env object, or future observations.",
                "Write a reusable turn-by-turn Python skill.",
                "Memory may store observations, outcomes, useful hypotheses, and control state.",
                "Simple hints: goals, keys, doors, walls, and unknown space may matter.",
                "Remembering what has been seen can be useful.",
                "Avoid repeating blocked or unproductive actions.",
                "Any search or planning loop must be finite.",
            ],
            font,
            small_font,
        )

    return img


def draw_cell(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, obj: str, color_idx: int, state: int, visible: bool) -> None:
    color = PALETTE.get(obj, PALETTE["empty"])
    if not visible:
        color = blend(color, (17, 24, 32), 0.68)
    draw.rectangle((x, y, x + size, y + size), fill=color, outline=(64, 72, 79))

    if obj == "door":
        door_color = (197, 139, 80) if state == 0 else (91, 58, 32)
        if not visible:
            door_color = blend(door_color, (17, 24, 32), 0.68)
        draw.rectangle((x + size * 0.22, y + size * 0.12, x + size * 0.78, y + size * 0.88), fill=door_color)

    if obj == "key":
        key_color = COLOR_BY_IDX.get(color_idx, PALETTE["key"])
        if not visible:
            key_color = blend(key_color, (17, 24, 32), 0.68)
        cx, cy = x + size * 0.42, y + size * 0.42
        r = size * 0.16
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=key_color)
        draw.rectangle((x + size * 0.52, y + size * 0.39, x + size * 0.80, y + size * 0.47), fill=key_color)

    if obj == "goal":
        goal_color = (226, 238, 230) if visible else (75, 92, 88)
        r = size * 0.18
        cx, cy = x + size / 2, y + size / 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=goal_color)

    if obj == "agent":
        points = triangle_points(x, y, size, state)
        draw.polygon(points, fill=PALETTE["agent"])


def triangle_points(x: int, y: int, size: int, direction: int):
    pad = size * 0.2
    cx = x + size / 2
    cy = y + size / 2
    if direction == 1:
        return [(cx, y + size - pad), (x + pad, y + pad), (x + size - pad, y + pad)]
    if direction == 2:
        return [(x + pad, cy), (x + size - pad, y + pad), (x + size - pad, y + size - pad)]
    if direction == 3:
        return [(cx, y + pad), (x + pad, y + size - pad), (x + size - pad, y + size - pad)]
    return [(x + size - pad, cy), (x + pad, y + pad), (x + pad, y + size - pad)]


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, font, width: int = 148) -> None:
    draw.rounded_rectangle((x, y, x + width, y + 62), radius=6, fill=(255, 255, 255), outline=(217, 210, 195))
    draw.text((x + 10, y + 8), label, fill=(100, 114, 127), font=font)
    draw.text((x + 10, y + 34), value, fill=(23, 32, 42), font=font)


def draw_section(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    title: str,
    lines: list[str],
    heading_font,
    body_font,
) -> None:
    draw.text((x, y), title, fill=(23, 32, 42), font=heading_font)
    y += 34
    for line in lines:
        wrapped = wrap_for_width(line, width - 26, body_font)
        first = True
        for chunk in wrapped:
            prefix = "- " if first else "  "
            draw.text((x, y), prefix + chunk, fill=(88, 101, 114), font=body_font)
            y += 24
            first = False
        y += 4


def wrapped_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, font, fill, line_gap: int = 4) -> int:
    for line in wrap_for_width(text, width, font):
        draw.text((x, y), line, fill=fill, font=font)
        y += text_height(font) + line_gap
    return y


def wrap_for_width(text: str, width: int, font) -> list[str]:
    approx_chars = max(12, width // max(8, text_width(font, "m")))
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=approx_chars) or [""])
    return lines


def load_font(size: int):
    for path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(font, text: str) -> int:
    left, _, right, _ = font.getbbox(text)
    return right - left


def text_height(font) -> int:
    _, top, _, bottom = font.getbbox("Ag")
    return bottom - top


def blend(color: tuple[int, int, int], base: tuple[int, int, int], amount: float):
    return tuple(int(color[i] * (1 - amount) + base[i] * amount) for i in range(3))


if __name__ == "__main__":
    raise SystemExit(main())
