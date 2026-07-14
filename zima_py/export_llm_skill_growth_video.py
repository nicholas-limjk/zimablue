from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FPS = 24
W, H = 1280, 720
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
    parser = argparse.ArgumentParser(description="Export a short MP4 illustrating LLM skill growth and reuse.")
    parser.add_argument("--out", default="artifacts/arc_curriculum/llm_skill_growth_demo.mp4")
    args = parser.parse_args()

    frames: list[Image.Image] = []
    add_main_solve_scene(frames, "4c4377d9", 9.0)
    add_transfer_3layer_scene(frames, "6fa7a44f", 4.2)
    add_transfer_3layer_scene(frames, "8be77c9e", 4.2)
    add_scene(frames, scene_summary(), 2.5)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, [frame.convert("RGB") for frame in frames], fps=FPS, macro_block_size=16)
    print(out)
    return 0


def scene_title() -> Image.Image:
    img, draw = canvas()
    draw.text((58, 68), "LLM-written skills become a growing body", fill=(22, 26, 33), font=font(34))
    draw.text((60, 118), "Failure -> skill writing -> verification -> reuse on unseen ARC tasks", fill=(73, 82, 96), font=font(18))
    steps = [
        ("1", "Base body searches", "crop / flip / rotate / fill"),
        ("2", "Search fails", "no existing operator fits"),
        ("3", "LLM writes skill", "small executable Python"),
        ("4", "Verifier keeps it", "train examples pass"),
        ("5", "Body reuses it", "later tasks compose skills"),
    ]
    for i, (num, title, desc) in enumerate(steps):
        x = 58 + i * 240
        y = 270
        draw.rounded_rectangle((x, y, x + 190, y + 160), radius=12, fill=(255, 255, 255), outline=(216, 222, 230))
        draw.ellipse((x + 18, y + 20, x + 52, y + 54), fill=(48, 118, 198))
        draw.text((x + 30, y + 27), num, fill=(255, 255, 255), font=font(14))
        draw.text((x + 18, y + 74), title, fill=(22, 26, 33), font=font(16))
        wrapped(draw, desc, x + 18, y + 104, 150, fill=(83, 91, 104), fnt=font(13), line_h=17)
        if i < len(steps) - 1:
            draw.line((x + 198, y + 82, x + 232, y + 82), fill=(48, 118, 198), width=3)
            draw.polygon([(x + 232, y + 82), (x + 222, y + 75), (x + 222, y + 89)], fill=(48, 118, 198))
    return img


def scene_failure() -> Image.Image:
    img, draw = canvas()
    task = load_task("4c4377d9")
    pair = task["train"][0]
    draw_header(draw, "Step 1: base body fails", "Task 4c4377d9. Existing primitives are searched first.")
    draw.text((110, 175), "train input", fill=(83, 91, 104), font=font(16))
    draw_grid(draw, pair["input"], 90, 210, 180)
    draw.text((390, 175), "target output", fill=(83, 91, 104), font=font(16))
    draw_grid(draw, pair["output"], 360, 210, 180)
    draw.text((690, 185), "Best existing primitives:", fill=(22, 26, 33), font=font(20))
    rows = [("identity", "0/4"), ("rotate90", "0/4"), ("flip_v", "0/4"), ("crop_nonzero_bbox", "0/4")]
    y = 235
    for name, score in rows:
        draw.rounded_rectangle((690, y, 1040, y + 44), radius=7, fill=(255, 255, 255), outline=(222, 227, 234))
        draw.text((710, y + 13), name, fill=(45, 52, 62), font=font(15))
        draw.text((990, y + 13), score, fill=(190, 64, 64), font=font(15))
        y += 54
    draw.text((690, 480), "No current body skill can make the output.", fill=(190, 64, 64), font=font(20))
    return img


def add_question_scene(frames: list[Image.Image], task_id: str, seconds: float) -> None:
    task = load_task(task_id)
    train = task["train"]
    test_input = task["test"][0]["input"]
    total = int(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        visible_pairs = min(len(train), int(p * 5.0) + 1)
        show_test = p > 0.82
        img, draw = canvas()
        draw_header(
            draw,
            "What the LLM sees",
            "Only visual input/output examples. It has to infer the transformation before writing a reusable skill.",
        )
        draw.text((72, 132), "ARC task 4c4377d9: train examples", fill=(22, 26, 33), font=font(20))
        positions = [(70, 175), (365, 175), (660, 175), (955, 175)]
        for idx, pair in enumerate(train):
            if idx < visible_pairs:
                draw_pair_card(draw, pair["input"], pair["output"], positions[idx][0], positions[idx][1], f"example {idx + 1}")
            else:
                draw_placeholder_card(draw, positions[idx][0], positions[idx][1], f"example {idx + 1}")
        if show_test:
            draw.rounded_rectangle((365, 520, 915, 665), radius=12, fill=(255, 255, 255), outline=(48, 118, 198), width=2)
            draw.text((390, 540), "then solve this test input", fill=(22, 26, 33), font=font(19))
            draw_grid(draw, test_input, 435, 570, 80)
            arrow(draw, 555, 610, 655, 610)
            draw.text((690, 592), "?", fill=(48, 118, 198), font=font(42))
        frames.append(img)


def add_main_solve_scene(frames: list[Image.Image], task_id: str, seconds: float) -> None:
    task = load_task(task_id)
    total = int(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        img, draw = canvas()
        draw_header(
            draw,
            "What the LLM sees",
            "Visual examples stay on screen. The lower layers show how the solver learns and applies a skill.",
        )
        draw.text((72, 122), f"ARC task {task_id}: train examples", fill=(22, 26, 33), font=font(20))
        draw_training_examples_strip(draw, task["train"], y=158)
        draw_solution_trace(draw, task, p)
        frames.append(img)


def draw_training_examples_strip(draw: ImageDraw.ImageDraw, train: list[dict], y: int) -> None:
    positions = [(70, y), (365, y), (660, y), (955, y)]
    for idx, pair in enumerate(train[:4]):
        draw_pair_card(draw, pair["input"], pair["output"], positions[idx][0], positions[idx][1], f"example {idx + 1}", scale=0.78)


def draw_solution_trace(draw: ImageDraw.ImageDraw, task: dict, p: float) -> None:
    pair = task["test"][0]
    test_input = pair["input"]
    solved = vertical_reflect_and_stack(test_input)
    draw_solver_layer(draw, p)
    draw_transform_layer(draw, test_input, solved, p)


def draw_solver_layer(draw: ImageDraw.ImageDraw, p: float) -> None:
    y0 = 410
    draw.rounded_rectangle((70, y0, 1210, y0 + 105), radius=14, fill=(255, 255, 255), outline=(212, 219, 229))
    draw.text((95, y0 + 18), "Layer 2: solver", fill=(22, 26, 33), font=font(18))
    steps = [
        ("1", "search body", "base primitives"),
        ("2", "fail", "no match"),
        ("3", "ask LLM", "write skill"),
        ("4", "verify", "train pass"),
    ]
    active = solver_stage(p)
    for idx, (num, title, desc) in enumerate(steps):
        x = 315 + idx * 195
        fill = (238, 246, 255) if idx == active else (249, 250, 252)
        outline = (48, 118, 198) if idx == active else (222, 227, 234)
        if idx < active:
            fill = (237, 250, 244)
            outline = (160, 214, 190)
        draw.rounded_rectangle((x, y0 + 18, x + 160, y0 + 86), radius=10, fill=fill, outline=outline, width=2 if idx == active else 1)
        draw.ellipse((x + 12, y0 + 34, x + 36, y0 + 58), fill=(48, 118, 198) if idx >= active else (32, 148, 102))
        draw.text((x + 20, y0 + 38), num, fill=(255, 255, 255), font=font(12))
        draw.text((x + 45, y0 + 30), title, fill=(22, 26, 33), font=font(13))
        draw.text((x + 45, y0 + 54), desc, fill=(83, 91, 104), font=font(10))
    if p > 0.86:
        draw.rounded_rectangle((95, y0 + 57, 250, y0 + 86), radius=8, fill=(32, 148, 102))
        draw.text((122, y0 + 64), "skill accepted", fill=(255, 255, 255), font=font(14))


def draw_transform_layer(draw: ImageDraw.ImageDraw, test_input: list[list[int]], solved: list[list[int]], p: float) -> None:
    y0 = 535
    draw.rounded_rectangle((70, y0, 1210, y0 + 150), radius=14, fill=(255, 255, 255), outline=(212, 219, 229))
    draw.text((95, y0 + 14), "Layer 3: how the test case is solved", fill=(22, 26, 33), font=font(18))

    draw.text((120, y0 + 48), "test input", fill=(83, 91, 104), font=font(13))
    draw_grid(draw, test_input, 116, y0 + 67, 74)

    if p < 0.08:
        draw.text((310, y0 + 74), "the output is still unknown", fill=(114, 123, 139), font=font(19))
        draw.text((310, y0 + 102), "?", fill=(48, 118, 198), font=font(38))
        return

    if p < 0.18:
        arrow(draw, 220, y0 + 104, 290, y0 + 104)
        draw.rounded_rectangle((315, y0 + 58, 555, y0 + 125), radius=10, fill=(255, 243, 243), outline=(232, 185, 185))
        draw.text((338, y0 + 75), "base primitives fail", fill=(190, 64, 64), font=font(18))
        draw.text((338, y0 + 101), "no solved output yet", fill=(114, 82, 82), font=font(13))
        return

    arrow(draw, 220, y0 + 104, 290, y0 + 104)
    draw.rounded_rectangle((315, y0 + 51, 590, y0 + 132), radius=10, fill=(30, 34, 40))
    code_lines = [
        "def solve(grid):",
        "  reflected = grid[::-1]",
        "  return reflected + grid",
    ]
    code_progress = smoothstep(clamp((p - 0.18) / 0.14))
    chars = "\n".join(code_lines)
    draw_code(draw, chars[: int(len(chars) * code_progress)], 335, y0 + 65, fill=(232, 236, 241), fnt=mono(14), line_h=20)
    if p < 0.32:
        draw.text((630, y0 + 78), "LLM writes the missing operation", fill=(20, 116, 81), font=font(18))
        return

    arrow(draw, 610, y0 + 104, 670, y0 + 104)
    mirror_progress = smootherstep(clamp((p - 0.32) / 0.22))
    draw.text((700, y0 + 48), "mirror", fill=(83, 91, 104), font=font(13))
    draw_grid_flip_v_animation(draw, test_input, 695, y0 + 63, 78, mirror_progress)
    if p < 0.54:
        draw.text((805, y0 + 80), "make reflected copy", fill=(20, 116, 81), font=font(16))
        return

    arrow(draw, 795, y0 + 104, 850, y0 + 104)
    stack_progress = smootherstep(clamp((p - 0.54) / 0.24))
    draw.text((880, y0 + 48), "extend", fill=(83, 91, 104), font=font(13))
    draw_stack_animation(draw, test_input, 872, y0 + 48, 100, stack_progress)
    if p < 0.78:
        draw.text((985, y0 + 80), "stack with original", fill=(20, 116, 81), font=font(16))
        return

    arrow(draw, 985, y0 + 104, 1030, y0 + 104)
    draw.text((1048, y0 + 48), "solved", fill=(83, 91, 104), font=font(13))
    draw_grid(draw, solved, 1038, y0 + 61, 88)


def solver_stage(p: float) -> int:
    if p < 0.12:
        return 0
    if p < 0.18:
        return 1
    if p < 0.72:
        return 2
    return 3


def add_failure_scene(frames: list[Image.Image], seconds: float) -> None:
    task = load_task("4c4377d9")
    pair = task["train"][0]
    rows = [("identity", "0/4"), ("rotate90", "0/4"), ("flip_v", "0/4"), ("crop_nonzero_bbox", "0/4")]
    total = int(seconds * FPS)
    for i in range(total):
        img, draw = canvas()
        draw_header(draw, "Step 2: base body fails", "The current body tries its existing primitives before asking the LLM.")
        draw.text((110, 175), "train input", fill=(83, 91, 104), font=font(16))
        draw_grid(draw, pair["input"], 90, 210, 180)
        draw.text((390, 175), "target output", fill=(83, 91, 104), font=font(16))
        draw_grid(draw, pair["output"], 360, 210, 180)
        draw.text((690, 185), "searching current body", fill=(22, 26, 33), font=font(20))

        active = min(len(rows) - 1, int((i / max(1, total - 1)) * len(rows)))
        y = 235
        for j, (name, score) in enumerate(rows):
            is_active = j == active and i < total * 0.78
            fill = (238, 246, 255) if is_active else (255, 255, 255)
            outline = (48, 118, 198) if is_active else (222, 227, 234)
            draw.rounded_rectangle((690, y, 1040, y + 44), radius=7, fill=fill, outline=outline, width=2 if is_active else 1)
            draw.text((710, y + 13), name, fill=(45, 52, 62), font=font(15))
            draw.text((990, y + 13), score, fill=(190, 64, 64), font=font(15))
            if j < active:
                draw.text((1060, y + 13), "rejected", fill=(150, 76, 76), font=font(14))
            elif is_active:
                draw.text((1060, y + 13), "testing...", fill=(48, 118, 198), font=font(14))
            y += 54

        if i > total * 0.78:
            draw.rounded_rectangle((690, 478, 1095, 540), radius=10, fill=(255, 238, 238), outline=(230, 178, 178))
            draw.text((715, 499), "failure: no existing skill fits", fill=(190, 64, 64), font=font(21))
        frames.append(img)


def add_typing_scene(frames: list[Image.Image], seconds: float) -> None:
    code = [
        "def solve(grid):",
        "    if not grid:",
        "        return []",
        "    orig = [list(row) for row in grid]",
        "    reflected = [list(row) for row in orig[::-1]]",
        "    out = []",
        "    for row in reflected:",
        "        out.append(list(row))",
        "    for row in orig:",
        "        out.append(list(row))",
        "    return out",
    ]
    total = int(seconds * FPS)
    chars = "\n".join(code)
    for i in range(total):
        img, draw = canvas()
        draw_header(draw, "Step 3: LLM writes a new skill", "The model proposes a small reusable operator, not a final answer.")
        draw.rounded_rectangle((80, 155, 1200, 610), radius=12, fill=(30, 34, 40))
        n = int(len(chars) * min(1, (i + 1) / (total * 0.8)))
        draw_code(draw, chars[:n], 110, 185, fill=(232, 236, 241), fnt=mono(18), line_h=26)
        draw.text((850, 555), "vertical_reflect_and_stack", fill=(108, 220, 170), font=font(22))
        frames.append(img)


def scene_verify(progress: float = 1.0) -> Image.Image:
    img, draw = canvas()
    task = load_task("4c4377d9")
    pair = task["train"][0]
    pred = vertical_reflect_and_stack(pair["input"])
    draw_header(draw, "Step 4: verifier accepts the skill", "The learned operator mirrors the whole grid, then extends it by stacking the original underneath.")
    draw.text((95, 165), "input", fill=(83, 91, 104), font=font(16))
    draw_grid(draw, pair["input"], 70, 210, 135)
    arrow(draw, 215, 280, 285, 280)
    draw.text((310, 165), "mirror vertically", fill=(83, 91, 104), font=font(16))
    draw_grid_flip_v_animation(draw, pair["input"], 300, 210, 135, smoothstep(clamp(progress / 0.45)))
    arrow(draw, 455, 280, 525, 280)
    draw.text((550, 165), "extend / stack", fill=(83, 91, 104), font=font(16))
    draw_stack_animation(draw, pair["input"], 535, 185, 190, smoothstep(clamp((progress - 0.35) / 0.5)))
    arrow(draw, 735, 280, 805, 280)
    draw.text((830, 165), "target", fill=(83, 91, 104), font=font(16))
    draw_grid(draw, pair["output"], 805, 185, 190)
    phase = "mirror" if progress < 0.45 else "mirror + stack"
    draw.text((535, 455), phase, fill=(20, 116, 81), font=font(24))
    if progress >= 0.98:
        draw.rounded_rectangle((470, 500, 820, 570), radius=10, fill=(32, 148, 102))
        draw.text((515, 523), "verified: 4/4 train pairs", fill=(255, 255, 255), font=font(22))
    return img


def add_verify_scene(frames: list[Image.Image], seconds: float) -> None:
    total = int(seconds * FPS)
    for i in range(total):
        progress = smoothstep(min(1.0, i / max(1, total * 0.78)))
        frames.append(scene_verify(progress))


def scene_unseen(task_id: str, flip_progress: float = 1.0, stack_progress: float = 1.0) -> Image.Image:
    img, draw = canvas()
    task = load_task(task_id)
    pair = task["train"][0]
    flipped = flip_v(pair["input"])
    pred = vertical_reflect_and_stack(flipped)
    draw_header(draw, f"Step 5: frozen body solves unseen task {task_id}", "No new LLM call. Search composes a base flip with the learned skill.")
    draw.text((70, 165), "unseen input", fill=(83, 91, 104), font=font(16))
    draw_grid(draw, pair["input"], 50, 200, 150)
    arrow(draw, 210, 275, 280, 275)
    draw.text((305, 165), "base skill: flip_v", fill=(83, 91, 104), font=font(16))
    draw_grid_flip_v_animation(draw, pair["input"], 300, 200, 150, flip_progress)
    arrow(draw, 470, 275, 540, 275)
    draw.text((565, 165), "learned skill", fill=(83, 91, 104), font=font(16))
    draw_stack_animation(draw, flipped, 565, 190, 180, stack_progress)
    arrow(draw, 755, 275, 825, 275)
    draw.text((850, 165), "target output", fill=(83, 91, 104), font=font(16))
    draw_grid(draw, pair["output"], 850, 200, 170)
    draw.text((250, 495), "winning sequence:", fill=(22, 26, 33), font=font(20))
    draw.text((440, 495), "flip_v -> vertical_reflect_and_stack", fill=(20, 116, 81), font=font(24))
    if stack_progress >= 0.98:
        draw.rounded_rectangle((450, 545, 830, 600), radius=9, fill=(32, 148, 102))
        draw.text((505, 562), "held-out transfer passed", fill=(255, 255, 255), font=font(21))
    return img


def add_unseen_scene(frames: list[Image.Image], task_id: str, seconds: float) -> None:
    total = int(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        flip_progress = smoothstep(clamp((p - 0.12) / 0.28))
        stack_progress = smoothstep(clamp((p - 0.45) / 0.35))
        frames.append(scene_unseen(task_id, flip_progress, stack_progress))


def add_transfer_3layer_scene(frames: list[Image.Image], task_id: str, seconds: float) -> None:
    task = load_task(task_id)
    total = int(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        img, draw = canvas()
        draw_header(
            draw,
            f"Step 5: reuse the grown body on unseen task {task_id}",
            "No new LLM call. Search composes base skills with the learned skill.",
        )
        pair = task["train"][0]
        draw_transfer_prompt_layer(draw, pair)
        draw_transfer_solver_layer(draw, p)
        draw_transfer_transform_layer(draw, pair, p)
        frames.append(img)


def draw_transfer_prompt_layer(draw: ImageDraw.ImageDraw, pair: dict) -> None:
    y0 = 135
    draw.rounded_rectangle((70, y0, 1210, y0 + 235), radius=14, fill=(255, 255, 255), outline=(212, 219, 229))
    draw.text((95, y0 + 18), "Layer 1: unseen visual task", fill=(22, 26, 33), font=font(18))
    draw.text((155, y0 + 62), "input", fill=(83, 91, 104), font=font(14))
    draw_grid(draw, pair["input"], 125, y0 + 92, 110)
    arrow(draw, 290, y0 + 145, 390, y0 + 145)
    draw.text((465, y0 + 58), "target", fill=(83, 91, 104), font=font(14))
    draw_grid(draw, pair["output"], 465, y0 + 72, 145)
    draw.text((730, y0 + 82), "Reuse grown body. No new LLM call.", fill=(73, 82, 96), font=font(20))


def draw_transfer_solver_layer(draw: ImageDraw.ImageDraw, p: float) -> None:
    y0 = 395
    draw.rounded_rectangle((70, y0, 1210, y0 + 92), radius=14, fill=(255, 255, 255), outline=(212, 219, 229))
    draw.text((95, y0 + 17), "Layer 2: solver", fill=(22, 26, 33), font=font(18))
    steps = [
        ("1", "search", "base + learned"),
        ("2", "compose", "flip_v + skill"),
        ("3", "verify", "target match"),
    ]
    active = 0 if p < 0.35 else 1 if p < 0.78 else 2
    for idx, (num, title, desc) in enumerate(steps):
        x = 330 + idx * 240
        fill = (238, 246, 255) if idx == active else (249, 250, 252)
        outline = (48, 118, 198) if idx == active else (222, 227, 234)
        if idx < active:
            fill = (237, 250, 244)
            outline = (160, 214, 190)
        draw.rounded_rectangle((x, y0 + 17, x + 185, y0 + 74), radius=10, fill=fill, outline=outline, width=2 if idx == active else 1)
        draw.ellipse((x + 13, y0 + 31, x + 37, y0 + 55), fill=(48, 118, 198) if idx >= active else (32, 148, 102))
        draw.text((x + 21, y0 + 35), num, fill=(255, 255, 255), font=font(12))
        draw.text((x + 48, y0 + 29), title, fill=(22, 26, 33), font=font(14))
        draw.text((x + 48, y0 + 52), desc, fill=(83, 91, 104), font=font(11))


def draw_transfer_transform_layer(draw: ImageDraw.ImageDraw, pair: dict, p: float) -> None:
    y0 = 510
    grid = pair["input"]
    flipped = flip_v(grid)
    solved = vertical_reflect_and_stack(flipped)
    draw.rounded_rectangle((70, y0, 1210, y0 + 175), radius=14, fill=(255, 255, 255), outline=(212, 219, 229))
    draw.text((95, y0 + 14), "Layer 3: transformations", fill=(22, 26, 33), font=font(18))
    draw.text((135, y0 + 50), "input", fill=(83, 91, 104), font=font(13))
    draw_grid(draw, grid, 115, y0 + 72, 82)
    arrow(draw, 220, y0 + 112, 285, y0 + 112)

    flip_p = smootherstep(clamp((p - 0.08) / 0.35))
    draw.text((320, y0 + 50), "flip_v", fill=(83, 91, 104), font=font(13))
    draw_grid_flip_v_animation(draw, grid, 310, y0 + 66, 95, flip_p)
    if p < 0.43:
        return

    arrow(draw, 425, y0 + 112, 495, y0 + 112)
    stack_p = smootherstep(clamp((p - 0.43) / 0.35))
    draw.text((535, y0 + 40), "skill", fill=(83, 91, 104), font=font(13))
    draw_stack_animation(draw, flipped, 540, y0 + 62, 105, stack_p)
    if p < 0.78:
        return

    arrow(draw, 675, y0 + 112, 750, y0 + 112)
    draw.text((790, y0 + 50), "solved", fill=(83, 91, 104), font=font(13))
    draw_grid(draw, solved, 780, y0 + 55, 115)
    draw.rounded_rectangle((960, y0 + 70, 1145, y0 + 115), radius=9, fill=(32, 148, 102))
    draw.text((1007, y0 + 83), "passed", fill=(255, 255, 255), font=font(18))


def scene_summary() -> Image.Image:
    img, draw = canvas()
    draw_header(draw, "Result: growth becomes reusable ability", "The skill is not just a cached answer; it changes what the body can search next.")
    bars = [
        ("Before tile-grid curriculum", 0, 5, (190, 64, 64)),
        ("After tile-grid curriculum", 2, 5, (32, 148, 102)),
        ("Broad one-step tile-grid sweep", 6, 12, (48, 118, 198)),
    ]
    x0, y0 = 210, 185
    for i, (label, val, total, color) in enumerate(bars):
        y = y0 + i * 115
        draw.text((x0, y), label, fill=(22, 26, 33), font=font(20))
        draw.rounded_rectangle((x0, y + 34, x0 + 650, y + 70), radius=8, fill=(225, 229, 235))
        draw.rounded_rectangle((x0, y + 34, x0 + int(650 * val / total), y + 70), radius=8, fill=color)
        draw.text((x0 + 680, y + 40), f"{val}/{total}", fill=(22, 26, 33), font=font(20))
    draw.text((210, 565), "The LLM discovered/wrote operators; verifier/search reused and composed them on later tasks.", fill=(73, 82, 96), font=font(20))
    return img


def add_scene(frames: list[Image.Image], img: Image.Image, seconds: float) -> None:
    frames.extend([img] * int(seconds * FPS))


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), (247, 248, 250))
    return img, ImageDraw.Draw(img)


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((48, 38), title, fill=(22, 26, 33), font=font(30))
    wrapped(draw, subtitle, 50, 80, 1120, fill=(73, 82, 96), fnt=font(17), line_h=22)


def arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int) -> None:
    draw.line((x1, y1, x2, y2), fill=(48, 118, 198), width=4)
    draw.polygon([(x2, y2), (x2 - 12, y2 - 8), (x2 - 12, y2 + 8)], fill=(48, 118, 198))


def draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, box: int) -> None:
    rows = len(grid)
    cols = len(grid[0])
    cell = max(4, min(box // max(rows, cols), 34))
    ox = x + (box - cols * cell) // 2
    oy = y + (box - rows * cell) // 2
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            draw.rectangle((ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 2, oy + (r + 1) * cell - 2), fill=COLORS.get(int(value), (60, 60, 60)))


def draw_pair_card(
    draw: ImageDraw.ImageDraw,
    input_grid: list[list[int]],
    output_grid: list[list[int]],
    x: int,
    y: int,
    label: str,
    scale: float = 1.0,
) -> None:
    w = int(250 * scale)
    h = int(260 * scale)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=(255, 255, 255), outline=(216, 222, 230))
    draw.text((x + int(18 * scale), y + int(18 * scale)), label, fill=(73, 82, 96), font=font(max(12, int(15 * scale))))
    draw.text((x + int(26 * scale), y + int(58 * scale)), "input", fill=(83, 91, 104), font=font(max(11, int(13 * scale))))
    draw.text((x + int(152 * scale), y + int(58 * scale)), "output", fill=(83, 91, 104), font=font(max(11, int(13 * scale))))
    draw_grid(draw, input_grid, x + int(20 * scale), y + int(85 * scale), int(85 * scale))
    arrow(draw, x + int(105 * scale), y + int(128 * scale), x + int(143 * scale), y + int(128 * scale))
    draw_grid(draw, output_grid, x + int(145 * scale), y + int(85 * scale), int(95 * scale))
    draw.text((x + int(24 * scale), y + int(215 * scale)), "visual rule only", fill=(114, 123, 139), font=font(max(11, int(13 * scale))))


def draw_placeholder_card(draw: ImageDraw.ImageDraw, x: int, y: int, label: str) -> None:
    draw.rounded_rectangle((x, y, x + 250, y + 260), radius=12, fill=(239, 242, 246), outline=(222, 227, 234))
    draw.text((x + 18, y + 18), label, fill=(140, 148, 160), font=font(15))
    draw.text((x + 98, y + 120), "...", fill=(166, 174, 187), font=font(28))


def draw_grid_revealed(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, box: int, reveal: float) -> None:
    rows = len(grid)
    cols = len(grid[0])
    cell = max(4, min(box // max(rows, cols), 34))
    ox = x + (box - cols * cell) // 2
    oy = y + (box - rows * cell) // 2
    total = rows * cols
    shown = int(total * clamp(reveal) + 0.5)
    draw.rounded_rectangle((x, y, x + box, y + box), radius=8, fill=(236, 239, 244), outline=(219, 225, 233))
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            idx = r * cols + c
            color = COLORS.get(int(value), (60, 60, 60)) if idx < shown else (220, 225, 232)
            draw.rectangle((ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 2, oy + (r + 1) * cell - 2), fill=color)
            if idx == shown and reveal < 0.98:
                draw.rectangle((ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 2, oy + (r + 1) * cell - 2), outline=(255, 255, 255), width=2)


def draw_grid_flip_v_animation(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, box: int, progress: float) -> None:
    rows = len(grid)
    cols = len(grid[0])
    cell = max(4, min(box // max(rows, cols), 34))
    width = cols * cell
    height = rows * cell
    ox = x + (box - width) // 2
    oy = y + (box - height) // 2
    p = smootherstep(progress)
    draw.rounded_rectangle((x, y, x + box, y + box), radius=8, fill=(236, 239, 244), outline=(219, 225, 233))
    draw.line((ox - 8, oy + height // 2, ox + width + 8, oy + height // 2), fill=(205, 213, 224), width=2)
    if 0.02 < p < 0.98:
        draw_grid_at_tinted(draw, flip_v(grid), ox, oy, cell, 0.16)
    for r, row in enumerate(grid):
        target_r = rows - 1 - r
        yy = oy + int((r + (target_r - r) * p) * cell)
        for c, value in enumerate(row):
            xx = ox + c * cell
            draw.rectangle((xx + 3, yy + 4, xx + cell + 1, yy + cell + 2), fill=(205, 211, 220))
            draw.rectangle((xx, yy, xx + cell - 2, yy + cell - 2), fill=COLORS.get(int(value), (60, 60, 60)))
    if 0.05 < p < 0.95:
        draw.arc((ox - 20, oy - 10, ox + width + 20, oy + height + 10), 80, 280, fill=(48, 118, 198), width=3)
        draw.polygon([(ox + width + 16, oy + height - 4), (ox + width + 5, oy + height - 8), (ox + width + 11, oy + height - 18)], fill=(48, 118, 198))


def draw_stack_animation(draw: ImageDraw.ImageDraw, grid: list[list[int]], x: int, y: int, box: int, progress: float) -> None:
    rows = len(grid)
    cols = len(grid[0])
    cell = max(4, min(box // max(rows * 2, cols), 34))
    width = cols * cell
    full_height = rows * 2 * cell
    ox = x + (box - width) // 2
    oy = y + (box - full_height) // 2
    p = smootherstep(progress)
    reflected = flip_v(grid)
    draw.rounded_rectangle((x, y, x + box, y + box), radius=8, fill=(236, 239, 244), outline=(219, 225, 233))
    center_y = y + (box - rows * cell) // 2
    reflected_y = int(center_y + (oy - center_y) * p)
    original_y = int(center_y + (oy + rows * cell - center_y) * p)
    if 0.02 < p < 0.98:
        draw_grid_at_tinted(draw, reflected, ox, oy, cell, 0.14)
        draw_grid_at_tinted(draw, grid, ox, oy + rows * cell, cell, 0.14)
    if p > 0.03:
        draw_grid_at_shadow(draw, reflected, ox, reflected_y, cell)
    draw_grid_at_shadow(draw, grid, ox, original_y, cell)
    if 0.05 < p < 0.98:
        draw.line((ox + width + 16, reflected_y + rows * cell, ox + width + 16, original_y), fill=(48, 118, 198), width=3)
        draw.polygon([(ox + width + 16, original_y), (ox + width + 9, original_y - 10), (ox + width + 23, original_y - 10)], fill=(48, 118, 198))


def draw_grid_at(draw: ImageDraw.ImageDraw, grid: list[list[int]], ox: int, oy: int, cell: int) -> None:
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            draw.rectangle(
                (ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 2, oy + (r + 1) * cell - 2),
                fill=COLORS.get(int(value), (60, 60, 60)),
            )


def draw_grid_at_shadow(draw: ImageDraw.ImageDraw, grid: list[list[int]], ox: int, oy: int, cell: int) -> None:
    rows = len(grid)
    cols = len(grid[0])
    draw.rounded_rectangle((ox + 3, oy + 4, ox + cols * cell + 1, oy + rows * cell + 2), radius=4, fill=(206, 212, 222))
    draw_grid_at(draw, grid, ox, oy, cell)


def draw_grid_at_tinted(draw: ImageDraw.ImageDraw, grid: list[list[int]], ox: int, oy: int, cell: int, amount: float) -> None:
    bg = (236, 239, 244)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            base = COLORS.get(int(value), (60, 60, 60))
            color = blend(bg, base, amount)
            draw.rectangle(
                (ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 2, oy + (r + 1) * cell - 2),
                fill=color,
            )


def wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fill, fnt, line_h: int) -> None:
    words = text.split()
    line = ""
    lines = []
    for word in words:
        probe = f"{line} {word}".strip()
        if draw.textlength(probe, font=fnt) <= width:
            line = probe
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    for i, item in enumerate(lines):
        draw.text((x, y + i * line_h), item, fill=fill, font=fnt)


def draw_code(draw: ImageDraw.ImageDraw, code: str, x: int, y: int, fill, fnt, line_h: int) -> None:
    for i, line in enumerate(code.splitlines()):
        draw.text((x, y + i * line_h), line, fill=fill, font=fnt)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3 - 2 * x)


def smootherstep(value: float) -> float:
    x = clamp(value)
    return x * x * x * (x * (x * 6 - 15) + 10)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    k = clamp(t)
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def vertical_reflect_and_stack(grid: list[list[int]]) -> list[list[int]]:
    orig = [list(row) for row in grid]
    return [list(row) for row in orig[::-1]] + [list(row) for row in orig]


def flip_v(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid[::-1]]


def load_task(task_id: str):
    return json.loads((ROOT / "arc_agi_source" / "data" / "training" / f"{task_id}.json").read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def mono(size: int):
    for name in ("consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
