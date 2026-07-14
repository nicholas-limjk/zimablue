from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

from .export_policy_evolution_grid import HARD_NAV_PANELS, render_frames, rollout


ROOT = Path(__file__).resolve().parents[1]


DOORKEY_PANELS = [
    {
        "policy": "FIRST SKILL",
        "env": "MiniGrid-DoorKey-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_general_skill.py",
        "headline": "reactive behavior",
        "caption": "The first policy reacts to nearby cells and simple interactions, but has no durable task model.",
        "accent": (160, 73, 58),
        "hold_after": 34,
    },
    {
        "policy": "REFLECTED",
        "env": "MiniGrid-DoorKey-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_reflected_skill.py",
        "headline": "writes a planner",
        "caption": "After failure, reflection writes a persistent key-door-goal planner with learned state and finite search.",
        "accent": (49, 112, 178),
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a paper-style stitched MiniGrid policy evolution demo.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--out", default="artifacts/minigrid-paper-demo.mp4")
    args = parser.parse_args()

    doorkey_runs = [rollout(panel, args.steps) for panel in DOORKEY_PANELS]
    hard_panels = [
        {**panel, "hold_after": 26} if panel["policy"] == "REFLECTED" else panel
        for panel in HARD_NAV_PANELS
    ]
    hard_runs = [rollout(panel, args.steps) for panel in hard_panels]

    frames = []
    frames.extend(title_frames("Initial problem: DoorKey", "A reactive skill fails; reflection writes a persistent planner.", 18))
    frames.extend(render_door_key_intro(doorkey_runs, args.steps))
    frames.extend(title_frames("Curriculum reflection", "The reflected skill fails outside its task family; curriculum reflection repairs general navigation.", 16))
    frames.extend(
        render_frames(
            hard_runs,
            args.steps,
            "MiniGrid Navigation Generalization",
            "CURRICULUM keeps object interaction but adds terminal-cue and frontier navigation.",
        )
    )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
    summary = {
        "title": "MiniGrid Policy Evolution Paper Demo",
        "video": str(out),
        "fps": args.fps,
        "segments": [
            {
                "name": "DoorKey reflection",
                "runs": compact_runs(doorkey_runs),
            },
            {
                "name": "Navigation generalization",
                "runs": compact_runs(hard_runs),
            },
        ],
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out} ({len(frames)} frames)")
    return 0


def compact_runs(runs):
    return [
        {
            "policy": run["policy"],
            "env": run["env"],
            "seed": run["seed"],
            "status": run["status"],
            "final_step": run["final_step"],
            "headline": run["headline"],
        }
        for run in runs
    ]


def title_frames(title: str, subtitle: str, count: int):
    frames = []
    title_font = load_font(52)
    body_font = load_font(28)
    for _ in range(count):
        img = Image.new("RGB", (1920, 1088), (244, 241, 232))
        draw = ImageDraw.Draw(img)
        draw.text((80, 420), title, fill=(23, 32, 42), font=title_font)
        draw.text((82, 490), subtitle, fill=(88, 101, 114), font=body_font)
        frames.append(img)
    return frames


def render_door_key_intro(runs, max_steps: int):
    width, height = 1920, 1088
    max_len = max(len(run["frames"]) for run in runs)
    frames = []
    title_font = load_font(38)
    header_font = load_font(32)
    font = load_font(22)
    small = load_font(17)
    for index in range(max_len + 18):
        img = Image.new("RGB", (width, height), (244, 241, 232))
        draw = ImageDraw.Draw(img)
        draw.text((42, 28), "DoorKey: reflection creates the first useful planner", fill=(23, 32, 42), font=title_font)
        draw.text(
            (44, 76),
            "Same initial problem, same seed: the reflected policy turns failure into a persistent key-door-goal skill.",
            fill=(88, 101, 114),
            font=font,
        )
        render_big_panel(draw, runs[0], min(index, len(runs[0]["frames"]) - 1), 52, 140, 870, 820, max_steps, header_font, font, small)
        render_big_panel(draw, runs[1], min(index, len(runs[1]["frames"]) - 1), 998, 140, 870, 820, max_steps, header_font, font, small)
        frames.append(img)
    return frames


def render_big_panel(draw, run, frame_index, x, y, w, h, max_steps, header_font, font, small):
    from .export_policy_evolution_grid import render_panel

    render_panel(draw, run, frame_index, x, y, w, h, max_steps, header_font, font, small)


def load_font(size: int):
    for path in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
