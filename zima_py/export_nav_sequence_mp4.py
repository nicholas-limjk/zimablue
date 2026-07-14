from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

from .export_minigrid_mp4 import draw_cell
from .minigrid_adapter import MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step, require_minigrid
from .skill_runtime import MiniGridSkillApi, SkillProgram, action_map, update_body_memory


ROOT = Path(__file__).resolve().parents[1]


SEQUENCE = [
    {
        "name": "simple_navigation_reflected_fail",
        "env": "MiniGrid-Empty-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_reflected_skill.py",
        "skill_label": "REFLECTED DOORKEY SKILL",
        "steps": 42,
        "hold_after": 24,
        "accent": (182, 68, 56),
    },
    {
        "name": "simple_navigation_curriculum_solve",
        "env": "MiniGrid-Empty-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "skill_label": "CURRICULUM NAV SKILL",
        "steps": 42,
        "accent": (54, 143, 93),
    },
    {
        "name": "key_door_first_fail",
        "env": "MiniGrid-DoorKey-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_general_skill.py",
        "skill_label": "FIRST GENERAL SKILL",
        "steps": 56,
        "hold_after": 34,
        "accent": (182, 68, 56),
    },
    {
        "name": "key_door_reflected_solve",
        "env": "MiniGrid-DoorKey-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_reflected_skill.py",
        "skill_label": "REFLECTED DOORKEY SKILL",
        "steps": 72,
        "accent": (50, 112, 178),
    },
    {
        "name": "complex_navigation_reflected_fail",
        "env": "MiniGrid-MultiRoom-N2-S4-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_reflected_skill.py",
        "skill_label": "REFLECTED DOORKEY SKILL",
        "steps": 58,
        "hold_after": 30,
        "accent": (182, 68, 56),
    },
    {
        "name": "complex_navigation_curriculum_solve",
        "env": "MiniGrid-MultiRoom-N2-S4-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "skill_label": "CURRICULUM NAV SKILL",
        "steps": 72,
        "accent": (54, 143, 93),
    },
    {
        "name": "four_rooms_curriculum_solve",
        "env": "MiniGrid-FourRooms-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "skill_label": "CURRICULUM NAV SKILL",
        "steps": 96,
        "accent": (54, 143, 93),
    },
    {
        "name": "crossing_curriculum_solve",
        "env": "MiniGrid-SimpleCrossingS9N1-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "skill_label": "CURRICULUM NAV SKILL",
        "steps": 96,
        "accent": (54, 143, 93),
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a clean sequential MiniGrid navigation reel.")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--out", default="artifacts/minigrid-nav-sequence-clean.mp4")
    parser.add_argument("--hold-final", type=int, default=8)
    args = parser.parse_args()

    runs = [rollout(segment) for segment in SEQUENCE]
    frames: list[Image.Image] = []
    for run in runs:
        frames.extend(render_run(run, hold_final=args.hold_final))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
    summary = {
        "title": "MiniGrid clean sequential navigation reel",
        "video": str(out),
        "fps": args.fps,
        "segments": [
            {
                "name": run["name"],
                "env": run["env"],
                "skill": run["skill"],
                "status": run["status"],
                "final_step": run["final_step"],
            }
            for run in runs
        ],
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out} ({len(frames)} frames)")
    return 0


def rollout(segment: dict[str, Any]) -> dict[str, Any]:
    max_steps = int(segment["steps"])
    env = make_minigrid(MiniGridSpec(env_id=segment["env"], seed=segment["seed"], max_steps=max_steps))
    _, _, _, object_to_idx, _ = require_minigrid()
    obs, _ = normalize_reset(env.reset(seed=segment["seed"]))
    memory: dict = {"blocked_or_low_reward_steps": 0, "skills_written": []}
    rng = random.Random(segment["seed"])
    program = SkillProgram()
    program.add_skill_file((ROOT / segment["skill"]).resolve(), reason=segment["name"])
    actions = action_map(env)
    idx_to_action = {value: name for name, value in actions.items()}
    frames = [frame_payload(env, 0, "-", obs, 0.0, "running")]
    status = "timeout"
    final_step = max_steps

    for step in range(max_steps):
        encoded = encode_observation(obs)
        api = MiniGridSkillApi(env, memory)
        action = program.act(encoded, memory, api)
        source = "skill"
        if action is None:
            action = rng.randrange(env.action_space.n)
            source = "base-random"
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        update_body_memory(memory, obs, next_obs, env, step, source, action, reward, terminated, truncated)
        status = "solved" if terminated and reward > 0 else "timeout" if truncated else "running"
        frames.append(frame_payload(env, step + 1, idx_to_action.get(int(action), str(int(action))), next_obs, reward, status))
        obs = next_obs
        if terminated or truncated:
            final_step = step + 1
            break

    env.close()
    if status != "solved" and segment.get("hold_after"):
        final_step = min(final_step, int(segment["hold_after"]))
        frames = frames[: final_step + 1]
        status = "stalled"
    return {**segment, "frames": frames, "status": status, "final_step": final_step, "object_to_idx": dict(object_to_idx)}


def frame_payload(env, step: int, action: str, obs, reward: float, status: str) -> dict[str, Any]:
    full_image, agent_pos, agent_dir, visible_mask = observer_frame(env)
    return {
        "step": int(step),
        "action": action,
        "reward": float(reward),
        "status": status,
        "full_image": full_image,
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "visible_mask": visible_mask,
    }


def observer_frame(env):
    unwrapped = getattr(env, "unwrapped", env)
    grid = unwrapped.grid.encode().astype(int)
    full_image = grid.tolist()
    agent_pos = [int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])]
    agent_dir = int(unwrapped.agent_dir)
    visible_mask = [[False for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    try:
        _, view_mask = unwrapped.gen_obs_grid()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                vx, vy = unwrapped.get_view_coords(x, y)
                if 0 <= vx < view_mask.shape[0] and 0 <= vy < view_mask.shape[1]:
                    visible_mask[x][y] = bool(view_mask[vx, vy])
    except Exception:
        visible_mask[agent_pos[0]][agent_pos[1]] = True
    visible_mask[agent_pos[0]][agent_pos[1]] = True
    return full_image, agent_pos, agent_dir, visible_mask


def render_run(run: dict[str, Any], hold_final: int) -> list[Image.Image]:
    frames = [render_frame(run, index) for index in range(len(run["frames"]))]
    if frames:
        frames.extend([frames[-1]] * max(0, hold_final))
    return frames


def render_frame(run: dict[str, Any], frame_index: int) -> Image.Image:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), (244, 241, 232))
    draw = ImageDraw.Draw(img)
    frame = run["frames"][frame_index]
    accent = run["accent"]
    solved = run["status"] == "solved"
    status_color = (54, 143, 93) if solved else (182, 68, 56)
    draw.rectangle((0, 0, width, height), outline=status_color, width=18)
    draw_header(draw, run, status_color, width)

    image = frame["full_image"]
    visible_mask = frame.get("visible_mask")
    object_by_idx = {value: key for key, value in run["object_to_idx"].items()}
    cols = len(image)
    rows = len(image[0])
    board = min(560, width - 120, height - 160)
    cell = max(1, board // max(cols, rows))
    board_w = cols * cell
    board_h = rows * cell
    ox = (width - board_w) // 2
    oy = 124 + (height - 148 - board_h) // 2

    draw.rectangle((ox - 10, oy - 10, ox + board_w + 10, oy + board_h + 10), fill=(17, 24, 32))
    for y in range(rows):
        for x in range(cols):
            object_idx, color_idx, state = image[x][y]
            obj = object_by_idx.get(object_idx, "empty")
            visible = True if visible_mask is None else bool(visible_mask[x][y])
            draw_cell(draw, ox + x * cell, oy + y * cell, cell, obj, color_idx, state, visible)

    ax, ay = frame["agent_pos"]
    draw_cell(draw, ox + ax * cell, oy + ay * cell, cell, "agent", 0, frame.get("agent_dir", 0), True)
    return img


def draw_header(draw: ImageDraw.ImageDraw, run: dict[str, Any], status_color: tuple[int, int, int], width: int) -> None:
    title_font = load_font(28)
    label_font = load_font(20)
    badge_font = load_font(30)
    env_name = clean_env_name(run["env"])
    skill = run.get("skill_label") or Path(run["skill"]).stem.replace("_", " ").upper()
    outcome = "SOLVED" if run["status"] == "solved" else "FAILED"

    draw.rectangle((18, 18, width - 18, 100), fill=(255, 253, 248))
    draw.text((44, 30), env_name, fill=(23, 32, 42), font=title_font)
    draw.text((44, 66), f"skill: {skill}", fill=(88, 101, 114), font=label_font)

    badge_w = 190
    bx = width - badge_w - 44
    draw.rounded_rectangle((bx, 30, bx + badge_w, 82), radius=8, fill=status_color)
    text_w = text_width(badge_font, outcome)
    draw.text((bx + (badge_w - text_w) / 2, 39), outcome, fill=(255, 255, 255), font=badge_font)


def clean_env_name(env: str) -> str:
    name = env.removeprefix("MiniGrid-").removesuffix("-v0")
    return name.replace("S9N1", "S9 N1").replace("N2-S4", "N2 S4")


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
