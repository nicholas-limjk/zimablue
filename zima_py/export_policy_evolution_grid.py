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


PANELS = [
    {
        "policy": "REFLECTED",
        "env": "MiniGrid-DoorKey-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_reflected_skill.py",
        "headline": "task-family planner",
        "caption": "After first reflection, the policy has learned key-door-goal structure and solves DoorKey.",
        "accent": (49, 112, 178),
    },
    {
        "policy": "REFLECTED",
        "env": "MiniGrid-Empty-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_reflected_skill.py",
        "headline": "does not generalize",
        "caption": "The same reflected skill is locked onto interaction logic and fails on simple navigation.",
        "accent": (160, 73, 58),
    },
    {
        "policy": "CURRICULUM",
        "env": "MiniGrid-Empty-8x8-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "headline": "navigation recovered",
        "caption": "Curriculum reflection adds terminal-cue and frontier navigation, fixing Empty.",
        "accent": (55, 142, 91),
    },
    {
        "policy": "CURRICULUM",
        "env": "MiniGrid-MultiRoom-N2-S4-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "headline": "broader transfer",
        "caption": "The curriculum skill keeps persistent state and handles a new room-navigation family.",
        "accent": (55, 142, 91),
    },
]

HARD_NAV_PANELS = [
    {
        "policy": "REFLECTED",
        "env": "MiniGrid-MultiRoom-N2-S4-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_reflected_skill.py",
        "headline": "task-family lock-in",
        "caption": "The reflected DoorKey-style planner stalls when the problem is room navigation instead of object interaction.",
        "accent": (160, 73, 58),
    },
    {
        "policy": "CURRICULUM",
        "env": "MiniGrid-MultiRoom-N2-S4-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "headline": "multi-room transfer",
        "caption": "The curriculum policy treats terminal cues and frontiers as first-class targets and solves the room sequence.",
        "accent": (55, 142, 91),
    },
    {
        "policy": "CURRICULUM",
        "env": "MiniGrid-FourRooms-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "headline": "four-room navigation",
        "caption": "The same generalized policy maintains exploration and reaches the goal through connected rooms.",
        "accent": (55, 142, 91),
    },
    {
        "policy": "CURRICULUM",
        "env": "MiniGrid-SimpleCrossingS9N1-v0",
        "seed": 0,
        "skill": "generated_skills/minigrid_curriculum_skill.py",
        "headline": "crossing obstacle layout",
        "caption": "The curriculum skill handles a crossing-style obstacle map without needing key-door assumptions.",
        "accent": (55, 142, 91),
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a 2x2 policy-evolution grid MP4.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--out", default="artifacts/minigrid-policy-evolution-grid.mp4")
    parser.add_argument("--scenario", choices=["evolution", "hard-nav"], default="evolution")
    args = parser.parse_args()

    panels = HARD_NAV_PANELS if args.scenario == "hard-nav" else PANELS
    title = "MiniGrid Navigation Generalization" if args.scenario == "hard-nav" else "MiniGrid Policy Evolution"
    subtitle = (
        "More interesting navigation problems: MultiRoom, FourRooms, and SimpleCrossing."
        if args.scenario == "hard-nav"
        else "REFLECTED learns one task family; CURRICULUM reflection repairs the policy into a broader navigator."
    )
    runs = [rollout(panel, args.steps) for panel in panels]
    frames = render_frames(runs, args.steps, title, subtitle)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)

    summary = {
        "title": title,
        "steps": args.steps,
        "video": str(out),
        "panels": [
            {
                "policy": run["policy"],
                "env": run["env"],
                "seed": run["seed"],
                "status": run["status"],
                "final_step": run["final_step"],
                "headline": run["headline"],
            }
            for run in runs
        ],
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out} ({len(frames)} frames)")
    return 0


def rollout(panel: dict[str, Any], max_steps: int) -> dict[str, Any]:
    env = make_minigrid(MiniGridSpec(env_id=panel["env"], seed=panel["seed"], max_steps=max_steps))
    _, _, _, object_to_idx, _ = require_minigrid()
    obs, _ = normalize_reset(env.reset(seed=panel["seed"]))
    memory: dict = {"blocked_or_low_reward_steps": 0, "skills_written": []}
    rng = random.Random(panel["seed"])
    program = SkillProgram()
    program.add_skill_file((ROOT / panel["skill"]).resolve(), reason=f"{panel['policy']}:{panel['env']}")
    actions = action_map(env)
    idx_to_action = {value: name for name, value in actions.items()}
    frames = [frame_payload(env, 0, "reset", "-", obs, 0.0, "running")]
    status = "timeout"
    final_step = max_steps

    panel_steps = int(panel.get("steps", max_steps))
    for step in range(panel_steps):
        encoded = encode_observation(obs)
        api = MiniGridSkillApi(env, memory)
        action = program.act(encoded, memory, api)
        source = "skill"
        if action is None:
            action = rng.randrange(env.action_space.n)
            source = "base-random"
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        update_body_memory(memory, obs, next_obs, env, step, source, action, reward, terminated, truncated)
        status = "solved" if terminated else "timeout" if truncated else "running"
        frames.append(frame_payload(env, step + 1, source, idx_to_action.get(int(action), str(int(action))), next_obs, reward, status))
        obs = next_obs
        if terminated or truncated:
            final_step = step + 1
            break

    env.close()
    if status != "solved" and panel.get("hold_after"):
        final_step = min(final_step, int(panel["hold_after"]))
        frames = frames[: final_step + 1]
        status = "stalled"
    return {**panel, "frames": frames, "status": status, "final_step": final_step, "object_to_idx": dict(object_to_idx)}


def frame_payload(env, step: int, source: str, action: str, obs, reward: float, status: str) -> dict[str, Any]:
    encoded = encode_observation(obs)
    full_image, agent_pos, agent_dir, visible_mask = observer_frame(env)
    return {
        "step": int(step),
        "source": source,
        "action": action,
        "reward": float(reward),
        "status": status,
        "mission": encoded["mission"],
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


def render_frames(runs: list[dict[str, Any]], max_steps: int, title: str, subtitle: str) -> list[Image.Image]:
    width, height = 1920, 1088
    max_len = max(len(run["frames"]) for run in runs)
    frames = []
    title_font = load_font(38)
    header_font = load_font(28)
    font = load_font(20)
    small = load_font(16)
    for index in range(max_len):
        img = Image.new("RGB", (width, height), (244, 241, 232))
        draw = ImageDraw.Draw(img)
        draw.text((42, 28), title, fill=(23, 32, 42), font=title_font)
        draw.text(
            (44, 76),
            subtitle,
            fill=(88, 101, 114),
            font=font,
        )
        for panel_index, run in enumerate(runs):
            col = panel_index % 2
            row = panel_index // 2
            render_panel(draw, run, min(index, len(run["frames"]) - 1), 44 + col * 938, 126 + row * 468, 894, 420, max_steps, header_font, font, small)
        frames.append(img)
    if frames:
        frames.extend([frames[-1]] * 18)
    return frames


def render_panel(draw, run, frame_index: int, x: int, y: int, w: int, h: int, max_steps: int, header_font, font, small) -> None:
    frame = run["frames"][frame_index]
    accent = run["accent"]
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(255, 253, 248), outline=(217, 210, 195))
    draw.rectangle((x, y, x + w, y + 58), fill=accent)
    draw.text((x + 18, y + 12), run["policy"], fill=(255, 255, 255), font=header_font)
    draw.text((x + 205, y + 20), run["headline"], fill=(244, 247, 249), font=font)
    status = run["status"].upper()
    draw.text((x + w - 124, y + 20), status, fill=(255, 255, 255), font=small)

    image = frame["full_image"]
    visible_mask = frame.get("visible_mask")
    object_by_idx = {value: key for key, value in run["object_to_idx"].items()}
    cols = len(image)
    rows = len(image[0])
    board = 300
    cell = board // max(cols, rows)
    ox = x + 18 + (board - cols * cell) // 2
    oy = y + 78 + (board - rows * cell) // 2
    draw.rectangle((ox, oy, ox + cols * cell, oy + rows * cell), fill=(17, 24, 32))
    for yy in range(rows):
        for xx in range(cols):
            object_idx, color_idx, state = image[xx][yy]
            obj = object_by_idx.get(object_idx, "empty")
            visible = True if visible_mask is None else bool(visible_mask[xx][yy])
            draw_cell(draw, ox + xx * cell, oy + yy * cell, cell, obj, color_idx, state, visible)
    ax, ay = frame["agent_pos"]
    draw_cell(draw, ox + ax * cell, oy + ay * cell, cell, "agent", 0, frame.get("agent_dir", 0), True)

    tx = x + 344
    ty = y + 82
    draw.text((tx, ty), run["env"], fill=(23, 32, 42), font=font)
    draw.text((tx, ty + 30), f"seed {run['seed']} | step {frame['step']}/{max_steps}", fill=(88, 101, 114), font=small)
    draw.text((tx, ty + 58), f"action: {frame['action']} | reward: {frame['reward']:.2f}", fill=(88, 101, 114), font=small)
    ty += 100
    for line in wrap(run["caption"], 54):
        draw.text((tx, ty), line, fill=(45, 57, 68), font=small)
        ty += 22
    ty += 12
    for line in wrap(evolution_note(run), 54):
        draw.text((tx, ty), line, fill=(88, 101, 114), font=small)
        ty += 22


def evolution_note(run: dict[str, Any]) -> str:
    if run["policy"] == "REFLECTED":
        return "Before curriculum: strong object-interaction assumptions, weaker when the world only asks for navigation."
    return "After curriculum: terminal cues and frontiers become first-class targets, while object interactions are preserved."


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    current = []
    for word in words:
        if current and sum(len(item) for item in current) + len(current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def load_font(size: int):
    for path in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
