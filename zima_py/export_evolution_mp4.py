from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

from .export_minigrid_mp4 import COLOR_BY_IDX, PALETTE, blend, draw_cell
from .minigrid_adapter import MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step, require_minigrid
from .skill_runtime import MiniGridSkillApi, SkillProgram, action_map, update_body_memory


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a side-by-side MP4 showing MiniGrid skill evolution.")
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--out", default="artifacts/minigrid-skill-evolution.mp4")
    args = parser.parse_args()

    runs = [
        rollout(args.env, args.seed, args.steps, "Base impulse", "random primitive actions", []),
        rollout(args.env, args.seed, args.steps, "First skill", "front-cell reflex / simple exploration", ["generated_skills/minigrid_general_skill.py"]),
        rollout(
            args.env,
            args.seed,
            args.steps,
            "Reflected skill",
            "persistent learned state + finite planning",
            ["generated_skills/minigrid_reflected_skill.py"],
        ),
    ]
    frames = render_video_frames(args.env, args.seed, runs, args.steps)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)

    summary = {
        "env": args.env,
        "seed": args.seed,
        "steps": args.steps,
        "runs": [
            {
                "label": run["label"],
                "subtitle": run["subtitle"],
                "status": run["status"],
                "final_step": run["final_step"],
                "solved": run["status"] == "solved",
            }
            for run in runs
        ],
        "video": str(out),
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out} ({len(frames)} frames)")
    return 0


def rollout(env_id: str, seed: int, max_steps: int, label: str, subtitle: str, skill_paths: list[str]) -> dict[str, Any]:
    env = make_minigrid(MiniGridSpec(env_id=env_id, seed=seed, max_steps=max_steps))
    _, _, _, object_to_idx, color_to_idx = require_minigrid()
    obs, _ = normalize_reset(env.reset(seed=seed))
    rng = random.Random(seed)
    memory: dict = {"blocked_or_low_reward_steps": 0, "skills_written": []}
    program = SkillProgram()
    for skill_path in skill_paths:
        resolved = (ROOT / skill_path).resolve()
        program.add_skill_file(resolved, reason=f"evolution:{label}")
        memory["skills_written"].append({"name": resolved.stem, "path": str(resolved), "reason": f"evolution:{label}"})

    actions = action_map(env)
    idx_to_action = {value: name for name, value in actions.items()}
    frames = [frame_payload(env, 0, "reset", "-", obs, 0.0, "running")]
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
        status = "solved" if terminated else "timeout" if truncated else "running"
        frames.append(frame_payload(env, step + 1, source, idx_to_action.get(int(action), str(int(action))), next_obs, reward, status))
        obs = next_obs
        if terminated or truncated:
            final_step = step + 1
            break

    env.close()
    return {
        "label": label,
        "subtitle": subtitle,
        "status": status,
        "final_step": final_step,
        "frames": frames,
        "actions": actions,
        "object_to_idx": dict(object_to_idx),
        "color_to_idx": dict(color_to_idx),
    }


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


def render_video_frames(env_id: str, seed: int, runs: list[dict[str, Any]], max_steps: int) -> list[Image.Image]:
    width, height = 1920, 1088
    fps_hold = 18
    frames = []
    font = load_font(24)
    small = load_font(18)
    title_font = load_font(40)
    panel_w = 590
    board = 510
    gap = 22
    x0 = 46
    y0 = 138
    max_len = max(len(run["frames"]) for run in runs)

    for index in range(max_len):
        img = Image.new("RGB", (width, height), (244, 241, 232))
        draw = ImageDraw.Draw(img)
        draw.text((46, 34), "Zima Blue MiniGrid: Skill Evolution", fill=(23, 32, 42), font=title_font)
        draw.text(
            (48, 84),
            f"{env_id} | seed {seed} | same world, three bodies with increasingly capable behavior",
            fill=(88, 101, 114),
            font=font,
        )
        for run_i, run in enumerate(runs):
            frame = run["frames"][min(index, len(run["frames"]) - 1)]
            x = x0 + run_i * (panel_w + gap)
            render_panel(draw, x, y0, panel_w, board, run, frame, index, max_steps, font, small)
        frames.append(img)
    if frames:
        frames.extend([frames[-1]] * fps_hold)
    return frames


def render_panel(draw, x: int, y: int, width: int, board: int, run: dict[str, Any], frame: dict[str, Any], index: int, max_steps: int, font, small):
    draw.rounded_rectangle((x, y, x + width, y + 852), radius=8, fill=(255, 253, 248), outline=(217, 210, 195))
    draw.text((x + 20, y + 18), run["label"], fill=(23, 32, 42), font=font)
    draw.text((x + 20, y + 50), run["subtitle"], fill=(88, 101, 114), font=small)
    status_color = (72, 168, 104) if run["status"] == "solved" else (154, 91, 66)
    draw.text((x + width - 150, y + 22), run["status"].upper(), fill=status_color, font=small)

    image = frame["full_image"]
    visible_mask = frame.get("visible_mask")
    object_by_idx = {value: key for key, value in run["object_to_idx"].items()}
    cols = len(image)
    rows = len(image[0])
    cell = board // max(cols, rows)
    ox = x + (width - cols * cell) // 2
    oy = y + 92
    draw.rectangle((ox, oy, ox + cols * cell, oy + rows * cell), fill=(17, 24, 32))
    for yy in range(rows):
        for xx in range(cols):
            object_idx, color_idx, state = image[xx][yy]
            obj = object_by_idx.get(object_idx, "empty")
            visible = True if visible_mask is None else bool(visible_mask[xx][yy])
            draw_cell(draw, ox + xx * cell, oy + yy * cell, cell, obj, color_idx, state, visible)
    ax, ay = frame["agent_pos"]
    draw_cell(draw, ox + ax * cell, oy + ay * cell, cell, "agent", 0, frame.get("agent_dir", 0), True)

    info_y = oy + rows * cell + 22
    metrics = [
        ("step", f"{frame['step']}/{max_steps}"),
        ("source", frame["source"]),
        ("action", frame["action"]),
        ("reward", f"{frame['reward']:.2f}"),
    ]
    for i, (label, value) in enumerate(metrics):
        mx = x + 20 + (i % 2) * 270
        my = info_y + (i // 2) * 62
        draw.rounded_rectangle((mx, my, mx + 250, my + 48), radius=6, fill=(255, 255, 255), outline=(217, 210, 195))
        draw.text((mx + 10, my + 6), label.upper(), fill=(100, 114, 127), font=small)
        draw.text((mx + 10, my + 25), str(value), fill=(23, 32, 42), font=small)

    caption_y = info_y + 144
    caption = evolution_caption(run["label"], frame, run)
    for line in wrap(caption, 48):
        draw.text((x + 20, caption_y), line, fill=(63, 75, 86), font=small)
        caption_y += 24


def evolution_caption(label: str, frame: dict[str, Any], run: dict[str, Any]) -> str:
    if label == "Base impulse":
        return "No skill has been written yet. The body only samples primitive actions, so success is accidental."
    if label == "First skill":
        return "The first behavior reacts to nearby cells. It can sometimes stumble through the task, but has no durable world model."
    if frame["status"] == "solved":
        return "The reflected skill has converted experience into a persistent planner and reached the terminal goal."
    return "The reflected skill stores learned state, remembered objects, action outcomes, and plans finite routes from its partial view."


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    current = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > width and current:
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
