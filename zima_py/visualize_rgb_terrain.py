from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path

import neat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from minigrid.wrappers import RGBImgPartialObsWrapper

from .embodied_map import EmbodiedMap
from .evolve_jepa_neat import neat_config_for_inputs, raw_body_features
from .jepa_control_agent import JepaControlAgent, belief_observation
from .minigrid_adapter import MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step
from .skill_runtime import MiniGridSkillApi
from .train_jepa_minigrid import update_carrying, useful_actions


ROOT = Path(__file__).resolve().parents[1]
TERRAIN_COLORS = {
    "unknown": (30, 34, 42),
    "empty": (224, 220, 203),
    "floor": (224, 220, 203),
    "wall": (91, 99, 110),
    "lava": (239, 74, 48),
    "goal": (50, 190, 105),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize partial RGB input beside observation-derived terrain memory.")
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--winner", default="artifacts/jepa/ablation-s9n1-map-winner.pkl")
    parser.add_argument("--output", default="artifacts/jepa/rgb-terrain-map-seed41.png")
    args = parser.parse_args()

    symbolic_env = make_minigrid(MiniGridSpec(env_id=args.env, seed=args.seed, max_steps=args.max_steps))
    rgb_env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(env_id=args.env, seed=args.seed, max_steps=args.max_steps)),
        tile_size=8,
    )
    symbolic_obs, _ = normalize_reset(symbolic_env.reset(seed=args.seed))
    rgb_obs, _ = normalize_reset(rgb_env.reset(seed=args.seed))

    winner = pickle.loads((ROOT / args.winner).read_bytes())
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config_for_inputs(282)),
    )
    network = neat.nn.RecurrentNetwork.create(winner, config)
    agent = JepaControlAgent(symbolic_env.action_space.n, seed=19)
    body_map = EmbodiedMap()
    previous_action = -1
    carrying = False
    snapshots = []
    solved = False

    for step in range(args.max_steps):
        encoded = encode_observation(symbolic_obs)
        api = MiniGridSkillApi(symbolic_env, {})
        front_before = api.front_object(encoded)
        body_map.observe(api.egocentric_cells(encoded), int(encoded.get("direction", body_map.direction)))
        snapshots.append(
            {
                "step": step,
                "rgb": np.asarray(rgb_obs["image"], dtype=np.uint8).copy(),
                "map": copy.deepcopy(body_map),
                "solved": False,
            }
        )
        state = belief_observation(encoded, previous_action, carrying)
        controller_features = np.concatenate(
            (raw_body_features(state, agent.action_count), body_map.features(carrying, False, include_terrain=True))
        )
        outputs = np.asarray(network.activate(controller_features.tolist()), dtype=np.float64)
        valid = useful_actions(api.actions, front_before)
        invalid = np.ones(agent.action_count, dtype=bool)
        invalid[valid] = False
        outputs[invalid] = -np.inf
        action = int(outputs.argmax())

        next_symbolic, reward, terminated, truncated, _ = normalize_step(symbolic_env.step(action))
        next_rgb, rgb_reward, rgb_terminated, rgb_truncated, _ = normalize_step(rgb_env.step(action))
        if (terminated, truncated) != (rgb_terminated, rgb_truncated) or float(reward) != float(rgb_reward):
            raise RuntimeError("Symbolic and RGB replay environments diverged.")
        next_encoded = encode_observation(next_symbolic)
        front_after = api.front_object(next_encoded)
        next_carrying = update_carrying(carrying, action, api.actions, front_before, front_after)
        blocked = bool(
            action == api.actions["forward"]
            and np.array_equal(encoded["image"], next_encoded["image"])
            and encoded.get("direction") == next_encoded.get("direction")
        )
        action_name = next((name for name, value in api.actions.items() if value == action), str(action))
        body_map.advance(action_name, blocked, int(next_encoded.get("direction", body_map.direction)))
        previous_action = action
        carrying = next_carrying
        symbolic_obs = next_symbolic
        rgb_obs = next_rgb
        if terminated or truncated:
            final_api = MiniGridSkillApi(symbolic_env, {})
            body_map.observe(
                final_api.egocentric_cells(next_encoded),
                int(next_encoded.get("direction", body_map.direction)),
            )
            solved = bool(terminated and float(reward) > 0.0)
            snapshots.append(
                {
                    "step": step + 1,
                    "rgb": np.asarray(rgb_obs["image"], dtype=np.uint8).copy(),
                    "map": copy.deepcopy(body_map),
                    "solved": solved,
                }
            )
            break

    selected = select_snapshots(snapshots, count=4)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    render_timeline(selected, output, args.env, args.seed, solved)
    symbolic_env.close()
    rgb_env.close()
    print(str(output))
    return 0


def select_snapshots(snapshots: list[dict], count: int) -> list[dict]:
    if len(snapshots) <= count:
        return snapshots
    indices = np.linspace(0, len(snapshots) - 1, count).round().astype(int)
    return [snapshots[int(index)] for index in indices]


def render_timeline(snapshots: list[dict], output: Path, env_id: str, seed: int, solved: bool) -> None:
    font = ImageFont.load_default()
    bold = font
    width = 940
    header = 64
    row_height = 250
    canvas = Image.new("RGB", (width, header + row_height * len(snapshots)), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "Partial RGB observation -> persistent observation-only terrain memory", fill=(20, 25, 34), font=bold)
    draw.text(
        (20, 34),
        f"{env_id}  seed={seed}  final={'solved' if solved else 'not solved'}",
        fill=(80, 87, 98),
        font=font,
    )

    final_map = snapshots[-1]["map"]
    xs = [position[0] for position in final_map.terrain] + [0]
    ys = [position[1] for position in final_map.terrain] + [0]
    bounds = (min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1)
    for row, snapshot in enumerate(snapshots):
        top = header + row * row_height
        draw.rounded_rectangle((12, top + 4, width - 12, top + row_height - 6), radius=10, fill=(255, 255, 255), outline=(220, 224, 230))
        label = f"step {snapshot['step']}" + ("  GOAL" if snapshot["solved"] else "")
        draw.text((28, top + 18), label, fill=(25, 31, 42), font=bold)
        draw.text((28, top + 38), "Current 56x56 RGB view (7x7 patches)", fill=(84, 91, 102), font=font)
        rgb = Image.fromarray(snapshot["rgb"], mode="RGB").resize((168, 168), Image.Resampling.NEAREST)
        rgb_draw = ImageDraw.Draw(rgb)
        for value in range(0, 169, 24):
            rgb_draw.line((value, 0, value, 168), fill=(255, 255, 255), width=1)
            rgb_draw.line((0, value, 168, value), fill=(255, 255, 255), width=1)
        canvas.paste(rgb, (28, top + 62))
        draw.text((226, top + 38), "Accumulated map in start-relative coordinates", fill=(84, 91, 102), font=font)
        render_map(draw, snapshot["map"], bounds, origin=(226, top + 62), area=(480, 140), font=font)
        render_legend(draw, origin=(730, top + 65), terrain_map=snapshot["map"], font=font)

    canvas.save(output)


def render_map(draw: ImageDraw.ImageDraw, terrain_map: EmbodiedMap, bounds, origin, area, font) -> None:
    min_x, max_x, min_y, max_y = bounds
    columns = max_x - min_x + 1
    rows = max_y - min_y + 1
    cell = max(8, min(area[0] // columns, area[1] // rows))
    map_width = columns * cell
    map_height = rows * cell
    ox = origin[0] + (area[0] - map_width) // 2
    oy = origin[1] + (area[1] - map_height) // 2
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            name = terrain_map.terrain.get((x, y), "unknown")
            color = TERRAIN_COLORS.get(name, TERRAIN_COLORS["empty"])
            left = ox + (x - min_x) * cell
            top = oy + (y - min_y) * cell
            draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=color, outline=(205, 209, 215))
            visits = terrain_map.visits.get((x, y), 0)
            if visits > 1:
                draw.ellipse((left + cell - 6, top + 2, left + cell - 2, top + 6), fill=(38, 111, 230))
    ax = ox + (terrain_map.x - min_x) * cell
    ay = oy + (terrain_map.y - min_y) * cell
    draw.ellipse((ax + 2, ay + 2, ax + cell - 3, ay + cell - 3), fill=(40, 105, 230), outline=(255, 255, 255))
    dx, dy = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}[terrain_map.direction]
    cx, cy = ax + cell // 2, ay + cell // 2
    draw.line((cx, cy, cx + dx * cell // 2, cy + dy * cell // 2), fill=(255, 255, 255), width=2)


def render_legend(draw: ImageDraw.ImageDraw, origin, terrain_map: EmbodiedMap, font) -> None:
    x, y = origin
    draw.text((x, y), "Map memory", fill=(28, 34, 44), font=font)
    for index, name in enumerate(("unknown", "empty", "wall", "lava", "goal")):
        top = y + 22 + index * 20
        draw.rectangle((x, top, x + 13, top + 13), fill=TERRAIN_COLORS[name], outline=(190, 195, 202))
        draw.text((x + 20, top), "free" if name == "empty" else name, fill=(69, 76, 88), font=font)
    draw.text((x, y + 130), f"known cells: {len(terrain_map.terrain)}", fill=(69, 76, 88), font=font)
    draw.text((x, y + 148), f"agent: ({terrain_map.x}, {terrain_map.y})", fill=(69, 76, 88), font=font)


if __name__ == "__main__":
    raise SystemExit(main())
