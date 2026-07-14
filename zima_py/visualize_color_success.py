from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import neat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evolve_jepa_neat import neat_config_for_inputs
from .evolve_rgb_neat import apply_rgb_style, controller_features
from .lexicase_reproduction import LexicaseReproduction
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = {0: "turn left", 1: "turn right", 2: "forward"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render paired successful normal and color-shifted JEPA+NEAT replays.")
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--evolution-seed", type=int, default=39)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--winner", default="artifacts/jepa/temporal-jepa-seed39-winner.pkl")
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-fresh.pt")
    parser.add_argument("--output", default="artifacts/jepa/temporal-jepa-color-success-seed52.mp4")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    agent = TemporalRgbJepaAgent(7, context_length=4, seed=args.evolution_seed)
    agent.load(ROOT / args.checkpoint)
    agent.eval()
    config = neat.Config(
        neat.DefaultGenome,
        LexicaseReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config_for_inputs(agent.feature_size + 8, lexicase=True)),
    )
    winner = pickle.loads((ROOT / args.winner).read_bytes())
    standard = replay(args, agent, winner, config, "standard")
    cyclic = replay(args, agent, winner, config, "cyclic")
    if not standard[-1]["solved"] or not cyclic[-1]["solved"]:
        raise RuntimeError("The selected policy and seed must solve both appearance conditions.")

    frames = render_video_frames(standard, cyclic, args.env, args.seed)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=args.fps, macro_block_size=16)
    print(str(output))
    return 0


def replay(args, agent, winner, config, style: str) -> list[dict]:
    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    obs, _ = normalize_reset(env.reset(seed=args.seed))
    first = apply_rgb_style(obs["image"], style)
    history_frames = deque([first.copy() for _ in range(agent.context_length)], maxlen=agent.context_length)
    history_actions = deque([-1 for _ in range(agent.context_length - 1)], maxlen=agent.context_length - 1)
    network = neat.nn.RecurrentNetwork.create(winner, config)
    cache = {}
    previous_action = -1
    rows = []
    for step in range(args.max_steps):
        image = apply_rgb_style(obs["image"], style)
        features = controller_features(
            image,
            previous_action,
            env.action_space.n,
            "temporal-jepa",
            agent,
            cache,
            np.stack(history_frames),
            list(history_actions),
        )
        outputs = np.asarray(network.activate(features.tolist()), dtype=np.float64)
        outputs[3:] = -np.inf
        action = int(outputs.argmax())
        rows.append({"image": image.copy(), "step": step, "action": action, "solved": False})
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        next_image = apply_rgb_style(next_obs["image"], style)
        history_frames.append(next_image.copy())
        history_actions.append(action)
        previous_action = action
        obs = next_obs
        if terminated or truncated:
            solved = bool(terminated and float(reward) > 0.0)
            rows.append(
                {"image": next_image.copy(), "step": step + 1, "action": None, "solved": solved}
            )
            break
    env.close()
    return rows


def render_video_frames(standard: list[dict], cyclic: list[dict], env_id: str, seed: int):
    font = ImageFont.load_default()
    total = max(len(standard), len(cyclic))
    frames = []
    for index in range(total):
        left = standard[min(index, len(standard) - 1)]
        right = cyclic[min(index, len(cyclic) - 1)]
        canvas = Image.new("RGB", (960, 608), (242, 244, 248))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), "One frozen temporal JEPA + NEAT policy, two RGB palettes", fill=(18, 24, 34), font=font)
        draw.text((24, 38), f"{env_id}  held-out seed={seed}", fill=(77, 84, 96), font=font)
        draw_panel(canvas, draw, left, (24, 76), "Standard RGB", "training appearance", font)
        draw_panel(canvas, draw, right, (496, 76), "Cyclic RGB → GBR", "unseen appearance", font)
        if left["solved"] and right["solved"]:
            draw.rounded_rectangle((275, 566, 685, 598), radius=10, fill=(32, 166, 95))
            draw.text((392, 576), "BOTH REPLAYS REACHED THE GOAL", fill=(255, 255, 255), font=font)
        else:
            draw.text((350, 576), "Each panel runs the policy independently", fill=(77, 84, 96), font=font)
        frames.append(np.asarray(canvas, dtype=np.uint8))
    # Hold on the successful final frame.
    frames.extend([frames[-1].copy() for _ in range(10)])
    return frames


def draw_panel(canvas, draw, row: dict, origin, title: str, subtitle: str, font) -> None:
    x, y = origin
    draw.rounded_rectangle((x, y, x + 440, y + 470), radius=12, fill=(255, 255, 255), outline=(213, 218, 226))
    draw.text((x + 16, y + 16), title, fill=(20, 27, 38), font=font)
    draw.text((x + 16, y + 35), subtitle, fill=(86, 93, 105), font=font)
    image = Image.fromarray(row["image"], mode="RGB").resize((392, 392), Image.Resampling.NEAREST)
    grid = ImageDraw.Draw(image)
    for value in range(0, 393, 56):
        grid.line((value, 0, value, 392), fill=(255, 255, 255), width=2)
        grid.line((0, value, 392, value), fill=(255, 255, 255), width=2)
    canvas.paste(image, (x + 24, y + 60))
    action = "GOAL" if row["solved"] else ACTION_NAMES.get(row["action"], "episode complete")
    color = (25, 151, 85) if row["solved"] else (51, 59, 72)
    draw.text((x + 24, y + 456), f"step {row['step']:02d}   action: {action}", fill=color, font=font)


if __name__ == "__main__":
    raise SystemExit(main())
