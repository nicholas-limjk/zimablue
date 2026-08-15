from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import neat
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evolve_jepa_neat import neat_config_for_inputs
from .evolve_rgb_neat import controller_features
from .lexicase_reproduction import LexicaseReproduction
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent
from .visualize_color_success import full_grid_frame


ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = ("LEFT", "RIGHT", "FORWARD")
ACTION_COLORS = ((90, 119, 240), (170, 95, 225), (28, 164, 110))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize a repaired action-conditioned JEPA and its evolved NEAT controller."
    )
    parser.add_argument("--env", default="MiniGrid-Dynamic-Obstacles-6x6-v0")
    parser.add_argument("--layout-seed", type=int, default=41)
    parser.add_argument("--evolution-seed", type=int, default=19)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument(
        "--checkpoint", default="artifacts/jepa/temporal-rgb-dynamic6-branched.pt"
    )
    parser.add_argument(
        "--winner", default="artifacts/jepa/dynamic6-racing-branched-jepa-cf-neat-seed19.pkl"
    )
    parser.add_argument(
        "--video", default="artifacts/jepa/repaired-jepa-neat-seed19-holdout41.mp4"
    )
    parser.add_argument(
        "--preview", default="artifacts/jepa/repaired-jepa-neat-seed19-preview.png"
    )
    parser.add_argument("--diagram", default="artifacts/jepa/repaired-jepa-system.png")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.layout_seed, args.max_steps)), tile_size=8
    )
    agent = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.evolution_seed)
    agent.load(ROOT / args.checkpoint)
    agent.eval()
    horizons = (1, 2, 4)
    input_count = (
        agent.feature_size
        + agent.counterfactual_feature_size(horizons)
        + env.action_space.n
        + 1
    )
    config = neat.Config(
        neat.DefaultGenome,
        LexicaseReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config_for_inputs(input_count, lexicase=True, feed_forward=True)),
    )
    genome = pickle.loads((ROOT / args.winner).read_bytes())
    network = neat.nn.FeedForwardNetwork.create(genome, config)

    rows = replay(env, agent, network, args.layout_seed, args.max_steps, horizons)
    if not rows[-1]["solved"]:
        raise RuntimeError(f"winner did not solve held-out layout seed {args.layout_seed}")

    video_path = ROOT / args.video
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_replay_frame(row, args.env, args.evolution_seed, args.layout_seed) for row in rows]
    frames.extend([frames[-1].copy() for _ in range(args.fps * 2)])
    imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=16)
    preview_path = ROOT / args.preview
    Image.fromarray(frames[min(15, len(rows) - 2)], mode="RGB").save(preview_path)

    diagram_path = ROOT / args.diagram
    diagram_path.parent.mkdir(parents=True, exist_ok=True)
    render_system_diagram().save(diagram_path)
    env.close()
    print(f"video={video_path}")
    print(f"preview={preview_path}")
    print(f"diagram={diagram_path}")
    return 0


def replay(env, agent, network, seed: int, max_steps: int, horizons: tuple[int, ...]) -> list[dict]:
    obs, _ = normalize_reset(env.reset(seed=seed))
    first = obs["image"].copy()
    history_frames = deque([first.copy() for _ in range(4)], maxlen=4)
    history_actions = deque([-1, -1, -1], maxlen=3)
    previous_action = -1
    cache: dict[bytes, np.ndarray] = {}
    rows = []
    last_diagnostics = None
    for step in range(max_steps):
        frames = np.stack(history_frames)
        actions = list(history_actions)
        features = controller_features(
            obs["image"],
            previous_action,
            env.action_space.n,
            "temporal-jepa",
            agent,
            cache,
            frames,
            actions,
            horizons,
        )
        outputs = np.asarray(network.activate(features.tolist()), dtype=np.float64)
        outputs[3:] = -np.inf
        selected = int(outputs.argmax())
        diagnostics = counterfactual_diagnostics(agent, frames, actions, horizons)
        last_diagnostics = diagnostics
        rows.append(
            {
                "full_image": full_grid_frame(env, "standard", tile_size=64),
                "step": step,
                "selected": selected,
                "outputs": outputs[:3].copy(),
                "diagnostics": diagnostics,
                "solved": False,
            }
        )
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(selected))
        history_frames.append(next_obs["image"].copy())
        history_actions.append(selected)
        previous_action = selected
        obs = next_obs
        if terminated or truncated:
            solved = bool(terminated and float(reward) > 0.0)
            rows.append(
                {
                    "full_image": full_grid_frame(env, "standard", tile_size=64),
                    "step": step + 1,
                    "selected": None,
                    "outputs": rows[-1]["outputs"],
                    "diagnostics": last_diagnostics,
                    "solved": solved,
                }
            )
            break
    return rows


@torch.no_grad()
def counterfactual_diagnostics(agent, frames, previous_actions, horizons):
    latent = agent.encode_context(frames, previous_actions)
    result = []
    for action in range(3):
        predicted = latent
        action_tensor = torch.as_tensor([action], dtype=torch.long, device=agent.device)
        magnitudes = {}
        heatmap = None
        for horizon in range(1, max(horizons) + 1):
            predicted = agent.predictor(predicted, action_tensor)
            if horizon in horizons:
                delta = predicted - latent
                spatial = delta.square().mean(dim=1).sqrt()[0].cpu().numpy()
                magnitudes[horizon] = float(spatial.mean())
                if horizon == max(horizons):
                    heatmap = spatial
        result.append({"magnitudes": magnitudes, "heatmap": heatmap})
    return result


def render_replay_frame(row: dict, env_id: str, evolution_seed: int, layout_seed: int) -> np.ndarray:
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (1120, 704), (239, 243, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), "REPAIRED JEPA + EVOLVED NEAT", fill=(18, 27, 42), font=font)
    draw.text(
        (28, 42),
        f"{env_id} | evolution seed {evolution_seed} | held-out layout {layout_seed}",
        fill=(83, 94, 111),
        font=font,
    )

    draw.rounded_rectangle((24, 76, 680, 674), radius=14, fill="white", outline=(207, 216, 228))
    draw.text((44, 94), "WHOLE GRID", fill=(31, 42, 59), font=font)
    draw.rectangle((44, 116, 56, 128), fill=(68, 145, 235))
    draw.text((63, 116), "blue shading = cells inside the policy's egocentric view", fill=(83, 94, 111), font=font)
    grid = Image.fromarray(row["full_image"], mode="RGB").resize((512, 512), Image.Resampling.NEAREST)
    canvas.paste(grid, (96, 146))

    draw.rounded_rectangle((704, 76, 1096, 674), radius=14, fill="white", outline=(207, 216, 228))
    draw.text((728, 96), "FROZEN JEPA COUNTERFACTUALS", fill=(31, 42, 59), font=font)
    draw.text((728, 116), "Repeated-action latent change at horizon 4", fill=(83, 94, 111), font=font)
    heatmaps = [item["heatmap"] for item in row["diagnostics"]]
    maximum = max(float(np.max(heatmap)) for heatmap in heatmaps) + 1e-9
    for action, heatmap in enumerate(heatmaps):
        x = 728 + action * 116
        draw.text((x, 145), ACTION_NAMES[action], fill=ACTION_COLORS[action], font=font)
        draw_heatmap(draw, heatmap / maximum, x, 166, 22)
        mags = row["diagnostics"][action]["magnitudes"]
        draw.text((x, 262), f"h1 {mags[1]:.3f}", fill=(68, 78, 94), font=font)
        draw.text((x, 278), f"h2 {mags[2]:.3f}", fill=(68, 78, 94), font=font)
        draw.text((x, 294), f"h4 {mags[4]:.3f}", fill=(68, 78, 94), font=font)

    draw.line((728, 326, 1072, 326), fill=(220, 226, 234), width=1)
    draw.text((728, 348), "NEAT ACTION SCORES", fill=(31, 42, 59), font=font)
    outputs = np.asarray(row["outputs"], dtype=np.float64)
    finite = outputs[np.isfinite(outputs)]
    low, high = float(finite.min()), float(finite.max())
    scale = max(1e-9, high - low)
    for action, value in enumerate(outputs):
        y = 384 + action * 58
        selected = row["selected"] == action
        if selected:
            draw.rounded_rectangle((720, y - 9, 1080, y + 39), radius=8, fill=(235, 250, 243))
        draw.text((736, y), ACTION_NAMES[action], fill=ACTION_COLORS[action], font=font)
        width = 175 * (float(value) - low) / scale if scale else 0
        draw.rounded_rectangle((824, y, 824 + max(3, width), y + 14), radius=5, fill=ACTION_COLORS[action])
        draw.text((1012, y), f"{value:+.3f}", fill=(58, 68, 84), font=font)
        if selected:
            draw.text((736, y + 20), "SELECTED", fill=(23, 139, 82), font=font)

    status = "GOAL REACHED" if row["solved"] else f"STEP {row['step']:02d}"
    color = (23, 139, 82) if row["solved"] else (50, 62, 80)
    draw.rounded_rectangle((728, 594, 1072, 642), radius=10, fill=(232, 247, 239) if row["solved"] else (240, 244, 249))
    draw.text((850, 612), status, fill=color, font=font)
    return np.asarray(canvas, dtype=np.uint8)


def draw_heatmap(draw: ImageDraw.ImageDraw, values: np.ndarray, x: int, y: int, cell: int) -> None:
    for row in range(4):
        for col in range(4):
            value = float(np.clip(values[row, col], 0.0, 1.0))
            color = (
                int(245 - 190 * value),
                int(248 - 105 * value),
                int(252 - 20 * value),
            )
            x0, y0 = x + col * cell, y + row * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color)


def render_system_diagram() -> Image.Image:
    font = ImageFont.load_default()
    image = Image.new("RGB", (1500, 900), (241, 245, 250))
    draw = ImageDraw.Draw(image)
    draw.text((50, 35), "HOW THE REPAIRED JEPA SYSTEM WORKS", fill=(17, 27, 43), font=font)
    draw.text((50, 58), "Matched interventions during pretraining; RGB-only closed-loop control at deployment", fill=(78, 91, 111), font=font)

    section(draw, (40, 100, 1460, 450), "1. SELF-SUPERVISED PRETRAINING (simulator cloning allowed)", (228, 238, 255), font)
    box(draw, (70, 185, 270, 275), "Same 4-frame\nRGB history", (255, 255, 255), font)
    box(draw, (350, 130, 540, 205), "Clone + LEFT", (235, 240, 255), font)
    box(draw, (350, 230, 540, 305), "Clone + RIGHT", (247, 237, 255), font)
    box(draw, (350, 330, 540, 405), "Clone + FORWARD", (230, 249, 240), font)
    box(draw, (620, 130, 840, 205), "Real RGB future\n1, 2, 4 steps", (255, 255, 255), font)
    box(draw, (620, 230, 840, 305), "Real RGB future\n1, 2, 4 steps", (255, 255, 255), font)
    box(draw, (620, 330, 840, 405), "Real RGB future\n1, 2, 4 steps", (255, 255, 255), font)
    box(draw, (930, 165, 1160, 255), "EMA target encoder\ncreates target latents", (255, 246, 221), font)
    box(draw, (930, 300, 1160, 390), "Context encoder +\naction predictor", (231, 244, 255), font)
    box(draw, (1240, 230, 1415, 325), "Match predicted\nand target latents", (255, 235, 235), font)
    for y in (167, 267, 367):
        arrow(draw, (270, 230), (350, y))
        arrow(draw, (540, y), (620, y))
    arrow(draw, (840, 168), (930, 205))
    arrow(draw, (840, 268), (930, 205))
    arrow(draw, (840, 368), (930, 205))
    arrow(draw, (1160, 210), (1240, 270))
    arrow(draw, (1160, 345), (1240, 290))

    section(draw, (40, 490, 1460, 850), "2. DEPLOYMENT (no cloning, map, planner, or privileged state)", (229, 247, 237), font)
    box(draw, (75, 605, 280, 700), "Live egocentric\nRGB history", (255, 255, 255), font)
    box(draw, (365, 565, 600, 740), "FROZEN JEPA\n\nEncode history\nPredict LEFT / RIGHT / FORWARD\nat horizons 1, 2, 4", (231, 244, 255), font)
    box(draw, (700, 605, 920, 700), "Spatial latent-change\ncounterfactuals", (247, 237, 255), font)
    box(draw, (1010, 605, 1210, 700), "Evolved feed-forward\nNEAT controller", (255, 246, 221), font)
    box(draw, (1300, 605, 1415, 700), "Primitive\naction", (230, 249, 240), font)
    arrow(draw, (280, 652), (365, 652))
    arrow(draw, (600, 652), (700, 652))
    arrow(draw, (920, 652), (1010, 652))
    arrow(draw, (1210, 652), (1300, 652))
    draw.text((495, 797), "JEPA models consequences", fill=(45, 104, 166), font=font)
    draw.text((895, 797), "NEAT chooses among them", fill=(137, 91, 18), font=font)
    return image


def section(draw, bounds, title, fill, font):
    draw.rounded_rectangle(bounds, radius=18, fill=fill, outline=(198, 210, 225), width=2)
    draw.text((bounds[0] + 24, bounds[1] + 20), title, fill=(38, 51, 70), font=font)


def box(draw, bounds, text, fill, font):
    draw.rounded_rectangle(bounds, radius=12, fill=fill, outline=(177, 190, 208), width=2)
    lines = text.split("\n")
    y = bounds[1] + (bounds[3] - bounds[1] - len(lines) * 16) // 2
    for line in lines:
        width = draw.textlength(line, font=font)
        draw.text(((bounds[0] + bounds[2] - width) / 2, y), line, fill=(35, 47, 65), font=font)
        y += 16


def arrow(draw, start, end):
    draw.line((start, end), fill=(91, 108, 132), width=3)
    x, y = end
    draw.polygon(((x, y), (x - 10, y - 6), (x - 10, y + 6)), fill=(91, 108, 132))


if __name__ == "__main__":
    raise SystemExit(main())
