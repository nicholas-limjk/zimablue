from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from minigrid.wrappers import RGBImgPartialObsWrapper

from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import RgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a spatial JEPA from full partial-view MiniGrid RGB frames.")
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--checkpoint", default="artifacts/jepa/rgb-spatial-s9n1.pt")
    parser.add_argument("--report", default="artifacts/jepa/rgb-spatial-s9n1-training.json")
    args = parser.parse_args()

    base_env = make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps))
    env = RGBImgPartialObsWrapper(base_env, tile_size=args.tile_size)
    agent = RgbJepaAgent(env.action_space.n, seed=args.seed)
    rng = random.Random(args.seed)
    metrics = None
    transitions = 0
    completed_updates = 0

    for episode in range(args.episodes):
        obs, _ = normalize_reset(env.reset(seed=args.seed + episode))
        for _ in range(args.max_steps):
            # Navigation-only random walks avoid semantic action hints while producing useful visual motion.
            action = rng.choices((0, 1, 2), weights=(1, 1, 3), k=1)[0]
            next_obs, _, terminated, truncated, _ = normalize_step(env.step(action))
            agent.observe(obs["image"], action, next_obs["image"])
            transitions += 1
            for _ in range(args.updates_per_step):
                current = agent.train_step()
                if current is not None:
                    metrics = current
                    completed_updates += 1
            obs = next_obs
            if terminated or truncated:
                break

    checkpoint = ROOT / args.checkpoint
    agent.save(checkpoint)
    report = {
        "environment": args.env,
        "seed": args.seed,
        "episodes": args.episodes,
        "transitions": transitions,
        "completed_updates": completed_updates,
        "image_shape": [7 * args.tile_size, 7 * args.tile_size, 3],
        "latent_shape": [agent.latent_channels, agent.spatial_size, agent.spatial_size],
        "feature_size": agent.feature_size,
        "parameters": agent.parameter_counts(),
        "latest_metrics": metrics or {},
        "checkpoint": str(checkpoint),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
