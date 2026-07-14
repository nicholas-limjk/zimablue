from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import neat
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evolve_jepa_neat import neat_config_for_inputs
from .evolve_rgb_neat import evaluation_summary, run_rgb_episode
from .lexicase_reproduction import LexicaseReproduction
from .minigrid_adapter import MiniGridSpec, make_minigrid
from .rgb_jepa import RgbJepaAgent, TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one frozen RGB+NEAT winner under paired appearance styles.")
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--input-mode", choices=("rgb", "random", "jepa", "temporal-random", "temporal-jepa"), required=True)
    parser.add_argument("--winner", required=True)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-fresh.pt")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--holdout-start", type=int, default=50)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    base_env = make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps))
    env = RGBImgPartialObsWrapper(base_env, tile_size=8)
    encoder = None
    if args.input_mode in {"random", "jepa", "temporal-random", "temporal-jepa"}:
        encoder = (
            TemporalRgbJepaAgent(env.action_space.n, context_length=args.context_length, seed=args.seed)
            if args.input_mode.startswith("temporal-")
            else RgbJepaAgent(env.action_space.n, seed=args.seed)
        )
        if args.input_mode in {"jepa", "temporal-jepa"}:
            encoder.load(ROOT / args.checkpoint)
        encoder.eval()

    visual_count = 192 if args.input_mode == "rgb" else encoder.feature_size
    config = neat.Config(
        neat.DefaultGenome,
        LexicaseReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config_for_inputs(visual_count + env.action_space.n + 1, lexicase=True)),
    )
    winner = pickle.loads((ROOT / args.winner).read_bytes())
    seeds = list(range(args.holdout_start, args.holdout_start + args.holdout_count))
    feature_cache = {}
    evaluations = {}
    for style in ("standard", "cyclic"):
        outcomes = [
            run_rgb_episode(
                env,
                neat.nn.RecurrentNetwork.create(winner, config),
                layout_seed,
                args.max_steps,
                args.input_mode,
                encoder,
                feature_cache,
                style,
            )
            for layout_seed in seeds
        ]
        evaluations[style] = evaluation_summary(seeds, outcomes)

    report = {
        "environment": args.env,
        "input_mode": args.input_mode,
        "evolution_seed": args.seed,
        "winner": str(ROOT / args.winner),
        "checkpoint": str(ROOT / args.checkpoint) if args.input_mode in {"jepa", "temporal-jepa"} else None,
        "appearance_shift": "RGB channels cyclically remapped to GBR",
        "evaluations": evaluations,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({style: {"solved": row["solved"], "success_rate": row["success_rate"]} for style, row in evaluations.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
