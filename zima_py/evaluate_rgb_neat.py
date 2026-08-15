from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import neat
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evolve_jepa_neat import neat_config_for_inputs
from .evolve_rgb_neat import evaluation_summary, parse_horizons, run_rgb_episode
from .jepa_belief import load_belief_head
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
    parser.add_argument("--controller", choices=("recurrent", "feedforward"), default="recurrent")
    parser.add_argument("--counterfactual-horizons", type=parse_horizons, default=())
    parser.add_argument(
        "--counterfactual-interface",
        choices=("full", "counterfactual-only", "relative", "relative-full"),
        default="full",
    )
    parser.add_argument(
        "--counterfactual-predictor",
        choices=("trained", "random"),
        default="trained",
    )
    parser.add_argument("--belief-checkpoint", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.counterfactual_horizons and not args.input_mode.startswith("temporal-"):
        raise ValueError("--counterfactual-horizons requires a temporal input mode")
    if args.counterfactual_predictor == "random" and not args.counterfactual_horizons:
        raise ValueError("a random counterfactual predictor requires --counterfactual-horizons")
    if args.counterfactual_interface != "full" and not args.counterfactual_horizons:
        raise ValueError("a reduced counterfactual interface requires --counterfactual-horizons")

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
        if args.counterfactual_horizons and args.counterfactual_predictor == "random":
            encoder.reset_predictor(args.seed + 7919)

    belief_head = (
        load_belief_head(ROOT / args.belief_checkpoint)
        if args.belief_checkpoint
        else None
    )

    base_visual_count = (
        0
        if args.counterfactual_horizons
        and args.counterfactual_interface in {"counterfactual-only", "relative"}
        else (192 if args.input_mode == "rgb" else encoder.feature_size)
    )
    counterfactual_count = (
        encoder.counterfactual_feature_size(args.counterfactual_horizons)
        if args.counterfactual_horizons
        else 0
    )
    visual_count = base_visual_count + counterfactual_count
    belief_count = 0 if belief_head is None else belief_head.feature_size
    config = neat.Config(
        neat.DefaultGenome,
        LexicaseReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(
            neat_config_for_inputs(
                visual_count + env.action_space.n + 1 + belief_count,
                lexicase=True,
                feed_forward=args.controller == "feedforward",
            )
        ),
    )
    winner = pickle.loads((ROOT / args.winner).read_bytes())
    network_type = (
        neat.nn.FeedForwardNetwork
        if args.controller == "feedforward"
        else neat.nn.RecurrentNetwork
    )
    seeds = list(range(args.holdout_start, args.holdout_start + args.holdout_count))
    feature_cache = {}
    evaluations = {}
    for style in ("standard", "cyclic"):
        outcomes = [
            run_rgb_episode(
                env,
                network_type.create(winner, config),
                layout_seed,
                args.max_steps,
                args.input_mode,
                encoder,
                feature_cache,
                style,
                False,
                belief_head,
                args.counterfactual_horizons,
                args.counterfactual_interface,
            )
            for layout_seed in seeds
        ]
        evaluations[style] = evaluation_summary(seeds, outcomes)

    report = {
        "environment": args.env,
        "input_mode": args.input_mode,
        "evolution_seed": args.seed,
        "controller": args.controller,
        "winner": str(ROOT / args.winner),
        "checkpoint": str(ROOT / args.checkpoint) if args.input_mode in {"jepa", "temporal-jepa"} else None,
        "belief_checkpoint": None if belief_head is None else str(ROOT / args.belief_checkpoint),
        "counterfactual_horizons": list(args.counterfactual_horizons),
        "counterfactual_feature_count": counterfactual_count,
        "counterfactual_interface": (
            args.counterfactual_interface if args.counterfactual_horizons else None
        ),
        "counterfactual_predictor": (
            args.counterfactual_predictor if args.counterfactual_horizons else None
        ),
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
