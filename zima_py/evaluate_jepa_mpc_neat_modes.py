from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import neat
import torch
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evaluate_jepa_mpc import CONTROL_ACTIONS
from .evolve_jepa_neat import neat_config_for_inputs
from .evolve_jepa_mpc_neat import (
    MODE_FEATURE_COUNT,
    evaluation_by_geometry,
    run_scorer_episode,
)
from .evolve_rgb_neat import evaluation_summary, json_scalar
from .jepa_rollout_outcome import JepaRolloutOutcomeHead
from .lexicase_reproduction import LexicaseReproduction
from .minigrid_adapter import MiniGridSpec, make_minigrid
from .rgb_jepa import TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved JEPA-MPC graph-mode NEAT controller."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--outcome-checkpoint", required=True)
    parser.add_argument("--winner", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--collision-logit-bias", type=float, default=-3.178054)
    parser.add_argument("--graph-capacity", type=int, default=128)
    parser.add_argument("--graph-match-threshold", type=float, default=0.97)
    parser.add_argument(
        "--frontier-policy",
        choices=("any-action", "forward-only", "forward-first"),
        default="any-action",
    )
    parser.add_argument("--mode-safety-margin", type=float, default=2.0)
    parser.add_argument("--commit-return", action="store_true")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--rgb-style", choices=("standard", "cyclic"), default="standard")
    args = parser.parse_args()

    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()
    outcome_head = JepaRolloutOutcomeHead.load(ROOT / args.outcome_checkpoint)
    sequences = torch.as_tensor(
        list(itertools.product(CONTROL_ACTIONS, repeat=args.depth)),
        dtype=torch.long,
        device=jepa.device,
    )
    config_path = neat_config_for_inputs(
        MODE_FEATURE_COUNT,
        lexicase=True,
        feed_forward=True,
        output_count=3,
    )
    config = neat.Config(
        neat.DefaultGenome,
        LexicaseReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    genome = pickle.loads((ROOT / args.winner).read_bytes())
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    seeds = list(range(args.start, args.start + args.count))
    feature_cache = {}
    outcomes = [
        run_scorer_episode(
            env,
            jepa,
            outcome_head,
            sequences,
            network,
            layout_seed,
            args.max_steps,
            args.rgb_style,
            args.collision_logit_bias,
            feature_cache,
            0,
            args.graph_capacity,
            True,
            0.0,
            "mode-selector",
            args.mode_safety_margin,
            args.commit_return,
            args.frontier_policy,
            args.graph_match_threshold,
        )
        for layout_seed in seeds
    ]
    report = {
        "environment": args.env,
        "winner": str((ROOT / args.winner).resolve()),
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "outcome_checkpoint": str((ROOT / args.outcome_checkpoint).resolve()),
        "graph_capacity": args.graph_capacity,
        "graph_match_threshold": args.graph_match_threshold,
        "frontier_policy": args.frontier_policy,
        "commit_return": args.commit_return,
        "mode_safety_margin": args.mode_safety_margin,
        "evaluation": evaluation_summary(seeds, outcomes),
        "by_geometry": evaluation_by_geometry(env, seeds, outcomes),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=json_scalar), encoding="utf-8")
    env.close()
    print(json.dumps(report["evaluation"] | {"report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
