from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import neat

from .embodied_map import BASE_FEATURE_SIZE, FULL_FEATURE_SIZE
from .evolve_jepa_neat import neat_config_for_inputs, run_controller_episode
from .jepa_control_agent import JepaControlAgent
from .minimal_jepa import MinimalJepaAgent
from .minigrid_adapter import MiniGridSpec, make_minigrid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVS = [
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen JEPA + recurrent NEAT winner without adaptation.")
    parser.add_argument("--env", action="append", default=[], help="Environment to evaluate; repeat for multiple environments.")
    parser.add_argument("--seed", type=int, action="append", default=[], help="Seed to evaluate; repeat for multiple seeds.")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--jepa-checkpoint", default="artifacts/jepa/doorkey-controller-seed19-v2.pt")
    parser.add_argument("--winner", default="artifacts/jepa/doorkey-neat-winner-seed19.pkl")
    parser.add_argument("--report", default="artifacts/jepa/doorkey-neat-generalization.json")
    parser.add_argument("--raw-input", action="store_true", help="Evaluate a winner evolved from raw observations without JEPA.")
    parser.add_argument("--minimal-jepa", action="store_true", help="Evaluate the prediction-only 16D JEPA interface.")
    parser.add_argument("--terrain-map", action="store_true", help="Use the persistent terrain-map controller interface.")
    args = parser.parse_args()

    env_ids = args.env or DEFAULT_ENVS
    seeds = args.seed or list(range(10))
    winner = pickle.loads((ROOT / args.winner).read_bytes())
    if args.raw_input and args.minimal_jepa:
        raise ValueError("--raw-input and --minimal-jepa are mutually exclusive.")
    include_body_map = args.terrain_map or (not args.raw_input and not args.minimal_jepa)
    map_feature_count = FULL_FEATURE_SIZE if args.terrain_map else BASE_FEATURE_SIZE if include_body_map else 0
    learned_feature_count = 160 if args.raw_input else 23 if args.minimal_jepa else 64 + 7 * 2
    input_count = learned_feature_count + map_feature_count
    config_path = neat_config_for_inputs(input_count)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    agent = MinimalJepaAgent(7, seed=19) if args.minimal_jepa else JepaControlAgent(7, seed=19)
    if not args.raw_input:
        agent.load(ROOT / args.jepa_checkpoint)
        if args.minimal_jepa:
            agent.eval()
        else:
            for module in (agent.encoder, agent.target_encoder, agent.predictor, agent.reward_head, agent.q_head):
                module.eval()

    environment_results: list[dict[str, Any]] = []
    for env_id in env_ids:
        runs = []
        feature_cache = {}
        for seed in seeds:
            env = make_minigrid(MiniGridSpec(env_id=env_id, seed=seed, max_steps=args.max_steps))
            network = neat.nn.RecurrentNetwork.create(winner, config)
            outcome = run_controller_episode(
                env,
                agent,
                network,
                seed,
                args.max_steps,
                feature_cache,
                raw_input=args.raw_input,
                terrain_map=args.terrain_map,
                include_body_map=include_body_map,
            )
            env.close()
            runs.append({"seed": seed, **outcome})
        solved = [run for run in runs if run["solved"]]
        row = {
            "environment": env_id,
            "seeds": seeds,
            "episodes": len(runs),
            "solved": len(solved),
            "success_rate": len(solved) / len(runs),
            "mean_solved_steps": None if not solved else sum(run["steps"] for run in solved) / len(solved),
            "runs": runs,
        }
        environment_results.append(row)
        print(json.dumps({key: row[key] for key in ("environment", "episodes", "solved", "success_rate", "mean_solved_steps")}))

    total_runs = sum(row["episodes"] for row in environment_results)
    total_solved = sum(row["solved"] for row in environment_results)
    report = {
        "adaptation": "none",
        "controller_input": (
            "raw_body_observation_plus_persistent_terrain_map"
            if args.raw_input and args.terrain_map
            else "raw_body_observation"
            if args.raw_input
            else "minimal_jepa_plus_persistent_terrain_map"
            if args.minimal_jepa and args.terrain_map
            else "minimal_jepa"
            if args.minimal_jepa
            else "frozen_jepa_plus_persistent_terrain_map"
            if args.terrain_map
            else "frozen_jepa_plus_embodied_map"
        ),
        "frozen_jepa_checkpoint": None if args.raw_input else str(ROOT / args.jepa_checkpoint),
        "frozen_neat_winner": str(ROOT / args.winner),
        "total_episodes": total_runs,
        "total_solved": total_solved,
        "overall_success_rate": total_solved / total_runs,
        "environments": environment_results,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall_success_rate": report["overall_success_rate"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
