from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evaluate_jepa_mpc import CONTROL_ACTIONS
from .evolve_jepa_neat import aggregate_layout_fitness
from .evolve_jepa_mpc_neat import (
    FRONTIER_FEATURE_COUNT,
    evaluation_by_geometry,
    geometry_balanced_sample,
    group_seeds_by_geometry,
    run_scorer_episode,
)
from .evolve_rgb_neat import evaluation_summary, json_scalar
from .jepa_rollout_outcome import JepaRolloutOutcomeHead
from .minigrid_adapter import MiniGridSpec, make_minigrid
from .rgb_jepa import TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_COUNT = 5
PARAMETER_COUNT = OUTPUT_COUNT * FRONTIER_FEATURE_COUNT + OUTPUT_COUNT


class LinearFrontierNetwork:
    def __init__(self, parameters: np.ndarray) -> None:
        vector = np.asarray(parameters, dtype=np.float64)
        if vector.shape != (PARAMETER_COUNT,):
            raise ValueError(f"expected {PARAMETER_COUNT} parameters")
        split = OUTPUT_COUNT * FRONTIER_FEATURE_COUNT
        self.weights = vector[:split].reshape(OUTPUT_COUNT, FRONTIER_FEATURE_COUNT)
        self.bias = vector[split:]

    def activate(self, inputs) -> list[float]:
        values = np.asarray(inputs, dtype=np.float64)
        return (self.weights @ values + self.bias).tolist()


def behavior_descriptor(outcomes: list[dict]) -> tuple[int, int, int, int]:
    steps = max(1, sum(int(outcome["steps"]) for outcome in outcomes))
    returns = sum(int(outcome["mode_counts"]["return"]) for outcome in outcomes)
    probes = sum(int(outcome["mode_counts"]["probe"]) for outcome in outcomes)
    exploration = int(
        sum(int(outcome["unique_positions"]) for outcome in outcomes)
        / max(1, len(outcomes))
        // 5
    )
    return (
        sum(bool(outcome["solved"]) for outcome in outcomes),
        min(4, int(5 * returns / steps)),
        min(4, int(5 * probes / steps)),
        min(10, exploration),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evolve a fixed-topology linear JEPA graph-frontier selector."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--outcome-checkpoint", required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--collision-logit-bias", type=float, default=-3.178054)
    parser.add_argument("--graph-capacity", type=int, default=128)
    parser.add_argument("--graph-match-threshold", type=float, default=0.97)
    parser.add_argument("--mode-safety-margin", type=float, default=2.0)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--parent-count", type=int, default=10)
    parser.add_argument("--mutation-sigma", type=float, default=0.15)
    parser.add_argument("--training-start", type=int, default=0)
    parser.add_argument("--training-seed-pool", type=int, default=50)
    parser.add_argument("--cases-per-generation", type=int, default=6)
    parser.add_argument("--validation-start", type=int, default=50)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--holdout-start", type=int, default=100)
    parser.add_argument("--holdout-count", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--winner", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not 1 <= args.elite_count <= args.parent_count <= args.population:
        raise ValueError("require 1 <= elite-count <= parent-count <= population")

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
    training_seeds = list(
        range(args.training_start, args.training_start + args.training_seed_pool)
    )
    groups = group_seeds_by_geometry(env, training_seeds)
    rng = np.random.default_rng(args.seed)
    population = rng.normal(0.0, 0.5, size=(args.population, PARAMETER_COUNT))
    feature_cache = {}
    archive: dict[tuple[int, int, int, int], tuple[float, np.ndarray]] = {}
    best_candidates = []
    fitness_history = []
    case_schedule = []

    for generation in range(args.generations):
        case_rng = random.Random(args.seed + 1009 * generation)
        cases = geometry_balanced_sample(groups, args.cases_per_generation, case_rng)
        case_schedule.append(cases)
        evaluated = []
        for parameters in population:
            network = LinearFrontierNetwork(parameters)
            outcomes = [
                run_scorer_episode(
                    env,
                    jepa,
                    outcome_head,
                    sequences,
                    network,
                    layout_seed,
                    args.max_steps,
                    "standard",
                    args.collision_logit_bias,
                    feature_cache,
                    0,
                    args.graph_capacity,
                    True,
                    0.0,
                    "frontier-selector",
                    args.mode_safety_margin,
                    True,
                    "any-action",
                    args.graph_match_threshold,
                )
                for layout_seed in cases
            ]
            fitness = aggregate_layout_fitness(outcomes)
            evaluated.append((fitness, parameters.copy(), outcomes))
            descriptor = behavior_descriptor(outcomes)
            current = archive.get(descriptor)
            if current is None or fitness > current[0]:
                archive[descriptor] = (fitness, parameters.copy())
        evaluated.sort(key=lambda row: row[0], reverse=True)
        fitness_history.append(float(evaluated[0][0]))
        best_candidates.append(evaluated[0][1].copy())
        elites = [row[1] for row in evaluated[: args.elite_count]]
        parents = [row[1] for row in evaluated[: args.parent_count]]
        next_population = [elite.copy() for elite in elites]
        while len(next_population) < args.population:
            parent = parents[int(rng.integers(0, len(parents)))]
            child = parent + rng.normal(0.0, args.mutation_sigma, size=parent.shape)
            next_population.append(child)
        population = np.stack(next_population)

    candidate_vectors = best_candidates + [row[1] for row in archive.values()]
    validation_seeds = list(
        range(args.validation_start, args.validation_start + args.validation_count)
    )
    validation_results = []
    for parameters in candidate_vectors:
        network = LinearFrontierNetwork(parameters)
        outcomes = [
            run_scorer_episode(
                env,
                jepa,
                outcome_head,
                sequences,
                network,
                layout_seed,
                args.max_steps,
                "standard",
                args.collision_logit_bias,
                feature_cache,
                0,
                args.graph_capacity,
                True,
                0.0,
                "frontier-selector",
                args.mode_safety_margin,
                True,
                "any-action",
                args.graph_match_threshold,
            )
            for layout_seed in validation_seeds
        ]
        validation_results.append(
            (aggregate_layout_fitness(outcomes), parameters, outcomes)
        )
    validation_fitness, winner, validation_outcomes = max(
        validation_results, key=lambda row: row[0]
    )
    test_seeds = list(range(args.holdout_start, args.holdout_start + args.holdout_count))
    network = LinearFrontierNetwork(winner)
    test_outcomes = [
        run_scorer_episode(
            env,
            jepa,
            outcome_head,
            sequences,
            network,
            layout_seed,
            args.max_steps,
            "standard",
            args.collision_logit_bias,
            feature_cache,
            0,
            args.graph_capacity,
            True,
            0.0,
            "frontier-selector",
            args.mode_safety_margin,
            True,
            "any-action",
            args.graph_match_threshold,
        )
        for layout_seed in test_seeds
    ]
    winner_path = ROOT / args.winner
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(winner_path, parameters=winner)
    report = {
        "environment": args.env,
        "controller": "fixed_topology_linear_frontier_ga",
        "parameter_count": PARAMETER_COUNT,
        "seed": args.seed,
        "generations": args.generations,
        "population": args.population,
        "cases_per_generation": args.cases_per_generation,
        "case_schedule": case_schedule,
        "mutation_sigma": args.mutation_sigma,
        "graph_match_threshold": args.graph_match_threshold,
        "validation_fitness": validation_fitness,
        "candidate_selection_evaluation": evaluation_summary(
            validation_seeds, validation_outcomes
        ),
        "holdout_evaluation": evaluation_summary(test_seeds, test_outcomes),
        "holdout_by_geometry": evaluation_by_geometry(env, test_seeds, test_outcomes),
        "fitness_history": fitness_history,
        "archive_size": len(archive),
        "winner": str(winner_path.resolve()),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=json_scalar), encoding="utf-8")
    env.close()
    print(json.dumps(report["holdout_evaluation"] | {"report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
