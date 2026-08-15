from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any

import neat
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evolve_jepa_neat import aggregate_layout_fitness, neat_config_for_inputs
from .jepa_belief import JepaBeliefHead, TinyRecursiveBeliefHead, load_belief_head
from .lexicase_reproduction import LexicaseReproduction
from .map_elites_archive import MiniGridMapElites
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import RgbJepaAgent, TemporalRgbJepaAgent
from .train_temporal_rgb_jepa import shortest_safe_plan


ROOT = Path(__file__).resolve().parents[1]


def parse_horizons(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    horizons = tuple(sorted(set(int(part) for part in value.split(","))))
    if not horizons or horizons[0] < 1:
        raise argparse.ArgumentTypeError("horizons must be positive comma-separated integers")
    return horizons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare RGB, random encoders, and frozen spatial or temporal JEPA inputs to recurrent NEAT."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument(
        "--input-mode",
        choices=("rgb", "random", "jepa", "temporal-random", "temporal-jepa"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument(
        "--fitness-mode",
        choices=("navigation", "reactive-racing"),
        default="navigation",
        help="Reactive racing rewards real motion and finish-line progress, not changing pixels.",
    )
    parser.add_argument("--training-seed-pool", type=int, default=50)
    parser.add_argument("--cases-per-generation", type=int, default=5)
    parser.add_argument("--holdout-start", type=int, default=50)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--train-rgb-style", choices=("standard", "cyclic"), default="standard")
    parser.add_argument("--holdout-rgb-style", choices=("standard", "cyclic"), default="standard")
    parser.add_argument("--checkpoint", default="artifacts/jepa/rgb-spatial-s9n1.pt")
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument(
        "--counterfactual-horizons",
        type=parse_horizons,
        default=(),
        help="Temporal JEPA rollout horizons to expose to NEAT, for example 1,2,4.",
    )
    parser.add_argument(
        "--counterfactual-predictor",
        choices=("trained", "random"),
        default="trained",
        help="Ablation: keep the loaded encoder but replace only its predictor with random weights.",
    )
    parser.add_argument(
        "--counterfactual-interface",
        choices=("full", "counterfactual-only", "relative", "relative-full"),
        default="full",
        help="Expose context plus predictions, predictions only, or action-relative predictions only.",
    )
    parser.add_argument("--winner", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--archive-injections", type=int, default=6)
    parser.add_argument("--controller", choices=("recurrent", "feedforward"), default="recurrent")
    parser.add_argument(
        "--oracle-action-hint",
        action="store_true",
        help="Diagnostic only: append the full-state safe planner's next action.",
    )
    parser.add_argument(
        "--belief-checkpoint",
        default="",
        help="Optional recurrent or tiny-recursive belief decoder trained from frozen JEPA latents.",
    )
    parser.add_argument(
        "--active-belief-lexicase",
        action="store_true",
        help="Select simultaneously on task, uncertainty reduction, coverage, and safe exploration.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.training_seed_pool < args.cases_per_generation:
        raise ValueError("--training-seed-pool must be at least --cases-per-generation")
    if args.counterfactual_horizons and not args.input_mode.startswith("temporal-"):
        raise ValueError("--counterfactual-horizons requires a temporal input mode")
    if args.counterfactual_predictor == "random" and not args.counterfactual_horizons:
        raise ValueError("a random counterfactual predictor requires --counterfactual-horizons")
    if args.counterfactual_interface != "full" and not args.counterfactual_horizons:
        raise ValueError("a reduced counterfactual interface requires --counterfactual-horizons")

    base_env = make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps))
    env = RGBImgPartialObsWrapper(base_env, tile_size=args.tile_size)
    encoder = None
    if args.input_mode in {"random", "jepa", "temporal-random", "temporal-jepa"}:
        encoder = (
            TemporalRgbJepaAgent(env.action_space.n, context_length=args.context_length, seed=args.seed)
            if args.input_mode.startswith("temporal-")
            else RgbJepaAgent(env.action_space.n, seed=args.seed)
        )
        if args.input_mode in {"jepa", "temporal-jepa"}:
            checkpoint = ROOT / args.checkpoint
            if not checkpoint.exists():
                raise FileNotFoundError(f"RGB JEPA checkpoint not found: {checkpoint}")
            encoder.load(checkpoint)
        encoder.eval()
        if args.counterfactual_horizons and args.counterfactual_predictor == "random":
            encoder.reset_predictor(args.seed + 7919)

    belief_head = None
    if args.belief_checkpoint:
        if not args.input_mode.startswith("temporal-"):
            raise ValueError("--belief-checkpoint requires a temporal input mode")
        belief_head = load_belief_head(ROOT / args.belief_checkpoint)
        if belief_head.latent_size != encoder.feature_size:
            raise ValueError("Belief checkpoint latent size does not match the JEPA encoder")
    if args.active_belief_lexicase and belief_head is None:
        raise ValueError("--active-belief-lexicase requires --belief-checkpoint")

    base_visual_features = (
        0
        if args.counterfactual_horizons
        and args.counterfactual_interface in {"counterfactual-only", "relative"}
        else (192 if args.input_mode == "rgb" else encoder.feature_size)
    )
    counterfactual_feature_count = (
        encoder.counterfactual_feature_size(args.counterfactual_horizons)
        if args.counterfactual_horizons
        else 0
    )
    visual_features = base_visual_features + counterfactual_feature_count
    oracle_feature_count = 3 if args.oracle_action_hint else 0
    belief_feature_count = 0 if belief_head is None else belief_head.feature_size
    input_count = (
        visual_features
        + env.action_space.n
        + 1
        + oracle_feature_count
        + belief_feature_count
    )
    config_path = neat_config_for_inputs(
        input_count,
        lexicase=True,
        feed_forward=args.controller == "feedforward",
    )
    config = neat.Config(
        neat.DefaultGenome,
        LexicaseReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    config.no_fitness_termination = True
    config.pop_size = args.population
    network_type = (
        neat.nn.FeedForwardNetwork
        if args.controller == "feedforward"
        else neat.nn.RecurrentNetwork
    )
    population = neat.Population(config, seed=args.seed)
    archive = MiniGridMapElites()
    population.reproduction.map_elites_archive = archive
    population.reproduction.archive_injections = args.archive_injections
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)
    if not args.quiet:
        population.add_reporter(neat.StdOutReporter(True))

    feature_cache: dict[bytes, np.ndarray] = {}
    case_schedule: list[list[int]] = []
    generation = 0

    def evaluate_genomes(genomes, neat_config) -> None:
        nonlocal generation
        rng = random.Random(args.seed + 1009 * generation)
        cases = rng.sample(range(args.training_seed_pool), args.cases_per_generation)
        case_schedule.append(cases)
        for _, genome in genomes:
            outcomes = [
                run_rgb_episode(
                    env,
                    network_type.create(genome, neat_config),
                    layout_seed,
                    args.max_steps,
                    args.input_mode,
                    encoder,
                    feature_cache,
                    args.train_rgb_style,
                    args.oracle_action_hint,
                    belief_head,
                    args.counterfactual_horizons,
                    args.counterfactual_interface,
                    args.fitness_mode,
                )
                for layout_seed in cases
            ]
            genome.fitness = aggregate_layout_fitness(outcomes)
            genome.lexicase_scores = (
                active_belief_lexicase_scores(outcomes, args.max_steps)
                if args.active_belief_lexicase
                else [float(outcome["fitness"]) for outcome in outcomes]
            )
            archive.update(genome, outcomes, genome.fitness, generation)
        generation += 1

    evolution_winner = population.run(evaluate_genomes, args.generations)
    final_training_seeds = list(range(min(10, args.training_seed_pool)))
    candidates = [("evolution_winner", evolution_winner)] + [
        (f"archive:{entry.descriptor}", entry.genome) for entry in archive.elites()
    ]
    candidate_results = []
    for source, candidate in candidates:
        outcomes = [
            run_rgb_episode(
                env,
                network_type.create(candidate, config),
                layout_seed,
                args.max_steps,
                args.input_mode,
                encoder,
                feature_cache,
                args.train_rgb_style,
                args.oracle_action_hint,
                belief_head,
                args.counterfactual_horizons,
                args.counterfactual_interface,
                args.fitness_mode,
            )
            for layout_seed in final_training_seeds
        ]
        candidate_results.append((aggregate_layout_fitness(outcomes), source, candidate, outcomes))
    robust_fitness, winner_source, winner, training_outcomes = max(candidate_results, key=lambda row: row[0])

    holdout_seeds = list(range(args.holdout_start, args.holdout_start + args.holdout_count))
    holdout_outcomes = [
        run_rgb_episode(
            env,
            network_type.create(winner, config),
            layout_seed,
            args.max_steps,
            args.input_mode,
            encoder,
            feature_cache,
            args.holdout_rgb_style,
            args.oracle_action_hint,
            belief_head,
            args.counterfactual_horizons,
            args.counterfactual_interface,
            args.fitness_mode,
        )
        for layout_seed in holdout_seeds
    ]
    standard_holdout_outcomes = (
        [
            run_rgb_episode(
                env,
                network_type.create(winner, config),
                layout_seed,
                args.max_steps,
                args.input_mode,
                encoder,
                feature_cache,
                "standard",
                args.oracle_action_hint,
                belief_head,
                args.counterfactual_horizons,
                args.counterfactual_interface,
                args.fitness_mode,
            )
            for layout_seed in holdout_seeds
        ]
        if args.holdout_rgb_style != "standard"
        else holdout_outcomes
    )
    diagnostic_suffix = ("-oracle" if args.oracle_action_hint else "") + (
        "-feedforward" if args.controller == "feedforward" else ""
    ) + ("-belief" if belief_head is not None else "") + (
        "-active" if args.active_belief_lexicase else ""
    )
    stem = f"rgb-{args.input_mode}-{args.env.lower().replace('minigrid-', '').replace('-v0', '')}-seed{args.seed}{diagnostic_suffix}"
    winner_path = ROOT / (args.winner or f"artifacts/jepa/{stem}-winner.pkl")
    report_path = ROOT / (args.report or f"artifacts/jepa/{stem}-report.json")
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_bytes(pickle.dumps(winner))

    report: dict[str, Any] = {
        "environment": args.env,
        "input_mode": args.input_mode,
        "semantic_inputs": False,
        "persistent_map": False,
        "train_rgb_style": args.train_rgb_style,
        "holdout_rgb_style": args.holdout_rgb_style,
        "visual_feature_count": visual_features,
        "base_visual_feature_count": base_visual_features,
        "counterfactual_horizons": list(args.counterfactual_horizons),
        "counterfactual_feature_count": counterfactual_feature_count,
        "counterfactual_predictor": (
            args.counterfactual_predictor if args.counterfactual_horizons else None
        ),
        "counterfactual_interface": (
            args.counterfactual_interface if args.counterfactual_horizons else None
        ),
        "controller_input_count": input_count,
        "controller": args.controller,
        "fitness_mode": args.fitness_mode,
        "oracle_action_hint": args.oracle_action_hint,
        "belief_checkpoint": None if belief_head is None else str(ROOT / args.belief_checkpoint),
        "belief_feature_count": belief_feature_count,
        "active_belief_lexicase": args.active_belief_lexicase,
        "seed": args.seed,
        "generations": args.generations,
        "population": args.population,
        "training_seed_pool": args.training_seed_pool,
        "cases_per_generation": args.cases_per_generation,
        "case_schedule": case_schedule,
        "selection": "epsilon_lexicase",
        "map_elites": True,
        "map_elites_archive": archive.summary(),
        "checkpoint": str(ROOT / args.checkpoint) if args.input_mode in {"jepa", "temporal-jepa"} else None,
        "context_length": args.context_length if args.input_mode.startswith("temporal-") else 1,
        "encoder_parameters": None if encoder is None else encoder.parameter_counts(),
        "winner_source": winner_source,
        "winner_fitness": robust_fitness,
        "winner_nodes": len(winner.nodes),
        "winner_connections": len(winner.connections),
        "training_evaluation": evaluation_summary(final_training_seeds, training_outcomes),
        "standard_holdout_evaluation": evaluation_summary(holdout_seeds, standard_holdout_outcomes),
        "holdout_evaluation": evaluation_summary(holdout_seeds, holdout_outcomes),
        "fitness_history": [genome.fitness for genome in statistics.most_fit_genomes],
        "winner": str(winner_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=json_scalar), encoding="utf-8")
    env.close()
    print(json.dumps(report["holdout_evaluation"] | {"report": str(report_path)}))
    return 0


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def visual_features(
    image: np.ndarray,
    mode: str,
    encoder: Any,
    cache: dict[bytes, np.ndarray],
    history_frames: np.ndarray | None = None,
    history_actions: list[int] | None = None,
    counterfactual_horizons: tuple[int, ...] = (),
    counterfactual_interface: str = "full",
) -> np.ndarray:
    if mode.startswith("temporal-"):
        assert history_frames is not None and history_actions is not None
        key = (
            history_frames.tobytes()
            + np.asarray(history_actions, dtype=np.int8).tobytes()
            + b"|cf|"
            + np.asarray(counterfactual_horizons, dtype=np.int16).tobytes()
            + counterfactual_interface.encode("ascii")
        )
    else:
        key = np.asarray(image, dtype=np.uint8).tobytes()
    if key in cache:
        return cache[key]
    if mode == "rgb":
        # 56x56 divides exactly into an 8x8 grid of 7x7 pixel averages.
        features = np.asarray(image, dtype=np.float64).reshape(8, 7, 8, 7, 3).mean(axis=(1, 3)).reshape(-1) / 255.0
    elif mode.startswith("temporal-"):
        assert isinstance(encoder, TemporalRgbJepaAgent)
        features = (
            encoder.features_with_counterfactuals(
                history_frames,
                history_actions,
                counterfactual_horizons,
                include_context=counterfactual_interface in {"full", "relative-full"},
                relative=counterfactual_interface in {"relative", "relative-full"},
            )
            if counterfactual_horizons
            else encoder.features(history_frames, history_actions)
        )
    else:
        assert encoder is not None
        features = encoder.features(image)
    cache[key] = features
    return features


def controller_features(
    image: np.ndarray,
    previous_action: int,
    action_count: int,
    mode: str,
    encoder: Any,
    cache: dict[bytes, np.ndarray],
    history_frames: np.ndarray | None = None,
    history_actions: list[int] | None = None,
    counterfactual_horizons: tuple[int, ...] = (),
    counterfactual_interface: str = "full",
) -> np.ndarray:
    previous = np.zeros(action_count + 1, dtype=np.float64)
    previous[previous_action if 0 <= previous_action < action_count else action_count] = 1.0
    return np.concatenate(
        (
            visual_features(
                image,
                mode,
                encoder,
                cache,
                history_frames,
                history_actions,
                counterfactual_horizons,
                counterfactual_interface,
            ),
            previous,
        )
    )


def run_rgb_episode(
    env,
    network,
    seed: int,
    max_steps: int,
    mode: str,
    encoder: Any,
    cache: dict[bytes, np.ndarray],
    rgb_style: str = "standard",
    oracle_action_hint: bool = False,
    belief_head: JepaBeliefHead | TinyRecursiveBeliefHead | None = None,
    counterfactual_horizons: tuple[int, ...] = (),
    counterfactual_interface: str = "full",
    fitness_mode: str = "navigation",
) -> dict[str, Any]:
    obs, _ = normalize_reset(env.reset(seed=seed))
    previous_action = -1
    first_image = apply_rgb_style(obs["image"], rgb_style)
    context_length = encoder.context_length if mode.startswith("temporal-") else 1
    history_frames = deque([first_image.copy() for _ in range(context_length)], maxlen=context_length)
    history_actions = deque([-1 for _ in range(context_length - 1)], maxlen=max(1, context_length - 1))
    visited_observations: set[bytes] = set()
    visited_positions = {(0, 0)}
    x = y = direction = 0
    blocked_steps = 0
    fitness = 0.0
    belief_hidden = None
    initial_belief_entropy = None
    previous_belief_entropy = None
    cumulative_entropy_reduction = 0.0
    goal = find_grid_object(env.unwrapped, "goal")
    start_world_position = tuple(int(value) for value in env.unwrapped.agent_pos)
    previous_goal_distance = manhattan(start_world_position, goal)
    for step in range(max_steps):
        image = apply_rgb_style(obs["image"], rgb_style)
        features = controller_features(
            image,
            previous_action,
            env.action_space.n,
            mode,
            encoder,
            cache,
            np.stack(history_frames) if mode.startswith("temporal-") else None,
            list(history_actions) if mode.startswith("temporal-") else None,
            counterfactual_horizons,
            counterfactual_interface,
        )
        if belief_head is not None:
            latent = features[: belief_head.latent_size]
            belief_features, belief_hidden = belief_head.step(
                latent,
                previous_action,
                belief_hidden,
            )
            if belief_head.route_size:
                route_start = belief_head.hidden_size
                route_probabilities = np.clip(
                    belief_features[route_start : route_start + belief_head.route_size],
                    1e-9,
                    1.0,
                )
                route_entropy = float(
                    -np.sum(route_probabilities * np.log(route_probabilities)) / math.log(3.0)
                )
                if initial_belief_entropy is None:
                    initial_belief_entropy = route_entropy
                if previous_belief_entropy is not None:
                    cumulative_entropy_reduction += max(
                        0.0,
                        previous_belief_entropy - route_entropy,
                    )
                previous_belief_entropy = route_entropy
            features = np.concatenate((features, belief_features))
        if oracle_action_hint:
            hint = np.zeros(3, dtype=np.float64)
            plan = shortest_safe_plan(env)
            if plan and 0 <= plan[0] < 3:
                hint[plan[0]] = 1.0
            features = np.concatenate((features, hint))
        outputs = np.asarray(network.activate(features.tolist()), dtype=np.float64)
        outputs[3:] = -np.inf  # LavaCrossing needs only left, right, and forward.
        action = int(outputs.argmax())
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        next_image = apply_rgb_style(next_obs["image"], rgb_style)

        signature = next_image.tobytes()
        if signature not in visited_observations:
            visited_observations.add(signature)
            if fitness_mode == "navigation":
                fitness += 0.005
        fitness -= 0.005 if fitness_mode == "reactive-racing" else 0.001
        blocked = bool(action == 2 and np.array_equal(image, next_image))
        if blocked:
            blocked_steps += 1
            if fitness_mode == "navigation":
                fitness -= 0.01
        if action == 0:
            direction = (direction - 1) % 4
        elif action == 1:
            direction = (direction + 1) % 4
        elif action == 2 and not blocked:
            dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
            x, y = x + dx, y + dy
            if (x, y) not in visited_positions:
                visited_positions.add((x, y))
                if fitness_mode == "navigation":
                    fitness += 0.01
        world_position = tuple(int(value) for value in env.unwrapped.agent_pos)
        if fitness_mode == "reactive-racing":
            goal_distance = manhattan(world_position, goal)
            fitness += 0.05 * (previous_goal_distance - goal_distance)
            previous_goal_distance = goal_distance
        solved = bool(terminated and float(reward) > 0.0)
        if solved:
            fitness += 10.0
        elif terminated:
            fitness -= 0.1 if fitness_mode == "reactive-racing" else 1.0
        obs = next_obs
        previous_action = action
        if mode.startswith("temporal-"):
            history_frames.append(next_image.copy())
            history_actions.append(action)
        if terminated or truncated:
            return episode_result(
                solved,
                step + 1,
                fitness,
                visited_observations,
                visited_positions,
                blocked_steps,
                initial_belief_entropy,
                previous_belief_entropy,
                cumulative_entropy_reduction,
            )
    return episode_result(
        False,
        max_steps,
        fitness,
        visited_observations,
        visited_positions,
        blocked_steps,
        initial_belief_entropy,
        previous_belief_entropy,
        cumulative_entropy_reduction,
    )


def find_grid_object(unwrapped, object_type: str) -> tuple[int, int]:
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == object_type:
                return int(x), int(y)
    raise RuntimeError(f"MiniGrid environment has no {object_type} object")


def manhattan(first: tuple[int, int], second: tuple[int, int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def episode_result(
    solved,
    steps,
    fitness,
    observations,
    positions,
    blocked_steps,
    initial_belief_entropy=None,
    final_belief_entropy=None,
    cumulative_entropy_reduction=0.0,
) -> dict[str, Any]:
    return {
        "solved": bool(solved),
        "steps": int(steps),
        "fitness": round(float(fitness), 6),
        "unique_observations": len(observations),
        "unique_positions": len(positions),
        "blocked_steps": int(blocked_steps),
        "belief_entropy_initial": None if initial_belief_entropy is None else float(initial_belief_entropy),
        "belief_entropy_final": None if final_belief_entropy is None else float(final_belief_entropy),
        "belief_uncertainty_reduction": (
            0.0
            if initial_belief_entropy is None or final_belief_entropy is None
            else max(0.0, float(initial_belief_entropy - final_belief_entropy))
        ),
        "belief_cumulative_reduction": float(cumulative_entropy_reduction),
        "collected": False,
        "opened_door": False,
    }


def apply_rgb_style(image: np.ndarray, style: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if style == "standard":
        return array
    if style == "cyclic":
        # An unseen palette change that preserves geometry and luminance structure.
        return array[..., [1, 2, 0]].copy()
    raise ValueError(f"Unknown RGB style: {style}")


def active_belief_lexicase_scores(
    outcomes: list[dict[str, Any]],
    max_steps: int,
) -> list[float]:
    """Expose concurrent task and active-perception objectives as lexicase cases."""
    task = [float(outcome["fitness"]) for outcome in outcomes]
    information = []
    coverage = []
    safe_exploration = []
    for outcome in outcomes:
        confirmed = min(
            1.0,
            (
                min(float(outcome["unique_observations"]), 20.0)
                + min(float(outcome["unique_positions"]), 20.0)
            )
            / 40.0,
        )
        uncertainty_reduction = min(
            1.0,
            float(outcome.get("belief_cumulative_reduction", 0.0)),
        )
        information.append(confirmed * uncertainty_reduction)
        coverage.append(confirmed)
        survival = min(1.0, float(outcome["steps"]) / max(1.0, float(max_steps)))
        position_coverage = min(1.0, float(outcome["unique_positions"]) / 15.0)
        blocked_fraction = min(1.0, float(outcome["blocked_steps"]) / max(1.0, float(max_steps)))
        safe_exploration.append(
            (2.0 if outcome["solved"] else 0.0)
            + math.sqrt(survival * position_coverage)
            - blocked_fraction
        )
    return task + information + coverage + safe_exploration


def evaluation_summary(seeds: list[int], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [{"seed": seed, **outcome} for seed, outcome in zip(seeds, outcomes)]
    solved = [row for row in runs if row["solved"]]
    return {
        "episodes": len(runs),
        "solved": len(solved),
        "success_rate": len(solved) / len(runs) if runs else 0.0,
        "mean_solved_steps": None if not solved else sum(row["steps"] for row in solved) / len(solved),
        "runs": runs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
