from __future__ import annotations

import argparse
import copy
import itertools
import json
import pickle
import random
from collections import deque
from pathlib import Path

import neat
import numpy as np
import torch
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evaluate_jepa_mpc import CONTROL_ACTIONS, outcome_sequence_features
from .episodic_latent_memory import EpisodicLatentMemory
from .episodic_latent_graph import EpisodicLatentGraph
from .evolve_jepa_neat import aggregate_layout_fitness, neat_config_for_inputs
from .evolve_rgb_neat import apply_rgb_style, episode_result, evaluation_summary, json_scalar
from .jepa_rollout_outcome import JepaRolloutOutcomeHead
from .lexicase_reproduction import LexicaseReproduction
from .map_elites_archive import MiniGridMapElites
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]
MODE_FEATURE_COUNT = 12
MODE_NAMES = ("local", "return", "probe")
FRONTIER_FEATURE_COUNT = 24
FRONTIER_STRATEGIES = ("nearest", "oldest", "least-visited")


def remap_genome_innovations(genome, reference_genomes, innovation_tracker) -> None:
    """Align a transplanted genome with a fresh population's innovation IDs."""
    key_to_innovation = {}
    for reference in reference_genomes:
        for key, connection in reference.connections.items():
            existing = key_to_innovation.get(key)
            if existing is not None and existing != connection.innovation:
                raise ValueError(f"fresh population has conflicting innovations for {key}")
            key_to_innovation[key] = connection.innovation
    for key, connection in genome.connections.items():
        innovation = key_to_innovation.get(key)
        if innovation is None:
            innovation = innovation_tracker.get_innovation_number(
                key[0], key[1], "warm_start_connection"
            )
            key_to_innovation[key] = innovation
        connection.innovation = innovation


def lava_layout_geometry(env, seed: int) -> tuple[str, int, int]:
    """Describe a LavaCrossing layout for dataset stratification only."""
    normalize_reset(env.reset(seed=int(seed)))
    base = env.unwrapped
    lava = {
        (x, y)
        for y in range(base.height)
        for x in range(base.width)
        if getattr(base.grid.get(x, y), "type", None) == "lava"
    }
    xs = {x for x, _ in lava}
    ys = {y for _, y in lava}
    if len(xs) == 1:
        line = next(iter(xs))
        gaps = [y for y in range(1, base.height - 1) if (line, y) not in lava]
        orientation = "vertical"
    elif len(ys) == 1:
        line = next(iter(ys))
        gaps = [x for x in range(1, base.width - 1) if (x, line) not in lava]
        orientation = "horizontal"
    else:
        raise ValueError(f"seed {seed} is not a single LavaCrossing barrier")
    if len(gaps) != 1:
        raise ValueError(f"seed {seed} has {len(gaps)} crossing gaps, expected one")
    return orientation, int(line), int(gaps[0])


def group_seeds_by_geometry(env, seeds: list[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for seed in seeds:
        orientation, _, gap = lava_layout_geometry(env, seed)
        groups.setdefault(f"{orientation}:gap={gap}", []).append(int(seed))
    return groups


def geometry_balanced_sample(
    groups: dict[str, list[int]],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample without replacement while spreading cases across gap geometries."""
    available = {key: list(values) for key, values in groups.items() if values}
    for values in available.values():
        rng.shuffle(values)
    selected = []
    while available and len(selected) < count:
        keys = list(available)
        rng.shuffle(keys)
        for key in keys:
            selected.append(available[key].pop())
            if not available[key]:
                del available[key]
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("not enough geometry-grouped seeds for requested cases")
    return selected


def evaluation_by_geometry(env, seeds: list[int], outcomes: list[dict]) -> dict[str, dict]:
    rows: dict[str, list[tuple[int, dict]]] = {}
    for seed, outcome in zip(seeds, outcomes):
        orientation, line, gap = lava_layout_geometry(env, seed)
        key = f"{orientation}:line={line}:gap={gap}"
        rows.setdefault(key, []).append((seed, outcome))
    return {
        key: {
            "episodes": len(group),
            "solved": sum(bool(outcome["solved"]) for _, outcome in group),
            "seeds": [seed for seed, _ in group],
            "solved_seeds": [seed for seed, outcome in group if outcome["solved"]],
        }
        for key, group in sorted(rows.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evolve a shared scalar scorer over JEPA-imagined action sequences."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--outcome-checkpoint", required=True)
    parser.add_argument("--predictor", choices=("trained", "random"), default="trained")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--collision-logit-bias", type=float, default=-3.178054)
    parser.add_argument("--episodic-memory-capacity", type=int, default=0)
    parser.add_argument("--episodic-graph-capacity", type=int, default=0)
    parser.add_argument("--graph-match-threshold", type=float, default=0.97)
    parser.add_argument("--fixed-energy-prior", action="store_true")
    parser.add_argument("--neat-residual-scale", type=float, default=0.5)
    parser.add_argument(
        "--controller-kind",
        choices=("sequence-residual", "mode-selector", "frontier-selector"),
        default="sequence-residual",
    )
    parser.add_argument("--mode-safety-margin", type=float, default=2.0)
    parser.add_argument("--commit-return", action="store_true")
    parser.add_argument(
        "--frontier-policy",
        choices=("any-action", "forward-only", "forward-first"),
        default="any-action",
    )
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--training-seed-pool", type=int, default=50)
    parser.add_argument("--training-start", type=int, default=0)
    parser.add_argument("--cases-per-generation", type=int, default=5)
    parser.add_argument("--geometry-balanced-cases", action="store_true")
    parser.add_argument("--validation-start", type=int, default=0)
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument("--holdout-start", type=int, default=50)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--train-rgb-style", choices=("standard", "cyclic"), default="standard")
    parser.add_argument("--holdout-rgb-style", choices=("standard", "cyclic"), default="standard")
    parser.add_argument("--archive-injections", type=int, default=6)
    parser.add_argument("--initial-genome", default="")
    parser.add_argument("--skip-archive-mode-diagnostics", action="store_true")
    parser.add_argument("--winner", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.depth < 1:
        raise ValueError("--depth must be positive")
    if args.training_seed_pool < args.cases_per_generation:
        raise ValueError("--training-seed-pool must be at least --cases-per-generation")
    if args.controller_kind in {"mode-selector", "frontier-selector"}:
        if args.episodic_graph_capacity < 1:
            raise ValueError("--mode-selector requires --episodic-graph-capacity")
        if not args.fixed_energy_prior:
            raise ValueError("--mode-selector requires --fixed-energy-prior")
    if args.commit_return and args.controller_kind not in {"mode-selector", "frontier-selector"}:
        raise ValueError("--commit-return requires a graph selector controller")
    training_range = set(range(args.training_start, args.training_start + args.training_seed_pool))
    validation_range = set(range(args.validation_start, args.validation_start + args.validation_count))
    holdout_range = set(range(args.holdout_start, args.holdout_start + args.holdout_count))
    if args.validation_count > 0 and (
        training_range & validation_range
        or training_range & holdout_range
        or validation_range & holdout_range
    ):
        raise ValueError("training, validation, and holdout seed ranges must be disjoint")

    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()
    if args.predictor == "random":
        jepa.reset_predictor(args.seed + 7919)
    outcome_head = JepaRolloutOutcomeHead.load(ROOT / args.outcome_checkpoint)
    sequences = torch.as_tensor(
        list(itertools.product(CONTROL_ACTIONS, repeat=args.depth)),
        dtype=torch.long,
        device=jepa.device,
    )
    memory_feature_count = args.depth + EpisodicLatentMemory.feature_size if args.episodic_memory_capacity else 0
    graph_feature_count = EpisodicLatentGraph.feature_size if args.episodic_graph_capacity else 0
    scorer_input_count = (
        FRONTIER_FEATURE_COUNT
        if args.controller_kind == "frontier-selector"
        else MODE_FEATURE_COUNT
        if args.controller_kind == "mode-selector"
        else 5 * args.depth + 8 + memory_feature_count + graph_feature_count
    )
    config_path = neat_config_for_inputs(
        scorer_input_count,
        lexicase=True,
        feed_forward=True,
        output_count=(
            5
            if args.controller_kind == "frontier-selector"
            else 3
            if args.controller_kind == "mode-selector"
            else 1
        ),
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
    population = neat.Population(config, seed=args.seed)
    initial_genome = None
    if args.initial_genome:
        initial_genome = pickle.loads((ROOT / args.initial_genome).read_bytes())
        seeded = copy.deepcopy(initial_genome)
        remap_genome_innovations(
            seeded,
            population.population.values(),
            population.reproduction.innovation_tracker,
        )
        seeded_key = min(population.population)
        seeded.key = seeded_key
        seeded.fitness = None
        population.population[seeded_key] = seeded
        population.species.speciate(config, population.population, 0)
    archive = MiniGridMapElites(
        include_modes=args.controller_kind in {"mode-selector", "frontier-selector"}
    )
    population.reproduction.map_elites_archive = archive
    population.reproduction.archive_injections = args.archive_injections
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)
    if not args.quiet:
        population.add_reporter(neat.StdOutReporter(True))

    feature_cache: dict[bytes, np.ndarray] = {}
    case_schedule: list[list[int]] = []
    training_seeds = list(
        range(args.training_start, args.training_start + args.training_seed_pool)
    )
    training_geometry_groups = group_seeds_by_geometry(env, training_seeds)
    generation = 0

    def evaluate_genomes(genomes, neat_config) -> None:
        nonlocal generation
        rng = random.Random(args.seed + 1009 * generation)
        cases = (
            geometry_balanced_sample(
                training_geometry_groups, args.cases_per_generation, rng
            )
            if args.geometry_balanced_cases
            else rng.sample(training_seeds, args.cases_per_generation)
        )
        case_schedule.append(cases)
        for _, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            outcomes = [
                run_scorer_episode(
                    env,
                    jepa,
                    outcome_head,
                    sequences,
                    network,
                    layout_seed,
                    args.max_steps,
                    args.train_rgb_style,
                    args.collision_logit_bias,
                    feature_cache,
                    args.episodic_memory_capacity,
                    args.episodic_graph_capacity,
                    args.fixed_energy_prior,
                    args.neat_residual_scale,
                    args.controller_kind,
                    args.mode_safety_margin,
                    args.commit_return,
                    args.frontier_policy,
                    args.graph_match_threshold,
                )
                for layout_seed in cases
            ]
            genome.fitness = aggregate_layout_fitness(outcomes)
            genome.lexicase_scores = [float(outcome["fitness"]) for outcome in outcomes]
            archive.update(genome, outcomes, genome.fitness, generation)
        generation += 1

    evolution_winner = population.run(evaluate_genomes, args.generations)
    final_training_seeds = (
        list(range(args.validation_start, args.validation_start + args.validation_count))
        if args.validation_count > 0
        else training_seeds[: min(10, len(training_seeds))]
    )
    candidates = (
        ([] if initial_genome is None else [("initial_genome", initial_genome)])
        + [("evolution_winner", evolution_winner)]
        + [(f"archive:{entry.descriptor}", entry.genome) for entry in archive.elites()]
    )
    candidate_results = []
    for source, candidate in candidates:
        network = neat.nn.FeedForwardNetwork.create(candidate, config)
        outcomes = [
            run_scorer_episode(
                env,
                jepa,
                outcome_head,
                sequences,
                network,
                layout_seed,
                args.max_steps,
                args.train_rgb_style,
                args.collision_logit_bias,
                feature_cache,
                args.episodic_memory_capacity,
                args.episodic_graph_capacity,
                args.fixed_energy_prior,
                args.neat_residual_scale,
                args.controller_kind,
                args.mode_safety_margin,
                args.commit_return,
                args.frontier_policy,
                args.graph_match_threshold,
            )
            for layout_seed in final_training_seeds
        ]
        candidate_results.append((aggregate_layout_fitness(outcomes), source, candidate, outcomes))
    robust_fitness, winner_source, winner, training_outcomes = max(
        candidate_results, key=lambda row: row[0]
    )

    holdout_seeds = list(range(args.holdout_start, args.holdout_start + args.holdout_count))
    network = neat.nn.FeedForwardNetwork.create(winner, config)
    holdout_outcomes = [
        run_scorer_episode(
            env,
            jepa,
            outcome_head,
            sequences,
            network,
            layout_seed,
            args.max_steps,
            args.holdout_rgb_style,
            args.collision_logit_bias,
            feature_cache,
            args.episodic_memory_capacity,
            args.episodic_graph_capacity,
            args.fixed_energy_prior,
            args.neat_residual_scale,
            args.controller_kind,
            args.mode_safety_margin,
            args.commit_return,
            args.frontier_policy,
            args.graph_match_threshold,
        )
        for layout_seed in holdout_seeds
    ]
    standard_outcomes = (
        [
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
                args.episodic_memory_capacity,
                args.episodic_graph_capacity,
                args.fixed_energy_prior,
                args.neat_residual_scale,
                args.controller_kind,
                args.mode_safety_margin,
                args.commit_return,
                args.frontier_policy,
                args.graph_match_threshold,
            )
            for layout_seed in holdout_seeds
        ]
        if args.holdout_rgb_style != "standard"
        else holdout_outcomes
    )
    archive_mode_diagnostics = {}
    archive_mode_genomes = {}
    for mode_name, descriptor_index in (() if args.skip_archive_mode_diagnostics else (("return", 5), ("probe", 6))):
        eligible = [
            entry
            for entry in archive.elites()
            if len(entry.descriptor) > descriptor_index and entry.descriptor[descriptor_index] > 0
        ]
        if not eligible:
            continue
        entry = max(
            eligible,
            key=lambda item: (
                item.descriptor[2],
                item.descriptor[descriptor_index],
                item.fitness,
            ),
        )
        diagnostic_network = neat.nn.FeedForwardNetwork.create(entry.genome, config)
        diagnostic_outcomes = [
            run_scorer_episode(
                env,
                jepa,
                outcome_head,
                sequences,
                diagnostic_network,
                layout_seed,
                args.max_steps,
                args.holdout_rgb_style,
                args.collision_logit_bias,
                feature_cache,
                args.episodic_memory_capacity,
                args.episodic_graph_capacity,
                args.fixed_energy_prior,
                args.neat_residual_scale,
                args.controller_kind,
                args.mode_safety_margin,
                args.commit_return,
                args.frontier_policy,
                args.graph_match_threshold,
            )
            for layout_seed in holdout_seeds
        ]
        archive_mode_diagnostics[mode_name] = {
            "training_descriptor": list(entry.descriptor),
            "training_fitness": entry.fitness,
            "generation": entry.generation,
            "holdout_evaluation": evaluation_summary(holdout_seeds, diagnostic_outcomes),
        }
        archive_mode_genomes[mode_name] = entry.genome
    winner_path = ROOT / args.winner
    report_path = ROOT / args.report
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_bytes(pickle.dumps(winner))
    for mode_name, genome in archive_mode_genomes.items():
        mode_path = winner_path.with_name(f"{winner_path.stem}-archive-{mode_name}{winner_path.suffix}")
        mode_path.write_bytes(pickle.dumps(genome))
        archive_mode_diagnostics[mode_name]["winner"] = str(mode_path)
    report = {
        "environment": args.env,
        "controller": (
            "feedforward_neat_graph_frontier_selector"
            if args.controller_kind == "frontier-selector"
            else "feedforward_neat_graph_mode_selector"
            if args.controller_kind == "mode-selector"
            else "feedforward_neat_shared_sequence_scorer"
        ),
        "controller_kind": args.controller_kind,
        "predictor": args.predictor,
        "jepa_checkpoint": str(ROOT / args.checkpoint),
        "outcome_checkpoint": str(ROOT / args.outcome_checkpoint),
        "visual_history_only_at_runtime": True,
        "depth": args.depth,
        "candidate_count": len(sequences),
        "scorer_input_count": scorer_input_count,
        "episodic_memory_capacity": args.episodic_memory_capacity,
        "episodic_memory_feature_count": memory_feature_count,
        "episodic_graph_capacity": args.episodic_graph_capacity,
        "episodic_graph_feature_count": graph_feature_count,
        "graph_match_threshold": args.graph_match_threshold,
        "fixed_energy_prior": args.fixed_energy_prior,
        "neat_residual_scale": args.neat_residual_scale,
        "mode_safety_margin": args.mode_safety_margin,
        "commit_return": args.commit_return,
        "frontier_policy": args.frontier_policy,
        "collision_logit_bias": args.collision_logit_bias,
        "seed": args.seed,
        "generations": args.generations,
        "population": args.population,
        "initial_genome": (
            None if not args.initial_genome else str((ROOT / args.initial_genome).resolve())
        ),
        "training_seed_pool": args.training_seed_pool,
        "training_start": args.training_start,
        "cases_per_generation": args.cases_per_generation,
        "geometry_balanced_cases": args.geometry_balanced_cases,
        "training_geometry_groups": {
            key: seeds for key, seeds in sorted(training_geometry_groups.items())
        },
        "validation_start": args.validation_start,
        "validation_count": args.validation_count,
        "case_schedule": case_schedule,
        "selection": "epsilon_lexicase",
        "map_elites": True,
        "map_elites_archive": archive.summary(),
        "train_rgb_style": args.train_rgb_style,
        "holdout_rgb_style": args.holdout_rgb_style,
        "winner_source": winner_source,
        "winner_fitness": robust_fitness,
        "winner_nodes": len(winner.nodes),
        "winner_connections": len(winner.connections),
        "candidate_selection_evaluation": evaluation_summary(final_training_seeds, training_outcomes),
        "standard_holdout_evaluation": evaluation_summary(holdout_seeds, standard_outcomes),
        "holdout_evaluation": evaluation_summary(holdout_seeds, holdout_outcomes),
        "holdout_by_geometry": evaluation_by_geometry(env, holdout_seeds, holdout_outcomes),
        "archive_mode_diagnostics": archive_mode_diagnostics,
        "fitness_history": [genome.fitness for genome in statistics.most_fit_genomes],
        "winner": str(winner_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=json_scalar), encoding="utf-8")
    env.close()
    print(json.dumps(report["holdout_evaluation"] | {"report": str(report_path)}))
    return 0


def run_scorer_episode(
    env,
    jepa: TemporalRgbJepaAgent,
    outcome_head: JepaRolloutOutcomeHead,
    sequences: torch.Tensor,
    network,
    seed: int,
    max_steps: int,
    rgb_style: str,
    collision_logit_bias: float,
    cache: dict[bytes, np.ndarray],
    episodic_memory_capacity: int = 0,
    episodic_graph_capacity: int = 0,
    fixed_energy_prior: bool = False,
    neat_residual_scale: float = 0.5,
    controller_kind: str = "sequence-residual",
    mode_safety_margin: float = 2.0,
    commit_return: bool = False,
    frontier_policy: str = "any-action",
    graph_match_threshold: float = 0.97,
) -> dict:
    obs, _ = normalize_reset(env.reset(seed=seed))
    first = apply_rgb_style(obs["image"], rgb_style)
    frames = deque([first.copy() for _ in range(jepa.context_length)], maxlen=jepa.context_length)
    actions = deque([-1] * (jepa.context_length - 1), maxlen=jepa.context_length - 1)
    visited_observations: set[bytes] = set()
    visited_positions = {(0, 0)}
    x = y = direction = 0
    blocked_steps = 0
    fitness = 0.0
    episodic_memory = (
        EpisodicLatentMemory(capacity=episodic_memory_capacity)
        if episodic_memory_capacity > 0
        else None
    )
    episodic_graph = (
        EpisodicLatentGraph(
            capacity=episodic_graph_capacity,
            match_threshold=graph_match_threshold,
        )
        if episodic_graph_capacity > 0
        else None
    )
    previous_action = -1
    mode_counts = {name: 0 for name in MODE_NAMES}
    frontier_mode_counts = {strategy: 0 for strategy in FRONTIER_STRATEGIES}
    committed_frontier: int | None = None
    commitment_counts = {"started": 0, "completed": 0, "aborted": 0}

    for step in range(max_steps):
        frames_array = np.stack(frames)
        latent = None
        if episodic_memory is not None or episodic_graph is not None:
            latent = jepa.encode_context(frames_array, list(actions))
            place_latent = jepa.encode_frame(frames_array[-1])
            if episodic_memory is not None:
                episodic_memory.add(place_latent)
            if episodic_graph is not None:
                episodic_graph.observe(place_latent, previous_action)
        key = (
            frames_array.tobytes()
            + np.asarray(actions, dtype=np.int8).tobytes()
            + np.asarray((sequences.shape[1],), dtype=np.int8).tobytes()
            + (episodic_memory.signature() if episodic_memory is not None else b"")
            + (episodic_graph.signature() if episodic_graph is not None else b"")
        )
        candidate_features = cache.get(key)
        if candidate_features is None:
            if latent is None:
                latent = jepa.encode_context(frames_array, list(actions))
            candidate_features = outcome_sequence_features(
                jepa,
                outcome_head,
                latent,
                sequences,
                collision_logit_bias,
                episodic_memory,
                episodic_graph,
            )
            cache[key] = candidate_features
        sequences_array = sequences.detach().cpu().numpy()
        if controller_kind in {"mode-selector", "frontier-selector"}:
            action = None
            selected_mode = "local"
            selected_frontier_mode = None
            if commit_return and committed_frontier is not None:
                path = episodic_graph.path_to_node(committed_frontier)
                if path == ():
                    commitment_counts["completed"] += 1
                    committed_frontier = None
                elif path is None:
                    commitment_counts["aborted"] += 1
                    committed_frontier = None
                else:
                    committed_choice = safe_required_action(
                        candidate_features,
                        sequences_array,
                        path[0],
                        mode_safety_margin,
                    )
                    if committed_choice is None:
                        commitment_counts["aborted"] += 1
                        committed_frontier = None
                    else:
                        action, selected_mode = committed_choice
                        selected_frontier_mode = "committed"
            if action is None:
                if controller_kind == "frontier-selector":
                    (
                        action,
                        selected_mode,
                        return_target,
                        selected_frontier_mode,
                    ) = select_frontier_action(
                        network,
                        candidate_features,
                        sequences_array,
                        episodic_graph,
                        step / max(1, max_steps - 1),
                        mode_safety_margin,
                    )
                else:
                    action, selected_mode, return_target = select_mode_action(
                        network,
                        candidate_features,
                        sequences_array,
                        episodic_graph,
                        step / max(1, max_steps - 1),
                        mode_safety_margin,
                        frontier_policy,
                    )
                if (
                    commit_return
                    and return_target is not None
                    and return_target != episodic_graph.current_node
                ):
                    committed_frontier = return_target
                    commitment_counts["started"] += 1
            mode_counts[selected_mode] += 1
            if selected_frontier_mode in frontier_mode_counts:
                frontier_mode_counts[selected_frontier_mode] += 1
        else:
            evolved_scores = np.asarray(
                [network.activate(features.tolist())[0] for features in candidate_features],
                dtype=np.float64,
            )
            scores = (
                fixed_sequence_energy(
                    candidate_features,
                    sequences.shape[1],
                    sequences_array,
                )
                + float(neat_residual_scale) * evolved_scores
                if fixed_energy_prior
                else evolved_scores
            )
            action = int(sequences[int(np.nanargmax(scores)), 0])
            mode_counts["local"] += 1
        image = apply_rgb_style(obs["image"], rgb_style)
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        next_image = apply_rgb_style(next_obs["image"], rgb_style)
        signature = next_image.tobytes()
        if signature not in visited_observations:
            visited_observations.add(signature)
            fitness += 0.005
        fitness -= 0.001
        blocked = bool(action == 2 and np.array_equal(image, next_image))
        if blocked:
            blocked_steps += 1
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
                fitness += 0.01
        solved = bool(terminated and float(reward) > 0.0)
        if solved:
            fitness += 10.0
        elif terminated:
            fitness -= 1.0
        frames.append(next_image.copy())
        actions.append(action)
        previous_action = action
        obs = next_obs
        if terminated or truncated:
            result = episode_result(
                solved,
                step + 1,
                fitness,
                visited_observations,
                visited_positions,
                blocked_steps,
            )
            result["mode_counts"] = mode_counts
            result["graph_nodes"] = 0 if episodic_graph is None else len(episodic_graph)
            result["return_commitments"] = commitment_counts
            result["frontier_mode_counts"] = frontier_mode_counts
            return result
    result = episode_result(
        False,
        max_steps,
        fitness,
        visited_observations,
        visited_positions,
        blocked_steps,
    )
    result["mode_counts"] = mode_counts
    result["graph_nodes"] = 0 if episodic_graph is None else len(episodic_graph)
    result["return_commitments"] = commitment_counts
    result["frontier_mode_counts"] = frontier_mode_counts
    return result


def selected_frontier(
    graph: EpisodicLatentGraph,
    frontier_policy: str,
) -> tuple[int, int, int] | None:
    if frontier_policy == "any-action":
        return graph.frontier_plan()
    if frontier_policy == "forward-only":
        return graph.frontier_plan((2,))
    if frontier_policy == "forward-first":
        return graph.frontier_plan((2,)) or graph.frontier_plan()
    raise ValueError(f"unknown frontier policy: {frontier_policy}")


def probe_action_groups(
    graph: EpisodicLatentGraph,
    frontier_policy: str,
) -> tuple[tuple[int, ...], ...]:
    if frontier_policy == "any-action":
        return (graph.current_untried_actions(),)
    if frontier_policy == "forward-only":
        return (graph.current_untried_actions((2,)),)
    if frontier_policy == "forward-first":
        return (
            graph.current_untried_actions((2,)),
            graph.current_untried_actions(),
        )
    raise ValueError(f"unknown frontier policy: {frontier_policy}")


def current_frontier_fraction(
    graph: EpisodicLatentGraph,
    frontier_policy: str,
) -> float:
    forward = graph.current_untried_actions((2,))
    if frontier_policy == "forward-only":
        return float(bool(forward))
    all_untried = graph.current_untried_actions()
    if frontier_policy == "any-action":
        return len(all_untried) / float(graph.action_count)
    if frontier_policy == "forward-first":
        return 1.0 if forward else len(all_untried) / float(graph.action_count)
    raise ValueError(f"unknown frontier policy: {frontier_policy}")


def mode_selector_features(
    candidate_features: np.ndarray,
    fixed_scores: np.ndarray,
    graph: EpisodicLatentGraph,
    step_fraction: float,
    frontier_policy: str = "any-action",
) -> np.ndarray:
    """Compact controller state for choosing a graph-navigation primitive."""
    scores = np.asarray(fixed_scores, dtype=np.float64)
    ordered = np.sort(scores)
    best = float(ordered[-1])
    margin = best - float(ordered[-2]) if len(ordered) > 1 else 0.0
    current = graph.nodes[graph.current_node] if graph.current_node is not None else None
    frontier = selected_frontier(graph, frontier_policy)
    graph_features = np.asarray(candidate_features[:, -EpisodicLatentGraph.feature_size :])
    return np.asarray(
        (
            np.tanh(best / 5.0),
            np.tanh(float(np.std(scores)) / 5.0),
            np.tanh(margin / 2.0),
            len(graph) / float(graph.capacity),
            0.0 if current is None else min(1.0, np.log1p(current.visits) / 5.0),
            current_frontier_fraction(graph, frontier_policy),
            float(frontier is not None),
            0.0 if frontier is None else 1.0 / (1.0 + frontier[1]),
            float(np.max(graph_features[:, 0])),
            float(np.max(graph_features[:, 3])),
            float(np.max(graph_features[:, 5])),
            float(np.clip(step_fraction, 0.0, 1.0)),
        ),
        dtype=np.float64,
    )


def select_mode_action(
    network,
    candidate_features: np.ndarray,
    sequences: np.ndarray,
    graph: EpisodicLatentGraph | None,
    step_fraction: float,
    safety_margin: float = 2.0,
    frontier_policy: str = "any-action",
) -> tuple[int, str, int | None]:
    """Choose among local MPC, return-to-frontier, and frontier probing."""
    if graph is None:
        raise ValueError("mode selection requires an episodic graph")
    depth = int(sequences.shape[1])
    fixed_scores = fixed_sequence_energy(candidate_features, depth, sequences)
    local_index = int(np.nanargmax(fixed_scores))
    local_score = float(fixed_scores[local_index])
    candidate_indices: list[int | None] = [local_index, None, None]

    frontier = selected_frontier(graph, frontier_policy)
    if frontier is not None:
        return_matches = np.flatnonzero(sequences[:, 0] == frontier[0])
        if len(return_matches):
            index = int(return_matches[np.argmax(fixed_scores[return_matches])])
            if fixed_scores[index] >= local_score - safety_margin:
                candidate_indices[1] = index

    local_action = int(sequences[local_index, 0])
    for untried in probe_action_groups(graph, frontier_policy):
        alternatives = tuple(action for action in untried if action != local_action)
        probe_matches = np.flatnonzero(np.isin(sequences[:, 0], alternatives))
        if not len(probe_matches):
            continue
        index = int(probe_matches[np.argmax(fixed_scores[probe_matches])])
        if fixed_scores[index] >= local_score - safety_margin:
            candidate_indices[2] = index
            break

    inputs = mode_selector_features(
        candidate_features, fixed_scores, graph, step_fraction, frontier_policy
    )
    logits = np.asarray(network.activate(inputs.tolist()), dtype=np.float64)
    if logits.shape != (3,):
        raise ValueError("mode selector must produce three outputs")
    logits[[index is None for index in candidate_indices]] = -np.inf
    selected_mode = int(np.argmax(logits))
    selected_index = candidate_indices[selected_mode]
    if selected_index is None:
        selected_mode = 0
        selected_index = local_index
    action = int(sequences[selected_index, 0])
    local_action = int(sequences[local_index, 0])
    effective_mode = MODE_NAMES[selected_mode] if action != local_action else "local"
    return_target = frontier[2] if selected_mode == 1 and frontier is not None else None
    return action, effective_mode, return_target


def frontier_selector_features(
    candidate_features: np.ndarray,
    fixed_scores: np.ndarray,
    graph: EpisodicLatentGraph,
    step_fraction: float,
) -> np.ndarray:
    """Mode state plus explicit summaries for three graph frontier proposals."""
    base = mode_selector_features(
        candidate_features,
        fixed_scores,
        graph,
        step_fraction,
        "any-action",
    )
    maximum_visits = max((node.visits for node in graph.nodes), default=1)
    proposal_features = []
    for strategy in FRONTIER_STRATEGIES:
        plan = graph.frontier_plan(strategy=strategy)
        if plan is None:
            proposal_features.extend((0.0, 0.0, 0.0, 0.0))
            continue
        _, distance, target = plan
        node = graph.nodes[target]
        proposal_features.extend(
            (
                1.0,
                1.0 / (1.0 + distance),
                min(1.0, (graph.step - 1 - node.last_seen) / float(graph.capacity)),
                1.0 - node.visits / float(maximum_visits),
            )
        )
    return np.concatenate((base, np.asarray(proposal_features, dtype=np.float64)))


def select_frontier_action(
    network,
    candidate_features: np.ndarray,
    sequences: np.ndarray,
    graph: EpisodicLatentGraph | None,
    step_fraction: float,
    safety_margin: float = 2.0,
) -> tuple[int, str, int | None, str | None]:
    """Choose local, one of three return targets, or a safe probe."""
    if graph is None:
        raise ValueError("frontier selection requires an episodic graph")
    fixed_scores = fixed_sequence_energy(
        candidate_features, int(sequences.shape[1]), sequences
    )
    local_index = int(np.nanargmax(fixed_scores))
    local_score = float(fixed_scores[local_index])
    candidate_indices: list[int | None] = [local_index, None, None, None, None]
    plans = [graph.frontier_plan(strategy=strategy) for strategy in FRONTIER_STRATEGIES]
    for offset, plan in enumerate(plans, start=1):
        if plan is None:
            continue
        matches = np.flatnonzero(sequences[:, 0] == plan[0])
        if not len(matches):
            continue
        index = int(matches[np.argmax(fixed_scores[matches])])
        if fixed_scores[index] >= local_score - safety_margin:
            candidate_indices[offset] = index

    local_action = int(sequences[local_index, 0])
    untried = graph.current_untried_actions()
    alternatives = tuple(action for action in untried if action != local_action)
    probe_matches = np.flatnonzero(np.isin(sequences[:, 0], alternatives))
    if len(probe_matches):
        index = int(probe_matches[np.argmax(fixed_scores[probe_matches])])
        if fixed_scores[index] >= local_score - safety_margin:
            candidate_indices[4] = index

    inputs = frontier_selector_features(
        candidate_features, fixed_scores, graph, step_fraction
    )
    logits = np.asarray(network.activate(inputs.tolist()), dtype=np.float64)
    if logits.shape != (5,):
        raise ValueError("frontier selector must produce five outputs")
    logits[[index is None for index in candidate_indices]] = -np.inf
    selected = int(np.argmax(logits))
    selected_index = candidate_indices[selected]
    if selected_index is None:
        selected = 0
        selected_index = local_index
    action = int(sequences[selected_index, 0])
    effective_mode = (
        "return"
        if selected in (1, 2, 3) and action != local_action
        else "probe"
        if selected == 4 and action != local_action
        else "local"
    )
    strategy = FRONTIER_STRATEGIES[selected - 1] if selected in (1, 2, 3) else None
    target = plans[selected - 1][2] if selected in (1, 2, 3) and plans[selected - 1] else None
    return action, effective_mode, target, strategy


def safe_required_action(
    candidate_features: np.ndarray,
    sequences: np.ndarray,
    required_action: int,
    safety_margin: float = 2.0,
) -> tuple[int, str] | None:
    """Ground a committed graph edge in the fixed JEPA-MPC safety energy."""
    fixed_scores = fixed_sequence_energy(candidate_features, int(sequences.shape[1]), sequences)
    local_index = int(np.nanargmax(fixed_scores))
    local_action = int(sequences[local_index, 0])
    matches = np.flatnonzero(sequences[:, 0] == int(required_action))
    if not len(matches):
        return None
    required_index = int(matches[np.argmax(fixed_scores[matches])])
    if fixed_scores[required_index] < fixed_scores[local_index] - safety_margin:
        return None
    action = int(sequences[required_index, 0])
    return action, "return" if action != local_action else "local"


def fixed_sequence_energy(
    candidate_features: np.ndarray,
    depth: int,
    sequences: np.ndarray | None = None,
) -> np.ndarray:
    """The calibrated MPC energy used as a safe prior for evolved residuals."""
    features = np.asarray(candidate_features, dtype=np.float64)
    energy = np.zeros(features.shape[0], dtype=np.float64)
    for step in range(depth):
        offset = 5 * step
        route_probability = np.clip(features[:, offset], 1e-8, 1.0)
        collision_risk = features[:, offset + 1]
        energy += (0.9**step) * (
            0.25 * np.log(route_probability) - 4.0 * collision_risk
        )
        if sequences is not None:
            energy -= (0.9**step) * (np.asarray(sequences)[:, step] != 2)
    final_offset = 5 * (depth - 1)
    energy += -4.0 * features[:, final_offset + 2] + 3.0 * features[:, final_offset + 3]
    if sequences is None:
        action_fraction_start = 5 * depth + 3
        forward_fraction = features[:, action_fraction_start + 2]
        energy -= depth * (1.0 - forward_fraction)
    return energy


if __name__ == "__main__":
    raise SystemExit(main())
