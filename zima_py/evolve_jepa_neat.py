from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import neat
import numpy as np

from .jepa_control_agent import ControlTransition, JepaControlAgent, belief_observation
from .embodied_map import BASE_FEATURE_SIZE, FULL_FEATURE_SIZE, EmbodiedMap
from .evolution_replay import BalancedEvolutionReplay
from .lexicase_reproduction import LexicaseReproduction
from .map_elites_archive import MiniGridMapElites
from .minimal_jepa import MinimalJepaAgent
from .minigrid_adapter import MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step
from .skill_runtime import MiniGridSkillApi
from .train_jepa_minigrid import update_carrying, useful_actions


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolve a recurrent NEAT controller over a frozen JEPA world model.")
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-seed", type=int, action="append", default=[], help="DoorKey layout seed used for fitness; repeat to train on several layouts.")
    parser.add_argument("--holdout-seed", type=int, action="append", default=[], help="Unseen seed used only after evolution.")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument(
        "--force-generations",
        action="store_true",
        help="Disable NEAT's fitness-threshold early stop for matched-budget experiments.",
    )
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--jepa-checkpoint", default="artifacts/jepa/doorkey-controller-seed19-v2.pt")
    parser.add_argument("--winner", default="artifacts/jepa/doorkey-neat-winner.pkl")
    parser.add_argument("--report", default="artifacts/jepa/doorkey-neat-report.json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--raw-input", action="store_true", help="Evolve NEAT directly from raw body observations; do not use JEPA or the embodied map.")
    parser.add_argument(
        "--minimal-jepa",
        action="store_true",
        help="Use the 16-dimensional prediction-only JEPA without reward or Q heads.",
    )
    parser.add_argument("--lexicase", action="store_true", help="Select parents with epsilon-lexicase over per-layout episode scores.")
    parser.add_argument("--random-training-seeds", action="store_true", help="Sample fresh layout cases from a seed pool every generation.")
    parser.add_argument("--training-seed-pool", type=int, default=50, help="Use training layout seeds in range [0, N).")
    parser.add_argument("--cases-per-generation", type=int, default=5)
    parser.add_argument("--map-elites", action="store_true", help="Archive behavioral elites by key/door/goal counts and reinject them.")
    parser.add_argument("--archive-injections", type=int, default=6)
    parser.add_argument(
        "--terrain-map",
        action="store_true",
        help="Add a persistent observation-only free/wall/lava map to the NEAT controller input.",
    )
    parser.add_argument(
        "--jepa-block-generations",
        type=int,
        default=0,
        help="Retrain JEPA between frozen blocks of this many generations; zero disables adaptation.",
    )
    parser.add_argument("--jepa-updates-per-block", type=int, default=200)
    parser.add_argument("--jepa-replay-per-block", type=int, default=4096)
    parser.add_argument(
        "--adapted-jepa-checkpoint",
        default="",
        help="Where to save the blockwise-adapted JEPA; defaults beside the winner.",
    )
    args = parser.parse_args()
    training_seeds = args.train_seed or [0, 1, 2, 3, 4]
    holdout_seeds = args.holdout_seed or (
        list(range(args.training_seed_pool, args.training_seed_pool + 20))
        if args.random_training_seeds
        else [10, 11, 12, 13, 14]
    )
    if args.map_elites and not args.lexicase:
        raise ValueError("--map-elites currently requires --lexicase so archived elites use the compatible reproduction path.")
    if args.raw_input and args.minimal_jepa:
        raise ValueError("--raw-input and --minimal-jepa are mutually exclusive.")
    if args.raw_input and args.jepa_block_generations:
        raise ValueError("Blockwise JEPA adaptation requires JEPA input mode.")
    if args.jepa_block_generations < 0:
        raise ValueError("--jepa-block-generations must be zero or positive.")

    env = make_minigrid(MiniGridSpec(env_id=args.env, seed=args.seed, max_steps=args.max_steps))
    agent = (
        MinimalJepaAgent(env.action_space.n, seed=args.seed)
        if args.minimal_jepa
        else JepaControlAgent(env.action_space.n, seed=args.seed)
    )
    checkpoint = ROOT / args.jepa_checkpoint
    if not args.raw_input:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Frozen JEPA checkpoint not found: {checkpoint}")
        agent.load(checkpoint)
        if args.minimal_jepa:
            agent.eval()
        else:
            agent.encoder.eval()
            agent.target_encoder.eval()
            agent.predictor.eval()
            agent.reward_head.eval()
            agent.q_head.eval()

    include_body_map = args.terrain_map or (not args.raw_input and not args.minimal_jepa)
    map_feature_count = FULL_FEATURE_SIZE if args.terrain_map else BASE_FEATURE_SIZE if include_body_map else 0
    learned_feature_count = (
        160
        if args.raw_input
        else agent.world_feature_size
        if args.minimal_jepa
        else agent.latent_dim + agent.action_count * 2
    )
    controller_input_count = learned_feature_count + map_feature_count
    config_path = neat_config_for_inputs(controller_input_count, lexicase=args.lexicase)
    reproduction_type = LexicaseReproduction if args.lexicase else neat.DefaultReproduction
    config = neat.Config(
        neat.DefaultGenome,
        reproduction_type,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    if args.force_generations:
        config.no_fitness_termination = True
    config.pop_size = int(args.population)
    population = neat.Population(config, seed=args.seed)
    archive = MiniGridMapElites() if args.map_elites else None
    if archive is not None:
        population.reproduction.map_elites_archive = archive
        population.reproduction.archive_injections = args.archive_injections
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    if not args.quiet:
        population.add_reporter(neat.StdOutReporter(True))

    feature_cache: dict[tuple[bytes, int, int, bool], np.ndarray] = {}
    replay_collector = (
        BalancedEvolutionReplay(seed=args.seed) if args.jepa_block_generations > 0 else None
    )
    generation_index = 0
    case_schedule: list[list[int]] = []
    jepa_retraining_history: list[dict[str, Any]] = []

    def retrain_jepa_between_blocks() -> None:
        if replay_collector is None:
            return
        replay_stats = replay_collector.feed(agent, args.jepa_replay_per_block)
        agent.encoder.train()
        agent.predictor.train()
        if not args.minimal_jepa:
            agent.q_head.train()
            agent.reward_head.train()
        metrics = None
        completed_updates = 0
        for _ in range(max(0, args.jepa_updates_per_block)):
            current = agent.train_step()
            if current is not None:
                completed_updates += 1
                metrics = current
        if args.minimal_jepa:
            agent.eval()
        else:
            agent.encoder.eval()
            agent.target_encoder.eval()
            agent.predictor.eval()
            agent.q_head.eval()
            agent.reward_head.eval()
        feature_cache.clear()
        jepa_retraining_history.append(
            {
                "after_generation": generation_index,
                "completed_updates": completed_updates,
                "replay": replay_stats,
                "metrics": metrics or {},
            }
        )

    def evaluate_genomes(genomes, neat_config) -> None:
        nonlocal generation_index
        if (
            replay_collector is not None
            and generation_index > 0
            and generation_index % args.jepa_block_generations == 0
        ):
            retrain_jepa_between_blocks()
        if args.random_training_seeds:
            case_rng = random.Random(args.seed + generation_index * 1009)
            generation_seeds = case_rng.sample(
                range(args.training_seed_pool), min(args.cases_per_generation, args.training_seed_pool)
            )
        else:
            generation_seeds = training_seeds
        case_schedule.append(list(generation_seeds))
        for _, genome in genomes:
            outcomes = []
            for layout_seed in generation_seeds:
                network = neat.nn.RecurrentNetwork.create(genome, neat_config)
                outcomes.append(
                    run_controller_episode(
                        env,
                        agent,
                        network,
                        layout_seed,
                        args.max_steps,
                        feature_cache,
                        raw_input=args.raw_input,
                        terrain_map=args.terrain_map,
                        include_body_map=include_body_map,
                        replay_collector=replay_collector,
                    )
                )
            genome.fitness = aggregate_layout_fitness(outcomes)
            genome.lexicase_scores = [float(outcome["fitness"]) for outcome in outcomes]
            if archive is not None:
                archive.update(genome, outcomes, genome.fitness, generation_index)
        generation_index += 1

    evolution_winner = population.run(evaluate_genomes, args.generations)
    final_training_seeds = (
        list(range(min(10, args.training_seed_pool))) if args.random_training_seeds else training_seeds
    )
    candidate_genomes = [("evolution_winner", evolution_winner)]
    if archive is not None:
        candidate_genomes.extend(
            (f"archive:{entry.descriptor}", entry.genome) for entry in archive.elites()
        )
    candidate_results = []
    for source, candidate in candidate_genomes:
        outcomes = []
        for layout_seed in final_training_seeds:
            network = neat.nn.RecurrentNetwork.create(candidate, config)
            outcomes.append(
                run_controller_episode(
                    env,
                    agent,
                    network,
                    layout_seed,
                    args.max_steps,
                    feature_cache,
                    raw_input=args.raw_input,
                    terrain_map=args.terrain_map,
                    include_body_map=include_body_map,
                )
            )
        robust_fitness = aggregate_layout_fitness(outcomes)
        candidate_results.append((robust_fitness, source, candidate, outcomes))
    robust_fitness, winner_source, winner, selected_training_outcomes = max(
        candidate_results, key=lambda item: item[0]
    )
    winner_path = ROOT / args.winner
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_bytes(pickle.dumps(winner))
    adapted_checkpoint = None
    if replay_collector is not None:
        adapted_checkpoint = (
            ROOT / args.adapted_jepa_checkpoint
            if args.adapted_jepa_checkpoint
            else winner_path.with_suffix(".jepa.pt")
        )
        agent.save(adapted_checkpoint)

    training_evaluations = []
    for layout_seed, outcome in zip(final_training_seeds, selected_training_outcomes):
        training_evaluations.append(
            {
                "seed": layout_seed,
                **outcome,
            }
        )
    evaluations = []
    for layout_seed in holdout_seeds:
        network = neat.nn.RecurrentNetwork.create(winner, config)
        evaluations.append(
            {
                "seed": layout_seed,
                **run_controller_episode(
                    env,
                    agent,
                    network,
                    layout_seed,
                    args.max_steps,
                    feature_cache,
                    raw_input=args.raw_input,
                    terrain_map=args.terrain_map,
                    include_body_map=include_body_map,
                ),
            }
        )
    solved = [row for row in evaluations if row["solved"]]
    training_solved = [row for row in training_evaluations if row["solved"]]
    report = {
        "environment": args.env,
        "seed": args.seed,
        "generations": args.generations,
        "generations_completed": len(case_schedule),
        "force_generations": args.force_generations,
        "population": args.population,
        "training_seeds": final_training_seeds,
        "random_training_seeds": args.random_training_seeds,
        "training_seed_pool": args.training_seed_pool if args.random_training_seeds else None,
        "cases_per_generation": len(case_schedule[0]) if case_schedule else 0,
        "case_schedule": case_schedule,
        "holdout_seeds": holdout_seeds,
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
        "controller_input_count": controller_input_count,
        "minimal_jepa": args.minimal_jepa,
        "selection": "epsilon_lexicase" if args.lexicase else "standard_neat",
        "map_elites": args.map_elites,
        "map_elites_archive": [] if archive is None else archive.summary(),
        "frozen_jepa_checkpoint": None if args.raw_input else str(checkpoint),
        "adapted_jepa_checkpoint": None if adapted_checkpoint is None else str(adapted_checkpoint),
        "jepa_block_generations": args.jepa_block_generations,
        "jepa_retraining_history": jepa_retraining_history,
        "evolution_winner_fitness": evolution_winner.fitness,
        "winner_source": winner_source,
        "winner_fitness": robust_fitness,
        "archive_candidate_count": len(candidate_genomes) - 1,
        "winner_nodes": len(winner.nodes),
        "winner_connections": len(winner.connections),
        "training_evaluation": {
            "episodes": len(training_evaluations),
            "solved": len(training_solved),
            "success_rate": len(training_solved) / len(training_evaluations),
            "runs": training_evaluations,
        },
        "holdout_evaluation": {
            "episodes": len(evaluations),
            "solved": len(solved),
            "success_rate": len(solved) / len(evaluations),
            "mean_solved_steps": None if not solved else sum(row["steps"] for row in solved) / len(solved),
            "runs": evaluations,
        },
        "fitness_history": [genome.fitness for genome in stats.most_fit_genomes],
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps(report["holdout_evaluation"] | {"winner_fitness": robust_fitness, "report": str(report_path)}))
    return 0 if solved else 1


def aggregate_layout_fitness(outcomes: list[dict[str, Any]]) -> float:
    """Reward average competence while penalizing seed-specific specialists."""
    fitnesses = [float(outcome["fitness"]) for outcome in outcomes]
    success_rate = sum(bool(outcome["solved"]) for outcome in outcomes) / len(outcomes)
    mean_fitness = sum(fitnesses) / len(fitnesses)
    worst_fitness = min(fitnesses)
    return mean_fitness + 0.25 * worst_fitness + 2.0 * success_rate


def neat_config_for_inputs(
    input_count: int,
    lexicase: bool = False,
    *,
    feed_forward: bool = False,
    output_count: int = 7,
) -> Path:
    template = Path(__file__).with_name("neat_jepa_config.ini").read_text(encoding="utf-8")
    generated = template.replace("num_inputs              = 78", f"num_inputs              = {int(input_count)}")
    generated = generated.replace("num_outputs             = 7", f"num_outputs             = {int(output_count)}")
    if feed_forward:
        generated = generated.replace("feed_forward            = False", "feed_forward            = True")
    if lexicase:
        generated = generated.replace("[DefaultReproduction]", "[LexicaseReproduction]")
    suffix = "_lexicase" if lexicase else ""
    controller_suffix = "_feedforward" if feed_forward else ""
    output_suffix = "" if output_count == 7 else f"_out{int(output_count)}"
    # Include the process ID so matched experiments can evolve concurrently
    # without observing a partially-written shared config file.
    path = ROOT / "artifacts" / "jepa" / (
        f"neat_config_{int(input_count)}{suffix}{controller_suffix}{output_suffix}"
        f"_pid{os.getpid()}.ini"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated, encoding="utf-8")
    return path


def run_controller_episode(
    env,
    agent: Any,
    network,
    seed: int,
    max_steps: int,
    feature_cache: dict[tuple[bytes, int, int, bool], np.ndarray] | None = None,
    *,
    raw_input: bool = False,
    terrain_map: bool = False,
    include_body_map: bool = True,
    replay_collector: BalancedEvolutionReplay | None = None,
) -> dict[str, Any]:
    obs, _ = normalize_reset(env.reset(seed=seed))
    memory: dict[str, Any] = {}
    previous_action = -1
    carrying = False
    pickup_rewarded = False
    door_rewarded = False
    visited: set[bytes] = set()
    visited_positions: set[tuple[int, int]] = {(0, 0)}
    fitness = 0.0
    blocked_steps = 0
    body_map = EmbodiedMap()
    for step in range(max_steps):
        encoded = encode_observation(obs)
        api = MiniGridSkillApi(env, memory)
        front_before = api.front_object(encoded)
        body_map.observe(api.egocentric_cells(encoded), int(encoded.get("direction", body_map.direction)))
        state = belief_observation(encoded, previous_action, carrying)
        cache_key = (state.image.tobytes(), state.direction, state.previous_action, state.carrying)
        if raw_input:
            controller_features = raw_body_features(state, agent.action_count)
            if include_body_map:
                controller_features = np.concatenate(
                    (
                        controller_features,
                        body_map.features(carrying, door_rewarded, include_terrain=terrain_map),
                    )
                )
        elif feature_cache is not None and cache_key in feature_cache:
            features = feature_cache[cache_key]
            controller_features = (
                np.concatenate(
                    (features, body_map.features(carrying, door_rewarded, include_terrain=terrain_map))
                )
                if include_body_map
                else features
            )
        else:
            features = agent.world_features(state)
            if feature_cache is not None:
                feature_cache[cache_key] = features
            controller_features = (
                np.concatenate(
                    (features, body_map.features(carrying, door_rewarded, include_terrain=terrain_map))
                )
                if include_body_map
                else features
            )
        outputs = np.asarray(network.activate(controller_features.tolist()), dtype=np.float64)
        valid = useful_actions(api.actions, front_before)
        invalid = np.ones(agent.action_count, dtype=bool)
        invalid[valid] = False
        outputs[invalid] = -np.inf
        action = int(outputs.argmax())
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        next_encoded = encode_observation(next_obs)
        front_after = api.front_object(next_encoded)
        next_carrying = update_carrying(carrying, action, api.actions, front_before, front_after)

        signature = next_encoded["image"].tobytes() + bytes([max(0, int(next_encoded.get("direction", 0)))])
        if signature not in visited:
            visited.add(signature)
            fitness += 0.005
        fitness -= 0.001
        blocked = bool(
            action == api.actions["forward"]
            and np.array_equal(encoded["image"], next_encoded["image"])
            and encoded.get("direction") == next_encoded.get("direction")
        )
        if blocked:
            fitness -= 0.01
            blocked_steps += 1
        body_map.advance(
            next((name for name, value in api.actions.items() if value == action), str(action)),
            blocked,
            int(next_encoded.get("direction", body_map.direction)),
        )
        position = (body_map.x, body_map.y)
        new_position = position not in visited_positions
        if new_position:
            visited_positions.add(position)
            fitness += 0.01
        if not pickup_rewarded and not carrying and next_carrying:
            fitness += 1.0
            pickup_rewarded = True
        door_event = bool(
            action == api.actions["toggle"]
            and front_before.get("object") == "door"
            and front_before.get("state_name") in {"locked", "closed"}
            and front_after.get("state_name") == "open"
        )
        if door_event and not door_rewarded:
            fitness += 2.0
            door_rewarded = True
        solved = bool(terminated and float(reward) > 0.0)
        if solved:
            fitness += 10.0
        elif terminated:
            fitness -= 1.0

        if replay_collector is not None:
            learning_reward = float(reward) - 0.001 + (0.01 if new_position else 0.0)
            if terminated and not solved:
                learning_reward -= 1.0
            transition = ControlTransition(
                state=state,
                action=action,
                reward=learning_reward,
                next_state=belief_observation(next_encoded, action, next_carrying),
                terminal=bool(terminated or truncated),
            )
            lava_index = api.object_to_idx.get("lava", -1)
            sees_lava = bool(lava_index >= 0 and np.any(state.image[..., 0] == lava_index))
            replay_collector.add(
                transition,
                priority=bool(new_position or sees_lava or terminated or truncated),
            )

        obs = next_obs
        previous_action = action
        carrying = next_carrying
        if terminated or truncated:
            return {
                "solved": solved,
                "steps": step + 1,
                "fitness": round(fitness, 6),
                "unique_observations": len(visited),
                "unique_positions": len(visited_positions),
                "blocked_steps": blocked_steps,
                "collected": pickup_rewarded,
                "opened_door": door_rewarded,
            }
    return {
        "solved": False,
        "steps": max_steps,
        "fitness": round(fitness, 6),
        "unique_observations": len(visited),
        "unique_positions": len(visited_positions),
        "blocked_steps": blocked_steps,
        "collected": pickup_rewarded,
        "opened_door": door_rewarded,
    }


def raw_body_features(state, action_count: int) -> np.ndarray:
    image = np.asarray(state.image, dtype=np.float64)
    direction = np.zeros(4, dtype=np.float64)
    if 0 <= state.direction < 4:
        direction[state.direction] = 1.0
    previous_action = np.zeros(action_count + 1, dtype=np.float64)
    previous_index = state.previous_action if 0 <= state.previous_action < action_count else action_count
    previous_action[previous_index] = 1.0
    return np.concatenate(
        (
            image[..., 0].reshape(-1) / 10.0,
            image[..., 1].reshape(-1) / 5.0,
            image[..., 2].reshape(-1) / 2.0,
            direction,
            previous_action,
            np.asarray([float(state.carrying)], dtype=np.float64),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
