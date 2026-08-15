from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from minigrid.wrappers import RGBImgPartialObsWrapper

from .episodic_latent_graph import EpisodicLatentGraph
from .evaluate_jepa_mpc import CONTROL_ACTIONS
from .evolve_jepa_mpc_neat import evaluation_by_geometry
from .evolve_rgb_neat import apply_rgb_style, episode_result, evaluation_summary
from .jepa_outcome_value import JepaOutcomeValueHead
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate generic JEPA outcome value over imagined and remembered latents."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--value-checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-outcome-value.pt")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--graph-capacity", type=int, default=128)
    parser.add_argument("--graph-match-threshold", type=float, default=0.97)
    parser.add_argument("--return-margin", type=float, default=0.0)
    parser.add_argument("--remaining-weight", type=float, default=1.0)
    parser.add_argument("--holdout-start", type=int, default=100)
    parser.add_argument("--holdout-count", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--rgb-style", choices=("standard", "cyclic"), default="standard")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()
    value = JepaOutcomeValueHead.load(ROOT / args.value_checkpoint)
    sequences = torch.as_tensor(
        list(itertools.product(CONTROL_ACTIONS, repeat=args.depth)),
        dtype=torch.long,
        device=jepa.device,
    )
    seeds = list(range(args.holdout_start, args.holdout_start + args.holdout_count))
    outcomes = [
        run_episode(
            env, jepa, value, sequences, seed, args.max_steps, args.rgb_style,
            args.graph_capacity, args.graph_match_threshold, args.return_margin,
            args.remaining_weight,
        )
        for seed in seeds
    ]
    summary = evaluation_summary(seeds, outcomes)
    report = {
        "environment": args.env,
        "controller": "generic_jepa_outcome_value_with_episodic_graph",
        "jepa_frozen": True,
        "runtime_inputs": ["visual_history", "actions", "experienced_latent_graph"],
        "semantic_or_map_inputs": False,
        "objective": ["eventual_success", "remaining_steps_fraction"],
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "value_checkpoint": str((ROOT / args.value_checkpoint).resolve()),
        "depth": args.depth,
        "graph_capacity": args.graph_capacity,
        "graph_match_threshold": args.graph_match_threshold,
        "return_margin": args.return_margin,
        "remaining_weight": args.remaining_weight,
        "holdout_evaluation": summary,
        "holdout_by_geometry": evaluation_by_geometry(env, seeds, outcomes),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({"solved": summary["solved"], "success_rate": summary["success_rate"], "report": str(report_path)}))
    return 0


@torch.no_grad()
def score_candidates(value, belief, candidates, distances, remaining_weight):
    count = candidates.shape[0]
    outputs = value(
        belief.expand(count, *belief.shape[1:]),
        candidates,
        distances,
    )
    return outputs["success_logit"] - float(remaining_weight) * outputs["remaining"]


@torch.no_grad()
def imagined_endpoints(jepa, belief, sequences):
    predicted = belief.expand(sequences.shape[0], *belief.shape[1:]).contiguous()
    for step in range(sequences.shape[1]):
        predicted = jepa.predictor(predicted, sequences[:, step])
    return predicted


def run_episode(
    env, jepa, value, sequences, seed, max_steps, rgb_style, graph_capacity,
    graph_match_threshold, return_margin, remaining_weight,
):
    obs, _ = normalize_reset(env.reset(seed=int(seed)))
    first = apply_rgb_style(obs["image"], rgb_style)
    frames = deque([first.copy() for _ in range(jepa.context_length)], maxlen=jepa.context_length)
    actions = deque([-1] * (jepa.context_length - 1), maxlen=jepa.context_length - 1)
    graph = (
        EpisodicLatentGraph(graph_capacity, match_threshold=graph_match_threshold)
        if graph_capacity > 0
        else None
    )
    previous_action = -1
    committed_target = None
    visited_observations = set()
    visited_positions = {(0, 0)}
    x = y = direction = 0
    blocked_steps = 0
    fitness = 0.0
    mode_counts = {"local": 0, "return": 0, "probe": 0}
    commitments = {"started": 0, "completed": 0, "aborted": 0}
    sequence_array = sequences.cpu().numpy()

    for step in range(max_steps):
        frames_array = np.stack(frames)
        belief = jepa.encode_context(frames_array, list(actions))
        place = jepa.encode_frame(frames_array[-1])
        if graph is not None:
            graph.observe(place, previous_action)
        action = None
        mode = "local"
        if committed_target is not None and graph is not None:
            path = graph.path_to_node(committed_target)
            if path == ():
                commitments["completed"] += 1
                committed_target = None
            elif path is None:
                commitments["aborted"] += 1
                committed_target = None
            else:
                action = int(path[0])
                mode = "return"

        if action is None:
            endpoints = imagined_endpoints(jepa, belief, sequences)
            local_distances = torch.full(
                (len(sequences),), sequences.shape[1] / float(max_steps), device=jepa.device
            )
            local_scores = score_candidates(
                value, belief, endpoints, local_distances, remaining_weight
            )
            local_index = int(local_scores.argmax())
            action = int(sequence_array[local_index, 0])
            best_local = float(local_scores[local_index])

            plans = [] if graph is None else graph.frontier_plans()
            if plans:
                targets = torch.stack([graph.nodes[target].prototype for _, _, target in plans]).to(jepa.device)
                graph_distances = torch.as_tensor(
                    [distance / float(max_steps) for _, distance, _ in plans],
                    dtype=torch.float32,
                    device=jepa.device,
                )
                graph_scores = score_candidates(
                    value, belief, targets, graph_distances, remaining_weight
                )
                best_graph = int(graph_scores.argmax())
                if float(graph_scores[best_graph]) > best_local + float(return_margin):
                    first_action, _, target = plans[best_graph]
                    action = int(first_action)
                    committed_target = int(target)
                    commitments["started"] += 1
                    mode = "return"

        mode_counts[mode] += 1
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
            result = episode_result(solved, step + 1, fitness, visited_observations, visited_positions, blocked_steps)
            result["mode_counts"] = mode_counts
            result["graph_nodes"] = 0 if graph is None else len(graph)
            result["return_commitments"] = commitments
            return result
    result = episode_result(False, max_steps, fitness, visited_observations, visited_positions, blocked_steps)
    result["mode_counts"] = mode_counts
    result["graph_nodes"] = 0 if graph is None else len(graph)
    result["return_commitments"] = commitments
    return result


if __name__ == "__main__":
    raise SystemExit(main())
