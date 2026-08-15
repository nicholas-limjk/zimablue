from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import RGBImgPartialObsWrapper

from .evolve_rgb_neat import apply_rgb_style, evaluation_summary
from .episodic_latent_memory import EpisodicLatentMemory
from .episodic_latent_graph import EpisodicLatentGraph
from .jepa_belief import JepaBeliefHead, load_belief_head
from .jepa_rollout_outcome import JepaRolloutOutcomeHead
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ACTIONS = (0, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate short-horizon JEPA model-predictive control from visual history only."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--belief-checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-belief-v2.pt")
    parser.add_argument(
        "--outcome-checkpoint",
        default="",
        help="Optional head calibrated directly on JEPA-imagined endpoints.",
    )
    parser.add_argument("--predictor", choices=("trained", "random"), default="trained")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--holdout-start", type=int, default=50)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--rgb-style", choices=("standard", "cyclic"), default="standard")
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=4.0)
    parser.add_argument(
        "--collision-logit-bias",
        type=float,
        default=0.0,
        help="Calibration offset; use -log(pos_weight) for a weighted-BCE outcome head.",
    )
    parser.add_argument("--distance-weight", type=float, default=2.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--exploration-weight", type=float, default=0.25)
    parser.add_argument("--turn-cost", type=float, default=0.0)
    parser.add_argument("--episodic-memory-capacity", type=int, default=0)
    parser.add_argument("--novelty-weight", type=float, default=0.0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.depth < 1:
        raise ValueError("--depth must be positive")

    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()
    if args.predictor == "random":
        jepa.reset_predictor(args.seed + 7919)
    outcome_head = (
        JepaRolloutOutcomeHead.load(ROOT / args.outcome_checkpoint)
        if args.outcome_checkpoint
        else None
    )
    belief = None if outcome_head is not None else load_belief_head(ROOT / args.belief_checkpoint)
    if belief is not None and not isinstance(belief, JepaBeliefHead):
        raise ValueError("JEPA MPC currently requires the GRU belief checkpoint")
    if belief is not None:
        belief.eval()

    sequences = torch.as_tensor(
        list(itertools.product(CONTROL_ACTIONS, repeat=args.depth)),
        dtype=torch.long,
        device=jepa.device,
    )
    weights = {
        "route": args.route_weight,
        "collision": args.collision_weight,
        "collision_logit_bias": args.collision_logit_bias,
        "distance": args.distance_weight,
        "progress": args.progress_weight,
        "exploration": args.exploration_weight,
        "turn_cost": args.turn_cost,
        "novelty": args.novelty_weight,
    }
    seeds = list(range(args.holdout_start, args.holdout_start + args.holdout_count))
    outcomes = [
        run_mpc_episode(
            env,
            jepa,
            belief,
            outcome_head,
            sequences,
            layout_seed,
            args.max_steps,
            args.rgb_style,
            weights,
            args.episodic_memory_capacity,
        )
        for layout_seed in seeds
    ]
    summary = evaluation_summary(seeds, outcomes)
    report = {
        "environment": args.env,
        "controller": "exhaustive_latent_mpc",
        "visual_history_only_at_runtime": True,
        "predictor": args.predictor,
        "checkpoint": str(ROOT / args.checkpoint),
        "belief_checkpoint": None if belief is None else str(ROOT / args.belief_checkpoint),
        "outcome_checkpoint": (
            None if outcome_head is None else str(ROOT / args.outcome_checkpoint)
        ),
        "depth": args.depth,
        "candidate_count": int(len(sequences)),
        "rgb_style": args.rgb_style,
        "weights": weights,
        "episodic_memory_capacity": args.episodic_memory_capacity,
        "evaluation": summary,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(
        json.dumps(
            {
                "predictor": args.predictor,
                "solved": summary["solved"],
                "success_rate": summary["success_rate"],
                "solved_seeds": [row["seed"] for row in summary["runs"] if row["solved"]],
                "report": str(report_path),
            }
        )
    )
    return 0


@torch.no_grad()
def choose_mpc_action(
    jepa: TemporalRgbJepaAgent,
    belief: JepaBeliefHead,
    latent: torch.Tensor,
    current_outputs: dict[str, torch.Tensor],
    current_hidden: torch.Tensor,
    sequences: torch.Tensor,
    weights: dict[str, float],
) -> tuple[int, dict[str, float]]:
    count = sequences.shape[0]
    predicted = latent.expand(count, -1, -1, -1).contiguous()
    hidden = current_hidden.expand(-1, count, -1).contiguous()
    outputs = {key: value.expand(count, -1, -1) for key, value in current_outputs.items()}
    scores = latent.new_zeros(count)
    total_risk = latent.new_zeros(count)

    for step in range(sequences.shape[1]):
        actions = sequences[:, step]
        row = torch.arange(count, device=actions.device)
        route = F.log_softmax(outputs["route_logits"][:, 0], dim=-1)[row, actions]
        collision = torch.sigmoid(outputs["collision_logits"][:, 0])[row, actions]
        exploration = torch.sigmoid(outputs["exploration_logits"][:, 0])[row, actions]
        discount = 0.9**step
        scores += discount * (
            weights["route"] * route
            - weights["collision"] * collision
            + weights["exploration"] * exploration
        )
        total_risk = torch.maximum(total_risk, collision)
        predicted = jepa.predictor(predicted, actions)
        outputs, hidden = belief(predicted.flatten(1).unsqueeze(1), actions.unsqueeze(1), hidden)

    endpoint_goal = outputs["goal"][:, 0]
    scores += (
        -weights["distance"] * endpoint_goal[:, 0]
        + weights["progress"] * endpoint_goal[:, 3]
    )
    best = int(scores.argmax())
    return int(sequences[best, 0]), {
        "score": float(scores[best]),
        "risk": float(total_risk[best]),
        "predicted_distance": float(endpoint_goal[best, 0]),
        "predicted_progress": float(endpoint_goal[best, 3]),
    }


@torch.no_grad()
def choose_outcome_mpc_action(
    jepa: TemporalRgbJepaAgent,
    outcome_head: JepaRolloutOutcomeHead,
    latent: torch.Tensor,
    sequences: torch.Tensor,
    weights: dict[str, float],
    episodic_memory: EpisodicLatentMemory | None = None,
) -> tuple[int, dict[str, float]]:
    count = sequences.shape[0]
    predicted = latent.expand(count, -1, -1, -1).contiguous()
    outputs = outcome_head(
        predicted,
        torch.zeros(count, dtype=torch.long, device=predicted.device),
    )
    scores = latent.new_zeros(count)
    total_risk = latent.new_zeros(count)
    row = torch.arange(count, device=predicted.device)
    for step in range(sequences.shape[1]):
        actions = sequences[:, step]
        route = F.log_softmax(outputs["route_logits"], dim=-1)[row, actions]
        collision = torch.sigmoid(
            outputs["collision_logits"] + weights["collision_logit_bias"]
        )[row, actions]
        scores += (0.9**step) * (
            weights["route"] * route
            - weights["collision"] * collision
            - weights["turn_cost"] * (actions != 2).to(dtype=scores.dtype)
        )
        total_risk = torch.maximum(total_risk, collision)
        predicted = jepa.predictor(predicted, actions)
        if episodic_memory is not None:
            novelty = episodic_memory.features(predicted)[:, 0]
            scores += (0.9**step) * weights["novelty"] * novelty
        outputs = outcome_head(
            predicted,
            torch.full((count,), step + 1, dtype=torch.long, device=predicted.device),
        )
    endpoint_goal = outputs["goal"]
    scores += (
        -weights["distance"] * endpoint_goal[:, 0]
        + weights["progress"] * endpoint_goal[:, 1]
    )
    best = int(scores.argmax())
    diagnostic = {
        "score": float(scores[best]),
        "risk": float(total_risk[best]),
        "predicted_distance": float(endpoint_goal[best, 0]),
        "predicted_progress": float(endpoint_goal[best, 1]),
    }
    for action in CONTROL_ACTIONS:
        diagnostic[f"best_score_action_{action}"] = float(
            scores[sequences[:, 0] == action].max()
        )
    return int(sequences[best, 0]), diagnostic


@torch.no_grad()
def outcome_sequence_features(
    jepa: TemporalRgbJepaAgent,
    outcome_head: JepaRolloutOutcomeHead,
    latent: torch.Tensor,
    sequences: torch.Tensor,
    collision_logit_bias: float,
    episodic_memory: EpisodicLatentMemory | None = None,
    episodic_graph: EpisodicLatentGraph | None = None,
) -> np.ndarray:
    """Return ordered, compact predicted consequences for every candidate."""
    count = sequences.shape[0]
    predicted = latent.expand(count, -1, -1, -1).contiguous()
    outputs = outcome_head(
        predicted,
        torch.zeros(count, dtype=torch.long, device=predicted.device),
    )
    rows = torch.arange(count, device=predicted.device)
    parts: list[torch.Tensor] = []
    memory_parts: list[torch.Tensor] = []
    for step in range(sequences.shape[1]):
        actions = sequences[:, step]
        route = torch.softmax(outputs["route_logits"], dim=-1)[rows, actions]
        collision = torch.sigmoid(
            outputs["collision_logits"] + float(collision_logit_bias)
        )[rows, actions]
        previous = predicted
        predicted = jepa.predictor(predicted, actions)
        latent_change = (predicted - previous).abs().flatten(1).mean(dim=1)
        if episodic_memory is not None:
            memory_parts.append(episodic_memory.features(predicted)[:, 0:1])
        outputs = outcome_head(
            predicted,
            torch.full((count,), step + 1, dtype=torch.long, device=predicted.device),
        )
        parts.extend(
            (
                route.unsqueeze(1),
                collision.unsqueeze(1),
                outputs["goal"][:, 0:1],
                outputs["goal"][:, 1:2],
                latent_change.unsqueeze(1),
            )
        )

    first_action = F.one_hot(sequences[:, 0], num_classes=3).float()
    action_fraction = F.one_hot(sequences, num_classes=3).float().mean(dim=1)
    endpoint_similarity = F.cosine_similarity(predicted.flatten(1), latent.expand_as(predicted).flatten(1)).unsqueeze(1)
    endpoint_change = (predicted - latent).abs().flatten(1).mean(dim=1, keepdim=True)
    endpoint_memory = (
        episodic_memory.features(predicted)
        if episodic_memory is not None
        else predicted.new_zeros((count, 0))
    )
    graph_features = (
        episodic_graph.candidate_features(predicted, sequences[:, 0])
        if episodic_graph is not None
        else predicted.new_zeros((count, 0))
    )
    combined = torch.cat(
        (
            *parts,
            first_action,
            action_fraction,
            endpoint_similarity,
            endpoint_change,
            *memory_parts,
            endpoint_memory,
            graph_features,
        ),
        dim=1,
    )
    return combined.cpu().numpy().astype(np.float64, copy=False)


def run_mpc_episode(
    env,
    jepa: TemporalRgbJepaAgent,
    belief: JepaBeliefHead | None,
    outcome_head: JepaRolloutOutcomeHead | None,
    sequences: torch.Tensor,
    seed: int,
    max_steps: int,
    rgb_style: str,
    weights: dict[str, float],
    episodic_memory_capacity: int = 0,
) -> dict:
    obs, _ = normalize_reset(env.reset(seed=seed))
    first = apply_rgb_style(obs["image"], rgb_style)
    frames = deque([first.copy() for _ in range(jepa.context_length)], maxlen=jepa.context_length)
    actions = deque([-1] * (jepa.context_length - 1), maxlen=jepa.context_length - 1)
    previous_action = -1
    belief_hidden = None
    visited_observations: set[bytes] = set()
    action_trace: list[int] = []
    diagnostics: list[dict[str, float]] = []
    episodic_memory = (
        EpisodicLatentMemory(capacity=episodic_memory_capacity)
        if episodic_memory_capacity > 0
        else None
    )

    for step in range(max_steps):
        latent = jepa.encode_context(np.stack(frames), list(actions))
        if episodic_memory is not None:
            episodic_memory.add(latent)
        if outcome_head is not None:
            action, diagnostic = choose_outcome_mpc_action(
                jepa,
                outcome_head,
                latent,
                sequences,
                weights,
                episodic_memory,
            )
            next_real_hidden = None
        else:
            assert belief is not None
            current_outputs, next_real_hidden = belief(
                latent.flatten(1).unsqueeze(1),
                torch.as_tensor([[previous_action]], dtype=torch.long, device=belief.device),
                belief_hidden,
            )
            action, diagnostic = choose_mpc_action(
                jepa,
                belief,
                latent,
                current_outputs,
                next_real_hidden,
                sequences,
                weights,
            )
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        next_image = apply_rgb_style(next_obs["image"], rgb_style)
        visited_observations.add(next_image.tobytes())
        frames.append(next_image.copy())
        actions.append(action)
        previous_action = action
        belief_hidden = next_real_hidden
        action_trace.append(action)
        diagnostics.append(diagnostic)
        obs = next_obs
        if terminated or truncated:
            return mpc_episode_result(
                seed,
                bool(terminated and float(reward) > 0.0),
                step + 1,
                visited_observations,
                action_trace,
                diagnostics,
            )
    return mpc_episode_result(
        seed, False, max_steps, visited_observations, action_trace, diagnostics
    )


def mpc_episode_result(seed, solved, steps, observations, action_trace, diagnostics) -> dict:
    return {
        "seed": int(seed),
        "solved": bool(solved),
        "steps": int(steps),
        "fitness": 10.0 if solved else 0.0,
        "unique_observations": len(observations),
        "unique_positions": 0,
        "blocked_steps": 0,
        "belief_entropy_initial": None,
        "belief_entropy_final": None,
        "belief_uncertainty_reduction": 0.0,
        "belief_cumulative_reduction": 0.0,
        "collected": False,
        "opened_door": False,
        "action_trace": action_trace,
        "mean_planned_risk": float(np.mean([row["risk"] for row in diagnostics])) if diagnostics else 0.0,
        "initial_plan": diagnostics[0] if diagnostics else {},
    }


if __name__ == "__main__":
    raise SystemExit(main())
