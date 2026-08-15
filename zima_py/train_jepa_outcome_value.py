from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import RGBImgPartialObsWrapper

from .jepa_outcome_value import JepaOutcomeValueHead
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent
from .train_temporal_rgb_jepa import shortest_safe_plan


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OutcomeEpisode:
    frames: list[np.ndarray]
    context_actions: list[np.ndarray]
    actions: list[int]
    observations: list[np.ndarray]
    solved: bool


@dataclass
class OutcomeValueDataset:
    beliefs: torch.Tensor
    candidates: torch.Tensor
    distances: torch.Tensor
    success: torch.Tensor
    remaining: torch.Tensor

    def __len__(self) -> int:
        return int(self.success.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a minimal success/remaining-step value head on frozen JEPA latents."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--random-episodes", type=int, default=120)
    parser.add_argument("--expert-episodes", type=int, default=120)
    parser.add_argument("--validation-random-episodes", type=int, default=30)
    parser.add_argument("--validation-expert-episodes", type=int, default=30)
    parser.add_argument("--training-start", type=int, default=0)
    parser.add_argument("--training-seed-count", type=int, default=40)
    parser.add_argument("--validation-start", type=int, default=40)
    parser.add_argument("--validation-seed-count", type=int, default=10)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--output", default="artifacts/jepa/temporal-rgb-s9n1-outcome-value.pt")
    parser.add_argument("--report", default="artifacts/jepa/temporal-rgb-s9n1-outcome-value-training.json")
    args = parser.parse_args()
    if args.max_horizon < 1:
        raise ValueError("--max-horizon must be positive")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()
    training_episodes = collect_episodes(
        env,
        args.random_episodes,
        args.expert_episodes,
        list(range(args.training_start, args.training_start + args.training_seed_count)),
        args.max_steps,
        jepa.context_length,
        rng,
    )
    validation_episodes = collect_episodes(
        env,
        args.validation_random_episodes,
        args.validation_expert_episodes,
        list(range(args.validation_start, args.validation_start + args.validation_seed_count)),
        args.max_steps,
        jepa.context_length,
        random.Random(args.seed + 1),
    )
    training = encode_dataset(jepa, training_episodes, args.max_horizon, args.max_steps)
    validation = encode_dataset(jepa, validation_episodes, args.max_horizon, args.max_steps)
    head = JepaOutcomeValueHead(jepa.feature_size, args.hidden_size)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positive_fraction = float(training.success.mean())
    positive_weight = (1.0 - positive_fraction) / max(positive_fraction, 1e-6)
    latest = {}
    for update in range(args.updates):
        indices = torch.as_tensor(
            rng.sample(range(len(training)), min(args.batch_size, len(training))),
            dtype=torch.long,
        )
        latest = train_step(head, optimizer, training, indices, positive_weight)
        if update % 100 == 0 or update + 1 == args.updates:
            print(json.dumps({"update": update + 1, **latest}))

    train_metrics = evaluate(head, training, positive_weight)
    validation_metrics = evaluate(head, validation, positive_weight)
    metadata = {
        "environment": args.env,
        "jepa_checkpoint": str((ROOT / args.checkpoint).resolve()),
        "objective": "eventual episode success plus normalized remaining steps",
        "targets": ["eventual_success", "remaining_steps_fraction"],
        "excluded_targets": ["terrain", "lava", "barrier", "goal_position", "novelty", "map", "pose"],
        "jepa_frozen": True,
        "candidate_sources": ["JEPA-predicted actual action endpoint", "EMA target-frame endpoint"],
    }
    output = ROOT / args.output
    head.save(output, metadata=metadata)
    report = {
        **metadata,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "training_episodes": len(training_episodes),
        "training_samples": len(training),
        "training_success_fraction": positive_fraction,
        "validation_episodes": len(validation_episodes),
        "validation_samples": len(validation),
        "updates": args.updates,
        "latest_batch": latest,
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "output": str(output.resolve()),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({"validation": validation_metrics, "report": str(report_path)}))
    return 0


def collect_episodes(
    env,
    random_episodes: int,
    expert_episodes: int,
    seeds: list[int],
    max_steps: int,
    context_length: int,
    rng: random.Random,
) -> list[OutcomeEpisode]:
    schedule = [False] * random_episodes + [True] * expert_episodes
    rng.shuffle(schedule)
    return [
        collect_episode(env, seeds[rng.randrange(len(seeds))], max_steps, context_length, rng, expert)
        for expert in schedule
    ]


def collect_episode(env, seed, max_steps, context_length, rng, expert) -> OutcomeEpisode:
    obs, _ = normalize_reset(env.reset(seed=int(seed)))
    image = np.asarray(obs["image"], dtype=np.uint8)
    history = deque([image.copy() for _ in range(context_length)], maxlen=context_length)
    action_history = deque([-1] * (context_length - 1), maxlen=context_length - 1)
    frames: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    actions: list[int] = []
    observations = [image.copy()]
    solved = False
    for _ in range(max_steps):
        frames.append(np.stack(history))
        contexts.append(np.asarray(action_history, dtype=np.int64))
        plan = shortest_safe_plan(env) if expert else None
        if expert and not plan:
            break
        action = int(plan[0]) if expert else rng.randrange(3)
        actions.append(action)
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        next_image = np.asarray(next_obs["image"], dtype=np.uint8)
        observations.append(next_image.copy())
        history.append(next_image.copy())
        action_history.append(action)
        solved = bool(terminated and float(reward) > 0.0)
        if terminated or truncated:
            break
    return OutcomeEpisode(frames, contexts, actions, observations, solved)


@torch.no_grad()
def encode_dataset(jepa, episodes, max_horizon, max_steps) -> OutcomeValueDataset:
    beliefs = []
    candidates = []
    distances = []
    success = []
    remaining = []
    for episode in episodes:
        for start in range(len(episode.actions)):
            belief = jepa.encode_context(episode.frames[start], episode.context_actions[start])
            predicted = belief
            for horizon in range(1, min(max_horizon, len(episode.actions) - start) + 1):
                action = episode.actions[start + horizon - 1]
                predicted = jepa.predictor(
                    predicted,
                    torch.as_tensor([action], dtype=torch.long, device=jepa.device),
                )
                endpoint_frame = episode.observations[start + horizon]
                actual = jepa.encode_frame(endpoint_frame)
                for candidate in (predicted, actual):
                    beliefs.append(belief[0].cpu())
                    candidates.append(candidate[0].cpu())
                    distances.append(horizon / float(max_steps))
                    success.append(float(episode.solved))
                    remaining.append(
                        (len(episode.actions) - start) / float(max_steps)
                        if episode.solved
                        else 1.0
                    )
    return OutcomeValueDataset(
        beliefs=torch.stack(beliefs),
        candidates=torch.stack(candidates),
        distances=torch.as_tensor(distances, dtype=torch.float32),
        success=torch.as_tensor(success, dtype=torch.float32),
        remaining=torch.as_tensor(remaining, dtype=torch.float32),
    )


def dataset_batch(dataset, indices, device):
    return {
        "beliefs": dataset.beliefs[indices].to(device),
        "candidates": dataset.candidates[indices].to(device),
        "distances": dataset.distances[indices].to(device),
        "success": dataset.success[indices].to(device),
        "remaining": dataset.remaining[indices].to(device),
    }


def loss_and_metrics(head, rows, positive_weight):
    outputs = head(rows["beliefs"], rows["candidates"], rows["distances"])
    success_loss = F.binary_cross_entropy_with_logits(
        outputs["success_logit"],
        rows["success"],
        pos_weight=torch.as_tensor(positive_weight, device=head.device),
    )
    remaining_loss = F.smooth_l1_loss(outputs["remaining"], rows["remaining"])
    loss = success_loss + remaining_loss
    probability = torch.sigmoid(outputs["success_logit"])
    return loss, {
        "loss": float(loss.detach()),
        "success_accuracy": float(((probability >= 0.5) == (rows["success"] >= 0.5)).float().mean()),
        "success_brier": float(((probability - rows["success"]) ** 2).mean().detach()),
        "remaining_mae": float(
            (outputs["remaining"] - rows["remaining"]).abs().mean().detach()
        ),
        "mean_predicted_success": float(probability.mean().detach()),
    }


def train_step(head, optimizer, dataset, indices, positive_weight):
    head.train()
    loss, metrics = loss_and_metrics(head, dataset_batch(dataset, indices, head.device), positive_weight)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step()
    return metrics


@torch.no_grad()
def evaluate(head, dataset, positive_weight, batch_size=512):
    head.eval()
    totals = {}
    count = 0
    for start in range(0, len(dataset), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(dataset)))
        _, metrics = loss_and_metrics(
            head, dataset_batch(dataset, indices, head.device), positive_weight
        )
        size = len(indices)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * size
        count += size
    return {key: value / count for key, value in totals.items()}


if __name__ == "__main__":
    raise SystemExit(main())
