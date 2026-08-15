from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import RGBImgPartialObsWrapper

from .jepa_rollout_outcome import JepaRolloutOutcomeHead
from .minigrid_adapter import MiniGridSpec, make_minigrid
from .rgb_jepa import TemporalRgbJepaAgent
from .train_jepa_belief import BeliefRow, collect_dataset


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OutcomeDataset:
    latents: torch.Tensor
    horizons: torch.Tensor
    route: torch.Tensor
    collision: torch.Tensor
    goal: torch.Tensor

    def __len__(self) -> int:
        return int(self.horizons.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate outcome heads on JEPA-imagined latents.")
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--predictor", choices=("trained", "random"), default="trained")
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--expert-episodes", type=int, default=50)
    parser.add_argument("--validation-random-episodes", type=int, default=25)
    parser.add_argument("--validation-expert-episodes", type=int, default=10)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--collision-positive-weight", type=float, default=24.0)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--output", default="artifacts/jepa/temporal-rgb-s9n1-rollout-outcome.pt")
    parser.add_argument("--report", default="artifacts/jepa/temporal-rgb-s9n1-rollout-outcome-training.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()
    if args.predictor == "random":
        jepa.reset_predictor(args.seed + 7919)
    training_rows = collect_dataset(
        env,
        random_episodes=args.random_episodes,
        expert_episodes=args.expert_episodes,
        seed_choices=list(range(40)),
        max_steps=args.max_steps,
        rng=rng,
        context_length=jepa.context_length,
    )
    validation_rows = collect_dataset(
        env,
        random_episodes=args.validation_random_episodes,
        expert_episodes=args.validation_expert_episodes,
        seed_choices=list(range(40, 50)),
        max_steps=args.max_steps,
        rng=random.Random(args.seed + 1),
        context_length=jepa.context_length,
    )
    training = encode_rollout_dataset(jepa, training_rows, args.max_horizon)
    validation = encode_rollout_dataset(jepa, validation_rows, args.max_horizon)
    head = JepaRolloutOutcomeHead(jepa.feature_size, args.max_horizon)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    latest = {}
    for update in range(args.updates):
        indices = torch.as_tensor(
            rng.sample(range(len(training)), min(args.batch_size, len(training))),
            dtype=torch.long,
        )
        latest = train_step(
            head, optimizer, training, indices, args.collision_positive_weight
        )
        if update % 100 == 0 or update + 1 == args.updates:
            print(json.dumps({"update": update + 1, **latest}))

    train_metrics = evaluate(head, training, args.collision_positive_weight)
    validation_metrics = evaluate(head, validation, args.collision_positive_weight)
    metadata = {
        "environment": args.env,
        "jepa_checkpoint": str(ROOT / args.checkpoint),
        "predictor": args.predictor,
        "trained_on": "JEPA-predicted endpoint latents aligned to real trajectory outcomes",
        "privileged_labels_training_only": True,
        "targets": ["safe route distribution", "collision by action", "safe goal distance", "rollout progress"],
    }
    output = ROOT / args.output
    head.save(output, metadata=metadata)
    report = {
        **metadata,
        "seed": args.seed,
        "max_horizon": args.max_horizon,
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "collision_positive_weight": args.collision_positive_weight,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "updates": args.updates,
        "latest_batch": latest,
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "output": str(output),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({"validation": validation_metrics, "report": str(report_path)}))
    return 0


@torch.no_grad()
def encode_rollout_dataset(
    jepa: TemporalRgbJepaAgent,
    episodes: list[list[BeliefRow]],
    max_horizon: int,
) -> OutcomeDataset:
    latents = []
    horizons = []
    route = []
    collision = []
    goal = []
    for episode in episodes:
        for start, row in enumerate(episode):
            predicted = jepa.encode_context(row.frames, row.context_actions)
            start_distance = float(row.goal[0])
            endpoint = row
            for horizon in range(0, min(max_horizon, len(episode) - start - 1) + 1):
                if horizon:
                    endpoint = episode[start + horizon]
                    action = int(endpoint.previous_action)
                    predicted = jepa.predictor(
                        predicted,
                        torch.as_tensor([action], dtype=torch.long, device=jepa.device),
                    )
                latents.append(predicted[0].cpu())
                horizons.append(horizon)
                route.append(torch.as_tensor(endpoint.route_distribution))
                collision.append(torch.as_tensor(endpoint.collision))
                goal.append(
                    torch.as_tensor(
                        (float(endpoint.goal[0]), start_distance - float(endpoint.goal[0])),
                        dtype=torch.float32,
                    )
                )
    return OutcomeDataset(
        latents=torch.stack(latents),
        horizons=torch.as_tensor(horizons, dtype=torch.long),
        route=torch.stack(route).float(),
        collision=torch.stack(collision).float(),
        goal=torch.stack(goal).float(),
    )


def batch(dataset: OutcomeDataset, indices: torch.Tensor, device) -> dict[str, torch.Tensor]:
    return {
        "latents": dataset.latents[indices].to(device),
        "horizons": dataset.horizons[indices].to(device),
        "route": dataset.route[indices].to(device),
        "collision": dataset.collision[indices].to(device),
        "goal": dataset.goal[indices].to(device),
    }


def metrics(
    head: JepaRolloutOutcomeHead,
    rows: dict[str, torch.Tensor],
    collision_positive_weight: float,
) -> tuple[torch.Tensor, dict]:
    outputs = head(rows["latents"], rows["horizons"])
    route_loss = -(rows["route"] * F.log_softmax(outputs["route_logits"], dim=-1)).sum(-1).mean()
    collision_loss = F.binary_cross_entropy_with_logits(
        outputs["collision_logits"],
        rows["collision"],
        pos_weight=torch.full(
            (3,), float(collision_positive_weight), device=head.device
        ),
    )
    goal_loss = F.smooth_l1_loss(outputs["goal"], rows["goal"])
    loss = route_loss + 0.5 * collision_loss + goal_loss
    collision_prediction = torch.sigmoid(outputs["collision_logits"]) >= 0.5
    collision_target = rows["collision"] >= 0.5
    positive_count = collision_target.sum().clamp(min=1)
    return loss, {
        "loss": float(loss.detach()),
        "route_accuracy": float((outputs["route_logits"].argmax(-1) == rows["route"].argmax(-1)).float().mean()),
        "collision_accuracy": float((collision_prediction == collision_target).float().mean()),
        "collision_recall": float(
            (collision_prediction & collision_target).sum().float().div(positive_count)
        ),
        "goal_mae": float((outputs["goal"] - rows["goal"]).abs().mean().detach()),
    }


def train_step(head, optimizer, dataset, indices, collision_positive_weight) -> dict:
    head.train()
    loss, result = metrics(
        head,
        batch(dataset, indices, head.device),
        collision_positive_weight,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step()
    return result


@torch.no_grad()
def evaluate(head, dataset, collision_positive_weight: float, batch_size: int = 512) -> dict:
    head.eval()
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, len(dataset), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(dataset)))
        rows = batch(dataset, indices, head.device)
        _, result = metrics(head, rows, collision_positive_weight)
        size = len(indices)
        for key, value in result.items():
            totals[key] = totals.get(key, 0.0) + value * size
        count += size
    return {key: value / count for key, value in totals.items()}


if __name__ == "__main__":
    raise SystemExit(main())
