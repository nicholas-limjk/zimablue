from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import RGBImgPartialObsWrapper

from .minigrid_adapter import MiniGridSpec, make_minigrid
from .rgb_jepa import TemporalRgbJepaAgent
from .train_jepa_belief import BeliefRow, collect_dataset
from .train_temporal_rgb_jepa import collect_branched_episode


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RolloutSample:
    frames: np.ndarray
    context_actions: np.ndarray
    future_actions: np.ndarray
    target_image: np.ndarray
    horizon: int
    category: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare trained, random, and persistence latent rollouts on unseen trajectories."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--expert-episodes", type=int, default=40)
    parser.add_argument(
        "--branched-episodes",
        type=int,
        default=0,
        help="Evaluate matched left/right/forward branches, including padded terminal futures.",
    )
    parser.add_argument("--seed-start", type=int, default=50)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    trained = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    trained.load(ROOT / args.checkpoint)
    trained.eval()
    random_predictor = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    random_predictor.load(ROOT / args.checkpoint)
    random_predictor.reset_predictor(args.seed + 7919)
    random_predictor.eval()

    if args.branched_episodes:
        branch_rows = []
        for index in range(args.branched_episodes):
            _, _, rows = collect_branched_episode(
                env,
                None,
                seed=args.seed_start + index % args.seed_count,
                max_steps=args.max_steps,
                rng=random.Random(args.seed + 101 + index),
            )
            branch_rows.extend(rows)
        episodes = []
        samples = build_branched_samples(branch_rows, horizons=(1, 2, 4))
    else:
        episodes = collect_dataset(
            env,
            random_episodes=args.random_episodes,
            expert_episodes=args.expert_episodes,
            seed_choices=list(range(args.seed_start, args.seed_start + args.seed_count)),
            max_steps=args.max_steps,
            rng=random.Random(args.seed + 101),
            context_length=trained.context_length,
        )
        samples = build_samples(episodes, horizons=(1, 2, 4))
    report = evaluate_samples(trained, random_predictor, samples, args.batch_size)
    if args.branched_episodes:
        report["matched_branch_retrieval"] = evaluate_branch_retrieval(
            trained, random_predictor, branch_rows, horizons=(1, 2, 4)
        )
    report.update(
        {
            "environment": args.env,
            "checkpoint": str(ROOT / args.checkpoint),
            "heldout_layout_seeds": list(
                range(args.seed_start, args.seed_start + args.seed_count)
            ),
            "random_episodes": args.random_episodes,
            "expert_episodes": args.expert_episodes,
            "branched_episodes": args.branched_episodes,
            "episode_count": len(episodes),
            "sample_count": len(samples),
            "target": "EMA target-frame encoder latent of the real future RGB observation",
            "baselines": {
                "random": "same trained context encoder and target encoder; predictor reset",
                "persistence": "current context latent copied unchanged to every horizon",
            },
        }
    )
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(
        json.dumps(
            {
                "samples": len(samples),
                "overall": report["overall"],
                "report": str(report_path),
            }
        )
    )
    return 0


def transition_category(current: BeliefRow, following: BeliefRow) -> str:
    terrain = current.terrain
    if terrain is not None and np.any(np.asarray(terrain) == 3):
        return "lava_visible"
    action = int(following.previous_action)
    if action in (0, 1):
        return "turn"
    if action != 2:
        return "other"
    if float(following.moved_last) > 0.5:
        return "free_forward"
    return "blocked_forward"


def build_samples(
    episodes: list[list[BeliefRow]],
    horizons: tuple[int, ...],
) -> list[RolloutSample]:
    samples = []
    for episode in episodes:
        for start, row in enumerate(episode):
            for horizon in horizons:
                endpoint_index = start + horizon
                if endpoint_index >= len(episode):
                    continue
                following = episode[start + 1]
                future_actions = np.asarray(
                    [episode[index].previous_action for index in range(start + 1, endpoint_index + 1)],
                    dtype=np.int64,
                )
                samples.append(
                    RolloutSample(
                        frames=row.frames,
                        context_actions=row.context_actions,
                        future_actions=future_actions,
                        target_image=episode[endpoint_index].frames[-1],
                        horizon=horizon,
                        category=transition_category(row, following),
                    )
                )
    return samples


def build_branched_samples(rows: list[dict], horizons: tuple[int, ...]) -> list[RolloutSample]:
    samples = []
    for row in rows:
        for horizon in horizons:
            samples.append(
                RolloutSample(
                    frames=row["frames"],
                    context_actions=row["previous_actions"],
                    future_actions=row["future_actions"][:horizon],
                    target_image=row["future_images"][horizon - 1],
                    horizon=horizon,
                    category=f"{row['category']}:action={row['first_action']}",
                )
            )
    return samples


@torch.no_grad()
def evaluate_branch_retrieval(
    trained: TemporalRgbJepaAgent,
    random_predictor: TemporalRgbJepaAgent,
    rows: list[dict],
    horizons: tuple[int, ...],
) -> dict:
    """Test whether action-conditioned predictions identify the correct branch.

    Rows are emitted in left/right/forward triples from exactly the same history.
    A model that merely predicts an average future should remain near chance.
    """
    results = {
        str(horizon): {
            name: {"correct": 0, "count": 0, "margin_sum": 0.0}
            for name in ("trained", "random", "persistence")
        }
        for horizon in horizons
    }
    for start in range(0, len(rows) - 2, 3):
        group = rows[start : start + 3]
        if [row["first_action"] for row in group] != [0, 1, 2]:
            continue
        current = trained.encode_context(group[0]["frames"], group[0]["previous_actions"])
        for horizon in horizons:
            targets = trained.target_frame_encoder(
                trained._images([row["future_images"][horizon - 1] for row in group])
            )
            model_predictions = {}
            for name, model in (("trained", trained), ("random", random_predictor)):
                predicted_rows = []
                for row in group:
                    predicted = current
                    for action in row["future_actions"][:horizon]:
                        predicted = model.predictor(
                            predicted,
                            torch.as_tensor([int(action)], dtype=torch.long, device=model.device),
                        )
                    predicted_rows.append(predicted[0])
                model_predictions[name] = torch.stack(predicted_rows)
            model_predictions["persistence"] = current.expand(3, *current.shape[1:])
            for name, predictions in model_predictions.items():
                distances = torch.cdist(predictions.flatten(1), targets.flatten(1))
                selected = distances.argmin(dim=1)
                correct_distances = distances.diag()
                masked = distances + torch.eye(3, device=distances.device) * 1e9
                margins = masked.min(dim=1).values - correct_distances
                row = results[str(horizon)][name]
                row["correct"] += int((selected == torch.arange(3, device=selected.device)).sum())
                row["count"] += 3
                row["margin_sum"] += float(margins.sum())
    return {
        horizon: {
            name: {
                "accuracy": values["correct"] / max(1, values["count"]),
                "mean_correct_margin": values["margin_sum"] / max(1, values["count"]),
                "count": values["count"],
            }
            for name, values in models.items()
        }
        for horizon, models in results.items()
    }


@torch.no_grad()
def evaluate_samples(
    trained: TemporalRgbJepaAgent,
    random_predictor: TemporalRgbJepaAgent,
    samples: list[RolloutSample],
    batch_size: int,
) -> dict:
    rows: list[dict] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        frames = trained._frames([sample.frames for sample in batch])
        context_actions = torch.as_tensor(
            np.stack([sample.context_actions for sample in batch]),
            dtype=torch.long,
            device=trained.device,
        )
        current = trained.context_encoder(frames, context_actions)
        targets = trained.target_frame_encoder(
            trained._images([sample.target_image for sample in batch])
        )
        trained_prediction = current
        random_prediction = current
        maximum_horizon = max(sample.horizon for sample in batch)
        for step in range(maximum_horizon):
            actions = torch.as_tensor(
                [
                    int(sample.future_actions[step])
                    if step < len(sample.future_actions)
                    else 0
                    for sample in batch
                ],
                dtype=torch.long,
                device=trained.device,
            )
            active = torch.as_tensor(
                [step < sample.horizon for sample in batch],
                dtype=torch.bool,
                device=trained.device,
            ).reshape(-1, 1, 1, 1)
            next_trained = trained.predictor(trained_prediction, actions)
            next_random = random_predictor.predictor(random_prediction, actions)
            trained_prediction = torch.where(active, next_trained, trained_prediction)
            random_prediction = torch.where(active, next_random, random_prediction)

        model_predictions = {
            "trained": trained_prediction,
            "random": random_prediction,
            "persistence": current,
        }
        for name, prediction in model_predictions.items():
            smooth_l1 = F.smooth_l1_loss(prediction, targets, reduction="none").flatten(1).mean(1)
            cosine = F.cosine_similarity(prediction.flatten(1), targets.flatten(1))
            l2 = torch.linalg.vector_norm(prediction.flatten(1) - targets.flatten(1), dim=1)
            for index, sample in enumerate(batch):
                rows.append(
                    {
                        "model": name,
                        "horizon": sample.horizon,
                        "category": sample.category,
                        "smooth_l1": float(smooth_l1[index]),
                        "cosine": float(cosine[index]),
                        "l2": float(l2[index]),
                    }
                )
    return summarize_rows(rows)


def summarize_rows(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"],)].append(row)
        grouped[(row["model"], row["horizon"])].append(row)
        grouped[(row["model"], row["horizon"], row["category"])].append(row)

    def summary(values: list[dict]) -> dict:
        return {
            "count": len(values),
            "smooth_l1": float(np.mean([row["smooth_l1"] for row in values])),
            "cosine": float(np.mean([row["cosine"] for row in values])),
            "l2": float(np.mean([row["l2"] for row in values])),
        }

    models = sorted({row["model"] for row in rows})
    horizons = sorted({row["horizon"] for row in rows})
    categories = sorted({row["category"] for row in rows})
    overall = {model: summary(grouped[(model,)]) for model in models}
    by_horizon = {
        str(horizon): {
            model: summary(grouped[(model, horizon)]) for model in models
        }
        for horizon in horizons
    }
    by_category = {
        str(horizon): {
            category: {
                model: summary(grouped[(model, horizon, category)])
                for model in models
            }
            for category in categories
            if grouped[(models[0], horizon, category)]
        }
        for horizon in horizons
    }
    return {
        "overall": overall,
        "by_horizon": by_horizon,
        "by_horizon_and_category": by_category,
    }


if __name__ == "__main__":
    raise SystemExit(main())
