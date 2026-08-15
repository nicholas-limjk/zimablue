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

from .jepa_evidence import JepaEvidenceMapHead
from .minigrid_adapter import MiniGridSpec, make_minigrid
from .rgb_jepa import TemporalRgbJepaAgent
from .train_jepa_belief import collect_dataset


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvidenceSamples:
    latent: torch.Tensor
    previous_latent: torch.Tensor
    previous_action: torch.Tensor
    terrain: torch.Tensor
    moved: torch.Tensor
    moved_mask: torch.Tensor

    def __len__(self) -> int:
        return int(self.latent.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train observable terrain and motion evidence heads on frozen temporal-JEPA latents."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--random-episodes", type=int, default=200)
    parser.add_argument("--expert-episodes", type=int, default=80)
    parser.add_argument("--validation-random-episodes", type=int, default=40)
    parser.add_argument("--validation-expert-episodes", type=int, default=20)
    parser.add_argument("--train-seed-pool", type=int, default=40)
    parser.add_argument("--validation-seed-start", type=int, default=40)
    parser.add_argument("--validation-seed-count", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--motion-weight", type=float, default=0.5)
    parser.add_argument("--rare-frame-fraction", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--output", default="artifacts/jepa/temporal-rgb-s9n1-evidence-map.pt")
    parser.add_argument("--report", default="artifacts/jepa/temporal-rgb-s9n1-evidence-map-training.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()

    training_rows = collect_dataset(
        env,
        random_episodes=args.random_episodes,
        expert_episodes=args.expert_episodes,
        seed_choices=list(range(args.train_seed_pool)),
        max_steps=args.max_steps,
        rng=rng,
        context_length=jepa.context_length,
    )
    validation_rows = collect_dataset(
        env,
        random_episodes=args.validation_random_episodes,
        expert_episodes=args.validation_expert_episodes,
        seed_choices=list(
            range(args.validation_seed_start, args.validation_seed_start + args.validation_seed_count)
        ),
        max_steps=args.max_steps,
        rng=random.Random(args.seed + 1),
        context_length=jepa.context_length,
    )
    training = encode_samples(jepa, training_rows)
    validation = encode_samples(jepa, validation_rows)
    model = JepaEvidenceMapHead(
        latent_size=jepa.feature_size,
        action_count=env.action_space.n,
        latent_channels=jepa.latent_channels,
        spatial_size=jepa.spatial_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rare_indices = torch.where(
        ((training.terrain == 3) | (training.terrain == 4)).flatten(1).any(dim=1)
    )[0].tolist()
    latest = {}
    for update in range(args.updates):
        indices = balanced_indices(
            len(training), rare_indices, args.batch_size, args.rare_frame_fraction, rng
        )
        latest = train_batch(model, optimizer, select(training, indices), args.motion_weight)
        if update % 100 == 0 or update + 1 == args.updates:
            print(json.dumps({"update": update + 1, **latest}))

    split_rng = torch.Generator().manual_seed(args.seed + 2)
    validation_order = torch.randperm(len(validation), generator=split_rng)
    split = len(validation_order) // 2
    calibration = select(validation, validation_order[:split])
    audit = select(validation, validation_order[split:])
    raw_audit_metrics = evaluate(model, audit, args.motion_weight)
    calibration_parameters = fit_calibration(model, calibration)
    training_metrics = evaluate(model, training, args.motion_weight)
    calibration_metrics = evaluate(model, calibration, args.motion_weight)
    validation_metrics = evaluate(model, audit, args.motion_weight)
    metadata = {
        "environment": args.env,
        "jepa_checkpoint": str(ROOT / args.checkpoint),
        "belief_architecture": "evidence_map",
        "terrain_classes": list(model.terrain_classes),
        "targets": {
            "terrain": "only view-visible unseen/free/wall/lava/goal cells",
            "motion": "whether the preceding forward action changed true position",
        },
        "privileged_labels_training_only": True,
        "inference_state": "predicted evidence plus action history only",
        "calibration": "scalar temperature, per-class bias, and observed terrain prior fitted on half of validation samples",
    }
    output = ROOT / args.output
    model.save(output, metadata=metadata)
    report = {
        **metadata,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_size": model.feature_size,
        "training_episodes": len(training_rows),
        "training_samples": len(training),
        "validation_episodes": len(validation_rows),
        "validation_samples": len(validation),
        "calibration_samples": len(calibration),
        "audit_samples": len(audit),
        "updates": args.updates,
        "rare_training_frames": len(rare_indices),
        "rare_frame_fraction": args.rare_frame_fraction,
        "latest_batch": latest,
        "raw_audit_metrics": raw_audit_metrics,
        "calibration_parameters": calibration_parameters,
        "training_metrics": training_metrics,
        "calibration_metrics": calibration_metrics,
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
def encode_samples(jepa, episodes) -> EvidenceSamples:
    latents = []
    previous_latents = []
    actions = []
    terrain = []
    moved = []
    moved_mask = []
    for episode in episodes:
        episode_latents = [
            jepa.features(row.frames, row.context_actions).astype(np.float32) for row in episode
        ]
        for index, row in enumerate(episode):
            latents.append(episode_latents[index])
            previous_latents.append(
                np.zeros_like(episode_latents[index]) if index == 0 else episode_latents[index - 1]
            )
            actions.append(row.previous_action)
            terrain.append(row.terrain)
            moved.append(row.moved_last)
            moved_mask.append(row.moved_mask)
    return EvidenceSamples(
        latent=torch.as_tensor(np.stack(latents), dtype=torch.float32),
        previous_latent=torch.as_tensor(np.stack(previous_latents), dtype=torch.float32),
        previous_action=torch.as_tensor(actions, dtype=torch.long),
        terrain=torch.as_tensor(np.stack(terrain), dtype=torch.long),
        moved=torch.as_tensor(moved, dtype=torch.float32),
        moved_mask=torch.as_tensor(moved_mask, dtype=torch.float32),
    )


def select(samples: EvidenceSamples, indices: torch.Tensor) -> EvidenceSamples:
    return EvidenceSamples(**{name: getattr(samples, name)[indices] for name in samples.__dataclass_fields__})


def balanced_indices(total, rare_indices, batch_size, rare_fraction, rng) -> torch.Tensor:
    size = min(int(batch_size), int(total))
    rare_size = min(len(rare_indices), int(round(size * float(rare_fraction))))
    rare = rng.sample(rare_indices, rare_size) if rare_size else []
    common_size = size - len(rare)
    common = rng.sample(range(total), common_size)
    indices = rare + common
    rng.shuffle(indices)
    return torch.as_tensor(indices, dtype=torch.long)


def losses_and_metrics(model, samples: EvidenceSamples, motion_weight: float) -> tuple[torch.Tensor, dict]:
    device = model.device
    latent = samples.latent.to(device)
    terrain = samples.terrain.to(device)
    terrain_logits = model.terrain_logits(latent)
    # Deliberately preserve rare-class recall during representation training;
    # the held-out calibration layer corrects the resulting posterior bias.
    class_weights = torch.as_tensor((0.25, 1.0, 1.5, 6.0, 10.0), device=device)
    terrain_loss = F.cross_entropy(
        terrain_logits.reshape(-1, len(model.terrain_classes)), terrain.reshape(-1), weight=class_weights
    )
    motion_logits = model.motion_logits(
        samples.previous_latent.to(device), latent, samples.previous_action.to(device)
    )
    motion_mask = samples.moved_mask.to(device)
    motion_element = F.binary_cross_entropy_with_logits(
        motion_logits, samples.moved.to(device), reduction="none"
    )
    motion_loss = (motion_element * motion_mask).sum() / motion_mask.sum().clamp_min(1.0)
    loss = terrain_loss + float(motion_weight) * motion_loss

    calibrated = model.calibrated_probabilities(terrain_logits)
    prediction = calibrated.argmax(dim=-1)
    visible = terrain != 0
    terrain_accuracy = (prediction[visible] == terrain[visible]).float().mean()
    per_class_recall = {}
    for index, name in enumerate(model.terrain_classes):
        mask = terrain == index
        predicted_mask = prediction == index
        true_positive = (predicted_mask & mask).sum().float()
        recall = true_positive / mask.sum().clamp_min(1)
        precision = true_positive / predicted_mask.sum().clamp_min(1)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-9)
        per_class_recall[f"{name}_recall"] = float(recall.detach())
        per_class_recall[f"{name}_precision"] = float(precision.detach())
        per_class_recall[f"{name}_f1"] = float(f1.detach())
    motion_prediction = torch.sigmoid(motion_logits) >= 0.5
    motion_accuracy = (
        ((motion_prediction == (samples.moved.to(device) >= 0.5)).float() * motion_mask).sum()
        / motion_mask.sum().clamp_min(1.0)
    )
    calibration_nll = F.nll_loss(
        torch.log(calibrated.clamp_min(1e-9)).reshape(-1, len(model.terrain_classes)),
        terrain.reshape(-1),
    )
    one_hot = F.one_hot(terrain, num_classes=len(model.terrain_classes)).to(calibrated.dtype)
    brier = ((calibrated - one_hot) ** 2).sum(dim=-1).mean()
    return loss, {
        "loss": float(loss.detach()),
        "terrain_loss": float(terrain_loss.detach()),
        "visible_terrain_accuracy": float(terrain_accuracy.detach()),
        "motion_loss": float(motion_loss.detach()),
        "motion_accuracy": float(motion_accuracy.detach()),
        "calibrated_nll": float(calibration_nll.detach()),
        "brier_score": float(brier.detach()),
        **per_class_recall,
    }


@torch.no_grad()
def calibration_tensors(model, samples: EvidenceSamples) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.terrain_logits(samples.latent.to(model.device))
    return logits.reshape(-1, len(model.terrain_classes)), samples.terrain.to(model.device).reshape(-1)


def fit_calibration(model, samples: EvidenceSamples, steps: int = 300) -> dict:
    model.eval()
    logits, labels = calibration_tensors(model, samples)
    logits = logits.detach()
    log_temperature = torch.zeros((), device=model.device, requires_grad=True)
    bias = torch.zeros(len(model.terrain_classes), device=model.device, requires_grad=True)
    optimizer = torch.optim.Adam((log_temperature, bias), lr=0.05)
    for _ in range(steps):
        temperature = F.softplus(log_temperature) + 0.05
        centered_bias = bias - bias.mean()
        loss = F.cross_entropy(logits / temperature + centered_bias, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        temperature = F.softplus(log_temperature) + 0.05
        centered_bias = bias - bias.mean()
        observed = labels[labels != 0] - 1
        counts = torch.bincount(observed, minlength=4).to(torch.float32) + 1.0
        prior = counts / counts.sum()
        model.set_calibration(temperature, centered_bias, prior)
    return {
        "temperature": float(temperature.detach()),
        "bias": [float(value) for value in centered_bias.detach().cpu()],
        "observed_terrain_prior": [float(value) for value in prior.detach().cpu()],
    }


def train_batch(model, optimizer, samples, motion_weight) -> dict:
    model.train()
    loss, metrics = losses_and_metrics(model, samples, motion_weight)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return metrics


@torch.no_grad()
def evaluate(model, samples, motion_weight, batch_size: int = 16384) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, len(samples), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(samples)))
        rows = select(samples, indices)
        _, metrics = losses_and_metrics(model, rows, motion_weight)
        size = len(rows)
        count += size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * size
    return {key: value / count for key, value in totals.items()}


if __name__ == "__main__":
    raise SystemExit(main())
