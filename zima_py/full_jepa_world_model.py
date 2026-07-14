from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MiniGridEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.object_embedding = nn.Embedding(32, 16)
        self.color_embedding = nn.Embedding(16, 8)
        self.state_embedding = nn.Embedding(8, 4)
        self.direction_embedding = nn.Embedding(5, 8)
        self.spatial = nn.Sequential(
            nn.Conv2d(28, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.project = nn.Sequential(nn.Linear(64 * 4 + 8, 256), nn.GELU(), nn.Linear(256, latent_dim))

    def forward(self, image: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        objects = self.object_embedding(image[..., 0].clamp(0, 31))
        colors = self.color_embedding(image[..., 1].clamp(0, 15))
        states = self.state_embedding(image[..., 2].clamp(0, 7))
        cells = torch.cat((objects, colors, states), dim=-1).permute(0, 3, 1, 2)
        spatial = self.spatial(cells).flatten(1)
        direction_index = torch.where((direction >= 0) & (direction < 4), direction, torch.full_like(direction, 4))
        latent = self.project(torch.cat((spatial, self.direction_embedding(direction_index)), dim=-1))
        return F.normalize(latent, dim=-1)


class ActionPredictor(nn.Module):
    def __init__(self, action_count: int, latent_dim: int = 128):
        super().__init__()
        self.action_embedding = nn.Embedding(action_count, 32)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + 32, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.transition(torch.cat((latent, self.action_embedding(action)), dim=-1))
        return F.normalize(latent + delta, dim=-1)


@dataclass(frozen=True)
class ReplayTransition:
    previous_image: np.ndarray
    previous_direction: int
    action: int
    next_image: np.ndarray
    next_direction: int
    terminal: bool
    step: int


class FullJepaWorldModel:
    """Action-conditioned JEPA trained online from one MiniGrid environment."""

    def __init__(
        self,
        action_count: int,
        *,
        latent_dim: int = 128,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        horizon: int = 3,
        replay_capacity: int = 20_000,
        warmup: int = 32,
        ema_decay: float = 0.99,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.action_count = int(action_count)
        self.latent_dim = int(latent_dim)
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.warmup = int(warmup)
        self.ema_decay = float(ema_decay)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.context_encoder = MiniGridEncoder(latent_dim).to(self.device)
        self.target_encoder = copy.deepcopy(self.context_encoder).to(self.device)
        self.target_encoder.requires_grad_(False)
        self.predictor = ActionPredictor(action_count, latent_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.replay: deque[ReplayTransition] = deque(maxlen=int(replay_capacity))
        self.total_train_steps = 0
        self.latest_loss: float | None = None
        self.recent_surprises: deque[dict[str, Any]] = deque(maxlen=128)
        self.action_stats: dict[int, dict[str, float]] = {}

    def observe(
        self,
        previous_obs: dict[str, Any],
        action: int,
        next_obs: dict[str, Any],
        *,
        step: int,
        action_name: str,
        blocked: bool,
        reward: float,
        terminal: bool,
        train_steps: int = 1,
    ) -> dict[str, Any]:
        surprise = self._prediction_error(previous_obs, int(action), next_obs)
        transition = ReplayTransition(
            previous_image=np.asarray(previous_obs["image"], dtype=np.int64).copy(),
            previous_direction=int(previous_obs.get("direction", -1)),
            action=int(action),
            next_image=np.asarray(next_obs["image"], dtype=np.int64).copy(),
            next_direction=int(next_obs.get("direction", -1)),
            terminal=bool(terminal),
            step=int(step),
        )
        self.replay.append(transition)
        for _ in range(max(0, int(train_steps))):
            self.train_step()

        stats = self.action_stats.setdefault(int(action), {"samples": 0.0, "error_ema": surprise})
        stats["samples"] += 1.0
        stats["error_ema"] = surprise if stats["samples"] == 1 else 0.9 * stats["error_ema"] + 0.1 * surprise
        event = {
            "step": int(step),
            "action": action_name,
            "action_value": int(action),
            "prediction_error": round(surprise, 8),
            "action_error_ema": round(stats["error_ema"], 8),
            "action_samples": int(stats["samples"]),
            "blocked": bool(blocked),
            "reward": float(reward),
            "train_loss": None if self.latest_loss is None else round(self.latest_loss, 8),
        }
        self.recent_surprises.append(event)
        return event

    def train_step(self) -> float | None:
        starts = self._valid_starts()
        if len(self.replay) < self.warmup or not starts:
            return None
        chosen = [self.rng.choice(starts) for _ in range(self.batch_size)]
        replay = list(self.replay)
        first = [replay[index] for index in chosen]
        image, direction = self._batch_observations(
            [row.previous_image for row in first], [row.previous_direction for row in first]
        )
        latent = self.context_encoder(image, direction)
        context_latent = latent
        prediction_losses = []
        target_latents = []
        for offset in range(self.horizon):
            rows = [replay[index + offset] for index in chosen]
            actions = torch.tensor([row.action for row in rows], dtype=torch.long, device=self.device)
            latent = self.predictor(latent, actions)
            target_image, target_direction = self._batch_observations(
                [row.next_image for row in rows], [row.next_direction for row in rows]
            )
            with torch.no_grad():
                target = self.target_encoder(target_image, target_direction)
            prediction_losses.append(F.smooth_l1_loss(latent, target))
            target_latents.append(target)

        prediction_loss = torch.stack(prediction_losses).mean()
        variance_loss = self._variance_loss(context_latent) + self._variance_loss(torch.cat(target_latents, dim=0))
        covariance_loss = self._covariance_loss(context_latent)
        loss = prediction_loss + 0.05 * variance_loss + 0.005 * covariance_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.context_encoder.parameters()) + list(self.predictor.parameters()), 1.0)
        self.optimizer.step()
        self._update_target_encoder()
        self.total_train_steps += 1
        self.latest_loss = float(loss.detach().cpu())
        return self.latest_loss

    def evidence(self, limit: int = 8) -> dict[str, Any]:
        ranked = sorted(self.recent_surprises, key=lambda row: row["prediction_error"], reverse=True)[:limit]
        per_action = [
            {
                "action_value": action,
                "samples": int(stats["samples"]),
                "prediction_error_ema": round(stats["error_ema"], 8),
            }
            for action, stats in sorted(self.action_stats.items())
        ]
        maturity = "warming_up" if len(self.replay) < self.warmup else "training"
        return {
            "kind": "full_jepa_action_conditioned_world_model",
            "maturity": maturity,
            "replay_transitions": len(self.replay),
            "train_steps": self.total_train_steps,
            "latest_train_loss": None if self.latest_loss is None else round(self.latest_loss, 8),
            "latent_dim": self.latent_dim,
            "prediction_horizon": self.horizon,
            "trained_components": ["context_encoder", "action_conditioned_predictor"],
            "ema_component": "target_encoder",
            "per_action": per_action,
            "highest_recent_prediction_errors": ranked,
            "interpretation": (
                "Use repeated errors after warmup as evidence. High error suggests missing dynamics knowledge; "
                "low error with continued task failure suggests the explicit skill or policy is deficient."
            ),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_count": self.action_count,
                "latent_dim": self.latent_dim,
                "context_encoder": self.context_encoder.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_train_steps": self.total_train_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if int(checkpoint["action_count"]) != self.action_count or int(checkpoint["latent_dim"]) != self.latent_dim:
            raise ValueError("JEPA checkpoint architecture does not match this run.")
        self.context_encoder.load_state_dict(checkpoint["context_encoder"])
        self.target_encoder.load_state_dict(checkpoint["target_encoder"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_train_steps = int(checkpoint.get("total_train_steps", 0))

    @torch.no_grad()
    def _prediction_error(self, previous_obs: dict[str, Any], action: int, next_obs: dict[str, Any]) -> float:
        image, direction = self._batch_observations(
            [np.asarray(previous_obs["image"])], [int(previous_obs.get("direction", -1))]
        )
        next_image, next_direction = self._batch_observations(
            [np.asarray(next_obs["image"])], [int(next_obs.get("direction", -1))]
        )
        latent = self.context_encoder(image, direction)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        prediction = self.predictor(latent, action_tensor)
        target = self.target_encoder(next_image, next_direction)
        return float(F.mse_loss(prediction, target).cpu())

    def _valid_starts(self) -> list[int]:
        replay = list(self.replay)
        starts = []
        for start in range(len(replay) - self.horizon + 1):
            rows = replay[start : start + self.horizon]
            if any(row.terminal for row in rows[:-1]):
                continue
            if any(rows[index + 1].step != rows[index].step + 1 for index in range(len(rows) - 1)):
                continue
            starts.append(start)
        return starts

    def _batch_observations(
        self, images: list[np.ndarray], directions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.as_tensor(np.stack(images), dtype=torch.long, device=self.device)
        direction = torch.tensor(directions, dtype=torch.long, device=self.device)
        return image, direction

    @staticmethod
    def _variance_loss(latent: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(0.05 - std).mean()

    @staticmethod
    def _covariance_loss(latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[0] < 2:
            return latent.new_tensor(0.0)
        centered = latent - latent.mean(dim=0)
        covariance = centered.T @ centered / (latent.shape[0] - 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        return off_diagonal.square().mean()

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.mul_(self.ema_decay).add_(context, alpha=1.0 - self.ema_decay)
