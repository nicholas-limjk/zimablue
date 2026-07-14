from __future__ import annotations

import copy
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .jepa_control_agent import BeliefObservation, ControlTransition


class MinimalEncoder(nn.Module):
    """Small local encoder deliberately incapable of storing an episode map."""

    def __init__(self, action_count: int, latent_dim: int = 16):
        super().__init__()
        self.object_embedding = nn.Embedding(32, 8)
        self.color_embedding = nn.Embedding(16, 3)
        self.state_embedding = nn.Embedding(8, 2)
        self.direction_embedding = nn.Embedding(5, 4)
        self.previous_action_embedding = nn.Embedding(action_count + 1, 4)
        self.vision = nn.Sequential(
            nn.Conv2d(13, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.project = nn.Sequential(nn.Linear(64 + 4 + 4 + 1, 32), nn.GELU(), nn.Linear(32, latent_dim))
        self.action_count = int(action_count)

    def forward(
        self,
        image: torch.Tensor,
        direction: torch.Tensor,
        previous_action: torch.Tensor,
        carrying: torch.Tensor,
    ) -> torch.Tensor:
        cells = torch.cat(
            (
                self.object_embedding(image[..., 0].clamp(0, 31)),
                self.color_embedding(image[..., 1].clamp(0, 15)),
                self.state_embedding(image[..., 2].clamp(0, 7)),
            ),
            dim=-1,
        ).permute(0, 3, 1, 2)
        vision = self.vision(cells).flatten(1)
        direction = torch.where((direction >= 0) & (direction < 4), direction, torch.full_like(direction, 4))
        previous_action = torch.where(
            (previous_action >= 0) & (previous_action < self.action_count),
            previous_action,
            torch.full_like(previous_action, self.action_count),
        )
        features = torch.cat(
            (
                vision,
                self.direction_embedding(direction),
                self.previous_action_embedding(previous_action),
                carrying.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        return F.normalize(self.project(features), dim=-1)


class MinimalPredictor(nn.Module):
    def __init__(self, action_count: int, latent_dim: int):
        super().__init__()
        self.action_embedding = nn.Embedding(action_count, 4)
        self.network = nn.Sequential(nn.Linear(latent_dim + 4, 32), nn.GELU(), nn.Linear(32, latent_dim))

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.network(torch.cat((latent, self.action_embedding(action)), dim=-1))
        return F.normalize(latent + delta, dim=-1)


class MinimalJepaAgent:
    """Prediction-only JEPA for testing representation value independently of control."""

    def __init__(
        self,
        action_count: int,
        *,
        latent_dim: int = 16,
        batch_size: int = 32,
        warmup: int = 256,
        replay_capacity: int = 50_000,
        learning_rate: float = 3e-4,
        ema_decay: float = 0.995,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.action_count = int(action_count)
        self.latent_dim = int(latent_dim)
        self.world_feature_size = self.latent_dim + self.action_count
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.ema_decay = float(ema_decay)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.encoder = MinimalEncoder(self.action_count, self.latent_dim).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device).requires_grad_(False)
        self.predictor = MinimalPredictor(self.action_count, self.latent_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.replay: deque[ControlTransition] = deque(maxlen=int(replay_capacity))
        self.total_train_steps = 0
        self.latest_metrics: dict[str, float] = {}

    def observe(self, transition: ControlTransition) -> None:
        self.replay.append(transition)

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < max(self.warmup, self.batch_size):
            return None
        batch = self.rng.sample(list(self.replay), self.batch_size)
        state = self._encode_batch([row.state for row in batch], target=False)
        actions = torch.tensor([row.action for row in batch], dtype=torch.long, device=self.device)
        with torch.no_grad():
            next_target = self._encode_batch([row.next_state for row in batch], target=True)
        predicted_next = self.predictor(state, actions)
        jepa_loss = F.smooth_l1_loss(predicted_next, next_target)
        variance_loss = self._variance_loss(state)
        loss = jepa_loss + 0.05 * variance_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.predictor.parameters()), 1.0)
        self.optimizer.step()
        self._ema_update()
        self.total_train_steps += 1
        self.latest_metrics = {
            "loss": float(loss.detach().cpu()),
            "jepa_loss": float(jepa_loss.detach().cpu()),
            "variance_loss": float(variance_loss.detach().cpu()),
        }
        return dict(self.latest_metrics)

    @torch.no_grad()
    def world_features(self, state: BeliefObservation) -> np.ndarray:
        latent = self._encode_batch([state], target=False)
        actions = torch.arange(self.action_count, dtype=torch.long, device=self.device)
        repeated = latent.expand(self.action_count, -1)
        imagined = self.predictor(repeated, actions)
        change = (imagined - repeated).square().mean(dim=1).sqrt()
        return torch.cat((latent[0], change), dim=0).cpu().numpy().astype(np.float64, copy=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_count": self.action_count,
                "latent_dim": self.latent_dim,
                "encoder": self.encoder.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_train_steps": self.total_train_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        if int(data["action_count"]) != self.action_count or int(data["latent_dim"]) != self.latent_dim:
            raise ValueError("Minimal JEPA checkpoint architecture mismatch.")
        for name in ("encoder", "target_encoder", "predictor"):
            getattr(self, name).load_state_dict(data[name])
        self.optimizer.load_state_dict(data["optimizer"])
        self.total_train_steps = int(data.get("total_train_steps", 0))

    def _encode_batch(self, states: list[BeliefObservation], *, target: bool) -> torch.Tensor:
        image = torch.as_tensor(np.stack([state.image for state in states]), dtype=torch.long, device=self.device)
        direction = torch.tensor([state.direction for state in states], dtype=torch.long, device=self.device)
        previous_action = torch.tensor([state.previous_action for state in states], dtype=torch.long, device=self.device)
        carrying = torch.tensor([state.carrying for state in states], dtype=torch.float32, device=self.device)
        encoder = self.target_encoder if target else self.encoder
        return encoder(image, direction, previous_action, carrying)

    @staticmethod
    def _variance_loss(latent: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(0.05 - std).mean()

    @torch.no_grad()
    def _ema_update(self) -> None:
        for target_parameter, online_parameter in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_parameter.mul_(self.ema_decay).add_(online_parameter, alpha=1.0 - self.ema_decay)

    def eval(self) -> None:
        self.encoder.eval()
        self.target_encoder.eval()
        self.predictor.eval()
