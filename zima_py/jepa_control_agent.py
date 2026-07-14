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


class BeliefEncoder(nn.Module):
    """Encode partial vision plus body-derived belief state."""

    def __init__(self, action_count: int, latent_dim: int = 64):
        super().__init__()
        self.object_embedding = nn.Embedding(32, 12)
        self.color_embedding = nn.Embedding(16, 6)
        self.state_embedding = nn.Embedding(8, 4)
        self.direction_embedding = nn.Embedding(5, 6)
        self.previous_action_embedding = nn.Embedding(action_count + 1, 6)
        self.vision = nn.Sequential(
            nn.Conv2d(22, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.project = nn.Sequential(nn.Linear(32 * 4 + 6 + 6 + 1, 128), nn.GELU(), nn.Linear(128, latent_dim))
        self.action_count = action_count

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


class LatentPredictor(nn.Module):
    def __init__(self, action_count: int, latent_dim: int):
        super().__init__()
        self.action_embedding = nn.Embedding(action_count, 16)
        self.network = nn.Sequential(nn.Linear(latent_dim + 16, 128), nn.GELU(), nn.Linear(128, latent_dim))

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.network(torch.cat((latent, self.action_embedding(action)), dim=-1))
        return F.normalize(latent + delta, dim=-1)


@dataclass(frozen=True)
class BeliefObservation:
    image: np.ndarray
    direction: int
    previous_action: int
    carrying: bool


@dataclass(frozen=True)
class ControlTransition:
    state: BeliefObservation
    action: int
    reward: float
    next_state: BeliefObservation
    terminal: bool


class JepaControlAgent:
    """Local JEPA world model plus reward/Q heads for primitive action control."""

    def __init__(
        self,
        action_count: int,
        *,
        latent_dim: int = 64,
        batch_size: int = 32,
        replay_capacity: int = 50_000,
        warmup: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        ema_decay: float = 0.995,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.action_count = int(action_count)
        self.latent_dim = int(latent_dim)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.gamma = float(gamma)
        self.ema_decay = float(ema_decay)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.encoder = BeliefEncoder(action_count, latent_dim).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device).requires_grad_(False)
        self.predictor = LatentPredictor(action_count, latent_dim).to(self.device)
        self.q_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, action_count)).to(self.device)
        self.target_q_head = copy.deepcopy(self.q_head).to(self.device).requires_grad_(False)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_count, 64), nn.GELU(), nn.Linear(64, 1)
        ).to(self.device)
        parameters = list(self.encoder.parameters()) + list(self.predictor.parameters())
        parameters += list(self.q_head.parameters()) + list(self.reward_head.parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
        self.replay: deque[ControlTransition] = deque(maxlen=int(replay_capacity))
        self.total_train_steps = 0
        self.latest_metrics: dict[str, float] = {}

    @torch.no_grad()
    def act(self, state: BeliefObservation, epsilon: float, valid_actions: list[int] | None = None) -> int:
        actions = valid_actions or list(range(self.action_count))
        if self.rng.random() < epsilon:
            return int(self.rng.choice(actions))
        latent = self._encode_batch([state], target=False)
        all_actions = torch.arange(self.action_count, dtype=torch.long, device=self.device)
        repeated_latent = latent.expand(self.action_count, -1)
        imagined_next = self.predictor(repeated_latent, all_actions)
        one_hot = F.one_hot(all_actions, self.action_count).float()
        predicted_reward = self.reward_head(torch.cat((repeated_latent, one_hot), dim=-1)).squeeze(1)
        continuation = self.target_q_head(imagined_next).max(dim=1).values
        q_values = predicted_reward + self.gamma * continuation
        invalid = torch.ones(self.action_count, dtype=torch.bool, device=self.device)
        invalid[torch.tensor(actions, dtype=torch.long, device=self.device)] = False
        q_values = q_values.masked_fill(invalid, -torch.inf)
        return int(q_values.argmax().item())

    @torch.no_grad()
    def world_features(self, state: BeliefObservation) -> np.ndarray:
        """Return frozen JEPA features suitable for an external controller."""
        latent = self._encode_batch([state], target=False)
        actions = torch.arange(self.action_count, dtype=torch.long, device=self.device)
        repeated = latent.expand(self.action_count, -1)
        imagined = self.predictor(repeated, actions)
        one_hot = F.one_hot(actions, self.action_count).float()
        rewards = self.reward_head(torch.cat((repeated, one_hot), dim=-1)).squeeze(1)
        change = (imagined - repeated).square().mean(dim=1).sqrt()
        features = torch.cat((latent[0], rewards, change), dim=0)
        return features.cpu().numpy().astype(np.float64, copy=False)

    def observe(self, transition: ControlTransition) -> None:
        self.replay.append(transition)

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < max(self.warmup, self.batch_size):
            return None
        batch = self.rng.sample(list(self.replay), self.batch_size)
        state = self._encode_batch([row.state for row in batch], target=False)
        actions = torch.tensor([row.action for row in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([row.reward for row in batch], dtype=torch.float32, device=self.device)
        terminals = torch.tensor([row.terminal for row in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_target = self._encode_batch([row.next_state for row in batch], target=True)

        predicted_next = self.predictor(state, actions)
        jepa_loss = F.smooth_l1_loss(predicted_next, next_target)
        variance_loss = self._variance_loss(state)

        q_values = self.q_head(state).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.q_head(next_target).argmax(dim=1)
            next_q = self.target_q_head(next_target).gather(1, next_online.unsqueeze(1)).squeeze(1)
            td_target = rewards + self.gamma * (1.0 - terminals) * next_q
        q_loss = F.smooth_l1_loss(q_values, td_target)

        one_hot_action = F.one_hot(actions, self.action_count).float()
        predicted_reward = self.reward_head(torch.cat((state, one_hot_action), dim=-1)).squeeze(1)
        reward_loss = F.smooth_l1_loss(predicted_reward, rewards)
        loss = jepa_loss + q_loss + 0.25 * reward_loss + 0.05 * variance_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 1.0)
        self.optimizer.step()
        self._ema_update(self.target_encoder, self.encoder)
        self._ema_update(self.target_q_head, self.q_head)
        self.total_train_steps += 1
        self.latest_metrics = {
            "loss": float(loss.detach().cpu()),
            "jepa_loss": float(jepa_loss.detach().cpu()),
            "q_loss": float(q_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
        }
        return dict(self.latest_metrics)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_count": self.action_count,
                "latent_dim": self.latent_dim,
                "encoder": self.encoder.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "q_head": self.q_head.state_dict(),
                "target_q_head": self.target_q_head.state_dict(),
                "reward_head": self.reward_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_train_steps": self.total_train_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        if int(data["action_count"]) != self.action_count or int(data["latent_dim"]) != self.latent_dim:
            raise ValueError("Controller checkpoint architecture mismatch.")
        for name in ("encoder", "target_encoder", "predictor", "q_head", "target_q_head", "reward_head"):
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
    def _ema_update(self, target: nn.Module, online: nn.Module) -> None:
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
            target_parameter.mul_(self.ema_decay).add_(online_parameter, alpha=1.0 - self.ema_decay)


def belief_observation(obs: dict[str, Any], previous_action: int, carrying: bool) -> BeliefObservation:
    return BeliefObservation(
        image=np.asarray(obs["image"], dtype=np.int64).copy(),
        direction=int(obs.get("direction", -1)),
        previous_action=int(previous_action),
        carrying=bool(carrying),
    )
