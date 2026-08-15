from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RgbTransition:
    image: np.ndarray
    action: int
    next_image: np.ndarray


@dataclass(frozen=True)
class TemporalRgbTransition:
    frames: np.ndarray
    previous_actions: np.ndarray
    future_actions: np.ndarray
    future_images: np.ndarray
    terminal_step: int | None = None


class SpatialRgbEncoder(nn.Module):
    """Encode a whole 56x56 RGB observation without collapsing spatial layout."""

    def __init__(self, latent_channels: int = 8, spatial_size: int = 4) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.spatial_size = int(spatial_size)
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size)),
            nn.Conv2d(24, self.latent_channels, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        latent = self.network(image)
        return F.normalize(latent.flatten(1), dim=1).reshape_as(latent)


class SpatialActionPredictor(nn.Module):
    def __init__(self, action_count: int, latent_channels: int = 8) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(int(action_count), 8)
        self.network = nn.Sequential(
            nn.Conv2d(latent_channels + 8, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        height, width = latent.shape[-2:]
        action_map = self.action_embedding(action)[:, :, None, None].expand(-1, -1, height, width)
        predicted = latent + self.network(torch.cat((latent, action_map), dim=1))
        return F.normalize(predicted.flatten(1), dim=1).reshape_as(latent)


class TemporalRgbEncoder(nn.Module):
    """Fuse a fixed short history while keeping the resulting representation spatial."""

    def __init__(self, action_count: int, context_length: int = 4, latent_channels: int = 8, spatial_size: int = 4) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.context_length = int(context_length)
        self.latent_channels = int(latent_channels)
        self.spatial_size = int(spatial_size)
        self.frame_encoder = SpatialRgbEncoder(self.latent_channels, self.spatial_size)
        self.previous_action_embedding = nn.Embedding(self.action_count + 1, 4)
        fusion_channels = self.context_length * self.latent_channels + (self.context_length - 1) * 4
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, self.latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, frames: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        if time != self.context_length:
            raise ValueError(f"Expected {self.context_length} frames, received {time}.")
        encoded = self.frame_encoder(frames.reshape(batch * time, channels, height, width))
        encoded = encoded.reshape(batch, time * self.latent_channels, self.spatial_size, self.spatial_size)
        action_indices = torch.where(
            (previous_actions >= 0) & (previous_actions < self.action_count),
            previous_actions,
            torch.full_like(previous_actions, self.action_count),
        )
        action_features = self.previous_action_embedding(action_indices)
        action_features = action_features.reshape(batch, -1, 1, 1).expand(-1, -1, self.spatial_size, self.spatial_size)
        fused = self.fusion(torch.cat((encoded, action_features), dim=1))
        return F.normalize(fused.flatten(1), dim=1).reshape_as(fused)


class RgbJepaAgent:
    """Small action-conditioned JEPA whose public representation is 8x4x4."""

    def __init__(
        self,
        action_count: int,
        *,
        latent_channels: int = 8,
        spatial_size: int = 4,
        batch_size: int = 64,
        warmup: int = 256,
        replay_capacity: int = 50_000,
        learning_rate: float = 3e-4,
        ema_decay: float = 0.995,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.action_count = int(action_count)
        self.latent_channels = int(latent_channels)
        self.spatial_size = int(spatial_size)
        self.feature_size = self.latent_channels * self.spatial_size * self.spatial_size
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.ema_decay = float(ema_decay)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.encoder = SpatialRgbEncoder(self.latent_channels, self.spatial_size).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device).requires_grad_(False)
        self.predictor = SpatialActionPredictor(self.action_count, self.latent_channels).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.replay: deque[RgbTransition] = deque(maxlen=int(replay_capacity))
        self.total_train_steps = 0
        self.latest_metrics: dict[str, float] = {}

    def observe(self, image: np.ndarray, action: int, next_image: np.ndarray) -> None:
        self.replay.append(
            RgbTransition(np.asarray(image, dtype=np.uint8), int(action), np.asarray(next_image, dtype=np.uint8))
        )

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < max(self.warmup, self.batch_size):
            return None
        batch = self.rng.sample(list(self.replay), self.batch_size)
        current = self.encoder(self._images([row.image for row in batch]))
        actions = torch.tensor([row.action for row in batch], dtype=torch.long, device=self.device)
        with torch.no_grad():
            target = self.target_encoder(self._images([row.next_image for row in batch]))
        predicted = self.predictor(current, actions)
        prediction_loss = F.smooth_l1_loss(predicted, target)
        variance_loss = self._variance_loss(current.flatten(1))
        loss = prediction_loss + 0.1 * variance_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.predictor.parameters()), 1.0)
        self.optimizer.step()
        self._ema_update()
        self.total_train_steps += 1
        self.latest_metrics = {
            "loss": float(loss.detach().cpu()),
            "prediction_loss": float(prediction_loss.detach().cpu()),
            "variance_loss": float(variance_loss.detach().cpu()),
        }
        return dict(self.latest_metrics)

    @torch.no_grad()
    def features(self, image: np.ndarray) -> np.ndarray:
        latent = self.encoder(self._images([image]))
        return latent[0].flatten().cpu().numpy().astype(np.float64, copy=False)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "encoder": sum(parameter.numel() for parameter in self.encoder.parameters()),
            "predictor": sum(parameter.numel() for parameter in self.predictor.parameters()),
            "total": sum(parameter.numel() for parameter in self.encoder.parameters())
            + sum(parameter.numel() for parameter in self.predictor.parameters()),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_count": self.action_count,
                "latent_channels": self.latent_channels,
                "spatial_size": self.spatial_size,
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
        architecture = (
            int(data["action_count"]),
            int(data["latent_channels"]),
            int(data["spatial_size"]),
        )
        expected = (self.action_count, self.latent_channels, self.spatial_size)
        if architecture != expected:
            raise ValueError(f"RGB JEPA checkpoint architecture {architecture} does not match {expected}.")
        self.encoder.load_state_dict(data["encoder"])
        self.target_encoder.load_state_dict(data["target_encoder"])
        self.predictor.load_state_dict(data["predictor"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.total_train_steps = int(data.get("total_train_steps", 0))

    def eval(self) -> None:
        self.encoder.eval()
        self.target_encoder.eval()
        self.predictor.eval()

    def _images(self, images: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(images)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0

    @staticmethod
    def _variance_loss(latent: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(0.05 - std).mean()

    @torch.no_grad()
    def _ema_update(self) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)


class TemporalRgbJepaAgent:
    """Four-frame JEPA: short visual history in, next-frame latent prediction out."""

    def __init__(
        self,
        action_count: int,
        *,
        context_length: int = 4,
        latent_channels: int = 8,
        spatial_size: int = 4,
        batch_size: int = 64,
        warmup: int = 256,
        replay_capacity: int = 50_000,
        learning_rate: float = 3e-4,
        ema_decay: float = 0.995,
        horizons: tuple[int, ...] = (1,),
        color_augmentation: bool = False,
        channel_permutation_invariant: bool = False,
        consistency_weight: float = 0.25,
        absorbing_weight: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.action_count = int(action_count)
        self.context_length = int(context_length)
        self.latent_channels = int(latent_channels)
        self.spatial_size = int(spatial_size)
        self.feature_size = self.latent_channels * self.spatial_size * self.spatial_size
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.ema_decay = float(ema_decay)
        self.horizons = tuple(sorted(set(int(horizon) for horizon in horizons)))
        if not self.horizons or self.horizons[0] < 1:
            raise ValueError("Temporal JEPA horizons must be positive.")
        self.max_horizon = self.horizons[-1]
        self.color_augmentation = bool(color_augmentation)
        self.channel_permutation_invariant = bool(channel_permutation_invariant)
        self.consistency_weight = float(consistency_weight)
        self.absorbing_weight = float(absorbing_weight)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.context_encoder = TemporalRgbEncoder(
            self.action_count, self.context_length, self.latent_channels, self.spatial_size
        ).to(self.device)
        self.target_frame_encoder = copy.deepcopy(self.context_encoder.frame_encoder).to(self.device).requires_grad_(False)
        self.predictor = SpatialActionPredictor(self.action_count, self.latent_channels).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.replay: deque[TemporalRgbTransition] = deque(maxlen=int(replay_capacity))
        self.total_train_steps = 0
        self.latest_metrics: dict[str, float] = {}

    def observe(
        self,
        frames: np.ndarray,
        previous_actions: np.ndarray | list[int],
        action: int,
        next_image: np.ndarray,
    ) -> None:
        frames_array = np.asarray(frames, dtype=np.uint8)
        actions_array = np.asarray(previous_actions, dtype=np.int64)
        if frames_array.shape[0] != self.context_length or actions_array.shape != (self.context_length - 1,):
            raise ValueError("Temporal JEPA observation has the wrong context length.")
        self.observe_sequence(
            frames_array,
            actions_array,
            [int(action)] * self.max_horizon,
            np.stack([np.asarray(next_image, dtype=np.uint8)] * self.max_horizon),
        )

    def observe_sequence(
        self,
        frames: np.ndarray,
        previous_actions: np.ndarray | list[int],
        future_actions: np.ndarray | list[int],
        future_images: np.ndarray,
        terminal_step: int | None = None,
    ) -> None:
        frames_array = np.asarray(frames, dtype=np.uint8)
        previous_array = np.asarray(previous_actions, dtype=np.int64)
        actions_array = np.asarray(future_actions, dtype=np.int64)
        images_array = np.asarray(future_images, dtype=np.uint8)
        if frames_array.shape[0] != self.context_length or previous_array.shape != (self.context_length - 1,):
            raise ValueError("Temporal JEPA observation has the wrong context length.")
        if actions_array.shape != (self.max_horizon,) or images_array.shape[0] != self.max_horizon:
            raise ValueError("Temporal JEPA observation has the wrong future horizon.")
        if terminal_step is not None and not 0 <= int(terminal_step) < self.max_horizon:
            raise ValueError("Temporal JEPA terminal step is outside the future horizon.")
        self.replay.append(
            TemporalRgbTransition(
                frames_array.copy(),
                previous_array.copy(),
                actions_array.copy(),
                images_array.copy(),
                None if terminal_step is None else int(terminal_step),
            )
        )

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < max(self.warmup, self.batch_size):
            return None
        batch = self.rng.sample(list(self.replay), self.batch_size)
        frames = self._frames([row.frames for row in batch])
        previous_actions = torch.as_tensor(
            np.stack([row.previous_actions for row in batch]), dtype=torch.long, device=self.device
        )
        original_context = self.context_encoder(frames, previous_actions)
        augmented_frames = self._augment_colors(frames) if self.color_augmentation else frames
        context = self.context_encoder(augmented_frames, previous_actions)
        cyclic_context = (
            self.context_encoder(self._cyclic_colors(frames), previous_actions)
            if self.color_augmentation
            else original_context
        )
        future_actions = torch.as_tensor(
            np.stack([row.future_actions for row in batch]), dtype=torch.long, device=self.device
        )
        future_images = self._future_images([row.future_images for row in batch])
        if self.color_augmentation:
            future_images = self._augment_colors(future_images)
        with torch.no_grad():
            batch_size, horizon_count, channels, height, width = future_images.shape
            targets = self.target_frame_encoder(
                future_images.reshape(batch_size * horizon_count, channels, height, width)
            ).reshape(batch_size, horizon_count, self.latent_channels, self.spatial_size, self.spatial_size)
        predicted = context
        horizon_losses = []
        absorbing_losses = []
        terminal_steps = torch.as_tensor(
            [-1 if row.terminal_step is None else row.terminal_step for row in batch],
            dtype=torch.long,
            device=self.device,
        )
        for step in range(self.max_horizon):
            predicted = self.predictor(predicted, future_actions[:, step])
            horizon = step + 1
            if horizon in self.horizons:
                horizon_losses.append(F.smooth_l1_loss(predicted, targets[:, step]))
            absorbing_mask = (terminal_steps >= 0) & (step > terminal_steps)
            if absorbing_mask.any():
                terminal_targets = targets[
                    absorbing_mask,
                    terminal_steps[absorbing_mask],
                ]
                absorbing_losses.append(
                    F.smooth_l1_loss(predicted[absorbing_mask], terminal_targets)
                )
        prediction_loss = torch.stack(horizon_losses).mean()
        absorbing_loss = (
            torch.stack(absorbing_losses).mean()
            if absorbing_losses
            else torch.zeros((), device=self.device)
        )
        consistency_loss = (
            torch.linalg.vector_norm(
                original_context.detach().flatten(1) - cyclic_context.flatten(1), dim=1
            ).mean()
            if self.color_augmentation
            else torch.zeros((), device=self.device)
        )
        variance_loss = RgbJepaAgent._variance_loss(context.flatten(1))
        loss = (
            prediction_loss
            + 0.1 * variance_loss
            + self.consistency_weight * consistency_loss
            + self.absorbing_weight * absorbing_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.context_encoder.parameters()) + list(self.predictor.parameters()), 1.0)
        self.optimizer.step()
        self._ema_update()
        self.total_train_steps += 1
        self.latest_metrics = {
            "loss": float(loss.detach().cpu()),
            "prediction_loss": float(prediction_loss.detach().cpu()),
            "variance_loss": float(variance_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "absorbing_loss": float(absorbing_loss.detach().cpu()),
        }
        return dict(self.latest_metrics)

    @torch.no_grad()
    def features(self, frames: np.ndarray, previous_actions: np.ndarray | list[int]) -> np.ndarray:
        latent = self.encode_context(frames, previous_actions)
        return latent[0].flatten().cpu().numpy().astype(np.float64, copy=False)

    @torch.no_grad()
    def encode_context(
        self,
        frames: np.ndarray,
        previous_actions: np.ndarray | list[int],
    ) -> torch.Tensor:
        """Return the spatial context latent for planning without flattening it."""
        return self.context_encoder(
            self._frames([np.asarray(frames, dtype=np.uint8)]),
            torch.as_tensor([list(previous_actions)], dtype=torch.long, device=self.device),
        )

    @torch.no_grad()
    def encode_frame(self, image: np.ndarray) -> torch.Tensor:
        """Encode one observation in the predictor's target-latent space."""
        return self.target_frame_encoder(
            self._images([np.asarray(image, dtype=np.uint8)])
        )

    @property
    def counterfactual_features_per_query(self) -> int:
        """Signed and magnitude change at every spatial cell, plus global similarity."""
        return 2 * self.spatial_size * self.spatial_size + 1

    def counterfactual_feature_size(
        self,
        horizons: tuple[int, ...],
        actions: tuple[int, ...] = (0, 1, 2),
    ) -> int:
        return len(actions) * len(horizons) * self.counterfactual_features_per_query

    def reset_predictor(self, seed: int) -> None:
        """Replace learned dynamics while leaving the trained context encoder intact."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))

            def reset(module: nn.Module) -> None:
                reset_parameters = getattr(module, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()

            self.predictor.apply(reset)
        self.predictor.eval()

    @torch.no_grad()
    def features_with_counterfactuals(
        self,
        frames: np.ndarray,
        previous_actions: np.ndarray | list[int],
        horizons: tuple[int, ...],
        actions: tuple[int, ...] = (0, 1, 2),
        *,
        include_context: bool = True,
        relative: bool = False,
    ) -> np.ndarray:
        """Encode history and summarize repeated-action latent rollouts.

        Each action/horizon query exposes the signed channel-mean change and the
        channel RMS change on the 4x4 latent grid, followed by cosine similarity
        to the current latent. This keeps spatial consequence information while
        avoiding one complete 128-value latent per imagined future.
        """
        normalized_horizons = tuple(sorted(set(int(horizon) for horizon in horizons)))
        if not normalized_horizons or normalized_horizons[0] < 1:
            raise ValueError("Counterfactual horizons must be positive.")
        normalized_actions = tuple(int(action) for action in actions)
        if any(action < 0 or action >= self.action_count for action in normalized_actions):
            raise ValueError("Counterfactual action is outside the JEPA action space.")

        latent = self.encode_context(frames, previous_actions)
        action_summaries: list[torch.Tensor] = []
        maximum_horizon = normalized_horizons[-1]
        for action in normalized_actions:
            predicted = latent
            action_tensor = torch.full((1,), action, dtype=torch.long, device=self.device)
            horizon_summaries: list[torch.Tensor] = []
            for step in range(1, maximum_horizon + 1):
                predicted = self.predictor(predicted, action_tensor)
                if step not in normalized_horizons:
                    continue
                delta = predicted - latent
                signed_spatial = delta.mean(dim=1).flatten(1)
                magnitude_spatial = delta.square().mean(dim=1).add(1e-12).sqrt().flatten(1)
                similarity = F.cosine_similarity(predicted.flatten(1), latent.flatten(1)).unsqueeze(1)
                horizon_summaries.append(
                    torch.cat((signed_spatial, magnitude_spatial, similarity), dim=1)
                )
            action_summaries.append(torch.cat(horizon_summaries, dim=1))

        counterfactuals = torch.stack(action_summaries, dim=1)
        if relative:
            counterfactuals = counterfactuals - counterfactuals.mean(dim=1, keepdim=True)
        flattened = counterfactuals.flatten(1)
        combined = (
            torch.cat((latent.flatten(1), flattened), dim=1)
            if include_context
            else flattened
        )
        return combined[0].cpu().numpy().astype(np.float64, copy=False)

    def parameter_counts(self) -> dict[str, int]:
        context = sum(parameter.numel() for parameter in self.context_encoder.parameters())
        predictor = sum(parameter.numel() for parameter in self.predictor.parameters())
        return {"context_encoder": context, "predictor": predictor, "total": context + predictor}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_count": self.action_count,
                "context_length": self.context_length,
                "latent_channels": self.latent_channels,
                "spatial_size": self.spatial_size,
                "horizons": self.horizons,
                "color_augmentation": self.color_augmentation,
                "channel_permutation_invariant": self.channel_permutation_invariant,
                "consistency_weight": self.consistency_weight,
                "absorbing_weight": self.absorbing_weight,
                "context_encoder": self.context_encoder.state_dict(),
                "target_frame_encoder": self.target_frame_encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_train_steps": self.total_train_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        architecture = tuple(int(data[name]) for name in ("action_count", "context_length", "latent_channels", "spatial_size"))
        expected = (self.action_count, self.context_length, self.latent_channels, self.spatial_size)
        if architecture != expected:
            raise ValueError(f"Temporal RGB JEPA checkpoint architecture {architecture} does not match {expected}.")
        self.context_encoder.load_state_dict(data["context_encoder"])
        self.target_frame_encoder.load_state_dict(data["target_frame_encoder"])
        self.predictor.load_state_dict(data["predictor"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.total_train_steps = int(data.get("total_train_steps", 0))
        self.horizons = tuple(int(value) for value in data.get("horizons", (1,)))
        self.max_horizon = max(self.horizons)
        self.color_augmentation = bool(data.get("color_augmentation", False))
        self.channel_permutation_invariant = bool(data.get("channel_permutation_invariant", False))
        self.consistency_weight = float(data.get("consistency_weight", self.consistency_weight))
        self.absorbing_weight = float(data.get("absorbing_weight", self.absorbing_weight))

    def eval(self) -> None:
        self.context_encoder.eval()
        self.target_frame_encoder.eval()
        self.predictor.eval()

    def _frames(self, frame_batches: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(frame_batches)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(0, 1, 4, 2, 3) / 255.0
        return torch.sort(tensor, dim=2).values if self.channel_permutation_invariant else tensor

    def _images(self, images: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(images)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        return torch.sort(tensor, dim=1).values if self.channel_permutation_invariant else tensor

    def _future_images(self, image_batches: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(image_batches)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(0, 1, 4, 2, 3) / 255.0
        return torch.sort(tensor, dim=2).values if self.channel_permutation_invariant else tensor

    def _augment_colors(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply an independently sampled channel permutation to each batch item."""
        if self.channel_permutation_invariant:
            return tensor
        permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1))
        rows = []
        for row in tensor:
            permutation = permutations[self.rng.randrange(len(permutations))]
            channel_axis = 1 if row.ndim == 4 else 0
            indices = torch.tensor(permutation, dtype=torch.long, device=row.device)
            rows.append(row.index_select(channel_axis, indices))
        return torch.stack(rows)

    def _cyclic_colors(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.channel_permutation_invariant:
            return tensor
        channel_axis = 2 if tensor.ndim == 5 else 1
        indices = torch.tensor((1, 2, 0), dtype=torch.long, device=tensor.device)
        return tensor.index_select(channel_axis, indices)

    @torch.no_grad()
    def _ema_update(self) -> None:
        for target, online in zip(self.target_frame_encoder.parameters(), self.context_encoder.frame_encoder.parameters()):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)
