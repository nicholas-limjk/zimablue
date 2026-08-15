from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


FORWARD_VECTORS = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


@dataclass
class EvidenceMapState:
    x: int = 0
    y: int = 0
    direction: int = 0
    evidence: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    evidence_strength: dict[tuple[int, int], float] = field(default_factory=dict)
    visits: dict[tuple[int, int], int] = field(default_factory=lambda: {(0, 0): 1})
    previous_latent: np.ndarray | None = None
    last_motion_probability: float = 0.0


class JepaEvidenceMapHead(nn.Module):
    """Decode observable terrain evidence and integrate it in start-relative coordinates."""

    terrain_classes = ("unseen", "free", "wall", "lava", "goal")
    view_size = 7
    feature_size = 25
    route_size = 0
    hidden_size = 0

    def __init__(
        self,
        latent_size: int = 128,
        action_count: int = 7,
        latent_channels: int = 8,
        spatial_size: int = 4,
        *,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.action_count = int(action_count)
        self.latent_channels = int(latent_channels)
        self.spatial_size = int(spatial_size)
        self.exploration_size = 0
        self.device = torch.device(device)
        if self.latent_channels * self.spatial_size * self.spatial_size != self.latent_size:
            raise ValueError("Evidence decoder latent geometry does not match latent_size")
        self.terrain_decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, len(self.terrain_classes), kernel_size=1),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(2 * self.latent_size + self.action_count + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("calibration_temperature", torch.ones(()))
        self.register_buffer("calibration_bias", torch.zeros(len(self.terrain_classes)))
        self.register_buffer("observed_terrain_prior", torch.full((4,), 0.25))
        self._terrain_cache: dict[bytes, np.ndarray] = {}
        self.to(self.device)

    def terrain_logits(self, latents: torch.Tensor) -> torch.Tensor:
        leading = latents.shape[:-1]
        spatial = latents.reshape(-1, self.latent_channels, self.spatial_size, self.spatial_size)
        spatial = F.interpolate(spatial, size=(self.view_size, self.view_size), mode="bilinear", align_corners=False)
        logits = self.terrain_decoder(spatial).permute(0, 2, 3, 1)
        return logits.reshape(*leading, self.view_size, self.view_size, len(self.terrain_classes))

    def motion_logits(
        self,
        previous_latents: torch.Tensor,
        current_latents: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:
        actions = previous_actions.clamp(min=-1, max=self.action_count - 1) + 1
        one_hot = F.one_hot(actions, num_classes=self.action_count + 1).to(current_latents.dtype)
        return self.motion_head(
            torch.cat((previous_latents, current_latents, one_hot), dim=-1)
        ).squeeze(-1)

    def forward(self, latents: torch.Tensor, previous_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        previous_latents = torch.cat((torch.zeros_like(latents[:, :1]), latents[:, :-1]), dim=1)
        return {
            "terrain_logits": self.terrain_logits(latents),
            "motion_logits": self.motion_logits(previous_latents, latents, previous_actions),
        }

    def calibrated_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.calibration_temperature.clamp_min(0.05)
        return torch.softmax(logits / temperature + self.calibration_bias, dim=-1)

    def set_calibration(
        self,
        temperature: torch.Tensor | float,
        bias: torch.Tensor,
        observed_prior: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.calibration_temperature.copy_(torch.as_tensor(temperature, device=self.device))
            self.calibration_bias.copy_(torch.as_tensor(bias, device=self.device))
            prior = torch.as_tensor(observed_prior, device=self.device).clamp_min(1e-6)
            self.observed_terrain_prior.copy_(prior / prior.sum())
        self._terrain_cache.clear()

    @torch.no_grad()
    def step(
        self,
        latent: np.ndarray,
        previous_action: int,
        hidden: EvidenceMapState | None = None,
    ) -> tuple[np.ndarray, EvidenceMapState]:
        state = EvidenceMapState() if hidden is None else hidden
        current = np.asarray(latent, dtype=np.float32)
        motion_probability = 0.0
        if previous_action == 0:
            state.direction = (state.direction - 1) % 4
        elif previous_action == 1:
            state.direction = (state.direction + 1) % 4
        elif previous_action == 2 and state.previous_latent is not None:
            previous_tensor = torch.as_tensor(state.previous_latent, device=self.device).reshape(1, -1)
            current_tensor = torch.as_tensor(current, device=self.device).reshape(1, -1)
            action_tensor = torch.as_tensor([previous_action], dtype=torch.long, device=self.device)
            motion_probability = float(
                torch.sigmoid(self.motion_logits(previous_tensor, current_tensor, action_tensor))[0]
            )
            if motion_probability >= 0.5:
                dx, dy = FORWARD_VECTORS[state.direction]
                state.x += dx
                state.y += dy
                position = (state.x, state.y)
                state.visits[position] = state.visits.get(position, 0) + 1
        state.last_motion_probability = motion_probability

        cache_key = current.tobytes()
        probabilities = self._terrain_cache.get(cache_key)
        if probabilities is None:
            latent_tensor = torch.as_tensor(current, device=self.device).reshape(1, -1)
            logits = self.terrain_logits(latent_tensor)[0]
            probabilities = self.calibrated_probabilities(logits).cpu().numpy()
            self._terrain_cache[cache_key] = probabilities
        self._integrate(state, probabilities)
        state.previous_latent = current.copy()
        return self._features(state), state

    def _integrate(self, state: EvidenceMapState, probabilities: np.ndarray) -> None:
        forward_x, forward_y = FORWARD_VECTORS[state.direction]
        right_x, right_y = -forward_y, forward_x
        for view_x in range(self.view_size):
            for view_y in range(self.view_size):
                row = probabilities[view_y, view_x]
                observed_weight = float(1.0 - row[0])
                if observed_weight < 0.2:
                    continue
                terrain = row[1:].astype(np.float64)
                terrain /= max(float(terrain.sum()), 1e-9)
                forward = self.view_size - 1 - view_y
                lateral = view_x - self.view_size // 2
                position = (
                    state.x + forward_x * forward + right_x * lateral,
                    state.y + forward_y * forward + right_y * lateral,
                )
                if position not in state.evidence:
                    state.evidence[position] = np.zeros(4, dtype=np.float64)
                prior = self.observed_terrain_prior.detach().cpu().numpy().astype(np.float64)
                likelihood_ratio = np.log(np.clip(terrain, 1e-6, 1.0) / np.clip(prior, 1e-6, 1.0))
                # Re-observation discounts stale claims before applying bounded
                # likelihood evidence, so contradictions can genuinely revise a cell.
                state.evidence[position] *= 0.85
                state.evidence[position] += observed_weight * np.clip(likelihood_ratio, -2.0, 2.0)
                state.evidence[position] = np.clip(state.evidence[position], -8.0, 8.0)
                old_strength = state.evidence_strength.get(position, 0.0)
                state.evidence_strength[position] = min(5.0, 0.85 * old_strength + observed_weight)
        current = (state.x, state.y)
        if current not in state.evidence:
            state.evidence[current] = np.zeros(4, dtype=np.float64)
        state.evidence[current] *= 0.85
        state.evidence[current] += np.asarray((2.0, -1.0, -2.0, -1.0))
        state.evidence[current] = np.clip(state.evidence[current], -8.0, 8.0)
        state.evidence_strength[current] = min(
            5.0, 0.85 * state.evidence_strength.get(current, 0.0) + 1.0
        )

    @staticmethod
    def _posterior(state: EvidenceMapState, position: tuple[int, int]) -> tuple[np.ndarray, float]:
        log_likelihood = state.evidence.get(position)
        strength = state.evidence_strength.get(position, 0.0)
        if log_likelihood is None or strength < 0.2:
            return np.zeros(4, dtype=np.float64), 1.0
        prior = np.asarray((0.25, 0.25, 0.25, 0.25), dtype=np.float64)
        # The stored values are evidence relative to the empirical class prior;
        # a uniform decision prior keeps action queries layout-agnostic.
        scores = np.log(prior) + log_likelihood
        scores -= scores.max()
        posterior = np.exp(scores)
        posterior /= posterior.sum()
        confidence = min(1.0, strength / 3.0)
        return posterior, 1.0 - confidence

    def _features(self, state: EvidenceMapState) -> np.ndarray:
        values: list[float] = []
        for direction in ((state.direction - 1) % 4, (state.direction + 1) % 4, state.direction):
            dx, dy = FORWARD_VECTORS[direction]
            position = (state.x + dx, state.y + dy)
            posterior, unknown = self._posterior(state, position)
            values.extend(
                (
                    float(posterior[0] + posterior[3]),
                    float(posterior[2]),
                    float(posterior[1]),
                    unknown,
                    min(1.0, state.visits.get(position, 0) / 5.0),
                )
            )

        known = [
            position for position in state.evidence if state.evidence_strength.get(position, 0.0) >= 0.2
        ]
        confidence = 0.0 if not known else float(
            np.mean([min(1.0, state.evidence_strength[position] / 3.0) for position in known])
        )
        values.extend(
            (
                min(1.0, len(known) / 81.0),
                confidence,
                state.last_motion_probability,
                min(1.0, state.visits.get((state.x, state.y), 0) / 5.0),
            )
        )
        values.extend(self._target_relative(state, class_index=3))
        values.extend(self._frontier_relative(state))
        return np.clip(np.asarray(values, dtype=np.float64), -2.0, 2.0)

    def _target_relative(self, state: EvidenceMapState, class_index: int) -> tuple[float, float, float]:
        candidates = []
        for position in state.evidence:
            posterior, unknown = self._posterior(state, position)
            if unknown < 0.8:
                candidates.append((float(posterior[class_index]), position))
        if not candidates or max(candidates)[0] < 0.4:
            return 0.0, 0.0, 0.0
        _, (target_x, target_y) = max(candidates)
        return self._relative(state, target_x, target_y) + (1.0,)

    def _frontier_relative(self, state: EvidenceMapState) -> tuple[float, float, float]:
        candidates = []
        for x, y in state.evidence:
            posterior, unknown = self._posterior(state, (x, y))
            if unknown >= 0.8 or posterior[0] + posterior[3] < 0.5:
                continue
            if any((x + dx, y + dy) not in state.evidence for dx, dy in FORWARD_VECTORS.values()):
                candidates.append((abs(x - state.x) + abs(y - state.y), x, y))
        if not candidates:
            return 0.0, 0.0, 0.0
        _, x, y = min(candidates)
        return self._relative(state, x, y) + (1.0,)

    @staticmethod
    def _relative(state: EvidenceMapState, x: int, y: int) -> tuple[float, float]:
        dx, dy = x - state.x, y - state.y
        forward_x, forward_y = FORWARD_VECTORS[state.direction]
        right_x, right_y = -forward_y, forward_x
        return (
            (dx * forward_x + dy * forward_y) / 8.0,
            (dx * right_x + dy * right_y) / 8.0,
        )

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": "evidence_map",
                "latent_size": self.latent_size,
                "action_count": self.action_count,
                "latent_channels": self.latent_channels,
                "spatial_size": self.spatial_size,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "JepaEvidenceMapHead":
        data = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            latent_size=int(data["latent_size"]),
            action_count=int(data["action_count"]),
            latent_channels=int(data["latent_channels"]),
            spatial_size=int(data["spatial_size"]),
            device=device,
        )
        model.load_state_dict(data["state_dict"], strict=False)
        model.eval()
        return model

    @classmethod
    def checkpoint_metadata(cls, path: Path) -> dict:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return dict(data.get("metadata", {}))
