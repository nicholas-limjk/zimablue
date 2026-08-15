from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class JepaRolloutOutcomeHead(nn.Module):
    """Decode control outcomes specifically from JEPA-imagined endpoint latents."""

    def __init__(
        self,
        latent_size: int = 128,
        max_horizon: int = 4,
        hidden_size: int = 64,
        *,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.max_horizon = int(max_horizon)
        self.hidden_size = int(hidden_size)
        self.device = torch.device(device)
        self.horizon_embedding = nn.Embedding(self.max_horizon + 1, 8)
        self.body = nn.Sequential(
            nn.Linear(self.latent_size + 8, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.route_head = nn.Linear(self.hidden_size, 3)
        self.collision_head = nn.Linear(self.hidden_size, 3)
        self.goal_head = nn.Linear(self.hidden_size, 2)
        self.to(self.device)

    def forward(self, latents: torch.Tensor, horizons: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = latents.flatten(1)
        horizon = horizons.clamp(min=0, max=self.max_horizon)
        hidden = self.body(torch.cat((flat, self.horizon_embedding(horizon)), dim=-1))
        return {
            "route_logits": self.route_head(hidden),
            "collision_logits": self.collision_head(hidden),
            "goal": torch.tanh(self.goal_head(hidden)),
        }

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_size": self.latent_size,
                "max_horizon": self.max_horizon,
                "hidden_size": self.hidden_size,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "JepaRolloutOutcomeHead":
        data = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            latent_size=int(data["latent_size"]),
            max_horizon=int(data["max_horizon"]),
            hidden_size=int(data["hidden_size"]),
            device=device,
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model

