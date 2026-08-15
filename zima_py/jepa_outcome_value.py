from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class JepaOutcomeValueHead(nn.Module):
    """Predict generic episode outcome from a JEPA belief and candidate latent.

    The head deliberately has no terrain-, goal-, barrier-, novelty-, or map-specific
    outputs.  ``distance`` is only the number of actions required to realize the
    candidate (an imagined rollout horizon or an experienced graph-path length).
    """

    def __init__(
        self,
        latent_size: int = 128,
        hidden_size: int = 128,
        *,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.hidden_size = int(hidden_size)
        self.device = torch.device(device)
        self.body = nn.Sequential(
            nn.Linear(2 * self.latent_size + 1, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.success_head = nn.Linear(self.hidden_size, 1)
        self.remaining_head = nn.Linear(self.hidden_size, 1)
        self.to(self.device)

    def forward(
        self,
        beliefs: torch.Tensor,
        candidates: torch.Tensor,
        distances: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        belief = beliefs.flatten(1)
        candidate = candidates.flatten(1)
        distance = distances.float().reshape(-1, 1).clamp(0.0, 1.0)
        hidden = self.body(torch.cat((belief, candidate, distance), dim=-1))
        return {
            "success_logit": self.success_head(hidden).squeeze(-1),
            "remaining": torch.sigmoid(self.remaining_head(hidden).squeeze(-1)),
        }

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_size": self.latent_size,
                "hidden_size": self.hidden_size,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "JepaOutcomeValueHead":
        data = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            latent_size=int(data["latent_size"]),
            hidden_size=int(data["hidden_size"]),
            device=device,
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model
