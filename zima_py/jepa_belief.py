from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class JepaBeliefHead(nn.Module):
    """Recurrent task-belief decoder trained on frozen temporal JEPA latents."""

    route_size = 3
    collision_size = 3
    goal_size = 4
    barrier_size = 2

    def __init__(
        self,
        latent_size: int = 128,
        action_count: int = 7,
        hidden_size: int = 64,
        exploration_size: int = 0,
        *,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.action_count = int(action_count)
        self.hidden_size = int(hidden_size)
        self.exploration_size = int(exploration_size)
        self.device = torch.device(device)
        self.gru = nn.GRU(
            self.latent_size + self.action_count + 1,
            self.hidden_size,
            batch_first=True,
        )
        self.route_head = nn.Linear(self.hidden_size, self.route_size)
        self.collision_head = nn.Linear(self.hidden_size, self.collision_size)
        self.goal_head = nn.Linear(self.hidden_size, self.goal_size)
        self.barrier_head = nn.Linear(self.hidden_size, self.barrier_size)
        self.exploration_head = (
            nn.Linear(self.hidden_size, self.exploration_size)
            if self.exploration_size
            else None
        )
        self.to(self.device)

    @property
    def feature_size(self) -> int:
        return (
            self.hidden_size
            + self.route_size
            + self.collision_size
            + self.goal_size
            + self.barrier_size
            + self.exploration_size
        )

    def forward(
        self,
        latents: torch.Tensor,
        previous_actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        actions = previous_actions.clamp(min=-1, max=self.action_count - 1) + 1
        action_one_hot = torch.nn.functional.one_hot(
            actions, num_classes=self.action_count + 1
        ).to(dtype=latents.dtype)
        recurrent, next_hidden = self.gru(torch.cat((latents, action_one_hot), dim=-1), hidden)
        outputs = {
            "hidden": recurrent,
            "route_logits": self.route_head(recurrent),
            "collision_logits": self.collision_head(recurrent),
            "goal": torch.tanh(self.goal_head(recurrent)),
            "barrier_logits": self.barrier_head(recurrent),
        }
        if self.exploration_head is not None:
            outputs["exploration_logits"] = self.exploration_head(recurrent)
        return outputs, next_hidden

    @torch.no_grad()
    def step(
        self,
        latent: np.ndarray,
        previous_action: int,
        hidden: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        latent_tensor = torch.as_tensor(
            np.asarray(latent, dtype=np.float32), device=self.device
        ).reshape(1, 1, self.latent_size)
        action_tensor = torch.as_tensor(
            [[int(previous_action)]], dtype=torch.long, device=self.device
        )
        outputs, next_hidden = self(latent_tensor, action_tensor, hidden)
        feature_parts = [
                outputs["hidden"],
                torch.softmax(outputs["route_logits"], dim=-1),
                torch.sigmoid(outputs["collision_logits"]),
                outputs["goal"],
                torch.sigmoid(outputs["barrier_logits"]),
        ]
        if "exploration_logits" in outputs:
            feature_parts.append(torch.sigmoid(outputs["exploration_logits"]))
        features = torch.cat(feature_parts, dim=-1)
        return features[0, 0].cpu().numpy().astype(np.float64), next_hidden

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": "gru",
                "latent_size": self.latent_size,
                "action_count": self.action_count,
                "hidden_size": self.hidden_size,
                "exploration_size": self.exploration_size,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "JepaBeliefHead":
        data = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            latent_size=int(data["latent_size"]),
            action_count=int(data["action_count"]),
            hidden_size=int(data["hidden_size"]),
            exploration_size=int(data.get("exploration_size", 0)),
            device=device,
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model

    @classmethod
    def checkpoint_metadata(cls, path: Path) -> dict:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return dict(data.get("metadata", {}))


class TinyRecursiveBeliefHead(nn.Module):
    """Persistent belief state refined repeatedly by one tiny shared network."""

    route_size = 3
    collision_size = 3
    goal_size = 4
    barrier_size = 2

    def __init__(
        self,
        latent_size: int = 128,
        action_count: int = 7,
        hidden_size: int = 64,
        exploration_size: int = 0,
        recursion_depth: int = 6,
        refinement_size: int = 96,
        *,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if recursion_depth < 1:
            raise ValueError("recursion_depth must be at least one")
        self.latent_size = int(latent_size)
        self.action_count = int(action_count)
        self.hidden_size = int(hidden_size)
        self.exploration_size = int(exploration_size)
        self.recursion_depth = int(recursion_depth)
        self.refinement_size = int(refinement_size)
        self.device = torch.device(device)
        evidence_size = self.latent_size + self.action_count + 1
        refinement_input = 3 * self.hidden_size
        self.evidence_projection = nn.Sequential(
            nn.Linear(evidence_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )
        self.refiner = nn.Sequential(
            nn.LayerNorm(refinement_input),
            nn.Linear(refinement_input, self.refinement_size),
            nn.GELU(),
            nn.Linear(self.refinement_size, 2 * self.hidden_size),
        )
        self.route_head = nn.Linear(self.hidden_size, self.route_size)
        self.collision_head = nn.Linear(self.hidden_size, self.collision_size)
        self.goal_head = nn.Linear(self.hidden_size, self.goal_size)
        self.barrier_head = nn.Linear(self.hidden_size, self.barrier_size)
        self.exploration_head = (
            nn.Linear(self.hidden_size, self.exploration_size)
            if self.exploration_size
            else None
        )
        self.to(self.device)

    @property
    def feature_size(self) -> int:
        return (
            self.hidden_size
            + self.route_size
            + self.collision_size
            + self.goal_size
            + self.barrier_size
            + self.exploration_size
        )

    def _decode(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {
            "hidden": hidden,
            "route_logits": self.route_head(hidden),
            "collision_logits": self.collision_head(hidden),
            "goal": torch.tanh(self.goal_head(hidden)),
            "barrier_logits": self.barrier_head(hidden),
        }
        if self.exploration_head is not None:
            outputs["exploration_logits"] = self.exploration_head(hidden)
        return outputs

    def forward(
        self,
        latents: torch.Tensor,
        previous_actions: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        actions = previous_actions.clamp(min=-1, max=self.action_count - 1) + 1
        action_one_hot = torch.nn.functional.one_hot(
            actions, num_classes=self.action_count + 1
        ).to(dtype=latents.dtype)
        evidence_sequence = self.evidence_projection(
            torch.cat((latents, action_one_hot), dim=-1)
        )
        batch_size = latents.shape[0]
        if hidden is None:
            answer = latents.new_zeros(batch_size, self.hidden_size)
            scratch = latents.new_zeros(batch_size, self.hidden_size)
        else:
            answer, scratch = hidden
            if answer.ndim == 3:
                answer = answer[0]
                scratch = scratch[0]

        final_by_time: list[torch.Tensor] = []
        refinements_by_depth: list[list[torch.Tensor]] = [
            [] for _ in range(self.recursion_depth)
        ]
        for time_index in range(latents.shape[1]):
            evidence = evidence_sequence[:, time_index]
            for depth_index in range(self.recursion_depth):
                delta = self.refiner(torch.cat((evidence, answer, scratch), dim=-1))
                delta_answer, delta_scratch = delta.chunk(2, dim=-1)
                # Bounded residual updates retain temporal state while allowing
                # repeated correction from the same current evidence.
                scratch = torch.tanh(scratch + delta_scratch)
                answer = torch.tanh(answer + delta_answer)
                refinements_by_depth[depth_index].append(answer)
            final_by_time.append(answer)

        final_hidden = torch.stack(final_by_time, dim=1)
        outputs = self._decode(final_hidden)
        outputs["refinement_outputs"] = [
            self._decode(torch.stack(rows, dim=1)) for rows in refinements_by_depth
        ]
        return outputs, (answer.unsqueeze(0), scratch.unsqueeze(0))

    @torch.no_grad()
    def step(
        self,
        latent: np.ndarray,
        previous_action: int,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        latent_tensor = torch.as_tensor(
            np.asarray(latent, dtype=np.float32), device=self.device
        ).reshape(1, 1, self.latent_size)
        action_tensor = torch.as_tensor(
            [[int(previous_action)]], dtype=torch.long, device=self.device
        )
        outputs, next_hidden = self(latent_tensor, action_tensor, hidden)
        feature_parts = [
            outputs["hidden"],
            torch.softmax(outputs["route_logits"], dim=-1),
            torch.sigmoid(outputs["collision_logits"]),
            outputs["goal"],
            torch.sigmoid(outputs["barrier_logits"]),
        ]
        if "exploration_logits" in outputs:
            feature_parts.append(torch.sigmoid(outputs["exploration_logits"]))
        features = torch.cat(feature_parts, dim=-1)
        return features[0, 0].cpu().numpy().astype(np.float64), next_hidden

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": "tiny_recursive",
                "latent_size": self.latent_size,
                "action_count": self.action_count,
                "hidden_size": self.hidden_size,
                "exploration_size": self.exploration_size,
                "recursion_depth": self.recursion_depth,
                "refinement_size": self.refinement_size,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "TinyRecursiveBeliefHead":
        data = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            latent_size=int(data["latent_size"]),
            action_count=int(data["action_count"]),
            hidden_size=int(data["hidden_size"]),
            exploration_size=int(data.get("exploration_size", 0)),
            recursion_depth=int(data["recursion_depth"]),
            refinement_size=int(data.get("refinement_size", 96)),
            device=device,
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model

    @classmethod
    def checkpoint_metadata(cls, path: Path) -> dict:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return dict(data.get("metadata", {}))


def load_belief_head(
    path: Path, *, device: str = "cpu"
):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("architecture") == "evidence_map":
        from .jepa_evidence import JepaEvidenceMapHead

        return JepaEvidenceMapHead.load(path, device=device)
    if data.get("architecture", "gru") == "tiny_recursive":
        return TinyRecursiveBeliefHead.load(path, device=device)
    return JepaBeliefHead.load(path, device=device)
