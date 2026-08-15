from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class EpisodicLatentMemory:
    """Bounded observation memory with no metric position or terrain representation."""

    feature_size = 5

    def __init__(
        self,
        capacity: int = 128,
        recent_size: int = 12,
        similarity_threshold: float = 0.95,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.recent_size = min(int(recent_size), self.capacity)
        self.similarity_threshold = float(similarity_threshold)
        self._latents: deque[torch.Tensor] = deque(maxlen=self.capacity)
        self._steps: deque[int] = deque(maxlen=self.capacity)
        self.step = 0

    def __len__(self) -> int:
        return len(self._latents)

    @torch.no_grad()
    def add(self, latent: torch.Tensor) -> None:
        vector = F.normalize(latent.detach().flatten().float(), dim=0).cpu()
        self._latents.append(vector)
        self._steps.append(self.step)
        self.step += 1

    @torch.no_grad()
    def features(self, predicted: torch.Tensor) -> torch.Tensor:
        """Return novelty, recent similarity, old similarity, density, and age."""
        vectors = F.normalize(predicted.flatten(1).float(), dim=1)
        if not self._latents:
            return torch.cat(
                (
                    torch.ones(vectors.shape[0], 1, device=vectors.device),
                    torch.zeros(vectors.shape[0], 4, device=vectors.device),
                ),
                dim=1,
            )
        bank = torch.stack(list(self._latents)).to(vectors.device)
        similarities = vectors @ bank.T
        closest_similarity, closest_index = similarities.max(dim=1)
        novelty = (1.0 - closest_similarity).clamp(min=0.0, max=2.0)
        recent_start = max(0, len(self._latents) - self.recent_size)
        recent_similarity = similarities[:, recent_start:].max(dim=1).values
        old_similarity = (
            similarities[:, :recent_start].max(dim=1).values
            if recent_start
            else similarities.new_zeros(similarities.shape[0])
        )
        density = (similarities >= self.similarity_threshold).float().mean(dim=1)
        stored_steps = torch.as_tensor(list(self._steps), device=vectors.device)
        closest_steps = stored_steps[closest_index]
        age = (self.step - 1 - closest_steps).float().clamp(min=0.0) / float(self.capacity)
        return torch.stack(
            (novelty, recent_similarity, old_similarity, density, age), dim=1
        )

    def signature(self) -> bytes:
        if not self._latents:
            return b"empty"
        quantized = np.stack(
            [torch.clamp(vector * 127.0, -128, 127).to(torch.int8).numpy() for vector in self._latents]
        )
        return hashlib.blake2b(quantized.tobytes(), digest_size=16).digest()

