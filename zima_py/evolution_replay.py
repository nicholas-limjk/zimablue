from __future__ import annotations

import random
from collections import deque

from .jepa_control_agent import ControlTransition, JepaControlAgent


class BalancedEvolutionReplay:
    """Keep informative evolutionary transitions without filling replay with loops."""

    def __init__(self, capacity: int = 50_000, seed: int = 0) -> None:
        half = max(1, int(capacity) // 2)
        self.priority: deque[ControlTransition] = deque(maxlen=half)
        self.ordinary: deque[ControlTransition] = deque(maxlen=half)
        self.rng = random.Random(seed)
        self.seen = 0

    def add(self, transition: ControlTransition, *, priority: bool) -> None:
        self.seen += 1
        if priority:
            self.priority.append(transition)
        elif self.rng.random() < 0.25:
            # Safe loops are extremely common in lava tasks. Retain a sample,
            # not every repeated turn, while preserving the original stream.
            self.ordinary.append(transition)

    def feed(self, agent: JepaControlAgent, count: int) -> dict[str, int]:
        requested = max(0, int(count))
        priority_count = min(len(self.priority), (requested + 1) // 2)
        ordinary_count = min(len(self.ordinary), requested - priority_count)
        remaining = requested - priority_count - ordinary_count
        if remaining:
            extra_priority = min(len(self.priority) - priority_count, remaining)
            priority_count += extra_priority
            remaining -= extra_priority
        if remaining:
            ordinary_count += min(len(self.ordinary) - ordinary_count, remaining)

        selected = self.rng.sample(list(self.priority), priority_count)
        selected += self.rng.sample(list(self.ordinary), ordinary_count)
        self.rng.shuffle(selected)
        for transition in selected:
            agent.observe(transition)
        return {
            "seen": self.seen,
            "priority_available": len(self.priority),
            "ordinary_available": len(self.ordinary),
            "fed": len(selected),
        }
