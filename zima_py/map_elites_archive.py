from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class ArchiveEntry:
    descriptor: tuple[int, int, int, int, int]
    fitness: float
    genome: Any
    generation: int


class MiniGridMapElites:
    """Archive elites by counts of key collections, door openings, and solves."""

    def __init__(self) -> None:
        self.entries: dict[tuple[int, int, int, int, int], ArchiveEntry] = {}

    def update(self, genome: Any, outcomes: list[dict], fitness: float, generation: int) -> bool:
        descriptor = (
            sum(bool(outcome["collected"]) for outcome in outcomes),
            sum(bool(outcome["opened_door"]) for outcome in outcomes),
            sum(bool(outcome["solved"]) for outcome in outcomes),
            min(
                10,
                int(
                    sum(
                        max(int(outcome["unique_observations"]), int(outcome.get("unique_positions", 0)))
                        for outcome in outcomes
                    )
                    / len(outcomes)
                    // 5
                ),
            ),
            sum(int(outcome["blocked_steps"]) < 5 for outcome in outcomes),
        )
        current = self.entries.get(descriptor)
        if current is not None and current.fitness >= fitness:
            return False
        self.entries[descriptor] = ArchiveEntry(descriptor, float(fitness), copy.deepcopy(genome), int(generation))
        return True

    def elites(self) -> list[ArchiveEntry]:
        return sorted(
            self.entries.values(),
            key=lambda entry: (
                entry.descriptor[2],
                entry.descriptor[1],
                entry.descriptor[0],
                entry.descriptor[3],
                entry.descriptor[4],
                entry.fitness,
            ),
            reverse=True,
        )

    def summary(self) -> list[dict]:
        return [
            {
                "descriptor": {
                    "seeds_collected": entry.descriptor[0],
                    "seeds_opened": entry.descriptor[1],
                    "seeds_solved": entry.descriptor[2],
                    "exploration_bucket": entry.descriptor[3],
                    "low_blocked_seeds": entry.descriptor[4],
                },
                "fitness": entry.fitness,
                "generation": entry.generation,
            }
            for entry in self.elites()
        ]
