from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


FORWARD_VECTORS = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

BASE_FEATURE_SIZE = 15
TERRAIN_RADIUS = 3
TERRAIN_CROP_WIDTH = TERRAIN_RADIUS * 2 + 1
# One categorical terrain value and one visit value per crop cell, followed by
# five global terrain ratios and four frontier/current-location statistics.
TERRAIN_FEATURE_SIZE = TERRAIN_CROP_WIDTH * TERRAIN_CROP_WIDTH * 2 + 9
FULL_FEATURE_SIZE = BASE_FEATURE_SIZE + TERRAIN_FEATURE_SIZE

TERRAIN_VALUES = {
    "unknown": 0.0,
    "empty": 0.25,
    "floor": 0.25,
    "agent": 0.25,
    "wall": -0.5,
    "lava": -1.0,
    "goal": 1.0,
}


@dataclass
class EmbodiedMap:
    """Dead-reckoned pose and object memory built only from body observations."""

    x: int = 0
    y: int = 0
    direction: int = 0
    remembered: dict[str, tuple[int, int]] = field(default_factory=dict)
    terrain: dict[tuple[int, int], str] = field(default_factory=dict)
    visits: dict[tuple[int, int], int] = field(default_factory=lambda: {(0, 0): 1})

    def observe(self, cells: list[dict], direction: int) -> None:
        self.direction = int(direction) if int(direction) in FORWARD_VECTORS else self.direction
        forward_x, forward_y = FORWARD_VECTORS[self.direction]
        right_x, right_y = -forward_y, forward_x
        for cell in cells:
            name = str(cell.get("object", ""))
            if name in {"", "unseen"}:
                continue
            forward = int(cell.get("forward", 0))
            lateral = int(cell.get("lateral", 0))
            object_x = self.x + forward_x * forward + right_x * lateral
            object_y = self.y + forward_y * forward + right_y * lateral
            position = (object_x, object_y)
            if name in {"key", "door", "goal"}:
                self.remembered[name] = position
            terrain_name = name if name in TERRAIN_VALUES else "empty"
            # Objects such as keys and doors occupy traversable floor for the
            # terrain memory. Closed doors remain represented by remembered[].
            if name in {"key", "door", "ball", "box"}:
                terrain_name = "empty"
            self.terrain[position] = terrain_name
        self.terrain[(self.x, self.y)] = "empty"

    def advance(self, action_name: str, blocked: bool, next_direction: int) -> None:
        if action_name == "forward" and not blocked:
            forward_x, forward_y = FORWARD_VECTORS[self.direction]
            self.x += forward_x
            self.y += forward_y
            position = (self.x, self.y)
            self.visits[position] = self.visits.get(position, 0) + 1
            self.terrain[position] = "empty"
        if int(next_direction) in FORWARD_VECTORS:
            self.direction = int(next_direction)

    def features(
        self,
        carrying: bool,
        door_opened: bool,
        scale: float = 8.0,
        *,
        include_terrain: bool = False,
    ) -> np.ndarray:
        forward_x, forward_y = FORWARD_VECTORS[self.direction]
        values = [self.x / scale, self.y / scale, float(forward_x), float(forward_y)]
        for name in ("key", "door", "goal"):
            position = self.remembered.get(name)
            if position is None:
                values.extend((0.0, 0.0, 0.0))
            else:
                values.extend(((position[0] - self.x) / scale, (position[1] - self.y) / scale, 1.0))
        values.extend((float(carrying), float(door_opened)))
        if include_terrain:
            values.extend(self._terrain_features(scale))
        return np.clip(np.asarray(values, dtype=np.float64), -2.0, 2.0)

    def _terrain_features(self, scale: float) -> list[float]:
        """Return an egocentric crop of persistent, observation-only terrain memory."""
        forward_x, forward_y = FORWARD_VECTORS[self.direction]
        right_x, right_y = -forward_y, forward_x
        terrain_values: list[float] = []
        visit_values: list[float] = []
        for forward in range(TERRAIN_RADIUS, -TERRAIN_RADIUS - 1, -1):
            for lateral in range(-TERRAIN_RADIUS, TERRAIN_RADIUS + 1):
                position = (
                    self.x + forward_x * forward + right_x * lateral,
                    self.y + forward_y * forward + right_y * lateral,
                )
                terrain_values.append(TERRAIN_VALUES.get(self.terrain.get(position, "unknown"), 0.25))
                visit_values.append(min(1.0, self.visits.get(position, 0) / 5.0))

        known_count = max(1, len(self.terrain))
        ratios = [
            sum(name in {"empty", "floor", "agent"} for name in self.terrain.values()) / known_count,
            sum(name == "wall" for name in self.terrain.values()) / known_count,
            sum(name == "lava" for name in self.terrain.values()) / known_count,
            sum(name == "goal" for name in self.terrain.values()) / known_count,
            min(1.0, known_count / 100.0),
        ]
        frontier = self.nearest_frontier()
        if frontier is None:
            frontier_features = [0.0, 0.0, 0.0]
        else:
            frontier_features = [
                (frontier[0] - self.x) / scale,
                (frontier[1] - self.y) / scale,
                1.0,
            ]
        current_visits = min(1.0, self.visits.get((self.x, self.y), 0) / 5.0)
        return terrain_values + visit_values + ratios + frontier_features + [current_visits]

    def nearest_frontier(self) -> tuple[int, int] | None:
        """Find the nearest known traversable cell adjacent to unknown terrain."""
        candidates = []
        for position, terrain_name in self.terrain.items():
            if terrain_name not in {"empty", "floor", "agent", "goal"}:
                continue
            x, y = position
            if any((x + dx, y + dy) not in self.terrain for dx, dy in FORWARD_VECTORS.values()):
                distance = abs(x - self.x) + abs(y - self.y)
                candidates.append((distance, x, y))
        if not candidates:
            return None
        _, x, y = min(candidates)
        return x, y
