from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LatentGraphNode:
    prototype: torch.Tensor
    visits: int = 0
    last_seen: int = 0
    tried_actions: set[int] = field(default_factory=set)


class EpisodicLatentGraph:
    """Action-linked topological memory over experienced JEPA observations."""

    feature_size = 8

    def __init__(
        self,
        capacity: int = 128,
        action_count: int = 3,
        match_threshold: float = 0.97,
        prototype_rate: float = 0.1,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.action_count = int(action_count)
        self.match_threshold = float(match_threshold)
        self.prototype_rate = float(prototype_rate)
        self.nodes: list[LatentGraphNode] = []
        self.edges: Counter[tuple[int, int, int]] = Counter()
        self.current_node: int | None = None
        self.step = 0

    def __len__(self) -> int:
        return len(self.nodes)

    def _vector(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(latent.detach().flatten().float(), dim=0).cpu()

    def _bank(self, device) -> torch.Tensor:
        return torch.stack([node.prototype for node in self.nodes]).to(device)

    @torch.no_grad()
    def match(self, latent: torch.Tensor) -> tuple[int | None, float]:
        if not self.nodes:
            return None, -1.0
        vector = self._vector(latent).to(latent.device)
        similarities = vector @ self._bank(latent.device).T
        similarity, index = similarities.max(dim=0)
        return int(index), float(similarity)

    @torch.no_grad()
    def observe(self, latent: torch.Tensor, previous_action: int = -1) -> int:
        vector = self._vector(latent)
        matched, similarity = self.match(latent)
        if matched is None or (similarity < self.match_threshold and len(self.nodes) < self.capacity):
            matched = len(self.nodes)
            self.nodes.append(LatentGraphNode(vector, visits=1, last_seen=self.step))
        else:
            node = self.nodes[matched]
            node.prototype = F.normalize(
                (1.0 - self.prototype_rate) * node.prototype + self.prototype_rate * vector,
                dim=0,
            )
            node.visits += 1
            node.last_seen = self.step
        if self.current_node is not None and 0 <= previous_action < self.action_count:
            self.nodes[self.current_node].tried_actions.add(int(previous_action))
            self.edges[(self.current_node, int(previous_action), matched)] += 1
        self.current_node = matched
        self.step += 1
        return matched

    def _directed_distances(self) -> dict[int, int]:
        if self.current_node is None:
            return {}
        adjacency: dict[int, set[int]] = {}
        for source, _, target in self.edges:
            adjacency.setdefault(source, set()).add(target)
        distances = {self.current_node: 0}
        queue = deque([self.current_node])
        while queue:
            source = queue.popleft()
            for target in adjacency.get(source, ()):
                if target not in distances:
                    distances[target] = distances[source] + 1
                    queue.append(target)
        return distances

    def current_untried_actions(
        self,
        action_subset: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Actions not yet observed from the currently matched latent node."""
        if self.current_node is None:
            return ()
        tried = self.nodes[self.current_node].tried_actions
        actions = range(self.action_count) if action_subset is None else action_subset
        return tuple(
            int(action)
            for action in actions
            if 0 <= int(action) < self.action_count and int(action) not in tried
        )

    def frontier_plan(
        self,
        action_subset: tuple[int, ...] | None = None,
        strategy: str = "nearest",
    ) -> tuple[int, int, int] | None:
        """Return (first action, distance, node) for the nearest reachable frontier.

        The current node is excluded: probing it is a separate controller mode.
        Only transitions actually experienced by the agent are traversed.
        """
        plans = self.frontier_plans(action_subset)
        if not plans:
            return None
        if strategy == "nearest":
            return min(plans, key=lambda plan: (plan[1], plan[2]))
        if strategy == "oldest":
            return min(
                plans,
                key=lambda plan: (
                    self.nodes[plan[2]].last_seen,
                    plan[1],
                    plan[2],
                ),
            )
        if strategy == "least-visited":
            return min(
                plans,
                key=lambda plan: (
                    self.nodes[plan[2]].visits,
                    plan[1],
                    self.nodes[plan[2]].last_seen,
                    plan[2],
                ),
            )
        raise ValueError(f"unknown frontier strategy: {strategy}")

    def frontier_plans(
        self,
        action_subset: tuple[int, ...] | None = None,
    ) -> list[tuple[int, int, int]]:
        """All reachable frontiers as (first action, distance, target node)."""
        if self.current_node is None:
            return []
        frontier_actions = (
            tuple(range(self.action_count)) if action_subset is None else action_subset
        )
        adjacency: dict[int, list[tuple[int, int, int]]] = {}
        for (source, action, target), count in self.edges.items():
            adjacency.setdefault(source, []).append((action, target, count))
        for transitions in adjacency.values():
            transitions.sort(key=lambda item: (-item[2], item[0], item[1]))

        parents: dict[int, tuple[int, int]] = {}
        distances = {self.current_node: 0}
        queue = deque([self.current_node])
        goals = []
        while queue:
            source = queue.popleft()
            if source != self.current_node and any(
                int(action) not in self.nodes[source].tried_actions
                for action in frontier_actions
            ):
                goals.append(source)
            for action, target, _ in adjacency.get(source, ()):
                if target in distances:
                    continue
                distances[target] = distances[source] + 1
                parents[target] = (source, action)
                queue.append(target)
        plans = []
        for goal in goals:
            cursor = goal
            first_action = -1
            while cursor != self.current_node:
                parent, action = parents[cursor]
                first_action = action
                cursor = parent
            plans.append((first_action, distances[goal], goal))
        return plans

    def path_to_node(self, target_node: int) -> tuple[int, ...] | None:
        """Shortest experienced action path from the current node to a target."""
        if self.current_node is None or not 0 <= int(target_node) < len(self.nodes):
            return None
        target_node = int(target_node)
        if target_node == self.current_node:
            return ()
        adjacency: dict[int, list[tuple[int, int, int]]] = {}
        for (source, action, target), count in self.edges.items():
            adjacency.setdefault(source, []).append((action, target, count))
        for transitions in adjacency.values():
            transitions.sort(key=lambda item: (-item[2], item[0], item[1]))
        parents: dict[int, tuple[int, int]] = {}
        seen = {self.current_node}
        queue = deque([self.current_node])
        while queue and target_node not in seen:
            source = queue.popleft()
            for action, target, _ in adjacency.get(source, ()):
                if target in seen:
                    continue
                seen.add(target)
                parents[target] = (source, action)
                queue.append(target)
        if target_node not in seen:
            return None
        reverse_actions = []
        cursor = target_node
        while cursor != self.current_node:
            parent, action = parents[cursor]
            reverse_actions.append(action)
            cursor = parent
        return tuple(reversed(reverse_actions))

    @torch.no_grad()
    def candidate_features(
        self,
        predicted_endpoints: torch.Tensor,
        first_actions: torch.Tensor,
    ) -> torch.Tensor:
        count = predicted_endpoints.shape[0]
        device = predicted_endpoints.device
        if not self.nodes:
            return torch.cat(
                (
                    torch.ones(count, 1, device=device),
                    torch.zeros(count, 1, device=device),
                    torch.ones(count, 3, device=device),
                    torch.zeros(count, 3, device=device),
                ),
                dim=1,
            )
        vectors = F.normalize(predicted_endpoints.flatten(1).float(), dim=1)
        similarities = vectors @ self._bank(device).T
        closest_similarity, closest = similarities.max(dim=1)
        novelty = (1.0 - closest_similarity).clamp(min=0.0, max=2.0)
        familiar = (closest_similarity >= self.match_threshold).float()
        maximum_visits = max(node.visits for node in self.nodes)
        inverse_visit = []
        endpoint_untried = []
        ages = []
        reachability = []
        distances = self._directed_distances()
        for index, is_familiar in zip(closest.tolist(), familiar.tolist()):
            node = self.nodes[index]
            inverse_visit.append(
                1.0 - math.log1p(node.visits) / max(math.log1p(maximum_visits), 1e-9)
            )
            endpoint_untried.append(
                (self.action_count - len(node.tried_actions)) / float(self.action_count)
                if is_familiar
                else 1.0
            )
            ages.append(min(1.0, (self.step - 1 - node.last_seen) / float(self.capacity)))
            distance = distances.get(index)
            reachability.append(0.0 if distance is None else 1.0 / (1.0 + distance))
        current = self.nodes[self.current_node] if self.current_node is not None else None
        first_untried = []
        first_confidence = []
        outgoing_total = sum(
            count_value
            for (source, _, _), count_value in self.edges.items()
            if source == self.current_node
        )
        for action in first_actions.tolist():
            first_untried.append(
                1.0 if current is None or int(action) not in current.tried_actions else 0.0
            )
            action_count = sum(
                count_value
                for (source, edge_action, _), count_value in self.edges.items()
                if source == self.current_node and edge_action == int(action)
            )
            first_confidence.append(action_count / max(1, outgoing_total))
        return torch.stack(
            (
                novelty,
                familiar,
                torch.as_tensor(inverse_visit, device=device),
                torch.as_tensor(endpoint_untried, device=device),
                torch.as_tensor(ages, device=device),
                torch.as_tensor(reachability, device=device),
                torch.as_tensor(first_untried, device=device),
                torch.as_tensor(first_confidence, device=device),
            ),
            dim=1,
        ).float()

    def signature(self) -> bytes:
        if not self.nodes:
            return b"empty-graph"
        quantized = np.stack(
            [
                torch.clamp(node.prototype * 127.0, -128, 127)
                .to(torch.int8)
                .numpy()
                for node in self.nodes
            ]
        )
        metadata = repr(
            (
                self.current_node,
                [(node.visits, node.last_seen, sorted(node.tried_actions)) for node in self.nodes],
                sorted(self.edges.items()),
            )
        ).encode("utf-8")
        return hashlib.blake2b(quantized.tobytes() + metadata, digest_size=16).digest()
