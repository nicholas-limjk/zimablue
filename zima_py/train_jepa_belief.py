from __future__ import annotations

import argparse
import json
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import RGBImgPartialObsWrapper

from .jepa_belief import JepaBeliefHead, TinyRecursiveBeliefHead
from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent
from .train_temporal_rgb_jepa import DIRECTION_VECTORS, shortest_safe_plan


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BeliefRow:
    frames: np.ndarray
    context_actions: np.ndarray
    previous_action: int
    route_action: int
    route_distribution: np.ndarray
    collision: np.ndarray
    goal: np.ndarray
    barrier: np.ndarray
    exploration: np.ndarray
    exploration_mask: np.ndarray
    terrain: np.ndarray | None = None
    moved_last: float = 0.0
    moved_mask: float = 0.0


@dataclass
class EncodedEpisode:
    latents: torch.Tensor
    previous_actions: torch.Tensor
    route_actions: torch.Tensor
    route_distribution: torch.Tensor
    collision: torch.Tensor
    goal: torch.Tensor
    barrier: torch.Tensor
    exploration: torch.Tensor
    exploration_mask: torch.Tensor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a recurrent or tiny-recursive belief decoder on frozen temporal-JEPA features."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--random-episodes", type=int, default=200)
    parser.add_argument("--expert-episodes", type=int, default=80)
    parser.add_argument("--validation-random-episodes", type=int, default=40)
    parser.add_argument("--validation-expert-episodes", type=int, default=20)
    parser.add_argument("--train-seed-pool", type=int, default=40)
    parser.add_argument("--validation-seed-start", type=int, default=40)
    parser.add_argument("--validation-seed-count", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--architecture", choices=("gru", "tiny-recursive"), default="gru")
    parser.add_argument("--recursion-depth", type=int, default=6)
    parser.add_argument("--refinement-size", type=int, default=96)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--output", default="artifacts/jepa/temporal-rgb-s9n1-belief-v2.pt")
    parser.add_argument("--report", default="artifacts/jepa/temporal-rgb-s9n1-belief-v2-training.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = RGBImgPartialObsWrapper(
        make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps)), tile_size=8
    )
    jepa = TemporalRgbJepaAgent(env.action_space.n, context_length=4, seed=args.seed)
    jepa.load(ROOT / args.checkpoint)
    jepa.eval()

    training_rows = collect_dataset(
        env,
        random_episodes=args.random_episodes,
        expert_episodes=args.expert_episodes,
        seed_choices=list(range(args.train_seed_pool)),
        max_steps=args.max_steps,
        rng=rng,
        context_length=jepa.context_length,
    )
    validation_rows = collect_dataset(
        env,
        random_episodes=args.validation_random_episodes,
        expert_episodes=args.validation_expert_episodes,
        seed_choices=list(
            range(
                args.validation_seed_start,
                args.validation_seed_start + args.validation_seed_count,
            )
        ),
        max_steps=args.max_steps,
        rng=random.Random(args.seed + 1),
        context_length=jepa.context_length,
    )
    training = encode_episodes(jepa, training_rows)
    validation = encode_episodes(jepa, validation_rows)

    if args.architecture == "tiny-recursive":
        belief = TinyRecursiveBeliefHead(
            latent_size=jepa.feature_size,
            action_count=env.action_space.n,
            hidden_size=args.hidden_size,
            exploration_size=4,
            recursion_depth=args.recursion_depth,
            refinement_size=args.refinement_size,
        )
    else:
        belief = JepaBeliefHead(
            latent_size=jepa.feature_size,
            action_count=env.action_space.n,
            hidden_size=args.hidden_size,
            exploration_size=4,
        )
    optimizer = torch.optim.AdamW(belief.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    latest = {}
    for update in range(args.updates):
        batch = rng.sample(training, min(args.batch_size, len(training)))
        latest = train_batch(belief, optimizer, batch)
        if update % 100 == 0 or update + 1 == args.updates:
            print(json.dumps({"update": update + 1, **latest}))

    train_metrics = evaluate(belief, training)
    validation_metrics = evaluate(belief, validation)
    metadata = {
        "environment": args.env,
        "jepa_checkpoint": str(ROOT / args.checkpoint),
        "targets": {
            "route": "uncertainty-mixed safe-route distribution: left/right/forward",
            "collision": "immediate collision risk for left/right/forward",
            "goal": "safe-distance, egocentric forward/right goal offset, distance progress",
            "barrier": "crossed-barrier event and persistent opposite-side belief",
            "exploration": "chosen-action information gain and history-derived coverage",
        },
        "objective_version": 2,
        "belief_architecture": args.architecture,
        "recursion_depth": args.recursion_depth if args.architecture == "tiny-recursive" else 1,
        "privileged_labels_training_only": True,
    }
    output = ROOT / args.output
    belief.save(output, metadata=metadata)
    report = {
        **metadata,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "architecture": args.architecture,
        "recursion_depth": args.recursion_depth if args.architecture == "tiny-recursive" else 1,
        "refinement_size": args.refinement_size if args.architecture == "tiny-recursive" else None,
        "belief_parameters": sum(parameter.numel() for parameter in belief.parameters()),
        "belief_feature_size": belief.feature_size,
        "training_episodes": len(training),
        "training_steps": sum(len(episode.route_actions) for episode in training),
        "validation_episodes": len(validation),
        "validation_steps": sum(len(episode.route_actions) for episode in validation),
        "updates": args.updates,
        "latest_batch": latest,
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "output": str(output),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({"validation": validation_metrics, "report": str(report_path)}))
    return 0


def collect_dataset(
    env,
    *,
    random_episodes: int,
    expert_episodes: int,
    seed_choices: list[int],
    max_steps: int,
    rng: random.Random,
    context_length: int,
) -> list[list[BeliefRow]]:
    episodes = []
    schedule = [False] * random_episodes + [True] * expert_episodes
    rng.shuffle(schedule)
    for expert in schedule:
        seed = seed_choices[rng.randrange(len(seed_choices))]
        rows = collect_episode(env, seed, max_steps, rng, context_length, expert)
        if rows:
            episodes.append(rows)
    return episodes


def collect_episode(
    env,
    seed: int,
    max_steps: int,
    rng: random.Random,
    context_length: int,
    expert: bool,
) -> list[BeliefRow]:
    obs, _ = normalize_reset(env.reset(seed=seed))
    first = np.asarray(obs["image"], dtype=np.uint8)
    frames = deque([first.copy() for _ in range(context_length)], maxlen=context_length)
    actions = deque([-1 for _ in range(context_length - 1)], maxlen=context_length - 1)
    previous_action = -1
    previous_distance = None
    crossed_last = False
    moved_last = 0.0
    moved_mask = 0.0
    unwrapped = env.unwrapped
    goal_position = find_goal(unwrapped)
    barrier = infer_barrier(unwrapped)
    barrier_side = barrier_side_for(tuple(unwrapped.agent_pos), barrier, None)
    seen_cells: set[tuple[int, int]] = set()
    rows = []
    for _ in range(max_steps):
        plan = shortest_safe_plan(env)
        if not plan:
            break
        distance = len(plan)
        seen_cells.update(visible_world_cells(unwrapped))
        route_distribution = uncertain_route_distribution(unwrapped, seen_cells)
        row = BeliefRow(
                frames=np.stack(frames),
                context_actions=np.asarray(actions, dtype=np.int64),
                previous_action=previous_action,
                route_action=int(plan[0]),
                route_distribution=route_distribution,
                collision=collision_targets(unwrapped),
                goal=goal_targets(
                    unwrapped,
                    goal_position,
                    distance,
                    previous_distance,
                ),
                barrier=np.asarray((float(crossed_last), float(barrier_side)), dtype=np.float32),
                exploration=np.asarray(
                    (0.0, 0.0, 0.0, len(seen_cells) / float(unwrapped.width * unwrapped.height)),
                    dtype=np.float32,
                ),
                exploration_mask=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
                terrain=visible_terrain_targets(unwrapped),
                moved_last=moved_last,
                moved_mask=moved_mask,
        )
        rows.append(row)
        action = int(plan[0]) if expert else rng.randrange(3)
        position_before = tuple(int(value) for value in unwrapped.agent_pos)
        next_obs, _, terminated, truncated, _ = normalize_step(env.step(action))
        position_after = tuple(int(value) for value in unwrapped.agent_pos)
        moved_last = float(position_after != position_before)
        moved_mask = float(action == 2)
        visible_after = visible_world_cells(unwrapped)
        new_cells = len(visible_after - seen_cells)
        row.exploration[action] = min(1.0, new_cells / 12.0)
        row.exploration_mask[action] = 1.0
        seen_cells.update(visible_after)
        next_image = np.asarray(next_obs["image"], dtype=np.uint8)
        frames.append(next_image.copy())
        actions.append(action)
        previous_action = action
        previous_distance = distance
        next_side = barrier_side_for(tuple(unwrapped.agent_pos), barrier, barrier_side)
        crossed_last = next_side != barrier_side
        barrier_side = next_side
        obs = next_obs
        if terminated or truncated:
            break
    return rows


def visible_terrain_targets(unwrapped) -> np.ndarray:
    """Return view-aligned observable labels: unseen, free, wall, lava, goal."""
    symbolic = np.asarray(unwrapped.gen_obs()["image"])
    _, visibility = unwrapped.gen_obs_grid()
    targets = np.zeros((unwrapped.agent_view_size, unwrapped.agent_view_size), dtype=np.int64)
    object_ids = symbolic[..., 0]
    # MiniGrid object ids: empty=1, wall=2, goal=8, lava=9, agent=10.
    targets[visibility] = 1
    targets[visibility & (object_ids == 2)] = 2
    targets[visibility & (object_ids == 9)] = 3
    targets[visibility & (object_ids == 8)] = 4
    # RGB tensors use image row/column order; MiniGrid's symbolic observation is
    # indexed view-x/view-y, so transpose labels to match the decoder pixels.
    return targets.T


def find_goal(unwrapped) -> tuple[int, int]:
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and getattr(cell, "type", "") == "goal":
                return x, y
    raise RuntimeError("No goal in MiniGrid layout.")


def collision_targets(unwrapped) -> np.ndarray:
    position = tuple(int(value) for value in unwrapped.agent_pos)
    direction = int(unwrapped.agent_dir)
    dx, dy = DIRECTION_VECTORS[direction]
    front = (position[0] + dx, position[1] + dy)
    blocked = not (0 <= front[0] < unwrapped.width and 0 <= front[1] < unwrapped.height)
    if not blocked:
        cell = unwrapped.grid.get(*front)
        blocked = cell is not None and getattr(cell, "type", "") in {"wall", "lava"}
    return np.asarray((0.0, 0.0, float(blocked)), dtype=np.float32)


def visible_world_cells(unwrapped) -> set[tuple[int, int]]:
    _, visibility = unwrapped.gen_obs_grid()
    forward = np.asarray(unwrapped.dir_vec, dtype=np.int64)
    right = np.asarray(unwrapped.right_vec, dtype=np.int64)
    top_left = (
        np.asarray(unwrapped.agent_pos, dtype=np.int64)
        + forward * (unwrapped.agent_view_size - 1)
        - right * (unwrapped.agent_view_size // 2)
    )
    cells = set()
    for view_y in range(unwrapped.agent_view_size):
        for view_x in range(unwrapped.agent_view_size):
            if not visibility[view_x, view_y]:
                continue
            absolute = top_left - forward * view_y + right * view_x
            x, y = int(absolute[0]), int(absolute[1])
            if 0 <= x < unwrapped.width and 0 <= y < unwrapped.height:
                cells.add((x, y))
    return cells


def uncertain_route_distribution(unwrapped, seen_cells: set[tuple[int, int]]) -> np.ndarray:
    position = tuple(int(value) for value in unwrapped.agent_pos)
    direction = int(unwrapped.agent_dir)
    candidate_states = [
        (position, (direction - 1) % 4),
        (position, (direction + 1) % 4),
    ]
    dx, dy = DIRECTION_VECTORS[direction]
    forward = (position[0] + dx, position[1] + dy)
    forward_safe = not bool(collision_targets(unwrapped)[2])
    candidate_states.append((forward, direction) if forward_safe else None)
    costs = np.asarray(
        [
            safe_path_length(unwrapped, state) if state is not None else np.inf
            for state in candidate_states
        ],
        dtype=np.float64,
    )
    safe = np.isfinite(costs)
    minimum = float(costs[safe].min())
    planner = np.zeros(3, dtype=np.float64)
    planner[safe] = np.exp(-(costs[safe] - minimum) / 2.0)
    planner /= planner.sum()
    uniform = safe.astype(np.float64) / safe.sum()
    unknown_fraction = 1.0 - len(seen_cells) / float(unwrapped.width * unwrapped.height)
    uncertainty = min(0.8, max(0.0, unknown_fraction))
    return ((1.0 - uncertainty) * planner + uncertainty * uniform).astype(np.float32)


def safe_path_length(unwrapped, start) -> float:
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (position, direction), distance = queue.popleft()
        cell = unwrapped.grid.get(*position)
        if cell is not None and getattr(cell, "type", "") == "goal":
            return float(distance)
        candidates = [(position, (direction - 1) % 4), (position, (direction + 1) % 4)]
        dx, dy = DIRECTION_VECTORS[direction]
        forward = (position[0] + dx, position[1] + dy)
        if 0 <= forward[0] < unwrapped.width and 0 <= forward[1] < unwrapped.height:
            front = unwrapped.grid.get(*forward)
            if front is None or getattr(front, "type", "") not in {"wall", "lava"}:
                candidates.append((forward, direction))
        for next_state in candidates:
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, distance + 1))
    return float("inf")


def goal_targets(unwrapped, goal_position, distance: int, previous_distance: int | None) -> np.ndarray:
    position = np.asarray(unwrapped.agent_pos, dtype=np.float32)
    delta = np.asarray(goal_position, dtype=np.float32) - position
    forward = np.asarray(DIRECTION_VECTORS[int(unwrapped.agent_dir)], dtype=np.float32)
    right = np.asarray((-forward[1], forward[0]), dtype=np.float32)
    scale = float(max(unwrapped.width, unwrapped.height))
    safe_distance = min(1.0, distance / float(unwrapped.width * unwrapped.height))
    progress = 0.0 if previous_distance is None else np.clip((previous_distance - distance) / scale, -1.0, 1.0)
    return np.asarray(
        (safe_distance, float(delta @ forward) / scale, float(delta @ right) / scale, progress),
        dtype=np.float32,
    )


def infer_barrier(unwrapped) -> tuple[int, int, int]:
    lava = []
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and getattr(cell, "type", "") == "lava":
                lava.append((x, y))
    if not lava:
        return 0, 0, 1
    x_value, x_count = Counter(x for x, _ in lava).most_common(1)[0]
    y_value, y_count = Counter(y for _, y in lava).most_common(1)[0]
    axis, coordinate = (0, x_value) if x_count > y_count else (1, y_value)
    start_delta = int(unwrapped.agent_pos[axis]) - coordinate
    start_sign = -1 if start_delta < 0 else 1
    return axis, coordinate, start_sign


def barrier_side_for(position, barrier: tuple[int, int, int], previous: int | None) -> int:
    axis, coordinate, start_sign = barrier
    delta = int(position[axis]) - coordinate
    if delta == 0:
        return 0 if previous is None else int(previous)
    return int((-1 if delta < 0 else 1) != start_sign)


@torch.no_grad()
def encode_episodes(
    jepa: TemporalRgbJepaAgent,
    episodes: list[list[BeliefRow]],
    batch_size: int = 256,
) -> list[EncodedEpisode]:
    flat = [row for episode in episodes for row in episode]
    latent_rows = []
    for start in range(0, len(flat), batch_size):
        rows = flat[start : start + batch_size]
        frames = jepa._frames([row.frames for row in rows])
        actions = torch.as_tensor(
            np.stack([row.context_actions for row in rows]),
            dtype=torch.long,
            device=jepa.device,
        )
        latent_rows.append(jepa.context_encoder(frames, actions).flatten(1).cpu())
    latents = torch.cat(latent_rows)
    encoded = []
    offset = 0
    for rows in episodes:
        size = len(rows)
        encoded.append(
            EncodedEpisode(
                latents=latents[offset : offset + size],
                previous_actions=torch.as_tensor([row.previous_action for row in rows], dtype=torch.long),
                route_actions=torch.as_tensor([row.route_action for row in rows], dtype=torch.long),
                route_distribution=torch.as_tensor(
                    np.stack([row.route_distribution for row in rows])
                ),
                collision=torch.as_tensor(np.stack([row.collision for row in rows])),
                goal=torch.as_tensor(np.stack([row.goal for row in rows])),
                barrier=torch.as_tensor(np.stack([row.barrier for row in rows])),
                exploration=torch.as_tensor(np.stack([row.exploration for row in rows])),
                exploration_mask=torch.as_tensor(
                    np.stack([row.exploration_mask for row in rows])
                ),
            )
        )
        offset += size
    return encoded


def padded_batch(episodes: list[EncodedEpisode], device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    length = max(len(episode.route_actions) for episode in episodes)
    batch_size = len(episodes)
    tensors = {
        "latents": torch.zeros(batch_size, length, episodes[0].latents.shape[-1], device=device),
        "previous_actions": torch.full((batch_size, length), -1, dtype=torch.long, device=device),
        "route": torch.zeros(batch_size, length, dtype=torch.long, device=device),
        "route_distribution": torch.zeros(batch_size, length, 3, device=device),
        "collision": torch.zeros(batch_size, length, 3, device=device),
        "goal": torch.zeros(batch_size, length, 4, device=device),
        "barrier": torch.zeros(batch_size, length, 2, device=device),
        "exploration": torch.zeros(batch_size, length, 4, device=device),
        "exploration_mask": torch.zeros(batch_size, length, 4, device=device),
    }
    mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    for index, episode in enumerate(episodes):
        size = len(episode.route_actions)
        mask[index, :size] = True
        for key, source in (
            ("latents", episode.latents),
            ("previous_actions", episode.previous_actions),
            ("route", episode.route_actions),
            ("route_distribution", episode.route_distribution),
            ("collision", episode.collision),
            ("goal", episode.goal),
            ("barrier", episode.barrier),
            ("exploration", episode.exploration),
            ("exploration_mask", episode.exploration_mask),
        ):
            tensors[key][index, :size] = source.to(device)
    return tensors, mask


def task_loss_components(outputs, batch, mask) -> dict[str, torch.Tensor]:
    route_log_probabilities = F.log_softmax(outputs["route_logits"][mask], dim=-1)
    route_loss = -(
        batch["route_distribution"][mask] * route_log_probabilities
    ).sum(dim=-1).mean()
    collision_loss = F.binary_cross_entropy_with_logits(
        outputs["collision_logits"][mask], batch["collision"][mask]
    )
    goal_loss = F.smooth_l1_loss(outputs["goal"][mask], batch["goal"][mask])
    barrier_element = F.binary_cross_entropy_with_logits(
        outputs["barrier_logits"][mask], batch["barrier"][mask], reduction="none"
    )
    barrier_element[:, 0] *= 8.0
    barrier_loss = barrier_element.mean()
    exploration_predictions = torch.sigmoid(outputs["exploration_logits"][mask])
    exploration_mask = batch["exploration_mask"][mask]
    exploration_error = F.smooth_l1_loss(
        exploration_predictions,
        batch["exploration"][mask],
        reduction="none",
    )
    exploration_loss = (exploration_error * exploration_mask).sum() / exploration_mask.sum()
    return {
        "route": route_loss,
        "collision": collision_loss,
        "goal": goal_loss,
        "barrier": barrier_loss,
        "exploration": exploration_loss,
    }


def combined_task_loss(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        components["route"]
        + 0.5 * components["collision"]
        + components["goal"]
        + 0.5 * components["barrier"]
        + components["exploration"]
    )


def losses_and_metrics(model, episodes: list[EncodedEpisode]) -> tuple[torch.Tensor, dict]:
    batch, mask = padded_batch(episodes, model.device)
    outputs, _ = model(batch["latents"], batch["previous_actions"])
    components = task_loss_components(outputs, batch, mask)
    route_loss = components["route"]
    collision_loss = components["collision"]
    goal_loss = components["goal"]
    barrier_loss = components["barrier"]
    exploration_predictions = torch.sigmoid(outputs["exploration_logits"][mask])
    exploration_mask = batch["exploration_mask"][mask]
    exploration_loss = components["exploration"]
    loss = combined_task_loss(components)
    intermediate = outputs.get("refinement_outputs", [])[:-1]
    if intermediate:
        auxiliary = torch.stack(
            [combined_task_loss(task_loss_components(row, batch, mask)) for row in intermediate]
        ).mean()
        loss = loss + 0.2 * auxiliary
    route_accuracy = (
        outputs["route_logits"][mask].argmax(dim=-1) == batch["route"][mask]
    ).float().mean()
    collision_accuracy = (
        (torch.sigmoid(outputs["collision_logits"][mask]) >= 0.5)
        == (batch["collision"][mask] >= 0.5)
    ).float().mean()
    barrier_accuracy = (
        (torch.sigmoid(outputs["barrier_logits"][mask]) >= 0.5)
        == (batch["barrier"][mask] >= 0.5)
    ).float().mean()
    metrics = {
        "loss": float(loss.detach()),
        "route_loss": float(route_loss.detach()),
        "route_accuracy": float(route_accuracy.detach()),
        "collision_accuracy": float(collision_accuracy.detach()),
        "goal_mae": float((outputs["goal"][mask] - batch["goal"][mask]).abs().mean().detach()),
        "barrier_accuracy": float(barrier_accuracy.detach()),
        "exploration_mae": float(
            (
                (exploration_predictions - batch["exploration"][mask]).abs()
                * exploration_mask
            ).sum().div(exploration_mask.sum()).detach()
        ),
    }
    return loss, metrics


def train_batch(model, optimizer, episodes) -> dict:
    model.train()
    loss, metrics = losses_and_metrics(model, episodes)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return metrics


@torch.no_grad()
def evaluate(model, episodes, batch_size: int = 16) -> dict:
    model.eval()
    weighted = Counter()
    total = 0
    for start in range(0, len(episodes), batch_size):
        rows = episodes[start : start + batch_size]
        _, metrics = losses_and_metrics(model, rows)
        count = sum(len(episode.route_actions) for episode in rows)
        total += count
        for key, value in metrics.items():
            weighted[key] += value * count
    return {key: value / total for key, value in weighted.items()}


if __name__ == "__main__":
    raise SystemExit(main())
