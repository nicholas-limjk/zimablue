from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper

from .minigrid_adapter import MiniGridSpec, make_minigrid, normalize_reset, normalize_step
from .rgb_jepa import TemporalRgbJepaAgent


ROOT = Path(__file__).resolve().parents[1]
DIRECTION_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a color-invariant, multi-horizon temporal JEPA from MiniGrid RGB sequences."
    )
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--episodes", type=int, default=120, help="Random-walk data episodes.")
    parser.add_argument("--expert-episodes", type=int, default=40, help="Successful planner trajectories used only as RGB/action data.")
    parser.add_argument("--expert-seed-pool", type=int, default=50)
    parser.add_argument("--expert-repeat", type=int, default=3, help="Replay multiplier for successful sequences.")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--horizon", action="append", type=int, default=[])
    parser.add_argument("--no-color-augmentation", action="store_true")
    parser.add_argument("--no-channel-canonicalization", action="store_true")
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--checkpoint", default="artifacts/jepa/temporal-rgb-s9n1-improved.pt")
    parser.add_argument("--report", default="artifacts/jepa/temporal-rgb-s9n1-improved-training.json")
    args = parser.parse_args()
    horizons = tuple(sorted(set(args.horizon or [1, 2, 4])))

    base_env = make_minigrid(MiniGridSpec(args.env, args.seed, args.max_steps))
    env = RGBImgPartialObsWrapper(base_env, tile_size=args.tile_size)
    agent = TemporalRgbJepaAgent(
        env.action_space.n,
        context_length=args.context_length,
        horizons=horizons,
        color_augmentation=not args.no_color_augmentation,
        channel_permutation_invariant=not args.no_channel_canonicalization,
        consistency_weight=args.consistency_weight,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    random_sequences = expert_sequences = successful_expert_episodes = 0

    for episode in range(args.episodes):
        frames, actions, _ = collect_episode(
            env,
            seed=args.seed + episode,
            max_steps=args.max_steps,
            actions=None,
            rng=rng,
        )
        random_sequences += add_episode_sequences(agent, frames, actions)

    for index in range(args.expert_episodes):
        layout_seed = index % args.expert_seed_pool
        obs, _ = normalize_reset(env.reset(seed=layout_seed))
        del obs
        planned_actions = shortest_safe_plan(env)
        frames, actions, solved = collect_episode(
            env,
            seed=layout_seed,
            max_steps=args.max_steps,
            actions=planned_actions,
            rng=rng,
        )
        if solved:
            successful_expert_episodes += 1
        for _ in range(max(1, args.expert_repeat)):
            expert_sequences += add_episode_sequences(agent, frames, actions)

    diagnostic_rows = list(agent.replay)[: min(128, len(agent.replay))]
    initial_invariance = invariance_diagnostics(agent, diagnostic_rows)
    metrics = None
    completed_updates = 0
    for _ in range(args.updates):
        current = agent.train_step()
        if current is not None:
            metrics = current
            completed_updates += 1
    final_invariance = invariance_diagnostics(agent, diagnostic_rows)

    checkpoint = ROOT / args.checkpoint
    agent.save(checkpoint)
    report = {
        "environment": args.env,
        "seed": args.seed,
        "random_episodes": args.episodes,
        "expert_episodes": args.expert_episodes,
        "successful_expert_episodes": successful_expert_episodes,
        "expert_data_source": "shortest safe planner; model receives only resulting RGB frames and actions",
        "random_sequences": random_sequences,
        "expert_sequences_after_repeat": expert_sequences,
        "replay_size": len(agent.replay),
        "updates_requested": args.updates,
        "completed_updates": completed_updates,
        "context_length": args.context_length,
        "horizons": list(horizons),
        "color_augmentation": agent.color_augmentation,
        "channel_permutation_invariant": agent.channel_permutation_invariant,
        "consistency_weight": agent.consistency_weight,
        "latent_shape": [agent.latent_channels, agent.spatial_size, agent.spatial_size],
        "feature_size": agent.feature_size,
        "parameters": agent.parameter_counts(),
        "initial_standard_cyclic_invariance": initial_invariance,
        "final_standard_cyclic_invariance": final_invariance,
        "latest_metrics": metrics or {},
        "checkpoint": str(checkpoint),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps(report))
    return 0


def collect_episode(env, *, seed: int, max_steps: int, actions: list[int] | None, rng: random.Random):
    obs, _ = normalize_reset(env.reset(seed=seed))
    frames = [np.asarray(obs["image"], dtype=np.uint8).copy()]
    taken_actions: list[int] = []
    solved = False
    for step in range(max_steps):
        action = (
            int(actions[step])
            if actions is not None and step < len(actions)
            else rng.choices((0, 1, 2), weights=(1, 1, 3), k=1)[0]
        )
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        frames.append(np.asarray(next_obs["image"], dtype=np.uint8).copy())
        taken_actions.append(action)
        if terminated or truncated:
            solved = bool(terminated and float(reward) > 0.0)
            break
        if actions is not None and step + 1 >= len(actions):
            break
    return frames, taken_actions, solved


def add_episode_sequences(agent: TemporalRgbJepaAgent, frames: list[np.ndarray], actions: list[int]) -> int:
    count = 0
    for time in range(max(0, len(actions) - agent.max_horizon + 1)):
        context_start = time - agent.context_length + 1
        context_frames = np.stack(
            [frames[max(0, context_start + offset)] for offset in range(agent.context_length)]
        )
        previous_actions = [
            actions[index] if index >= 0 else -1
            for index in range(context_start, context_start + agent.context_length - 1)
        ]
        future_actions = actions[time : time + agent.max_horizon]
        future_images = np.stack(frames[time + 1 : time + agent.max_horizon + 1])
        agent.observe_sequence(context_frames, previous_actions, future_actions, future_images)
        count += 1
    return count


def shortest_safe_plan(env) -> list[int]:
    """Plan with simulator state for data collection; no planner state is stored in JEPA replay."""
    unwrapped = env.unwrapped
    start = (tuple(int(value) for value in unwrapped.agent_pos), int(unwrapped.agent_dir))
    queue = deque([start])
    parent: dict[tuple[tuple[int, int], int], tuple[tuple[tuple[int, int], int], int] | None] = {start: None}
    goal_state = None
    while queue:
        state = queue.popleft()
        position, direction = state
        cell = unwrapped.grid.get(*position)
        if cell is not None and getattr(cell, "type", "") == "goal":
            goal_state = state
            break
        candidates = [((position, (direction - 1) % 4), 0), ((position, (direction + 1) % 4), 1)]
        dx, dy = DIRECTION_VECTORS[direction]
        forward = (position[0] + dx, position[1] + dy)
        if 0 <= forward[0] < unwrapped.width and 0 <= forward[1] < unwrapped.height:
            front_cell = unwrapped.grid.get(*forward)
            if front_cell is None or getattr(front_cell, "type", "") not in {"wall", "lava"}:
                candidates.append(((forward, direction), 2))
        for next_state, action in candidates:
            if next_state not in parent:
                parent[next_state] = (state, action)
                queue.append(next_state)
    if goal_state is None:
        raise RuntimeError("No safe planner path to the MiniGrid goal.")
    actions = []
    current = goal_state
    while parent[current] is not None:
        previous, action = parent[current]
        actions.append(action)
        current = previous
    return list(reversed(actions))


def invariance_diagnostics(agent: TemporalRgbJepaAgent, rows) -> dict | None:
    if not rows:
        return None
    agent.eval()
    similarities = []
    distances = []
    for row in rows:
        standard = agent.features(row.frames, row.previous_actions)
        cyclic = agent.features(row.frames[..., [1, 2, 0]], row.previous_actions)
        denominator = float(np.linalg.norm(standard) * np.linalg.norm(cyclic))
        similarities.append(float(np.dot(standard, cyclic) / denominator) if denominator else 0.0)
        distances.append(float(np.linalg.norm(standard - cyclic)))
    agent.context_encoder.train()
    agent.predictor.train()
    return {
        "mean_cosine": float(np.mean(similarities)),
        "mean_l2": float(np.mean(distances)),
        "max_l2": float(np.max(distances)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
