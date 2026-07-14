from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .jepa_control_agent import ControlTransition, belief_observation
from .minimal_jepa import MinimalJepaAgent
from .minigrid_adapter import MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step
from .skill_runtime import MiniGridSkillApi
from .train_jepa_minigrid import update_carrying, useful_actions


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a small prediction-only JEPA on MiniGrid transitions.")
    parser.add_argument("--env", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-seed", type=int, action="append", default=[])
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--checkpoint", default="artifacts/jepa/minimal-jepa.pt")
    parser.add_argument("--report", default="artifacts/jepa/minimal-jepa-report.json")
    args = parser.parse_args()
    seeds = args.train_seed or list(range(10))
    rng = random.Random(args.seed)
    env = make_minigrid(MiniGridSpec(env_id=args.env, seed=args.seed, max_steps=args.max_steps))
    agent = MinimalJepaAgent(env.action_space.n, seed=args.seed)
    total_steps = 0
    episode_rows = []

    for episode in range(args.episodes):
        layout_seed = seeds[episode % len(seeds)]
        obs, _ = normalize_reset(env.reset(seed=layout_seed))
        previous_action = -1
        carrying = False
        terminated_failure = False
        for step in range(args.max_steps):
            encoded = encode_observation(obs)
            api = MiniGridSkillApi(env, {})
            front_before = api.front_object(encoded)
            state = belief_observation(encoded, previous_action, carrying)
            action = int(rng.choice(useful_actions(api.actions, front_before)))
            next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
            next_encoded = encode_observation(next_obs)
            front_after = api.front_object(next_encoded)
            next_carrying = update_carrying(carrying, action, api.actions, front_before, front_after)
            agent.observe(
                ControlTransition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=belief_observation(next_encoded, action, next_carrying),
                    terminal=bool(terminated or truncated),
                )
            )
            total_steps += 1
            if total_steps % args.train_every == 0:
                for _ in range(args.updates_per_step):
                    agent.train_step()
            previous_action = action
            carrying = next_carrying
            obs = next_obs
            if terminated or truncated:
                terminated_failure = bool(terminated and float(reward) <= 0.0)
                break
        episode_rows.append(
            {
                "episode": episode + 1,
                "seed": layout_seed,
                "steps": step + 1,
                "terminated_failure": terminated_failure,
                "train_steps": agent.total_train_steps,
                "metrics": agent.latest_metrics,
            }
        )
        if episode == 0 or (episode + 1) % 20 == 0:
            print(json.dumps(episode_rows[-1]))

    checkpoint = ROOT / args.checkpoint
    agent.save(checkpoint)
    parameter_count = sum(parameter.numel() for module in (agent.encoder, agent.target_encoder, agent.predictor) for parameter in module.parameters())
    report = {
        "environment": args.env,
        "seed": args.seed,
        "training_seeds": seeds,
        "episodes": args.episodes,
        "environment_steps": total_steps,
        "train_steps": agent.total_train_steps,
        "latent_dim": agent.latent_dim,
        "world_feature_size": agent.world_feature_size,
        "stored_parameter_count": parameter_count,
        "final_metrics": agent.latest_metrics,
        "history": episode_rows,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({"checkpoint": str(checkpoint), "report": str(report_path), **report | {"history": None}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
