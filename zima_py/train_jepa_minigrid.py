from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .jepa_control_agent import ControlTransition, JepaControlAgent, belief_observation
from .minigrid_adapter import MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step
from .skill_runtime import MiniGridSkillApi


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a fully local JEPA + Q controller on one MiniGrid environment.")
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-seed", type=int, action="append", default=[], help="Layout seed used for training; repeat to mix several layouts.")
    parser.add_argument("--eval-seed", type=int, action="append", default=[], help="Layout seed used for frozen greedy evaluation.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=220)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint", default="artifacts/jepa/doorkey-controller.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", default="artifacts/jepa/doorkey-controller-report.json")
    args = parser.parse_args()
    training_seeds = args.train_seed or [args.seed]
    evaluation_seeds = args.eval_seed or training_seeds

    env = make_minigrid(MiniGridSpec(env_id=args.env, seed=args.seed, max_steps=args.max_steps))
    agent = JepaControlAgent(
        env.action_space.n,
        batch_size=args.batch_size,
        warmup=args.warmup,
        seed=args.seed,
    )
    checkpoint = ROOT / args.checkpoint
    if args.resume and checkpoint.exists():
        agent.load(checkpoint)

    history: list[dict[str, Any]] = []
    total_steps = 0
    solved_window: list[bool] = []
    for episode in range(args.episodes):
        layout_seed = training_seeds[episode % len(training_seeds)]
        epsilon = epsilon_at(episode, args.epsilon_start, args.epsilon_end, args.epsilon_decay_episodes)
        result, steps, shaped_return, total_steps = run_episode(
            env,
            agent,
            seed=layout_seed,
            max_steps=args.max_steps,
            epsilon=epsilon,
            total_steps=total_steps,
            train_every=args.train_every,
            updates=args.updates,
            training=True,
        )
        solved_window.append(result)
        del solved_window[:-25]
        row = {
            "episode": episode + 1,
            "layout_seed": layout_seed,
            "solved": result,
            "steps": steps,
            "epsilon": round(epsilon, 4),
            "shaped_return": round(shaped_return, 4),
            "train_steps": agent.total_train_steps,
            "metrics": agent.latest_metrics,
            "recent_success_rate": sum(solved_window) / len(solved_window),
        }
        history.append(row)
        if episode == 0 or (episode + 1) % 10 == 0:
            print(json.dumps(row))

        if (episode + 1) % args.eval_every == 0:
            eval_result = evaluate_seeds(env, agent, evaluation_seeds, args.max_steps)
            history.append({"episode": episode + 1, "evaluation": eval_result})
            print(json.dumps({"episode": episode + 1, "evaluation": eval_result}))
            agent.save(checkpoint)
            if eval_result["success_rate"] >= 1.0:
                break

    final_evaluation = evaluate_seeds(env, agent, evaluation_seeds, args.max_steps)
    agent.save(checkpoint)
    report = {
        "environment": args.env,
        "seed": args.seed,
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seeds,
        "episodes_requested": args.episodes,
        "episodes_completed": max((row.get("episode", 0) for row in history), default=0),
        "total_environment_steps": total_steps,
        "total_train_steps": agent.total_train_steps,
        "final_evaluation": final_evaluation,
        "history": history,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({"final_evaluation": final_evaluation, "checkpoint": str(checkpoint), "report": str(report_path)}))
    return 0 if final_evaluation["success_rate"] > 0 else 1


def run_episode(
    env,
    agent: JepaControlAgent,
    *,
    seed: int,
    max_steps: int,
    epsilon: float,
    total_steps: int,
    train_every: int,
    updates: int,
    training: bool,
) -> tuple[bool, int, float, int]:
    obs, _ = normalize_reset(env.reset(seed=seed))
    memory: dict[str, Any] = {}
    carrying = False
    previous_action = -1
    shaped_return = 0.0
    pickup_rewarded = False
    door_rewarded = False
    for step in range(max_steps):
        encoded = encode_observation(obs)
        api = MiniGridSkillApi(env, memory)
        front_before = api.front_object(encoded)
        state = belief_observation(encoded, previous_action, carrying)
        valid_actions = useful_actions(api.actions, front_before)
        action = agent.act(state, epsilon=epsilon, valid_actions=valid_actions)
        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        next_encoded = encode_observation(next_obs)
        front_after = api.front_object(next_encoded)
        next_carrying = update_carrying(carrying, action, api.actions, front_before, front_after)
        pickup_event = not carrying and next_carrying
        door_event = bool(
            action == api.actions["toggle"]
            and front_before.get("object") == "door"
            and front_before.get("state_name") in {"locked", "closed"}
            and front_after.get("state_name") == "open"
        )
        learning_reward = shaped_reward(
            float(reward),
            action,
            api.actions,
            encoded,
            next_encoded,
            carrying,
            next_carrying,
            front_before,
            front_after,
            award_pickup=pickup_event and not pickup_rewarded,
            award_door=door_event and not door_rewarded,
        )
        pickup_rewarded = pickup_rewarded or pickup_event
        door_rewarded = door_rewarded or door_event
        shaped_return += learning_reward
        if training:
            agent.observe(
                ControlTransition(
                    state=state,
                    action=int(action),
                    reward=learning_reward,
                    next_state=belief_observation(next_encoded, int(action), next_carrying),
                    terminal=bool(terminated or truncated),
                )
            )
            total_steps += 1
            if total_steps % train_every == 0:
                for _ in range(updates):
                    agent.train_step()
        previous_action = int(action)
        carrying = next_carrying
        obs = next_obs
        if terminated or truncated:
            solved = bool(terminated and float(reward) > 0.0)
            return solved, step + 1, shaped_return, total_steps
    return False, max_steps, shaped_return, total_steps


def evaluate(env, agent: JepaControlAgent, seed: int, max_steps: int, episodes: int) -> dict[str, Any]:
    results = []
    for _ in range(episodes):
        solved, steps, _, _ = run_episode(
            env,
            agent,
            seed=seed,
            max_steps=max_steps,
            epsilon=0.0,
            total_steps=0,
            train_every=1,
            updates=0,
            training=False,
        )
        results.append({"solved": solved, "steps": steps})
    solved_steps = [row["steps"] for row in results if row["solved"]]
    return {
        "episodes": episodes,
        "solved": len(solved_steps),
        "success_rate": len(solved_steps) / episodes,
        "mean_solved_steps": None if not solved_steps else sum(solved_steps) / len(solved_steps),
    }


def evaluate_seeds(env, agent: JepaControlAgent, seeds: list[int], max_steps: int) -> dict[str, Any]:
    results = []
    for seed in seeds:
        solved, steps, _, _ = run_episode(
            env,
            agent,
            seed=seed,
            max_steps=max_steps,
            epsilon=0.0,
            total_steps=0,
            train_every=1,
            updates=0,
            training=False,
        )
        results.append({"seed": seed, "solved": solved, "steps": steps})
    solved_steps = [row["steps"] for row in results if row["solved"]]
    return {
        "episodes": len(results),
        "solved": len(solved_steps),
        "success_rate": len(solved_steps) / len(results),
        "mean_solved_steps": None if not solved_steps else sum(solved_steps) / len(solved_steps),
        "runs": results,
    }


def epsilon_at(episode: int, start: float, end: float, decay_episodes: int) -> float:
    progress = min(1.0, episode / max(1, decay_episodes))
    return start + progress * (end - start)


def useful_actions(actions: dict[str, int], front: dict[str, Any]) -> list[int]:
    valid = [actions["left"], actions["right"], actions["forward"]]
    if front.get("object") in {"key", "ball", "box"}:
        valid.append(actions["pickup"])
    if front.get("object") == "door":
        valid.append(actions["toggle"])
    return valid


def update_carrying(
    carrying: bool,
    action: int,
    actions: dict[str, int],
    front_before: dict[str, Any],
    front_after: dict[str, Any],
) -> bool:
    if action == actions["pickup"] and front_before.get("object") in {"key", "ball", "box"}:
        return front_after.get("object") != front_before.get("object")
    if action == actions["drop"]:
        return False
    return carrying


def shaped_reward(
    reward: float,
    action: int,
    actions: dict[str, int],
    previous_obs: dict[str, Any],
    next_obs: dict[str, Any],
    carrying: bool,
    next_carrying: bool,
    front_before: dict[str, Any],
    front_after: dict[str, Any],
    *,
    award_pickup: bool = True,
    award_door: bool = True,
) -> float:
    shaped = reward - 0.001
    if award_pickup and not carrying and next_carrying:
        shaped += 0.2
    if (
        award_door
        and action == actions["toggle"]
        and front_before.get("object") == "door"
        and front_before.get("state_name") in {"locked", "closed"}
        and front_after.get("state_name") == "open"
    ):
        shaped += 0.2
    if action == actions["forward"] and np.array_equal(previous_obs["image"], next_obs["image"]):
        shaped -= 0.01
    return shaped


if __name__ == "__main__":
    raise SystemExit(main())
