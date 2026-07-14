from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .minigrid_adapter import MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step
from .skill_runtime import MiniGridSkillApi, SkillProgram, action_map, update_body_memory


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MiniGrid skills across seeds without LLM calls.")
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--out", default="artifacts/minigrid_eval.json")
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Named skill spec label:path. Can be passed multiple times.",
    )
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    specs = [("base_random", [])]
    for item in args.skill:
        if ":" not in item:
            raise SystemExit(f"--skill must be label:path, got {item!r}")
        label, path = item.split(":", 1)
        specs.append((label, [path]))

    results = []
    for label, skill_paths in specs:
        for seed in seeds:
            results.append(run_once(args.env, seed, args.steps, label, skill_paths))

    summary = summarize(results)
    payload = {"env": args.env, "steps": args.steps, "seeds": seeds, "summary": summary, "results": results}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


def run_once(env_id: str, seed: int, max_steps: int, label: str, skill_paths: list[str]) -> dict:
    env = make_minigrid(MiniGridSpec(env_id=env_id, seed=seed, max_steps=max_steps))
    obs, _ = normalize_reset(env.reset(seed=seed))
    rng = random.Random(seed)
    memory: dict = {"blocked_or_low_reward_steps": 0, "skills_written": []}
    program = SkillProgram()
    for skill_path in skill_paths:
        resolved = (ROOT / skill_path).resolve()
        program.add_skill_file(resolved, reason=f"eval:{label}")
        memory["skills_written"].append({"name": resolved.stem, "path": str(resolved), "reason": f"eval:{label}"})

    actions = action_map(env)
    idx_to_action = {value: name for name, value in actions.items()}
    trace = []
    status = "timeout"
    final_step = max_steps
    reward_value = 0.0

    for step in range(max_steps):
        encoded = encode_observation(obs)
        api = MiniGridSkillApi(env, memory)
        action = program.act(encoded, memory, api)
        source = "skill"
        if action is None:
            action = rng.randrange(env.action_space.n)
            source = "base-random"

        next_obs, reward, terminated, truncated, _ = normalize_step(env.step(action))
        update_body_memory(memory, obs, next_obs, env, step, source, action, reward, terminated, truncated)
        reward_value = float(reward)
        trace.append(
            {
                "step": int(step),
                "source": source,
                "action": idx_to_action.get(int(action), str(int(action))),
                "reward": reward_value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "blocked": bool(memory.get("last_action_blocked", False)),
            }
        )
        obs = next_obs
        if terminated or truncated:
            status = "solved" if terminated and reward_value > 0 else "failed" if terminated else "timeout"
            final_step = step + 1
            break

    env.close()
    return {
        "label": label,
        "seed": seed,
        "status": status,
        "solved": status == "solved",
        "steps": int(final_step),
        "reward": reward_value,
        "blocked_forward_steps": int(memory.get("blocked_forward_steps", 0)),
        "learned_structure_keys": sorted(memory.get("learned_structure", {}).keys())
        if isinstance(memory.get("learned_structure"), dict)
        else [],
        "last_actions": [event["action"] for event in trace[-12:]],
    }


def summarize(results: list[dict]) -> dict:
    labels = sorted({item["label"] for item in results})
    summary = {}
    for label in labels:
        rows = [item for item in results if item["label"] == label]
        solved = [item for item in rows if item["solved"]]
        summary[label] = {
            "runs": len(rows),
            "solved": len(solved),
            "success_rate": len(solved) / len(rows) if rows else 0.0,
            "mean_steps": sum(item["steps"] for item in rows) / len(rows) if rows else 0.0,
            "mean_solved_steps": sum(item["steps"] for item in solved) / len(solved) if solved else None,
            "mean_blocked_forward_steps": sum(item["blocked_forward_steps"] for item in rows) / len(rows)
            if rows
            else 0.0,
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
