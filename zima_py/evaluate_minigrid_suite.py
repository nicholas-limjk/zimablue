from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_minigrid import run_once, summarize


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ENVS = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-DoorKey-16x16-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MiniGrid skills across multiple environments.")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--out", default="artifacts/minigrid_suite_eval.json")
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Named skill spec label:path. Can be passed multiple times.",
    )
    args = parser.parse_args()

    envs = [item.strip() for item in args.envs.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not args.skill:
        args.skill = [
            "general:generated_skills/minigrid_general_skill.py",
            "reflected:generated_skills/minigrid_reflected_skill.py",
        ]
    specs = [("base_random", [])]
    for item in args.skill:
        if ":" not in item:
            raise SystemExit(f"--skill must be label:path, got {item!r}")
        label, path = item.split(":", 1)
        specs.append((label, [path]))

    env_payloads = []
    for env_id in envs:
        print(f"evaluating {env_id}")
        results = []
        errors = []
        for label, skill_paths in specs:
            for seed in seeds:
                try:
                    results.append(run_once(env_id, seed, args.steps, label, skill_paths))
                except Exception as exc:
                    errors.append({"label": label, "seed": seed, "error": repr(exc)})
        env_payloads.append(
            {
                "env": env_id,
                "summary": summarize(results),
                "errors": errors,
                "results": results,
            }
        )

    payload = {"steps": args.steps, "seeds": seeds, "envs": env_payloads}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(env_payloads)
    print(f"wrote {out}")
    return 0


def print_summary(env_payloads: list[dict]) -> None:
    labels = sorted({label for item in env_payloads for label in item["summary"]})
    print("env," + ",".join(f"{label}_success" for label in labels))
    for item in env_payloads:
        values = []
        for label in labels:
            row = item["summary"].get(label, {})
            values.append(f"{row.get('solved', 0)}/{row.get('runs', 0)}")
        print(item["env"] + "," + ",".join(values))


if __name__ == "__main__":
    raise SystemExit(main())
