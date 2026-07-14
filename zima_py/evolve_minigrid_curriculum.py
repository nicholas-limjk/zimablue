from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_minigrid import run_once, summarize
from .export_minigrid_viz import normalize_persistent_state_namespace, skill_self_audit
from .llm import OpenAICompatiblePythonSkillWriter
from .skill_runtime import write_skill


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENVS = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-Unlock-v0",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reflect a MiniGrid skill over a small cross-env curriculum.")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--current-skill", default="generated_skills/minigrid_reflected_skill.py")
    parser.add_argument("--skill-out", default="generated_skills/minigrid_curriculum_skill.py")
    parser.add_argument("--report-out", default="artifacts/minigrid_curriculum_reflection.json")
    parser.add_argument("--no-llm", action="store_true", help="Only evaluate current skill; do not ask for a rewrite.")
    args = parser.parse_args()

    envs = [item.strip() for item in args.envs.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    current_results = evaluate_skill(envs, seeds, args.steps, "current", args.current_skill)
    current_summary = summarize(current_results)
    current_source = (ROOT / args.current_skill).read_text(encoding="utf-8")
    report = {
        "curriculum_envs": envs,
        "seeds": seeds,
        "steps": args.steps,
        "current_skill": args.current_skill,
        "current_audit": skill_self_audit(current_source),
        "current_summary": current_summary,
        "current_results": current_results,
    }

    if args.no_llm or all(item["solved"] for item in current_results):
        write_report(args.report_out, report)
        print(json.dumps(current_summary, indent=2))
        print(f"wrote {ROOT / args.report_out}")
        return 0

    writer = OpenAICompatiblePythonSkillWriter()
    payload = build_curriculum_payload(envs, seeds, args.steps, current_source, current_summary, current_results)
    proposal = writer.propose(payload)
    normalized_source, normalization = normalize_persistent_state_namespace(proposal.source)
    proposed_audit = skill_self_audit(normalized_source)
    record = write_skill(ROOT / args.skill_out, normalized_source)
    record.reason = proposal.reason

    proposed_results = evaluate_skill(envs, seeds, args.steps, "proposed", args.skill_out)
    proposed_summary = summarize(proposed_results)
    report.update(
        {
            "proposal": {
                "name": proposal.name,
                "reason": proposal.reason,
                "design": proposal.design,
                "normalization": normalization,
                "path": str((ROOT / args.skill_out).resolve()),
                "audit": proposed_audit,
            },
            "proposed_summary": proposed_summary,
            "proposed_results": proposed_results,
        }
    )
    write_report(args.report_out, report)
    print(json.dumps({"current": current_summary, "proposed": proposed_summary, "proposal": report["proposal"]}, indent=2))
    print(f"wrote {ROOT / args.report_out}")
    return 0


def evaluate_skill(envs: list[str], seeds: list[int], steps: int, label: str, skill_path: str) -> list[dict]:
    results = []
    for env_id in envs:
        for seed in seeds:
            row = run_once(env_id, seed, steps, label, [skill_path])
            row["env"] = env_id
            results.append(row)
    return results


def build_curriculum_payload(
    envs: list[str],
    seeds: list[int],
    steps: int,
    current_source: str,
    current_summary: dict,
    current_results: list[dict],
) -> dict:
    failures = [item for item in current_results if not item["solved"]]
    successes = [item for item in current_results if item["solved"]]
    return {
        "impulse": "reach_goal",
        "information_boundary": [
            "The skill writer sees aggregate body-only evaluation results across a small curriculum.",
            "At runtime the skill still receives only obs, memory, and api.",
            "No hidden grid, env object, oracle path, or future observations may be used.",
        ],
        "current_body_observation": {
            "type": "curriculum_summary",
            "curriculum_envs": envs,
            "seeds": seeds,
            "step_limit": steps,
        },
        "body_learned_memory": {
            "current_summary": current_summary,
            "success_examples": compact_examples(successes),
            "failure_examples": compact_examples(failures),
        },
        "recent_trace": compact_examples(failures)[-12:],
        "skill_contract": {
            "required_function": "def act(obs, memory, api):",
            "allowed_libraries": "numpy is available as api.np; persistent state is available through memory",
            "body_only_helpers": [
                "api.front_object(obs)",
                "api.egocentric_cells(obs)",
                "api.visible_cells(obs)",
                "api.visible_objects(obs)",
                "api.actions",
            ],
            "return_value": "one MiniGrid primitive action int",
        },
        "repair_request": {
            "reason": "Curriculum reflection: the current skill solves some environments but fails others. Write a more general replacement.",
            "current_skill": {"source": current_source},
            "current_skill_self_audit": skill_self_audit(current_source),
            "curriculum_summary": current_summary,
            "failed_envs": sorted({item["env"] for item in failures}),
            "reflection_questions": [
                "What assumptions helped on the solved environments but failed on the unsolved ones?",
                "How should the skill behave when no interactive object is needed and only navigation to a terminal cue matters?",
                "How should it preserve learned structure while avoiding task-family lock-in?",
                "What finite exploration strategy should work when remembered targets are absent?",
            ],
            "acceptance_contract": {
                "required_static_checks": [
                    "source initializes or aliases memory['learned_structure']",
                    "source updates learned_structure from body observations and action outcomes",
                    "source handles both object-interaction tasks and no-object navigation tasks",
                    "source uses finite planning or exploration and avoids turn-only loops",
                ]
            },
            "instruction": (
                "Write a single reusable MiniGrid skill for the curriculum. "
                "Use task-agnostic language and behavior: terminal cues, obstacles, interactive objects, remembered targets, and frontiers. "
                "Do not assume every environment has a key or door. "
                "If no useful interactive object is known, explore and navigate toward terminal cues/frontiers. "
                "Preserve learned state in memory['learned_structure'] or a single private namespace that the harness can normalize. "
                "All loops/search must be finite. Return JSON with name, reason, design, source."
            ),
        },
    }


def compact_examples(rows: list[dict]) -> list[dict]:
    return [
        {
            "env": item["env"],
            "seed": item["seed"],
            "status": item["status"],
            "steps": item["steps"],
            "learned_structure_keys": item.get("learned_structure_keys", []),
            "last_actions": item.get("last_actions", []),
        }
        for item in rows[:18]
    ]


def write_report(path: str, payload: dict) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
