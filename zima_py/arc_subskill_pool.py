from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .arc_reflect import evaluate_source, predict_tests, static_checks
from .llm import load_env_file, parse_json_object, supports_custom_temperature


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="ARC reflection with independently tested candidate subskills.")
    parser.add_argument("task")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--skill-out", default=None)
    args = parser.parse_args()

    task_path = (ROOT / args.task).resolve() if not Path(args.task).is_absolute() else Path(args.task)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task_id = task_path.stem
    out = ROOT / (args.out or f"artifacts/arc_subskill_pool/{task_id}.json")
    skill_out = ROOT / (args.skill_out or f"generated_skills/arc_subskill_pool/{task_id}.py")

    client = SubskillClient(model=args.model, reasoning_effort=args.reasoning_effort)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for round_index in range(args.rounds):
        response = client.propose(build_payload(task_path, task, args.candidates, history))
        candidates = response.get("candidates", [])
        round_rows = []
        for idx, candidate in enumerate(candidates):
            source = str(candidate.get("source", ""))
            checks = static_checks(source)
            if checks:
                evaluation = {"passed": False, "correct": 0, "total": len(task["train"]), "errors": checks, "cases": []}
            else:
                evaluation = evaluate_source(source, task["train"])
            row = {
                "round": round_index + 1,
                "index": idx,
                "name": str(candidate.get("name", f"candidate_{idx}")),
                "design": str(candidate.get("design", "")),
                "source": source,
                "evaluation": evaluation,
            }
            round_rows.append(row)
            if best is None or evaluation.get("correct", 0) > best["evaluation"].get("correct", 0):
                best = row
        history.append({"round": round_index + 1, "candidates": compact_candidates(round_rows)})
        print(json.dumps({"round": round_index + 1, "scores": [row["evaluation"].get("correct", 0) for row in round_rows]}, indent=2))
        if best and best["evaluation"].get("passed"):
            break

    if best is None:
        raise RuntimeError("No candidates generated.")
    skill_out.parent.mkdir(parents=True, exist_ok=True)
    skill_out.write_text(best["source"].strip() + "\n", encoding="utf-8")
    test_predictions = predict_tests(best["source"], task["test"]) if best["evaluation"].get("passed") else []
    expected = task["test"][0].get("output")
    test_ok = bool(test_predictions and expected is not None and test_predictions[0]["prediction"] == expected)
    report = {
        "task": str(task_path),
        "rounds": history,
        "best": {
            "name": best["name"],
            "design": best["design"],
            "skill_path": str(skill_out),
            "evaluation": best["evaluation"],
            "test_ok": test_ok,
            "test_predictions": test_predictions,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"task": task_id, "best_train": f"{best['evaluation'].get('correct', 0)}/{best['evaluation'].get('total', len(task['train']))}", "test_ok": test_ok, "report": str(out)}, indent=2))
    return 0


class SubskillClient:
    def __init__(self, model: str | None = None, reasoning_effort: str | None = None, timeout_s: int = 300):
        load_env_file()
        self.api_key = os.getenv("AGENT_SKILL_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")
        self.endpoint = os.getenv("AGENT_SKILL_ENDPOINT") or "https://api.openai.com/v1/chat/completions"
        self.model = model or os.getenv("AGENT_SKILL_MODEL") or "gpt-4.1-mini"
        self.reasoning_effort = reasoning_effort
        self.timeout_s = int(os.getenv("AGENT_SKILL_TIMEOUT_S", str(timeout_s)))
        if not self.api_key:
            raise RuntimeError("Missing AGENT_SKILL_API_KEY / OPENAI_API_KEY / openai_key.")

    def propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate a small pool of named ARC candidate skills. "
                        "Return JSON only with key candidates. candidates is a list of objects with keys name, design, source. "
                        "Each source must be a complete Python module defining def solve(grid):. "
                        "Do not import anything; np is already available. No file, network, subprocess, eval, exec, randomness."
                    ),
                },
                {"role": "user", "content": build_prompt(payload)},
            ],
        }
        if supports_custom_temperature(self.model):
            body["temperature"] = 0.4
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        return parse_json_object(data["choices"][0]["message"]["content"])


def build_payload(task_path: Path, task: dict[str, Any], count: int, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": task_path.stem,
        "base_impulse": "make output grids match the ARC training examples",
        "candidate_count": count,
        "train_pairs": task["train"],
        "test_inputs": [{"input": item["input"]} for item in task["test"]],
        "history": history[-4:],
        "instructions": [
            "Generate diverse candidate subskills, not small variations of one idea.",
            "Each candidate should test a different representation or rule: object selection, fill/line construction, symmetry, mask extraction, crop/scale, color rule, etc.",
            "If history exists, mutate the strongest partial candidate and also include at least one different hypothesis.",
            "The harness will execute each candidate independently on all train pairs and keep the best.",
        ],
    }


def build_prompt(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "ARC candidate-subskill pool episode.",
            "You are not writing one final answer; you are proposing a population of named executable hypotheses.",
            "The environment will test each candidate independently and feed back the scores/diffs.",
            "Return JSON only: {\"candidates\": [{\"name\": ..., \"design\": ..., \"source\": ...}, ...]}",
            "",
            json.dumps(payload, indent=2),
        ]
    )


def compact_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        evaluation = row["evaluation"]
        compact.append(
            {
                "name": row["name"],
                "design": row["design"],
                "train_score": f"{evaluation.get('correct', 0)}/{evaluation.get('total', 0)}",
                "errors": evaluation.get("errors", []),
                "failed_cases": [case for case in evaluation.get("cases", []) if not case.get("ok")][:3],
                "source_excerpt": row["source"][:3000],
            }
        )
    compact.sort(key=lambda item: int(item["train_score"].split("/")[0]), reverse=True)
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
