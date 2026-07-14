from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .arc_primitive_search import Candidate, build_candidates, copy_grid, diff_summary, normalize, shape
from .arc_reflect import load_solve, static_checks
from .llm import load_env_file, parse_json_object, supports_custom_temperature


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="ARC agent with base body primitives plus persistent learned primitives.")
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--library-in", default=None)
    parser.add_argument("--library-out", default="artifacts/arc_growing_body/library.json")
    parser.add_argument("--report-out", default="artifacts/arc_growing_body/report.json")
    parser.add_argument("--skill-dir", default="generated_skills/arc_growing_body")
    args = parser.parse_args()

    client = PrimitiveWriter(model=args.model, reasoning_effort=args.reasoning_effort)
    learned: list[dict[str, Any]] = load_library(args.library_in)
    runs = []
    skill_dir = ROOT / args.skill_dir
    skill_dir.mkdir(parents=True, exist_ok=True)

    for task_arg in args.tasks:
        task_path = (ROOT / task_arg).resolve() if not Path(task_arg).is_absolute() else Path(task_arg)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task_id = task_path.stem
        print(f"task {task_id}")

        search = search_with_body(task, learned)
        if search["train_passed"]:
            run = finish_from_search(task_id, task_path, task, search, "search")
            runs.append(run)
            print(json.dumps({"task": task_id, "mode": "search", "best": search["best_name"], "train": search["train_score"], "test_ok": run["test_ok"], "library_size": len(learned)}, indent=2))
            continue

        history = []
        best_written = None
        previous_eval = None
        for iteration in range(args.iters):
            payload = build_payload(task_path, task, learned, search, history, previous_eval)
            proposal = client.propose(payload)
            source = str(proposal["source"])
            checks = static_checks(source)
            if checks:
                evaluation = {"passed": False, "correct": 0, "total": len(task["train"]), "errors": checks, "cases": []}
            else:
                evaluation = evaluate_source_as_primitive(source, task["train"])
            row = {
                "iteration": iteration + 1,
                "name": str(proposal.get("name", f"{task_id}_primitive")),
                "reason": str(proposal.get("reason", "")),
                "design": str(proposal.get("design", "")),
                "source": source,
                "evaluation": evaluation,
            }
            history.append(compact_written(row))
            if best_written is None or evaluation.get("correct", 0) > best_written["evaluation"].get("correct", 0):
                best_written = row
            previous_eval = evaluation
            print(json.dumps({"iteration": iteration + 1, "correct": evaluation.get("correct", 0), "total": evaluation.get("total", len(task["train"]))}, indent=2))
            if evaluation.get("passed"):
                break

        if best_written is None:
            raise RuntimeError(f"No primitive proposal for {task_id}")
        skill_path = skill_dir / f"{task_id}.py"
        skill_path.write_text(best_written["source"].strip() + "\n", encoding="utf-8")
        test_prediction = None
        test_ok = False
        if best_written["evaluation"].get("passed"):
            solve = load_solve(best_written["source"])
            test_prediction = normalize(solve(copy_grid(task["test"][0]["input"])))
            expected = task["test"][0].get("output")
            test_ok = bool(expected is not None and test_prediction == expected)
        run = {
            "task": task_id,
            "path": str(task_path),
            "mode": "write_new_primitive",
            "best_name": best_written["name"],
            "train_score": f"{best_written['evaluation'].get('correct', 0)}/{best_written['evaluation'].get('total', len(task['train']))}",
            "train_passed": bool(best_written["evaluation"].get("passed")),
            "test_ok": test_ok,
            "test_prediction": test_prediction,
            "skill_path": str(skill_path),
            "search_before_write": public_search(search),
            "write_history": history,
        }
        runs.append(run)
        if best_written["evaluation"].get("passed"):
            learned.append(library_record(task_id, run, best_written["source"], best_written["name"], best_written["design"]))
        print(json.dumps({"task": task_id, "mode": "write", "train": run["train_score"], "test_ok": test_ok, "library_size": len(learned)}, indent=2))

    report = {
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "total": len(runs),
        "test_correct": sum(1 for run in runs if run["test_ok"]),
        "library_size": len(learned),
        "runs": runs,
    }
    library_out = ROOT / args.library_out
    report_out = ROOT / args.report_out
    library_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    library_out.write_text(json.dumps({"skills": learned}, indent=2), encoding="utf-8")
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_correct": report["test_correct"], "total": report["total"], "library_size": len(learned), "report": str(report_out)}, indent=2))
    return 0


def load_library(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    resolved = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not resolved.exists():
        return []
    data = json.loads(resolved.read_text(encoding="utf-8"))
    skills = data.get("skills", [])
    if not isinstance(skills, list):
        raise ValueError(f"library file must contain a skills list: {resolved}")
    return skills


class PrimitiveWriter:
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
                        "You write one new ARC body primitive. Return JSON only with keys name, reason, design, source. "
                        "source must define def solve(grid): and return list[list[int]]. "
                        "Use pure Python list operations. Do not import anything. Avoid numpy unless the operation is extremely simple. "
                        "No file/network/subprocess/eval/exec/randomness. "
                        "The primitive should be reusable and train-verifiable, not a hard-coded test answer."
                    ),
                },
                {"role": "user", "content": build_prompt(payload)},
            ],
        }
        if supports_custom_temperature(self.model):
            body["temperature"] = 0.2
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


def search_with_body(task: dict, learned: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = build_candidates(task["train"])
    for item in learned:
        try:
            solve = load_solve(item["source"])
        except Exception:
            continue
        candidates.append(Candidate(f"learned::{item['task']}::{item['name']}", solve))
    scored = []
    for candidate in candidates:
        evaluation = evaluate_candidate(candidate, task["train"])
        source = None
        if candidate.name.startswith("learned::"):
            task_id = candidate.name.split("::")[1]
            source = next((item["source"] for item in learned if item["task"] == task_id), None)
        scored.append({"name": candidate.name, "candidate": candidate, "evaluation": evaluation, "source": source})
    scored.sort(key=lambda row: row["evaluation"].get("correct", 0), reverse=True)
    best = scored[0]
    return {
        "best_name": best["name"],
        "train_score": f"{best['evaluation'].get('correct', 0)}/{best['evaluation'].get('total', len(task['train']))}",
        "train_passed": bool(best["evaluation"].get("passed")),
        "top_candidates": [
            {
                "name": row["name"],
                "train_score": f"{row['evaluation'].get('correct', 0)}/{row['evaluation'].get('total', len(task['train']))}",
                "failed_cases": [case for case in row["evaluation"].get("cases", []) if not case.get("ok")][:2],
            }
            for row in scored[:10]
        ],
        "source": best["source"],
        "_candidate": best["candidate"],
    }


def public_search(search: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in search.items() if k != "_candidate"}


def finish_from_search(task_id: str, task_path: Path, task: dict, search: dict[str, Any], mode: str) -> dict[str, Any]:
    pred = normalize(search["_candidate"].fn(copy_grid(task["test"][0]["input"])))
    expected = task["test"][0].get("output")
    result = {
        "task": task_id,
        "path": str(task_path),
        "mode": mode,
        "best_name": search["best_name"],
        "train_score": search["train_score"],
        "train_passed": True,
        "test_ok": bool(expected is not None and pred == expected),
        "test_prediction": pred,
        "search": public_search(search),
    }
    return result


def evaluate_candidate(candidate: Candidate, train_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    correct = 0
    for idx, pair in enumerate(train_pairs):
        try:
            pred = normalize(candidate.fn(copy_grid(pair["input"])))
            expected = normalize(pair["output"])
            ok = pred == expected
        except Exception as exc:
            cases.append({"index": idx, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        correct += int(ok)
        cases.append({"index": idx, "ok": ok, "predicted_shape": shape(pred), "expected_shape": shape(expected), "diff": diff_summary(pred, expected)})
    return {"passed": correct == len(train_pairs), "correct": correct, "total": len(train_pairs), "cases": cases}


def evaluate_source_as_primitive(source: str, train_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        solve = load_solve(source)
    except Exception as exc:
        return {"passed": False, "correct": 0, "total": len(train_pairs), "errors": [f"load error: {type(exc).__name__}: {exc}"], "cases": []}
    return evaluate_candidate(Candidate("proposed", solve), train_pairs)


def build_payload(
    task_path: Path,
    task: dict[str, Any],
    learned: list[dict[str, Any]],
    search: dict[str, Any],
    history: list[dict[str, Any]],
    previous_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_id": task_path.stem,
        "base_impulse": "grow the agent body by adding one reusable primitive when current primitives fail",
        "train_pairs": task["train"],
        "test_inputs": [{"input": item["input"]} for item in task["test"]],
        "current_body_search": {k: v for k, v in search.items() if k != "_candidate" and k != "source"},
        "learned_primitives": [
            {
                "task": item["task"],
                "name": item["name"],
                "train_score": item["train_score"],
                "test_ok": item["test_ok"],
                "design": item["design"],
                "source_excerpt": item["source"][:3500],
            }
            for item in learned[-8:]
        ],
        "previous_written_attempts": history[-4:],
        "previous_eval": previous_eval,
        "instructions": [
            "The base body primitives and learned primitives were already searched and failed.",
            "Write exactly one new primitive as def solve(grid):.",
            "Use current_body_search top candidate failures to identify the missing operation.",
            "Prefer reusable operations over task-specific literal outputs.",
            "A useful new primitive may combine object detection, geometry, color rules, masks, fills, or output construction.",
            "If a learned primitive is close, mutate it; otherwise invent a new body primitive.",
        ],
    }


def build_prompt(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "ARC growing-body episode.",
            "The agent has a base body of simple primitives. Search over that body has failed on this task.",
            "Your job is to add one new reusable body primitive, not merely answer the puzzle.",
            "The harness will verify your primitive on all training pairs; only train-passing primitives persist.",
            "Return JSON only: {\"name\": ..., \"reason\": ..., \"design\": ..., \"source\": ...}",
            "",
            json.dumps(payload, indent=2),
        ]
    )


def compact_written(row: dict[str, Any]) -> dict[str, Any]:
    evaluation = row["evaluation"]
    return {
        "name": row["name"],
        "design": row["design"],
        "train_score": f"{evaluation.get('correct', 0)}/{evaluation.get('total', 0)}",
        "errors": evaluation.get("errors", []),
        "failed_cases": [case for case in evaluation.get("cases", []) if not case.get("ok")][:3],
        "source_excerpt": row["source"][:2500],
    }


def library_record(task_id: str, run: dict[str, Any], source: str, name: str, design: str) -> dict[str, Any]:
    return {
        "task": task_id,
        "name": name,
        "train_score": run["train_score"],
        "test_ok": run["test_ok"],
        "design": design,
        "source": source,
    }


if __name__ == "__main__":
    raise SystemExit(main())
