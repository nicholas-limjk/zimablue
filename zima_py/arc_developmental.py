from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arc_reflect import ArcSkillWriter, build_prompt, build_prompt_payload, evaluate_source, predict_tests, static_checks


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential ARC agent with persistent learned skill library.")
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--library-out", default="artifacts/arc_developmental/library.json")
    parser.add_argument("--report-out", default="artifacts/arc_developmental/report.json")
    parser.add_argument("--skill-dir", default="generated_skills/arc_developmental")
    args = parser.parse_args()

    writer = ArcSkillWriter(model=args.model, reasoning_effort=args.reasoning_effort)
    library: list[dict[str, Any]] = []
    runs = []
    skill_dir = ROOT / args.skill_dir
    skill_dir.mkdir(parents=True, exist_ok=True)

    for task_arg in args.tasks:
        task_path = (ROOT / task_arg).resolve() if not Path(task_arg).is_absolute() else Path(task_arg)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task_id = task_path.stem
        print(f"task {task_id}")

        reuse = try_library(task, library)
        if reuse:
            runs.append({**reuse, "task": task_id, "path": str(task_path), "mode": "reuse"})
            print(json.dumps({"task": task_id, "mode": "reuse", "train": reuse["train_score"], "test_ok": reuse["test_ok"]}, indent=2))
            continue

        partial_matches = score_library(task, library)
        previous_source = ""
        previous_eval = None
        best = None
        iterations = []
        for iteration in range(args.iters):
            payload = build_prompt_payload(task_path, task, iteration, previous_source, previous_eval, library)
            if partial_matches:
                payload["prior_skill_partial_matches"] = partial_matches[:5]
                payload["partial_match_instruction"] = (
                    "These are concrete evaluations of prior skills on the current training pairs. "
                    "If a prior skill partially works, mutate the best partial match. "
                    "Use its failure diffs to identify exactly which selector, geometry, color, or output-construction assumption broke."
                )
            proposal = writer.propose(payload)
            checks = static_checks(proposal.source)
            if checks:
                eval_result = {"passed": False, "correct": 0, "total": len(task["train"]), "errors": checks, "cases": []}
            else:
                eval_result = evaluate_source(proposal.source, task["train"])
            iterations.append(
                {
                    "iteration": iteration + 1,
                    "name": proposal.name,
                    "reason": proposal.reason,
                    "design": proposal.design,
                    "static_errors": checks,
                    "evaluation": eval_result,
                }
            )
            if best is None or eval_result.get("correct", 0) > best["evaluation"].get("correct", 0):
                best = {"proposal": proposal, "evaluation": eval_result}
            previous_source = proposal.source
            previous_eval = eval_result
            print(json.dumps({"iteration": iteration + 1, "correct": eval_result.get("correct", 0), "total": len(task["train"])}, indent=2))
            if eval_result.get("passed"):
                break

        if best is None:
            raise RuntimeError(f"No proposal for {task_id}")
        proposal = best["proposal"]
        evaluation = best["evaluation"]
        skill_path = skill_dir / f"{task_id}.py"
        skill_path.write_text(proposal.source.strip() + "\n", encoding="utf-8")
        test_predictions = predict_tests(proposal.source, task["test"]) if evaluation.get("passed") else []
        expected = task["test"][0].get("output")
        test_ok = bool(test_predictions and expected is not None and test_predictions[0]["prediction"] == expected)
        train_score = f"{evaluation.get('correct', 0)}/{evaluation.get('total', len(task['train']))}"
        run = {
            "task": task_id,
            "path": str(task_path),
            "mode": "write",
            "skill_path": str(skill_path),
            "train_score": train_score,
            "train_passed": bool(evaluation.get("passed")),
            "test_ok": test_ok,
            "iterations": iterations,
            "test_predictions": test_predictions,
        }
        runs.append(run)
        if evaluation.get("passed"):
            library.append(
                {
                    "task": task_id,
                    "skill_path": str(skill_path),
                    "train_score": train_score,
                    "test_ok": test_ok,
                    "reason": proposal.reason,
                    "design": proposal.design,
                    "source": proposal.source,
                }
            )
        print(json.dumps({"task": task_id, "train": train_score, "test_ok": test_ok, "library_size": len(library)}, indent=2))

    report = {
        "model": writer.model,
        "reasoning_effort": writer.reasoning_effort,
        "total": len(runs),
        "test_correct": sum(1 for run in runs if run["test_ok"]),
        "library_size": len(library),
        "runs": runs,
    }
    library_out = ROOT / args.library_out
    report_out = ROOT / args.report_out
    library_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    library_out.write_text(json.dumps({"skills": library}, indent=2), encoding="utf-8")
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_correct": report["test_correct"], "total": report["total"], "library_size": len(library), "report": str(report_out)}, indent=2))
    return 0


def try_library(task: dict[str, Any], library: list[dict[str, Any]]) -> dict[str, Any] | None:
    expected = task["test"][0].get("output")
    for item in reversed(library):
        source = item.get("source", "")
        eval_result = evaluate_source(source, task["train"])
        if not eval_result.get("passed"):
            continue
        predictions = predict_tests(source, task["test"])
        test_ok = bool(predictions and expected is not None and predictions[0]["prediction"] == expected)
        return {
            "source_task": item.get("task"),
            "train_score": f"{eval_result['correct']}/{eval_result['total']}",
            "train_passed": True,
            "test_ok": test_ok,
            "test_predictions": predictions,
        }
    return None


def score_library(task: dict[str, Any], library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = []
    for item in reversed(library):
        source = item.get("source", "")
        eval_result = evaluate_source(source, task["train"])
        scores.append(
            {
                "source_task": item.get("task"),
                "train_score": f"{eval_result.get('correct', 0)}/{eval_result.get('total', len(task['train']))}",
                "passed": bool(eval_result.get("passed")),
                "reason": item.get("reason", ""),
                "design": item.get("design", ""),
                "source_excerpt": source[:5000],
                "failure_cases": [
                    case
                    for case in eval_result.get("cases", [])
                    if not case.get("ok")
                ][:3],
            }
        )
    scores.sort(key=lambda row: (int(row["train_score"].split("/")[0]), row["passed"]), reverse=True)
    return scores


if __name__ == "__main__":
    raise SystemExit(main())
