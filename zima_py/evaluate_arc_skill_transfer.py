from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arc_growing_body import evaluate_candidate, load_library, public_search, search_with_body
from .arc_primitive_search import Candidate, copy_grid, normalize
from .arc_reflect import load_solve


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether learned ARC body primitives transfer to held-out-looking ARC training tasks.")
    parser.add_argument("--library", required=True)
    parser.add_argument("--tasks-dir", default="arc_agi_source/data/training")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--out", default="artifacts/arc_growing_body/transfer_eval.json")
    args = parser.parse_args()

    learned = load_library(args.library)
    tasks_dir = resolve(args.tasks_dir)
    excluded = {Path(item).stem for item in args.exclude}
    rows = []

    for task_path in sorted(tasks_dir.glob("*.json")):
        if task_path.stem in excluded:
            continue
        task = json.loads(task_path.read_text(encoding="utf-8"))
        learned_result = learned_only_search(task, learned)
        body_result = search_with_body(task, learned)
        body_test_ok = False
        body_pred = None
        if body_result["train_passed"]:
            body_pred = normalize(body_result["_candidate"].fn(copy_grid(task["test"][0]["input"])))
            expected = task["test"][0].get("output")
            body_test_ok = bool(expected is not None and body_pred == expected)
        rows.append(
            {
                "task": task_path.stem,
                "learned_best": public_search(learned_result),
                "learned_test_ok": learned_result["test_ok"],
                "body_best": public_search(body_result),
                "body_test_ok": body_test_ok,
            }
        )

    summary = {
        "library": str(resolve(args.library)),
        "skills": [{"task": item["task"], "name": item["name"], "test_ok": item["test_ok"]} for item in learned],
        "excluded": sorted(excluded),
        "total": len(rows),
        "learned_train_passed": sum(1 for row in rows if row["learned_best"]["train_passed"]),
        "learned_test_correct": sum(1 for row in rows if row["learned_test_ok"]),
        "body_train_passed": sum(1 for row in rows if row["body_best"]["train_passed"]),
        "body_test_correct": sum(1 for row in rows if row["body_test_ok"]),
        "learned_transfer_hits": [row for row in rows if row["learned_test_ok"]],
        "learned_train_pass_test_fail": [row for row in rows if row["learned_best"]["train_passed"] and not row["learned_test_ok"]],
        "learned_partial": top_partial(rows),
        "body_hits": [row for row in rows if row["body_test_ok"]],
    }

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("total", "learned_train_passed", "learned_test_correct", "body_train_passed", "body_test_correct")}, indent=2))
    print(out)
    return 0


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def learned_only_search(task: dict[str, Any], learned: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for item in learned:
        try:
            solve = load_solve(item["source"])
        except Exception:
            continue
        candidates.append(Candidate(f"learned::{item['task']}::{item['name']}", solve))
    if not candidates:
        return {"best_name": None, "train_score": "0/0", "train_passed": False, "top_candidates": [], "test_ok": False, "_candidate": None}
    scored = []
    for candidate in candidates:
        evaluation = evaluate_candidate(candidate, task["train"])
        scored.append({"name": candidate.name, "candidate": candidate, "evaluation": evaluation})
    scored.sort(key=lambda row: row["evaluation"].get("correct", 0), reverse=True)
    best = scored[0]
    test_ok = False
    if best["evaluation"].get("passed"):
        pred = normalize(best["candidate"].fn(copy_grid(task["test"][0]["input"])))
        expected = task["test"][0].get("output")
        test_ok = bool(expected is not None and pred == expected)
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
            for row in scored
        ],
        "test_ok": test_ok,
        "_candidate": best["candidate"],
    }


def top_partial(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    partial = []
    for row in rows:
        score = row["learned_best"]["train_score"]
        try:
            correct, total = [int(item) for item in score.split("/")]
        except ValueError:
            continue
        if correct > 0 and correct < total:
            partial.append(row)
    partial.sort(key=lambda row: int(row["learned_best"]["train_score"].split("/")[0]), reverse=True)
    return partial[:20]


if __name__ == "__main__":
    raise SystemExit(main())
