from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .arc_growing_body import evaluate_candidate, load_library
from .arc_primitive_search import Candidate, copy_grid, normalize
from .arc_reflect import load_solve


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Find visually similar ARC tasks and test learned skills on them.")
    parser.add_argument("--library", required=True)
    parser.add_argument("--tasks-dir", default="arc_agi_source/data/training")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default="artifacts/arc_growing_body/similar_transfer_eval.json")
    args = parser.parse_args()

    tasks_dir = resolve(args.tasks_dir)
    excluded = {Path(item).stem for item in args.exclude}
    task_paths = {path.stem: path for path in tasks_dir.glob("*.json")}
    task_features = {
        task_id: feature(json.loads(path.read_text(encoding="utf-8")))
        for task_id, path in task_paths.items()
    }
    skills = load_library(args.library)
    rows = []

    for skill in skills:
        source_task = skill["task"]
        if source_task not in task_features:
            continue
        solve = load_solve(skill["source"])
        candidate = Candidate(f"learned::{source_task}::{skill['name']}", solve)
        neighbors = []
        for task_id, vec in task_features.items():
            if task_id == source_task or task_id in excluded:
                continue
            neighbors.append((distance(task_features[source_task], vec), task_id))
        neighbors.sort()
        tested = []
        for dist, task_id in neighbors[: args.top_k]:
            task = json.loads(task_paths[task_id].read_text(encoding="utf-8"))
            eval_result = evaluate_candidate(candidate, task["train"])
            test_ok = False
            if eval_result["passed"]:
                pred = normalize(solve(copy_grid(task["test"][0]["input"])))
                expected = task["test"][0].get("output")
                test_ok = bool(expected is not None and pred == expected)
            tested.append(
                {
                    "task": task_id,
                    "distance": round(dist, 4),
                    "train_score": f"{eval_result.get('correct', 0)}/{eval_result.get('total', len(task['train']))}",
                    "train_passed": bool(eval_result["passed"]),
                    "test_ok": test_ok,
                    "first_failure": next((case for case in eval_result.get("cases", []) if not case.get("ok")), None),
                }
            )
        rows.append({"source_task": source_task, "skill": skill["name"], "neighbors": tested})

    summary = {
        "library": str(resolve(args.library)),
        "excluded": sorted(excluded),
        "top_k": args.top_k,
        "skills_tested": len(rows),
        "neighbor_cases": sum(len(row["neighbors"]) for row in rows),
        "train_passed": sum(1 for row in rows for item in row["neighbors"] if item["train_passed"]),
        "test_correct": sum(1 for row in rows for item in row["neighbors"] if item["test_ok"]),
        "rows": rows,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("skills_tested", "neighbor_cases", "train_passed", "test_correct")}, indent=2))
    print(out)
    return 0


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def feature(task: dict[str, Any]) -> list[float]:
    train = task["train"]
    values: list[float] = []
    values.extend(avg_shape(pair["input"] for pair in train))
    values.extend(avg_shape(pair["output"] for pair in train))
    values.append(sum(1 for pair in train if shape(pair["input"]) == shape(pair["output"])) / max(1, len(train)))
    values.extend(color_hist(pair["input"] for pair in train))
    values.extend(color_hist(pair["output"] for pair in train))
    values.append(avg_density(pair["input"] for pair in train))
    values.append(avg_density(pair["output"] for pair in train))
    return values


def avg_shape(grids: Any) -> list[float]:
    pairs = [shape(grid) for grid in grids]
    return [sum(h for h, _ in pairs) / len(pairs), sum(w for _, w in pairs) / len(pairs)]


def shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def color_hist(grids: Any) -> list[float]:
    counts = [0] * 10
    total = 0
    for grid in grids:
        for row in grid:
            for value in row:
                if 0 <= int(value) < 10:
                    counts[int(value)] += 1
                total += 1
    return [count / max(1, total) for count in counts]


def avg_density(grids: Any) -> float:
    vals = []
    for grid in grids:
        total = sum(len(row) for row in grid)
        nonzero = sum(1 for row in grid for value in row if value != 0)
        vals.append(nonzero / max(1, total))
    return sum(vals) / max(1, len(vals))


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


if __name__ == "__main__":
    raise SystemExit(main())
