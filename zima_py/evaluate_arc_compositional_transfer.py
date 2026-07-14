from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arc_compositional_body import composition_search, finish_from_composition, public_composition
from .arc_growing_body import load_library


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ARC compositional body search on many tasks without new LLM calls.")
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--library", required=True)
    parser.add_argument("--tasks-dir", default="arc_agi_source/data/training")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--beam", type=int, default=40)
    parser.add_argument("--out", default="artifacts/arc_compositional_body/transfer_eval.json")
    args = parser.parse_args()

    learned = load_library(args.library)
    tasks_dir = resolve(args.tasks_dir)
    excluded = {Path(item).stem for item in args.exclude}
    if args.tasks:
        task_paths = [resolve(task) for task in args.tasks]
    else:
        all_paths = [path for path in sorted(tasks_dir.glob("*.json")) if path.stem not in excluded]
        task_paths = all_paths[args.offset : args.offset + args.limit if args.limit is not None else None]
    rows = []

    for index, task_path in enumerate(task_paths, start=1):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        search = composition_search(task, learned, max_depth=args.depth, beam=args.beam)
        test_ok = False
        test_prediction = None
        if search["train_passed"]:
            result = finish_from_composition(task_path.stem, task_path, task, search, "composition_search")
            test_ok = result["test_ok"]
            test_prediction = result["test_prediction"]
        row = {
            "task": task_path.stem,
            "train_score": search["train_score"],
            "train_passed": search["train_passed"],
            "test_ok": test_ok,
            "test_prediction": test_prediction,
            "composition": public_composition(search),
        }
        rows.append(row)
        print(json.dumps({"batch_index": index, "task": task_path.stem, "train": row["train_score"], "test_ok": test_ok, **progress(rows)}, indent=2))

    summary = {
        "library": str(resolve(args.library)),
        "depth": args.depth,
        "beam": args.beam,
        "excluded": sorted(excluded),
        "offset": args.offset,
        "limit": args.limit,
        "total": len(rows),
        "train_passed": sum(1 for row in rows if row["train_passed"]),
        "test_correct": sum(1 for row in rows if row["test_ok"]),
        "rows": rows,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("total", "train_passed", "test_correct")}, indent=2))
    print(out)
    return 0


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def progress(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seen": len(rows),
        "train_passed": sum(1 for row in rows if row["train_passed"]),
        "test_correct": sum(1 for row in rows if row["test_ok"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
