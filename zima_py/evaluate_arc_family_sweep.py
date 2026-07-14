from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arc_compositional_body import composition_search, finish_from_composition, public_composition
from .arc_growing_body import load_library


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen ARC body across operator-family curriculum bins.")
    parser.add_argument("--families", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--tasks-dir", default="arc_agi_source/data/training")
    parser.add_argument("--per-family", type=int, default=12)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--beam", type=int, default=40)
    parser.add_argument("--out", default="artifacts/arc_curriculum/family_sweep.json")
    args = parser.parse_args()

    family_data = json.loads(resolve(args.families).read_text(encoding="utf-8"))["families"]
    learned = load_library(args.library)
    tasks_dir = resolve(args.tasks_dir)
    rows = []

    for family, info in family_data.items():
        task_ids = info["tasks"][: args.per_family]
        family_rows = []
        for task_id in task_ids:
            task_path = tasks_dir / f"{task_id}.json"
            task = json.loads(task_path.read_text(encoding="utf-8"))
            search = composition_search(task, learned, max_depth=args.depth, beam=args.beam)
            test_ok = False
            if search["train_passed"]:
                test_ok = finish_from_composition(task_id, task_path, task, search, "composition_search")["test_ok"]
            family_rows.append(
                {
                    "task": task_id,
                    "train_score": search["train_score"],
                    "train_passed": search["train_passed"],
                    "test_ok": test_ok,
                    "composition": public_composition(search),
                }
            )
        rows.append(
            {
                "family": family,
                "total": len(family_rows),
                "train_passed": sum(1 for row in family_rows if row["train_passed"]),
                "test_correct": sum(1 for row in family_rows if row["test_ok"]),
                "partials": [
                    row
                    for row in family_rows
                    if not row["train_passed"] and int(row["train_score"].split("/")[0]) > 0
                ],
                "rows": family_rows,
            }
        )

    rows.sort(key=lambda row: (row["test_correct"], row["train_passed"], len(row["partials"])), reverse=True)
    summary = {
        "library": str(resolve(args.library)),
        "families": str(resolve(args.families)),
        "per_family": args.per_family,
        "depth": args.depth,
        "beam": args.beam,
        "rows": rows,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps([
        {
            "family": row["family"],
            "test_correct": f"{row['test_correct']}/{row['total']}",
            "train_passed": f"{row['train_passed']}/{row['total']}",
            "partials": len(row["partials"]),
        }
        for row in rows
    ], indent=2))
    print(out)
    return 0


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


if __name__ == "__main__":
    raise SystemExit(main())
