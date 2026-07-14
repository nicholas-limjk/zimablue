from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .arc_compositional_body import composition_search, public_composition
from .arc_growing_body import load_library


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build simple ARC curriculum bins from observable task signatures.")
    parser.add_argument("--library", default=None)
    parser.add_argument("--tasks-dir", default="arc_agi_source/data/training")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam", type=int, default=25)
    parser.add_argument("--out", default="artifacts/arc_curriculum/bins.json")
    args = parser.parse_args()

    learned = load_library(args.library) if args.library else []
    excluded = {Path(item).stem for item in args.exclude}
    rows = []
    for path in sorted(resolve(args.tasks_dir).glob("*.json")):
        if path.stem in excluded:
            continue
        task = json.loads(path.read_text(encoding="utf-8"))
        row = describe_task(path.stem, task)
        if args.library:
            search = composition_search(task, learned, max_depth=args.depth, beam=args.beam)
            row["body_train_score"] = search["train_score"]
            row["body_train_passed"] = search["train_passed"]
            row["body_best_sequence"] = public_composition(search)["best_sequence"]
        rows.append(row)

    bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for tag in row["tags"]:
            bins[tag].append(row)

    summary = {
        "total": len(rows),
        "bins": {
            tag: {
                "count": len(items),
                "unsolved_count": sum(1 for item in items if not item.get("body_train_passed", False)),
                "tasks": [item["task"] for item in items],
            }
            for tag, items in sorted(bins.items(), key=lambda pair: (-len(pair[1]), pair[0]))
        },
        "rows": rows,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"total": summary["total"], "top_bins": [(tag, info["count"], info["unsolved_count"]) for tag, info in list(summary["bins"].items())[:12]]}, indent=2))
    print(out)
    return 0


def describe_task(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    train = task["train"]
    input_shapes = [shape(pair["input"]) for pair in train]
    output_shapes = [shape(pair["output"]) for pair in train]
    same_shape_ratio = sum(1 for a, b in zip(input_shapes, output_shapes) if a == b) / len(train)
    area_ratios = [(oh * ow) / max(1, ih * iw) for (ih, iw), (oh, ow) in zip(input_shapes, output_shapes)]
    color_delta = avg(color_count(pair["output"]) - color_count(pair["input"]) for pair in train)
    density_delta = avg(density(pair["output"]) - density(pair["input"]) for pair in train)
    tags = []
    if same_shape_ratio == 1:
        tags.append("same_shape")
    if same_shape_ratio == 0:
        tags.append("shape_change")
    if avg(area_ratios) < 0.55:
        tags.append("crop_or_extract")
    if avg(area_ratios) > 1.5:
        tags.append("expand_or_tile")
    if abs(avg(area_ratios) - 1.0) < 0.05 and abs(color_delta) >= 1:
        tags.append("recolor_or_mask")
    if density_delta > 0.15:
        tags.append("fill_or_complete")
    if density_delta < -0.15:
        tags.append("erase_or_select")
    if output_objectish(train):
        tags.append("object_selection")
    if not tags:
        tags.append("misc")
    return {
        "task": task_id,
        "tags": tags,
        "train_count": len(train),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "avg_area_ratio": round(avg(area_ratios), 3),
        "avg_color_delta": round(color_delta, 3),
        "avg_density_delta": round(density_delta, 3),
    }


def output_objectish(train: list[dict[str, Any]]) -> bool:
    return all(shape(pair["output"])[0] <= shape(pair["input"])[0] and shape(pair["output"])[1] <= shape(pair["input"])[1] for pair in train)


def shape(grid: list[list[int]]) -> list[int]:
    return [len(grid), len(grid[0]) if grid else 0]


def color_count(grid: list[list[int]]) -> int:
    return len({v for row in grid for v in row})


def density(grid: list[list[int]]) -> float:
    total = sum(len(row) for row in grid)
    return sum(1 for row in grid for v in row if v != 0) / max(1, total)


def avg(values: Any) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


if __name__ == "__main__":
    raise SystemExit(main())
