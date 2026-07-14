from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build narrower ARC operator-family bins from train-pair signatures.")
    parser.add_argument("--tasks-dir", default="arc_agi_source/data/training")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--out", default="artifacts/arc_curriculum/operator_families.json")
    args = parser.parse_args()

    excluded = {Path(item).stem for item in args.exclude}
    rows = []
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(resolve(args.tasks_dir).glob("*.json")):
        if path.stem in excluded:
            continue
        task = json.loads(path.read_text(encoding="utf-8"))
        row = describe(path.stem, task)
        rows.append(row)
        for family in row["families"]:
            families[family].append(row)

    summary = {
        "total": len(rows),
        "families": {
            name: {
                "count": len(items),
                "tasks": [item["task"] for item in items],
            }
            for name, items in sorted(families.items(), key=lambda pair: (-len(pair[1]), pair[0]))
        },
        "rows": rows,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"total": summary["total"], "top_families": [(k, v["count"]) for k, v in list(summary["families"].items())[:20]]}, indent=2))
    print(out)
    return 0


def describe(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    pairs = task["train"]
    in_shapes = [shape(pair["input"]) for pair in pairs]
    out_shapes = [shape(pair["output"]) for pair in pairs]
    ratios = []
    for (ih, iw), (oh, ow) in zip(in_shapes, out_shapes):
        ratios.append((ratio(oh, ih), ratio(ow, iw)))
    families = set()
    if all(r == (2.0, 2.0) for r in ratios):
        families.add("scale_2x")
    if all(r == (3.0, 3.0) for r in ratios):
        families.add("scale_3x")
    if all(r[0] == r[1] and r[0] in {2.0, 3.0, 4.0} for r in ratios):
        families.add("uniform_integer_scale")
    if all(r[0] >= 1.5 and r[1] >= 1.5 for r in ratios):
        families.add("expand_both_axes")
    if all(r[0] > 1.2 and abs(r[1] - 1.0) < 0.01 for r in ratios):
        families.add("expand_rows_only")
    if all(abs(r[0] - 1.0) < 0.01 and r[1] > 1.2 for r in ratios):
        families.add("expand_cols_only")
    if all(r[0] < 0.8 and r[1] < 0.8 for r in ratios):
        families.add("crop_both_axes")
    if all(shape(pair["input"]) == shape(pair["output"]) for pair in pairs):
        if avg_density_delta(pairs) > 0.15:
            families.add("same_shape_fill")
        if abs(avg_color_delta(pairs)) >= 1:
            families.add("same_shape_recolor")
    if output_is_tiled_input_size(pairs):
        families.add("output_grid_of_input_tiles")
    if not families:
        families.add("misc")
    return {
        "task": task_id,
        "families": sorted(families),
        "input_shapes": in_shapes,
        "output_shapes": out_shapes,
        "ratios": ratios,
        "density_delta": round(avg_density_delta(pairs), 3),
        "color_delta": round(avg_color_delta(pairs), 3),
    }


def output_is_tiled_input_size(pairs: list[dict[str, Any]]) -> bool:
    hits = 0
    for pair in pairs:
        ih, iw = shape(pair["input"])
        oh, ow = shape(pair["output"])
        if ih and iw and oh % ih == 0 and ow % iw == 0 and (oh > ih or ow > iw):
            hits += 1
    return hits == len(pairs)


def avg_density_delta(pairs: list[dict[str, Any]]) -> float:
    return avg(density(pair["output"]) - density(pair["input"]) for pair in pairs)


def avg_color_delta(pairs: list[dict[str, Any]]) -> float:
    return avg(color_count(pair["output"]) - color_count(pair["input"]) for pair in pairs)


def shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def ratio(a: int, b: int) -> float:
    return round(a / max(1, b), 3)


def density(grid: list[list[int]]) -> float:
    total = sum(len(row) for row in grid)
    return sum(1 for row in grid for value in row if value != 0) / max(1, total)


def color_count(grid: list[list[int]]) -> int:
    return len({value for row in grid for value in row})


def avg(values: Any) -> float:
    vals = list(values)
    return sum(vals) / max(1, len(vals))


def resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


if __name__ == "__main__":
    raise SystemExit(main())
