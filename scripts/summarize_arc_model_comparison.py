from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    "007bbfb7_grid_expansion",
    "1cf80156_object_extraction",
    "0520fde7_spatial_mapping",
    "0ca9ddb6_marker_to_shape_revision",
    "28e73c20_boundary_fill_revision",
    "3de23699_selector_revision",
]
TASK_FILES = {
    "007bbfb7_grid_expansion": ROOT / "examples/arc/category_1_good_fit/007bbfb7_grid_expansion.json",
    "1cf80156_object_extraction": ROOT / "examples/arc/category_1_good_fit/1cf80156_object_extraction.json",
    "0520fde7_spatial_mapping": ROOT / "examples/arc/category_1_good_fit/0520fde7_spatial_mapping.json",
    "0ca9ddb6_marker_to_shape_revision": ROOT / "examples/arc/category_2_reflection_helps/0ca9ddb6_marker_to_shape_revision.json",
    "28e73c20_boundary_fill_revision": ROOT / "examples/arc/category_2_reflection_helps/28e73c20_boundary_fill_revision.json",
    "3de23699_selector_revision": ROOT / "examples/arc/category_2_reflection_helps/3de23699_selector_revision.json",
}


def main() -> int:
    vanilla = json.loads((ROOT / "artifacts/arc_vanilla_selected6_gpt5mini_low.json").read_text(encoding="utf-8"))
    vanilla_by_task = {Path(row["path"]).stem: row for row in vanilla["tasks"]}
    rows = []
    for task in TASKS:
        task_data = json.loads(TASK_FILES[task].read_text(encoding="utf-8"))
        expected = task_data["test"][0].get("output")
        vanilla_row = vanilla_by_task[task]
        reflection_path = ROOT / "artifacts/arc_gpt5mini_low" / f"{task}_reflection.json"
        if reflection_path.exists():
            reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
            train_eval = reflection["best_train_evaluation"]
            predictions = reflection.get("test_predictions") or []
            reflected_prediction = predictions[0]["prediction"] if predictions else None
            reflected_test_ok = bool(expected is not None and reflected_prediction == expected)
            reflected_train = f"{train_eval['correct']}/{train_eval['total']}"
        else:
            reflected_test_ok = None
            reflected_train = "not run"
        rows.append(
            {
                "task": task,
                "vanilla_test_ok": bool(vanilla_row["correct"]),
                "reflection_train": reflected_train,
                "reflection_test_ok": reflected_test_ok,
            }
        )
    out = ROOT / "artifacts/arc_gpt5mini_low_comparison.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
