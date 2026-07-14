from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARD_IDS = [
    "00d62c1b",
    "045e512c",
    "025d127b",
    "1f85a75f",
    "91714a58",
    "9ecd008a",
    "a8c38be5",
    "b8825c91",
    "c8f0f002",
    "d511f180",
    "e5062a87",
    "f8a8fe49",
]


def main() -> int:
    vanilla = json.loads((ROOT / "artifacts/arc_vanilla_hard_candidates_gpt5mini_low.json").read_text(encoding="utf-8"))
    vanilla_by_id = {Path(row["path"]).stem: row for row in vanilla["tasks"]}
    rows = []
    for task_id in HARD_IDS:
        task = json.loads((ROOT / "arc_agi_source/data/training" / f"{task_id}.json").read_text(encoding="utf-8"))
        expected = task["test"][0].get("output")
        vanilla_ok = bool(vanilla_by_id[task_id]["correct"])
        reflection_path = ROOT / "artifacts/arc_gpt5mini_low_hard" / f"{task_id}_reflection.json"
        reflection_train = "not run"
        reflection_test_ok = None
        if reflection_path.exists():
            reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
            train_eval = reflection["best_train_evaluation"]
            reflection_train = f"{train_eval['correct']}/{train_eval['total']}"
            preds = reflection.get("test_predictions") or []
            reflection_test_ok = bool(preds and expected is not None and preds[0]["prediction"] == expected)
        combined_ok = vanilla_ok or bool(reflection_test_ok)
        rows.append(
            {
                "task": task_id,
                "vanilla_test_ok": vanilla_ok,
                "reflection_train": reflection_train,
                "reflection_test_ok": reflection_test_ok,
                "combined_ok": combined_ok,
            }
        )
    summary = {
        "vanilla_correct": sum(1 for row in rows if row["vanilla_test_ok"]),
        "combined_correct": sum(1 for row in rows if row["combined_ok"]),
        "total": len(rows),
        "repaired": sum(1 for row in rows if not row["vanilla_test_ok"] and row["reflection_test_ok"]),
        "rows": rows,
    }
    out = ROOT / "artifacts/arc_gpt5mini_low_hard_comparison.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
