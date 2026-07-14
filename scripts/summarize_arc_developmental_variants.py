from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "basic_library": ROOT / "artifacts/arc_developmental/gpt5mini_low_report.json",
    "mutation_prompt": ROOT / "artifacts/arc_developmental/gpt5mini_low_mutation_report.json",
    "partial_match": ROOT / "artifacts/arc_developmental/gpt5mini_low_partial_report.json",
}


def main() -> int:
    rows = []
    for name, path in VARIANTS.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": name,
                "test_correct": f"{report['test_correct']}/{report['total']}",
                "library_size": report["library_size"],
                "task_train_scores": {run["task"]: run["train_score"] for run in report["runs"]},
            }
        )
    out = ROOT / "artifacts/arc_developmental/variant_comparison.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
