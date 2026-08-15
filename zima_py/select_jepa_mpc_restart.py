from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .evolve_jepa_mpc_neat import lava_layout_geometry
from .minigrid_adapter import MiniGridSpec, make_minigrid


ROOT = Path(__file__).resolve().parents[1]


def validation_metrics(report: dict, env) -> dict:
    evaluation = report["candidate_selection_evaluation"]
    solved_runs = [run for run in evaluation["runs"] if run["solved"]]
    geometries = sorted(
        {
            (orientation, gap)
            for run in solved_runs
            for orientation, _, gap in [lava_layout_geometry(env, int(run["seed"]))]
        }
    )
    return {
        "validation_solved": len(solved_runs),
        "validation_geometry_count": len(geometries),
        "validation_geometries": [f"{orientation}:gap={gap}" for orientation, gap in geometries],
        "validation_mean_fitness": (
            sum(float(run["fitness"]) for run in evaluation["runs"])
            / max(1, len(evaluation["runs"]))
        ),
    }


def selection_key(row: dict) -> tuple[int, int, float]:
    """Geometry breadth, then task successes, then ordinary mean fitness."""
    return (
        int(row["validation_geometry_count"]),
        int(row["validation_solved"]),
        float(row["validation_mean_fitness"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a JEPA-MPC neuroevolution restart using validation results only."
    )
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--winner", default="")
    args = parser.parse_args()

    reports = []
    for value in args.reports:
        path = (ROOT / value).resolve()
        reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
    first = reports[0][1]
    env = make_minigrid(MiniGridSpec(first["environment"], 0, 192))
    try:
        rows = []
        for path, report in reports:
            metrics = validation_metrics(report, env)
            holdout = report.get("holdout_evaluation", {})
            rows.append(
                {
                    "report": str(path),
                    "winner": report["winner"],
                    "evolution_seed": int(report["seed"]),
                    **metrics,
                    # Recorded for audit only; never used by selection_key.
                    "test_solved": holdout.get("solved"),
                    "test_episodes": holdout.get("episodes"),
                }
            )
    finally:
        env.close()
    selected = max(rows, key=selection_key)
    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "selection_rule": "validation geometry count, validation solves, validation mean fitness",
        "test_metrics_used_for_selection": False,
        "candidates": rows,
        "selected": selected,
    }
    if args.winner:
        winner_path = (ROOT / args.winner).resolve()
        winner_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected["winner"], winner_path)
        result["selected_winner"] = str(winner_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["selected"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
