from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .arc_growing_body import (
    PrimitiveWriter,
    compact_written,
    evaluate_source_as_primitive,
    library_record,
    load_library,
)
from .arc_primitive_search import Candidate, build_candidates, copy_grid, diff_summary, normalize, shape
from .arc_reflect import load_solve, static_checks
from .llm import parse_json_object, supports_custom_temperature


ROOT = Path(__file__).resolve().parents[1]
Grid = list[list[int]]


@dataclass
class SearchState:
    names: list[str]
    candidates: list[Candidate]
    outputs: list[Grid | None]
    correct: int
    total_diff: int


def main() -> int:
    parser = argparse.ArgumentParser(description="ARC growing body with small primitives and bounded composition search.")
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--beam", type=int, default=40)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--library-in", default=None)
    parser.add_argument("--library-out", default="artifacts/arc_compositional_body/library.json")
    parser.add_argument("--report-out", default="artifacts/arc_compositional_body/report.json")
    parser.add_argument("--skill-dir", default="generated_skills/arc_compositional_body")
    parser.add_argument("--persist-partials", action="store_true", help="Persist a new operator if composition search improves train score even without full train pass.")
    args = parser.parse_args()

    client = SmallPrimitiveWriter(model=args.model, reasoning_effort=args.reasoning_effort)
    learned = load_library(args.library_in)
    runs = []
    skill_dir = ROOT / args.skill_dir
    skill_dir.mkdir(parents=True, exist_ok=True)

    task_args = selected_task_args(args)
    if not task_args:
        raise ValueError("No tasks selected.")

    for task_arg in task_args:
        task_path = (ROOT / task_arg).resolve() if not Path(task_arg).is_absolute() else Path(task_arg)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task_id = task_path.stem
        print(f"task {task_id}")

        search = composition_search(task, learned, max_depth=args.depth, beam=args.beam)
        initial_best_correct = search["best_correct"]
        if search["train_passed"]:
            run = finish_from_composition(task_id, task_path, task, search, "composition_search")
            runs.append(run)
            print(json.dumps({"task": task_id, "mode": "search", "sequence": search["best_sequence"], "train": search["train_score"], "test_ok": run["test_ok"], "library_size": len(learned)}, indent=2))
            continue

        history = []
        best_written = None
        best_search = search
        previous_eval = None
        for iteration in range(args.iters):
            payload = build_payload(task_path, task, learned, best_search, history, previous_eval)
            proposal = client.propose(payload)
            source = str(proposal["source"])
            checks = static_checks(source)
            if checks:
                evaluation = {"passed": False, "correct": 0, "total": len(task["train"]), "errors": checks, "cases": []}
                composed = best_search
            else:
                evaluation = evaluate_source_as_primitive(source, task["train"])
                temp_learned = learned + [library_record(task_id, {"train_score": "unverified", "test_ok": False}, source, str(proposal.get("name", f"{task_id}_operator")), str(proposal.get("design", "")))]
                composed = composition_search(task, temp_learned, max_depth=args.depth, beam=args.beam)
            row = {
                "iteration": iteration + 1,
                "name": str(proposal.get("name", f"{task_id}_operator")),
                "reason": str(proposal.get("reason", "")),
                "design": str(proposal.get("design", "")),
                "source": source,
                "operator_alone_evaluation": evaluation,
                "composition_search": public_composition(composed),
            }
            history.append(compact_composition_written(row))
            if best_written is None or composed["best_correct"] > best_search["best_correct"]:
                best_written = row
                best_search = composed
            previous_eval = {"operator_alone": evaluation, "composition": public_composition(composed)}
            print(json.dumps({"iteration": iteration + 1, "operator_alone": f"{evaluation.get('correct', 0)}/{evaluation.get('total', len(task['train']))}", "best_composition": composed["train_score"], "sequence": composed["best_sequence"]}, indent=2))
            if composed["train_passed"]:
                break

        run = {
            "task": task_id,
            "path": str(task_path),
            "mode": "write_small_operator",
            "best_name": best_written["name"] if best_written else None,
            "train_score": best_search["train_score"],
            "train_passed": best_search["train_passed"],
            "test_ok": False,
            "test_prediction": None,
            "composition": public_composition(best_search),
            "write_history": history,
        }
        if best_written:
            skill_path = skill_dir / f"{task_id}.py"
            skill_path.write_text(best_written["source"].strip() + "\n", encoding="utf-8")
            run["skill_path"] = str(skill_path)
        if best_search["train_passed"]:
            completed = finish_from_composition(task_id, task_path, task, best_search, "write_small_operator")
            run.update({k: completed[k] for k in ("test_ok", "test_prediction")})
            if best_written and new_operator_used(best_search, task_id):
                learned.append(library_record(task_id, run, best_written["source"], best_written["name"], best_written["design"]))
        elif (
            args.persist_partials
            and best_written
            and best_search["best_correct"] > initial_best_correct
            and new_operator_used(best_search, task_id)
        ):
            run["persisted_partial"] = True
            learned.append(library_record(task_id, run, best_written["source"], best_written["name"], best_written["design"]))
        runs.append(run)
        print(json.dumps({"task": task_id, "mode": "write", "train": run["train_score"], "test_ok": run["test_ok"], "library_size": len(learned)}, indent=2))

    report = {
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "depth": args.depth,
        "beam": args.beam,
        "total": len(runs),
        "test_correct": sum(1 for run in runs if run["test_ok"]),
        "library_size": len(learned),
        "runs": runs,
    }
    library_out = ROOT / args.library_out
    report_out = ROOT / args.report_out
    library_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    library_out.write_text(json.dumps({"skills": learned}, indent=2), encoding="utf-8")
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_correct": report["test_correct"], "total": report["total"], "library_size": len(learned), "report": str(report_out)}, indent=2))
    return 0


def selected_task_args(args: argparse.Namespace) -> list[str]:
    if args.tasks:
        return args.tasks
    if not args.tasks_dir:
        return []
    tasks_dir = (ROOT / args.tasks_dir).resolve() if not Path(args.tasks_dir).is_absolute() else Path(args.tasks_dir)
    excluded = {Path(item).stem for item in args.exclude}
    paths = [path for path in sorted(tasks_dir.glob("*.json")) if path.stem not in excluded]
    end = args.offset + args.limit if args.limit is not None else None
    return [str(path) for path in paths[args.offset : end]]


class SmallPrimitiveWriter(PrimitiveWriter):
    def propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write one small ARC body operator. Return JSON only with keys name, reason, design, source. "
                        "source must define def solve(grid): and return list[list[int]]. "
                        "Use pure Python list operations. Do not import anything. "
                        "No file/network/subprocess/eval/exec/randomness. "
                        "The operator should be reusable and composable, not a whole-task hard-coded solver. "
                        "Prefer the smallest train-verifiable reusable operator: micro-skill if it can help search, otherwise a concise complete operator built from general substeps."
                    ),
                },
                {"role": "user", "content": build_prompt(payload)},
            ],
        }
        if supports_custom_temperature(self.model):
            body["temperature"] = 0.2
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 + attempt * 3)
        assert data is not None
        return parse_json_object(data["choices"][0]["message"]["content"])


def composition_search(task: dict[str, Any], learned: list[dict[str, Any]], max_depth: int, beam: int) -> dict[str, Any]:
    candidates = build_composition_candidates(task, learned)
    targets = [normalize(pair["output"]) for pair in task["train"]]
    states = [
        SearchState(
            names=[],
            candidates=[],
            outputs=[normalize(copy_grid(pair["input"])) for pair in task["train"]],
            correct=0,
            total_diff=10**9,
        )
    ]
    best: SearchState | None = None
    top_by_depth = []
    for depth in range(1, max_depth + 1):
        expanded: list[SearchState] = []
        for state in states:
            for candidate in candidates:
                outputs = []
                failed = False
                for grid in state.outputs:
                    if grid is None:
                        outputs.append(None)
                        failed = True
                        continue
                    try:
                        outputs.append(normalize(candidate.fn(copy_grid(grid))))
                    except Exception:
                        outputs.append(None)
                        failed = True
                correct, total_diff = score_outputs(outputs, targets)
                expanded.append(SearchState(state.names + [candidate.name], state.candidates + [candidate], outputs, correct, total_diff + (100000 if failed else 0)))
        expanded.sort(key=lambda item: (item.correct, -item.total_diff, -len(item.names)), reverse=True)
        deduped = dedupe_states(expanded)
        top_by_depth.append([state_summary(item, len(targets)) for item in deduped[:10]])
        if best is None or (deduped[0].correct, -deduped[0].total_diff) > (best.correct, -best.total_diff):
            best = deduped[0]
        if best.correct == len(targets):
            break
        states = deduped[:beam]
    assert best is not None
    return {
        "best_sequence": best.names,
        "train_score": f"{best.correct}/{len(targets)}",
        "train_passed": best.correct == len(targets),
        "best_correct": best.correct,
        "best_total_diff": best.total_diff,
        "top_by_depth": top_by_depth,
        "_candidates": best.candidates,
    }


def build_composition_candidates(task: dict[str, Any], learned: list[dict[str, Any]]) -> list[Candidate]:
    candidates = build_candidates(task["train"])
    for item in learned:
        try:
            solve = load_solve(item["source"])
        except Exception:
            continue
        candidates.append(Candidate(f"learned::{item['task']}::{item['name']}", solve))
    return candidates


def score_outputs(outputs: list[Grid | None], targets: list[Grid]) -> tuple[int, int]:
    correct = 0
    total_diff = 0
    for pred, expected in zip(outputs, targets):
        if pred == expected:
            correct += 1
            continue
        if pred is None or shape(pred) != shape(expected):
            total_diff += 1000
            continue
        total_diff += sum(1 for prow, erow in zip(pred, expected) for p, e in zip(prow, erow) if p != e)
    return correct, total_diff


def dedupe_states(states: list[SearchState]) -> list[SearchState]:
    seen = set()
    out = []
    for state in states:
        sig = json.dumps(state.outputs, separators=(",", ":"))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(state)
    return out


def state_summary(state: SearchState, total: int) -> dict[str, Any]:
    return {"sequence": state.names, "train_score": f"{state.correct}/{total}", "total_diff": state.total_diff}


def public_composition(search: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in search.items() if k != "_candidates"}


def finish_from_composition(task_id: str, task_path: Path, task: dict[str, Any], search: dict[str, Any], mode: str) -> dict[str, Any]:
    pred = normalize(apply_sequence(search["_candidates"], copy_grid(task["test"][0]["input"])))
    expected = task["test"][0].get("output")
    return {
        "task": task_id,
        "path": str(task_path),
        "mode": mode,
        "best_name": " | ".join(search["best_sequence"]),
        "best_sequence": search["best_sequence"],
        "train_score": search["train_score"],
        "train_passed": True,
        "test_ok": bool(expected is not None and pred == expected),
        "test_prediction": pred,
        "composition": public_composition(search),
    }


def apply_sequence(candidates: list[Candidate], grid: Grid) -> Grid:
    out = copy_grid(grid)
    for candidate in candidates:
        out = normalize(candidate.fn(out))
    return out


def build_payload(
    task_path: Path,
    task: dict[str, Any],
    learned: list[dict[str, Any]],
    search: dict[str, Any],
    history: list[dict[str, Any]],
    previous_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_id": task_path.stem,
        "base_impulse": "grow a body of small reusable ARC operators and compose them with search",
        "train_pairs": task["train"],
        "test_inputs": [{"input": item["input"]} for item in task["test"]],
        "current_composition_search": public_composition(search),
        "learned_operators": [
            {
                "task": item["task"],
                "name": item["name"],
                "train_score": item["train_score"],
                "test_ok": item["test_ok"],
                "design": item["design"],
                "source_excerpt": item["source"][:2500],
            }
            for item in learned[-10:]
        ],
        "previous_written_attempts": history[-4:],
        "previous_eval": previous_eval,
        "instructions": [
            "Write one small reusable operator as def solve(grid):.",
            "The harness will search compositions of base operators, learned operators, and your new operator.",
            "Do not solve the whole task in one giant special-case program.",
            "Prefer the smallest train-verifiable reusable operator: one general transformation, one selection rule, or one compact complete operator built from reusable substeps.",
            "Good micro-skills include: crop_bbox, extract_connected_components, select_component_by_color, select_largest_component, reflect, tile, scale, replace_color, fill_enclosed_region, draw_line, extend_pattern, or make_mask.",
            "Bad skills bake in the whole task: avoid long special-case programs, fixed dimensions, or brittle task-specific object counts.",
            "If a tiny micro-skill will not verify or improve composition, write a concise complete operator, but structure it around reusable helpers or generic rules.",
            "The name should be short and abstract, like crop_bbox_by_color or tile_reflections_2x2, not a sentence describing this exact puzzle.",
            "Your operator may be useful even if it does not solve the task alone, because composition search will test it in sequences.",
            "Use pure Python list operations; do not import anything.",
        ],
    }


def build_prompt(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "ARC compositional growing-body episode.",
            "The current body was searched as sequences and failed.",
            "Add exactly one small reusable operator. The verifier will compose it with existing body operators.",
            "Aim for an operation that could plausibly transfer to another ARC task, even if it only partially helps this one.",
            "Return JSON only: {\"name\": ..., \"reason\": ..., \"design\": ..., \"source\": ...}",
            "",
            json.dumps(payload, indent=2),
        ]
    )


def compact_composition_written(row: dict[str, Any]) -> dict[str, Any]:
    alone = row["operator_alone_evaluation"]
    return {
        "name": row["name"],
        "design": row["design"],
        "operator_alone_train_score": f"{alone.get('correct', 0)}/{alone.get('total', 0)}",
        "operator_alone_errors": alone.get("errors", []),
        "composition": row["composition_search"],
        "source_excerpt": row["source"][:2200],
    }


def new_operator_used(search: dict[str, Any], task_id: str) -> bool:
    return any(name.startswith(f"learned::{task_id}::") for name in search["best_sequence"])


if __name__ == "__main__":
    raise SystemExit(main())
