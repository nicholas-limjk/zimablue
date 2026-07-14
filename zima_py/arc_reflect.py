from __future__ import annotations

import argparse
import ast
import json
import os
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .llm import load_env_file, parse_json_object, supports_custom_temperature


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ArcProposal:
    name: str
    reason: str
    design: str
    source: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Reflectively synthesize ARC solve(grid) skills.")
    parser.add_argument("task", help="Path to an ARC task JSON.")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--skill-out", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--library", default=None, help="Optional JSON skill library to expose to the writer.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    task_path = (ROOT / args.task).resolve() if not Path(args.task).is_absolute() else Path(args.task)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    skill_out = ROOT / (args.skill_out or f"generated_skills/arc/{task_path.stem}.py")
    report_out = ROOT / (args.report_out or f"artifacts/arc/{task_path.stem}_reflection.json")

    report: dict[str, Any] = {
        "task": str(task_path),
        "train_count": len(task["train"]),
        "test_count": len(task["test"]),
        "iterations": [],
    }
    if args.no_llm:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    writer = ArcSkillWriter(model=args.model, reasoning_effort=args.reasoning_effort)
    previous_source = ""
    previous_eval: dict[str, Any] | None = None
    best: tuple[int, ArcProposal, dict[str, Any]] | None = None

    for iteration in range(args.iters):
        payload = build_prompt_payload(task_path, task, iteration, previous_source, previous_eval, load_skill_library(args.library))
        proposal = writer.propose(payload)
        checks = static_checks(proposal.source)
        if checks:
            eval_result = {"passed": False, "correct": 0, "total": len(task["train"]), "errors": checks, "cases": []}
        else:
            eval_result = evaluate_source(proposal.source, task["train"])
        report["iterations"].append(
            {
                "iteration": iteration + 1,
                "proposal": {"name": proposal.name, "reason": proposal.reason, "design": proposal.design},
                "static_errors": checks,
                "evaluation": eval_result,
            }
        )
        if best is None or eval_result.get("correct", 0) > best[0]:
            best = (int(eval_result.get("correct", 0)), proposal, eval_result)
        previous_source = proposal.source
        previous_eval = eval_result
        print(json.dumps({"iteration": iteration + 1, "correct": eval_result.get("correct", 0), "total": len(task["train"])}, indent=2))
        if eval_result.get("passed"):
            break

    if best is None:
        raise RuntimeError("No proposal was produced.")
    _, best_proposal, best_eval = best
    skill_out.parent.mkdir(parents=True, exist_ok=True)
    skill_out.write_text(best_proposal.source.strip() + "\n", encoding="utf-8")
    test_predictions = predict_tests(best_proposal.source, task["test"]) if best_eval.get("passed") else []
    report.update(
        {
            "best_skill": str(skill_out),
            "best_train_evaluation": best_eval,
            "test_predictions": test_predictions,
        }
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_skill": str(skill_out), "report": str(report_out), "train": best_eval}, indent=2))
    return 0


class ArcSkillWriter:
    def __init__(self, model: str | None = None, reasoning_effort: str | None = None, timeout_s: int = 300):
        load_env_file()
        self.api_key = os.getenv("AGENT_SKILL_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")
        self.endpoint = os.getenv("AGENT_SKILL_ENDPOINT") or "https://api.openai.com/v1/chat/completions"
        self.model = model or os.getenv("AGENT_SKILL_MODEL") or "gpt-4.1-mini"
        self.reasoning_effort = reasoning_effort
        self.timeout_s = int(os.getenv("AGENT_SKILL_TIMEOUT_S", str(timeout_s)))
        if not self.api_key:
            raise RuntimeError("Missing AGENT_SKILL_API_KEY / OPENAI_API_KEY / openai_key.")

    def propose(self, payload: dict[str, Any]) -> ArcProposal:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write one Python ARC grid-transformation skill. "
                        "Return JSON only with keys name, reason, design, source. "
                        "source must define def solve(grid): and return a list[list[int]]. "
                        "You may use helper functions in the same source file and may use np if needed. "
                        "Do not import modules, read files, use network, subprocess, eval, exec, or randomness. "
                        "Prefer compact deterministic programs that infer the rule from the provided training pairs."
                    ),
                },
                {"role": "user", "content": build_prompt(payload)},
            ],
        }
        if supports_custom_temperature(self.model):
            request_body["temperature"] = 0.2
        if self.reasoning_effort:
            request_body["reasoning_effort"] = self.reasoning_effort
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        parsed = parse_json_object(data["choices"][0]["message"]["content"])
        return ArcProposal(
            name=str(parsed.get("name", "arc_skill")),
            reason=str(parsed.get("reason", "")),
            design=str(parsed.get("design", "")),
            source=str(parsed["source"]),
        )


def build_prompt_payload(
    task_path: Path,
    task: dict[str, Any],
    iteration: int,
    previous_source: str,
    previous_eval: dict[str, Any] | None,
    skill_library: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "task_id": task_path.stem,
        "base_impulse": "make the output grid match the training examples and solve the test input",
        "information_boundary": [
            "You may inspect all training input/output pairs.",
            "At runtime solve(grid) receives one input grid only.",
            "Do not hard-code the test output; infer a reusable transformation from train pairs.",
            "The code will be run against every train pair after you propose it.",
        ],
        "train_pairs": task["train"],
        "test_inputs": [{"input": item["input"]} for item in task["test"]],
        "skill_contract": {
            "required_function": "def solve(grid):",
            "input": "grid is a rectangular list[list[int]] with ARC colors 0..9",
            "output": "rectangular list[list[int]]",
            "allowed_runtime": "pure Python builtins and np",
        },
    }
    if skill_library:
        payload["persistent_skill_library"] = skill_library[-12:]
        payload["developmental_instruction"] = (
            "This agent accumulates reusable ARC skills over tasks. "
            "Inspect the persistent_skill_library and reuse or mutate earlier ideas when relevant. "
            "Do not blindly copy a prior skill; adapt it to the current train pairs."
        )
        payload["mutation_protocol"] = {
            "required_in_design": [
                "name the closest prior skill by task id, or say none",
                "state which part of that prior skill transfers",
                "state which assumption breaks on the current train pairs",
                "describe the mutation that repairs the broken assumption",
            ],
            "instruction": (
                "If any prior skill is even partially relevant, mutate it instead of writing from scratch. "
                "A good mutation preserves useful structure and changes only the selector, geometry rule, color rule, or output construction that failed."
            ),
        }
    if iteration and previous_eval is not None:
        payload["reflection_request"] = {
            "previous_source": previous_source,
            "previous_evaluation": previous_eval,
            "instruction": (
                "The previous skill failed at least one training pair. "
                "Study the mismatches, revise the inferred rule, and return a replacement source. "
                "If previous_evaluation.errors contains a static validation error, fix that first before changing the ARC hypothesis."
            ),
        }
    return payload


def load_skill_library(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    library_path = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not library_path.exists():
        return []
    data = json.loads(library_path.read_text(encoding="utf-8"))
    skills = data.get("skills", data if isinstance(data, list) else [])
    compact = []
    for item in skills:
        compact.append(
            {
                "task": item.get("task"),
                "train_score": item.get("train_score"),
                "test_ok": item.get("test_ok"),
                "reason": item.get("reason", ""),
                "design": item.get("design", ""),
                "source": item.get("source", "")[:5000],
            }
        )
    return compact


def build_prompt(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "ARC reflection episode.",
            "Infer the simplest transformation that fits every training pair.",
            "Write executable Python, not prose.",
            "Do not import anything. np is already available in the execution namespace if you need array helpers.",
            "If reflection_request.previous_evaluation.errors mentions imports or static validation, the next source must remove imports completely.",
            "Useful primitives often include connected components, bounding boxes, color counts, symmetry, tiling, cropping, scaling, line/region fill, and object selection.",
            "If a previous attempt failed, use the diffs to change the hypothesis rather than patching one cell.",
            "If persistent_skill_library is present, your design must explicitly choose the closest prior skill and explain the mutation, or explain why none applies.",
            "Return JSON only: {\"name\": ..., \"reason\": ..., \"design\": ..., \"source\": ...}",
            "",
            json.dumps(payload, indent=2),
        ]
    )


def static_checks(source: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    has_solve = any(isinstance(node, ast.FunctionDef) and node.name == "solve" for node in tree.body)
    if not has_solve:
        errors.append("source must define def solve(grid):")
    banned_nodes = (ast.Import, ast.ImportFrom)
    banned_calls = {"open", "eval", "exec", "__import__", "compile", "input", "globals", "locals"}
    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            errors.append("imports are not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned_calls:
            errors.append(f"banned call: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "remove", "unlink", "rmdir"}:
            errors.append(f"banned attribute: {node.attr}")
    return sorted(set(errors))


def evaluate_source(source: str, train_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        solve = load_solve(source)
    except Exception as exc:
        return {"passed": False, "correct": 0, "total": len(train_pairs), "errors": [f"load error: {type(exc).__name__}: {exc}"], "cases": []}
    cases = []
    correct = 0
    for index, pair in enumerate(train_pairs):
        try:
            predicted = normalize_grid(solve(copy_grid(pair["input"])))
            expected = normalize_grid(pair["output"])
            ok = predicted == expected
            if ok:
                correct += 1
            cases.append(
                {
                    "index": index,
                    "ok": ok,
                    "input_shape": shape(pair["input"]),
                    "expected_shape": shape(expected),
                    "predicted_shape": shape(predicted),
                    "diff": diff_summary(predicted, expected),
                }
            )
        except Exception as exc:
            cases.append({"index": index, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"passed": correct == len(train_pairs), "correct": correct, "total": len(train_pairs), "errors": [], "cases": cases}


def predict_tests(source: str, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    solve = load_solve(source)
    return [{"index": i, "prediction": normalize_grid(solve(copy_grid(item["input"])))} for i, item in enumerate(tests)]


def load_solve(source: str):
    namespace: dict[str, Any] = {"np": np, "__builtins__": safe_builtins()}
    exec(compile(source, "<arc_skill>", "exec"), namespace)
    solve = namespace.get("solve")
    if not callable(solve):
        raise ValueError("source did not define callable solve(grid)")
    return solve


def safe_builtins() -> dict[str, Any]:
    names = [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    ]
    import builtins

    return {name: getattr(builtins, name) for name in names}


def normalize_grid(value: Any) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise ValueError("output must be a non-empty list[list[int]]")
    width = len(value[0])
    if width == 0 or any(len(row) != width for row in value):
        raise ValueError("output grid must be rectangular")
    return [[int(cell) for cell in row] for row in value]


def copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


def shape(grid: list[list[int]]) -> list[int]:
    return [len(grid), len(grid[0]) if grid else 0]


def diff_summary(predicted: list[list[int]], expected: list[list[int]], limit: int = 20) -> dict[str, Any]:
    if shape(predicted) != shape(expected):
        return {"shape_mismatch": True, "predicted_shape": shape(predicted), "expected_shape": shape(expected)}
    diffs = []
    for r, (prow, erow) in enumerate(zip(predicted, expected)):
        for c, (p, e) in enumerate(zip(prow, erow)):
            if p != e:
                diffs.append({"row": r, "col": c, "predicted": p, "expected": e})
                if len(diffs) >= limit:
                    return {"shape_mismatch": False, "diff_count_at_least": len(diffs), "sample": diffs}
    return {"shape_mismatch": False, "diff_count": len(diffs), "sample": diffs}


if __name__ == "__main__":
    raise SystemExit(main())
