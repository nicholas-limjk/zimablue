from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .llm import load_env_file, parse_json_object, supports_custom_temperature


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate vanilla LLM ARC predictions without code/reflection.")
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--out", default="artifacts/arc_vanilla_baseline.json")
    args = parser.parse_args()

    client = VanillaArcClient(model=args.model, reasoning_effort=args.reasoning_effort)
    rows = []
    for task_arg in args.tasks:
        task_path = (ROOT / task_arg).resolve() if not Path(task_arg).is_absolute() else Path(task_arg)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        try:
            response = client.predict(task_path.stem, task)
            predictions = response.get("predictions", [])
            error = None
        except Exception as exc:
            response = {"reason": ""}
            predictions = []
            error = f"{type(exc).__name__}: {exc}"
        test_rows = []
        correct = 0
        for index, item in enumerate(task["test"]):
            expected = item.get("output")
            prediction = normalize_grid(predictions[index]) if index < len(predictions) else None
            ok = expected is not None and prediction == expected
            correct += int(ok)
            test_rows.append({"index": index, "ok": ok, "prediction": prediction, "expected": expected})
        row = {
            "task": task_path.stem,
            "path": str(task_path),
            "correct": correct,
            "total": len(task["test"]),
            "reason": response.get("reason", ""),
            "error": error,
            "tests": test_rows,
        }
        rows.append(row)
        print(json.dumps({"task": row["task"], "correct": correct, "total": row["total"]}, indent=2))

    summary = {
        "model": client.model,
        "correct": sum(row["correct"] for row in rows),
        "total": sum(row["total"] for row in rows),
        "tasks": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"model": client.model, "correct": summary["correct"], "total": summary["total"], "out": str(out)}, indent=2))
    return 0


class VanillaArcClient:
    def __init__(self, model: str | None = None, reasoning_effort: str | None = None, timeout_s: int = 300):
        load_env_file()
        self.api_key = os.getenv("AGENT_SKILL_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")
        self.endpoint = os.getenv("AGENT_SKILL_ENDPOINT") or "https://api.openai.com/v1/chat/completions"
        self.model = model or os.getenv("AGENT_SKILL_MODEL") or "gpt-4.1-mini"
        self.reasoning_effort = reasoning_effort
        self.timeout_s = int(os.getenv("AGENT_SKILL_TIMEOUT_S", str(timeout_s)))
        if not self.api_key:
            raise RuntimeError("Missing AGENT_SKILL_API_KEY / OPENAI_API_KEY / openai_key.")

    def predict(self, task_id: str, task: dict[str, Any]) -> dict[str, Any]:
        prompt = {
            "task_id": task_id,
            "instruction": (
                "Solve this ARC task directly. Infer the transformation from the train input/output pairs, "
                "then provide the output grid for each test input. Do not write code."
            ),
            "train": task["train"],
            "test_inputs": [{"input": item["input"]} for item in task["test"]],
            "response_schema": {"reason": "short explanation", "predictions": ["list of output grids, one per test input"]},
        }
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You solve ARC grid puzzles. Return JSON only with keys reason and predictions.",
                },
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
        }
        if supports_custom_temperature(self.model):
            request_body["temperature"] = 0.0
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
        return parse_json_object(data["choices"][0]["message"]["content"])


def normalize_grid(value: Any) -> list[list[int]] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        return None
    width = len(value[0])
    if width == 0 or any(len(row) != width for row in value):
        return None
    try:
        return [[int(cell) for cell in row] for row in value]
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
