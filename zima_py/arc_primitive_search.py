from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
Grid = list[list[int]]
Transform = Callable[[Grid], Grid]


@dataclass
class Candidate:
    name: str
    fn: Transform


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a small library of ARC primitive transformations.")
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--out", default="artifacts/arc_primitive_search.json")
    args = parser.parse_args()

    rows = []
    for task_arg in args.tasks:
        task_path = (ROOT / task_arg).resolve() if not Path(task_arg).is_absolute() else Path(task_arg)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        result = search_task(task_path.stem, task)
        rows.append(result)
        print(json.dumps({"task": result["task"], "best_train": result["best_train"], "test_ok": result["test_ok"], "best": result["best_name"]}, indent=2))
    summary = {
        "correct": sum(1 for row in rows if row["test_ok"]),
        "total": len(rows),
        "rows": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"correct": summary["correct"], "total": summary["total"], "out": str(out)}, indent=2))
    return 0


def search_task(task_id: str, task: dict) -> dict:
    candidates = build_candidates(task["train"])
    scored = []
    for candidate in candidates:
        cases = []
        correct = 0
        for idx, pair in enumerate(task["train"]):
            try:
                pred = normalize(candidate.fn(copy_grid(pair["input"])))
                expected = normalize(pair["output"])
                ok = pred == expected
            except Exception as exc:
                pred = None
                expected = pair["output"]
                ok = False
                cases.append({"index": idx, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                continue
            correct += int(ok)
            cases.append({"index": idx, "ok": ok, "predicted_shape": shape(pred), "expected_shape": shape(expected), "diff": diff_summary(pred, expected)})
        scored.append({"candidate": candidate, "correct": correct, "total": len(task["train"]), "cases": cases})
    scored.sort(key=lambda row: (row["correct"], -len(row["candidate"].name)), reverse=True)
    best = scored[0]
    test_pred = None
    test_ok = False
    if best["correct"] == best["total"]:
        test_pred = normalize(best["candidate"].fn(copy_grid(task["test"][0]["input"])))
        expected = task["test"][0].get("output")
        test_ok = bool(expected is not None and test_pred == expected)
    return {
        "task": task_id,
        "best_name": best["candidate"].name,
        "best_train": f"{best['correct']}/{best['total']}",
        "test_ok": test_ok,
        "test_prediction": test_pred,
        "top_candidates": [
            {
                "name": row["candidate"].name,
                "train": f"{row['correct']}/{row['total']}",
                "cases": row["cases"][:3],
            }
            for row in scored[:12]
        ],
    }


def build_candidates(train_pairs: list[dict]) -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate("identity", lambda g: copy_grid(g)),
        Candidate("rotate90", rotate90),
        Candidate("rotate180", rotate180),
        Candidate("rotate270", rotate270),
        Candidate("flip_h", flip_h),
        Candidate("flip_v", flip_v),
        Candidate("crop_nonzero_bbox", crop_nonzero_bbox),
        Candidate("largest_component_crop", largest_component_crop),
        Candidate("remove_black_rows_cols", remove_black_rows_cols),
    ]
    colors = sorted({v for pair in train_pairs for grid in (pair["input"], pair["output"]) for row in grid for v in row})
    nonzero = [c for c in colors if c != 0]
    for src in colors:
        for dst in colors:
            if src != dst:
                candidates.append(Candidate(f"replace_{src}_with_{dst}", lambda g, src=src, dst=dst: replace_color(g, src, dst)))
    for barrier in nonzero:
        for fill in nonzero:
            if fill != barrier:
                candidates.append(Candidate(f"fill_regions_barrier_{barrier}_with_{fill}", lambda g, barrier=barrier, fill=fill: fill_enclosed(g, barrier, fill)))
    for color in nonzero:
        candidates.extend(
            [
                Candidate(f"complete_rows_color_{color}", lambda g, color=color: complete_rows(g, color)),
                Candidate(f"complete_cols_color_{color}", lambda g, color=color: complete_cols(g, color)),
                Candidate(f"complete_bbox_color_{color}", lambda g, color=color: complete_bbox(g, color)),
                Candidate(f"connect_same_color_lines_{color}", lambda g, color=color: connect_same_color_lines(g, color)),
            ]
        )
    candidates.extend(mask_pattern_candidates(train_pairs))
    return candidates


def mask_pattern_candidates(train_pairs: list[dict]) -> list[Candidate]:
    candidates: list[Candidate] = []
    # If every pair is same shape, learn simple per-color input/output mask deltas.
    same_shape = all(shape(pair["input"]) == shape(pair["output"]) for pair in train_pairs)
    if not same_shape:
        return candidates
    colors = sorted({v for pair in train_pairs for grid in (pair["input"], pair["output"]) for row in grid for v in row})
    for color in colors:
        for bg in colors:
            if color == bg:
                continue
            candidates.append(Candidate(f"preserve_{color}_erase_other_to_{bg}", lambda g, color=color, bg=bg: preserve_color(g, color, bg)))
    return candidates


def rotate90(g: Grid) -> Grid:
    return [list(row) for row in zip(*g[::-1])]


def rotate180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]


def rotate270(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)][::-1]


def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def flip_v(g: Grid) -> Grid:
    return g[::-1]


def crop_nonzero_bbox(g: Grid) -> Grid:
    coords = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != 0]
    if not coords:
        return copy_grid(g)
    r0, r1 = min(r for r, _ in coords), max(r for r, _ in coords)
    c0, c1 = min(c for _, c in coords), max(c for _, c in coords)
    return [row[c0 : c1 + 1] for row in g[r0 : r1 + 1]]


def largest_component_crop(g: Grid) -> Grid:
    comps = components(g, include_zero=False)
    if not comps:
        return copy_grid(g)
    comp = max(comps, key=lambda item: len(item[1]))
    color, coords = comp
    r0, r1 = min(r for r, _ in coords), max(r for r, _ in coords)
    c0, c1 = min(c for _, c in coords), max(c for _, c in coords)
    out = [[0 for _ in range(c1 - c0 + 1)] for _ in range(r1 - r0 + 1)]
    for r, c in coords:
        out[r - r0][c - c0] = color
    return out


def remove_black_rows_cols(g: Grid) -> Grid:
    rows = [i for i, row in enumerate(g) if any(v != 0 for v in row)]
    cols = [j for j in range(len(g[0])) if any(g[i][j] != 0 for i in range(len(g)))]
    if not rows or not cols:
        return copy_grid(g)
    return [[g[i][j] for j in cols] for i in rows]


def replace_color(g: Grid, src: int, dst: int) -> Grid:
    return [[dst if v == src else v for v in row] for row in g]


def preserve_color(g: Grid, color: int, bg: int) -> Grid:
    return [[v if v == color else bg for v in row] for row in g]


def fill_enclosed(g: Grid, barrier: int, fill: int) -> Grid:
    h, w = len(g), len(g[0])
    out = copy_grid(g)
    seen = [[False] * w for _ in range(h)]
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if g[r][c] != barrier and not seen[r][c]:
                seen[r][c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r][c] != barrier and not seen[r][c]:
                seen[r][c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] != barrier:
                seen[nr][nc] = True
                stack.append((nr, nc))
    for r in range(h):
        for c in range(w):
            if g[r][c] != barrier and not seen[r][c]:
                out[r][c] = fill
    return out


def complete_rows(g: Grid, color: int) -> Grid:
    out = copy_grid(g)
    for r, row in enumerate(g):
        cols = [c for c, v in enumerate(row) if v == color]
        if len(cols) >= 2:
            for c in range(min(cols), max(cols) + 1):
                out[r][c] = color
    return out


def complete_cols(g: Grid, color: int) -> Grid:
    out = copy_grid(g)
    for c in range(len(g[0])):
        rows = [r for r in range(len(g)) if g[r][c] == color]
        if len(rows) >= 2:
            for r in range(min(rows), max(rows) + 1):
                out[r][c] = color
    return out


def complete_bbox(g: Grid, color: int) -> Grid:
    coords = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color]
    if not coords:
        return copy_grid(g)
    out = copy_grid(g)
    r0, r1 = min(r for r, _ in coords), max(r for r, _ in coords)
    c0, c1 = min(c for _, c in coords), max(c for _, c in coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r][c] = color
    return out


def connect_same_color_lines(g: Grid, color: int) -> Grid:
    out = complete_cols(complete_rows(g, color), color)
    pts = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color]
    for i, (r1, c1) in enumerate(pts):
        for r2, c2 in pts[i + 1 :]:
            dr, dc = r2 - r1, c2 - c1
            if abs(dr) == abs(dc) and dr != 0:
                sr = 1 if dr > 0 else -1
                sc = 1 if dc > 0 else -1
                for k in range(abs(dr) + 1):
                    out[r1 + k * sr][c1 + k * sc] = color
    return out


def components(g: Grid, include_zero: bool) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or (g[r][c] == 0 and not include_zero):
                continue
            color = g[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append((color, coords))
    return comps


def normalize(g: Grid) -> Grid:
    return [[int(v) for v in row] for row in g]


def copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def shape(g: Grid | None) -> list[int]:
    if not g:
        return [0, 0]
    return [len(g), len(g[0])]


def diff_summary(pred: Grid | None, expected: Grid, limit: int = 12) -> dict:
    if pred is None:
        return {"missing": True}
    if shape(pred) != shape(expected):
        return {"shape_mismatch": True, "predicted_shape": shape(pred), "expected_shape": shape(expected)}
    diffs = []
    for r, (prow, erow) in enumerate(zip(pred, expected)):
        for c, (p, e) in enumerate(zip(prow, erow)):
            if p != e:
                diffs.append({"row": r, "col": c, "predicted": p, "expected": e})
                if len(diffs) >= limit:
                    return {"shape_mismatch": False, "diff_count_at_least": len(diffs), "sample": diffs}
    return {"shape_mismatch": False, "diff_count": len(diffs), "sample": diffs}


if __name__ == "__main__":
    raise SystemExit(main())
