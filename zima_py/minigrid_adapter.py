from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class MissingMiniGridError(RuntimeError):
    pass


@dataclass(frozen=True)
class MiniGridSpec:
    env_id: str = "MiniGrid-DoorKey-8x8-v0"
    seed: int = 7
    max_steps: Optional[int] = None
    fully_observed: bool = False
    render_mode: Optional[str] = None


def require_minigrid() -> tuple[Any, Any, Any, dict[str, int], dict[str, int]]:
    try:
        import gymnasium as gym
        import numpy as np
        from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
    except Exception as exc:  # pragma: no cover - exercised when deps are absent.
        raise MissingMiniGridError(
            "Python MiniGrid dependencies are missing. Install them with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements-python.txt"
        ) from exc

    return gym, np, None, OBJECT_TO_IDX, COLOR_TO_IDX


def make_minigrid(spec: MiniGridSpec):
    gym, _, _, _, _ = require_minigrid()

    try:
        env = gym.make(spec.env_id, render_mode=spec.render_mode, max_steps=spec.max_steps)
    except TypeError:
        env = gym.make(spec.env_id, render_mode=spec.render_mode)

    if spec.fully_observed:
        try:
            from minigrid.wrappers import FullyObsWrapper

            env = FullyObsWrapper(env)
        except Exception:
            # Older MiniGrid versions may not expose the wrapper in the same place.
            pass

    return env


def normalize_reset(reset_result):
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def normalize_step(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated, truncated, info
    obs, reward, done, info = step_result
    return obs, reward, done, False, info


def encode_observation(obs: Any) -> dict[str, Any]:
    _, np, _, _, _ = require_minigrid()

    if isinstance(obs, dict):
        image = np.asarray(obs.get("image"))
        return {
            "image": image,
            "direction": int(obs.get("direction", -1)),
            "mission": str(obs.get("mission", "")),
        }

    return {
        "image": np.asarray(obs),
        "direction": -1,
        "mission": "",
    }


def summarize_grid(image, object_to_idx: dict[str, int]) -> dict[str, Any]:
    _, np, _, _, _ = require_minigrid()
    arr = np.asarray(image)
    idx_to_object = {value: key for key, value in object_to_idx.items()}
    counts: dict[str, int] = {}
    cells = []

    if arr.ndim >= 3 and arr.shape[-1] >= 1:
        objects = arr[:, :, 0]
        for value in np.unique(objects):
            name = idx_to_object.get(int(value), f"object:{int(value)}")
            count = int((objects == value).sum())
            counts[name] = count
        for x in range(objects.shape[0]):
            for y in range(objects.shape[1]):
                name = idx_to_object.get(int(objects[x, y]), f"object:{int(objects[x, y])}")
                if name not in {"empty", "wall", "unseen"}:
                    cells.append({"x": x, "y": y, "object": name})

    return {"shape": list(arr.shape), "counts": counts, "interesting_cells": cells[:80]}
