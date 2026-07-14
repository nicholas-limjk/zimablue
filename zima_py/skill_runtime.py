from __future__ import annotations

import importlib.util
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

from .minigrid_adapter import encode_observation, require_minigrid, summarize_grid


ActionFn = Callable[[dict[str, Any], dict[str, Any], Any], Optional[int]]


@dataclass
class SkillRecord:
    name: str
    path: Path
    reason: str = ""


@dataclass
class SkillProgram:
    records: list[SkillRecord] = field(default_factory=list)
    _functions: list[ActionFn] = field(default_factory=list)

    def clear(self) -> None:
        self.records.clear()
        self._functions.clear()

    def add_skill_file(self, path: Path, reason: str = "") -> None:
        self.add(SkillRecord(name=path.stem, path=path, reason=reason))

    def add(self, record: SkillRecord) -> None:
        module = load_skill_module(record.path)
        act = getattr(module, "act", None)
        if not callable(act):
            raise ValueError(f"{record.path} must define act(obs, memory, api).")
        self.records.append(record)
        self._functions.append(act)

    def act(self, obs: dict[str, Any], memory: dict[str, Any], api: "MiniGridSkillApi") -> Optional[int]:
        enriched_obs = {
            **obs,
            "actions": api.actions,
            "object_to_idx": api.object_to_idx,
            "color_to_idx": api.color_to_idx,
        }
        for fn in self._functions:
            action = fn(enriched_obs, memory, api)
            if action is not None:
                return int(action)
        return None


class MiniGridSkillApi:
    def __init__(self, env: Any, memory: dict[str, Any]):
        _, np, _, object_to_idx, color_to_idx = require_minigrid()
        self.np = np
        self.memory = memory
        self.object_to_idx = dict(object_to_idx)
        self.color_to_idx = dict(color_to_idx)
        self.actions = action_map(env)

    def describe(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": obs.get("mission", ""),
            "direction": obs.get("direction"),
            "grid": summarize_grid(obs["image"], self.object_to_idx),
            "front_cell": describe_cell(self.front_cell(obs), self.object_to_idx),
            "egocentric_cells": self.egocentric_cells(obs),
            "actions": self.actions,
            "object_to_idx": self.object_to_idx,
            "color_to_idx": self.color_to_idx,
        }

    def front_cell(self, obs: dict[str, Any]) -> Any:
        image = obs["image"]
        # MiniGrid observations are indexed [x, y, channel]. The default partial
        # observation is egocentric, with the agent at x=3, y=6 and the cell
        # directly ahead at x=3, y=5.
        if image.shape[0] >= 4 and image.shape[1] >= 6:
            return image[3, 5]
        return image[image.shape[0] // 2, image.shape[1] // 2]

    def front_object(self, obs: dict[str, Any]) -> dict[str, Any]:
        return describe_cell(self.front_cell(obs), self.object_to_idx)

    def visible_objects(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        return summarize_grid(obs["image"], self.object_to_idx)["interesting_cells"]

    def egocentric_cells(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        image = obs["image"]
        center_x = image.shape[0] // 2
        agent_y = image.shape[1] - 1
        cells: list[dict[str, Any]] = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                described = describe_cell(image[x, y], self.object_to_idx)
                cells.append(
                    {
                        "view_x": int(x),
                        "view_y": int(y),
                        "lateral": int(x - center_x),
                        "forward": int(agent_y - y),
                        **described,
                    }
                )
        return cells

    def visible_cells(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        return self.egocentric_cells(obs)

    def remember(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self.memory.get(key, default)


def action_map(env: Any) -> dict[str, int]:
    actions = getattr(getattr(env, "unwrapped", env), "actions", None)
    if actions is None:
        return {
            "left": 0,
            "right": 1,
            "forward": 2,
            "pickup": 3,
            "drop": 4,
            "toggle": 5,
            "done": 6,
        }
    return {name: int(value) for name, value in actions.__members__.items()}


def load_skill_module(path: Path) -> ModuleType:
    importlib.invalidate_caches()
    stat = path.stat()
    digest = hashlib.sha1(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")).hexdigest()[:12]
    module_name = f"{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import skill module {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_skill(path: Path, source: str) -> SkillRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.strip() + "\n", encoding="utf-8")
    return SkillRecord(name=path.stem, path=path)


def trace_payload(
    obs: Any,
    env: Any,
    memory: dict[str, Any],
    recent: list[dict[str, Any]],
    repair_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    encoded = encode_observation(obs)
    api = MiniGridSkillApi(env, memory)
    payload = {
        "impulse": "reach_goal",
        "information_boundary": [
            "The skill writer receives only the body observation and learned memory below.",
            "The environment is real Farama MiniGrid, but no full hidden map, env object, grid object, or oracle state is provided.",
            "At runtime the generated skill receives only obs, memory, and api; obs['image'] is the current partial MiniGrid observation.",
        ],
        "current_body_observation": {
            **api.describe(encoded),
            "raw_image": encoded["image"].astype(int).tolist(),
        },
        "body_learned_memory": learned_memory_for_prompt(memory),
        "world_model_evidence": memory.get("world_model_evidence"),
        "recent_trace": recent[-12:],
        "skill_contract": {
            "file": "generated Python module",
            "required_function": "def act(obs, memory, api):",
            "return_value": "one MiniGrid action int, e.g. api.actions['forward']",
            "allowed_libraries": "numpy is available as api.np; persistent state is available through the memory dict",
            "body_only_helpers": [
                "api.front_object(obs)",
                "api.egocentric_cells(obs)",
                "api.visible_cells(obs)",
                "api.visible_objects(obs)",
                "api.actions",
            ],
            "important": "Write an executable mapping/planning policy, not a one-shot answer. The policy must continue exploring, using memory, until it solves the task.",
        },
    }
    if repair_context:
        payload["repair_request"] = repair_context
    return payload


def learned_memory_for_prompt(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "mission": memory.get("mission"),
        "step_count": memory.get("step_count", 0),
        "blocked_or_low_reward_steps": memory.get("blocked_or_low_reward_steps", 0),
        "blocked_forward_steps": memory.get("blocked_forward_steps", 0),
        "last_action_blocked": memory.get("last_action_blocked", False),
        "seen_object_counts": memory.get("seen_object_counts", {}),
        "seen_interesting_objects": memory.get("seen_interesting_objects", []),
        "recent_observations": memory.get("recent_observations", [])[-8:],
        "recent_actions": memory.get("recent_actions", [])[-12:],
        "recent_action_values": memory.get("recent_action_values", [])[-12:],
        "recent_action_names": memory.get("recent_action_names", [])[-12:],
        "skills_written": memory.get("skills_written", []),
        "world_model_evidence": memory.get("world_model_evidence"),
    }


def update_body_memory(
    memory: dict[str, Any],
    previous_obs: Any,
    obs: Any,
    env: Any,
    step: int,
    source: str,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> None:
    previous_encoded = encode_observation(previous_obs)
    encoded = encode_observation(obs)
    _, np, _, object_to_idx, _ = require_minigrid()
    actions = action_map(env)
    idx_to_action = {value: name for name, value in actions.items()}
    summary = summarize_grid(encoded["image"], object_to_idx)

    memory["mission"] = encoded.get("mission", "")
    memory["step_count"] = int(step) + 1
    blocked = bool(
        idx_to_action.get(int(action)) == "forward"
        and float(reward) == 0.0
        and np.array_equal(previous_encoded["image"], encoded["image"])
        and int(previous_encoded.get("direction", -1)) == int(encoded.get("direction", -1))
    )
    memory["last_action_blocked"] = blocked
    if blocked:
        memory["blocked_forward_steps"] = int(memory.get("blocked_forward_steps", 0)) + 1

    counts = memory.setdefault("seen_object_counts", {})
    for name, count in summary["counts"].items():
        counts[name] = int(counts.get(name, 0)) + int(count)

    seen = memory.setdefault("seen_interesting_objects", [])
    known = {(item["object"], item["x"], item["y"]) for item in seen}
    for item in summary["interesting_cells"]:
        key = (item["object"], item["x"], item["y"])
        if key not in known:
            seen.append({**item, "first_seen_step": int(step)})
            known.add(key)
    del seen[:-40]

    observations = memory.setdefault("recent_observations", [])
    observations.append(
        {
            "step": int(step),
            "direction": int(encoded.get("direction", -1)),
            "shape": list(np.asarray(encoded["image"]).shape),
            "counts": summary["counts"],
            "interesting_cells": summary["interesting_cells"][:12],
        }
    )
    del observations[:-12]

    actions_memory = memory.setdefault("recent_actions", [])
    action_values = memory.setdefault("recent_action_values", [])
    action_names = memory.setdefault("recent_action_names", [])
    action_name = idx_to_action.get(int(action), str(int(action)))
    actions_memory.append(
        {
            "step": int(step),
            "source": source,
            "action": action_name,
            "action_value": int(action),
            "action_name": action_name,
            "blocked": blocked,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    )
    action_values.append(int(action))
    action_names.append(action_name)
    del actions_memory[:-20]
    del action_values[:-20]
    del action_names[:-20]


def dump_trace(path: Path, trace: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def describe_cell(cell: Any, object_to_idx: dict[str, int]) -> dict[str, Any]:
    idx_to_object = {value: key for key, value in object_to_idx.items()}
    if cell is None:
        return {"object": "unknown", "raw": None}
    values = [int(value) for value in cell.tolist()]
    obj = idx_to_object.get(values[0], f"object:{values[0]}")
    return {
        "object": obj,
        "color_idx": values[1] if len(values) > 1 else None,
        "state": values[2] if len(values) > 2 else None,
        "state_name": door_state_name(values[2]) if len(values) > 2 and obj == "door" else None,
        "raw": values,
    }


def door_state_name(value: int) -> str:
    return {0: "open", 1: "closed", 2: "locked"}.get(int(value), f"state:{int(value)}")
