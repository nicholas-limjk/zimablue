from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SkillProposal:
    name: str
    reason: str
    design: str
    source: str


class OpenAICompatiblePythonSkillWriter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 120,
    ):
        load_env_file()
        self.api_key = api_key or os.getenv("AGENT_SKILL_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")
        self.endpoint = endpoint or os.getenv("AGENT_SKILL_ENDPOINT") or "https://api.openai.com/v1/chat/completions"
        self.model = model or os.getenv("AGENT_SKILL_MODEL") or "gpt-4.1-mini"
        self.timeout_s = int(os.getenv("AGENT_SKILL_TIMEOUT_S", str(timeout_s)))
        if not self.api_key:
            raise RuntimeError("Missing AGENT_SKILL_API_KEY / OPENAI_API_KEY / openai_key.")

    def propose(self, payload: dict[str, Any]) -> SkillProposal:
        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write one Python skill file for a MiniGrid agent. "
                        "Return JSON only with keys name, reason, design, source. "
                        "source must define def act(obs, memory, api): and may use api.np. "
                        "Prefer compact, executable code over prose. "
                        "For repairs, design must describe the persistent state machine before source implements it. "
                        "If compiler_feedback is present, treat it as a static compiler error and fix every failed check before doing anything else. "
                        "If a repair request includes an acceptance contract, write code that satisfies it exactly."
                    ),
                },
                {"role": "user", "content": build_prompt(payload)},
            ],
        }
        if supports_custom_temperature(self.model):
            request_body["temperature"] = 0.2
        body = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        parsed = parse_json_object(text)
        return SkillProposal(
            name=str(parsed.get("name", "generated_skill")),
            reason=str(parsed.get("reason", "")),
            design=str(parsed.get("design", "")),
            source=str(parsed["source"]),
        )


def supports_custom_temperature(model: str) -> bool:
    default_temperature_only_prefixes = ("gpt-5",)
    default_temperature_models = {"o3", "o3-pro"}
    if model in default_temperature_models:
        return False
    return not model.startswith(default_temperature_only_prefixes)


def build_prompt(payload: dict[str, Any]) -> str:
    lines = [
            "Base impulse: reach_goal.",
            "Base actor: random primitive MiniGrid actions.",
            "You are the thinking layer. Inspect only the partial body observation, learned memory, and recent failures.",
            "You do not receive the full grid, hidden map, env object, oracle position, or future observations.",
            "Write a reusable policy skill that keeps acting turn by turn.",
            "Use NumPy through api.np. Do not import network, filesystem, subprocess, or external services.",
            "Use memory for anything the body has learned across turns: observations, outcomes, objects encountered, useful hypotheses, and control state.",
            "Current perception is not the same as learned state. Important facts may disappear from view and still be useful later.",
            "Preserve task-relevant observations, interaction outcomes, and hypotheses so the policy can act on them after they are no longer visible.",
            "Prefer policies that can adapt from experience rather than one-off scripts for the current screen.",
            "Look at recent failures. If a behavior is unproductive, change strategy instead of repeating it.",
            "If world_model_evidence is present, use it diagnostically: recurring high prediction error suggests missing world knowledge or a need to gather evidence; low prediction error with continued failure suggests the current policy should change.",
            "Do not treat a single high-error transition as a discovered rule. Prefer repeated action-conditioned evidence and connect any proposed skill change to observed outcomes.",
            "If the world model reports maturity='warming_up', do not use its residuals as strong evidence yet.",
            "Do not guess MiniGrid partial-observation coordinates. Use api.front_cell(obs), api.front_object(obs), api.visible_cells(obs), and api.egocentric_cells(obs).",
            "api.front_object(obs) returns a dict like {'object': 'wall', 'state': 0, 'raw': [...]}.",
            "For doors, state_name is provided: open=0, closed=1, locked=2. Open doors are traversable; closed/locked doors need toggle/key handling.",
            "api.egocentric_cells(obs) returns cells with view_x/view_y plus body-relative lateral and forward offsets. forward=1,lateral=0 is directly ahead.",
            "memory['recent_actions'] contains dict records; memory['recent_action_values'] contains raw action ints; memory['recent_action_names'] contains strings.",
            "If memory['last_action_blocked'] is true after a forward action, the policy must turn or change exploration mode instead of repeatedly moving forward.",
            "Simple hints: objects, obstacles, interactions, terminal cues, and unknown space may matter; remembering what has been seen can be useful; any search or planning loop you write must be finite.",
            "If the payload contains a repair_request, treat it as a reflection episode: study the failed skill and trace, identify the missing capability, then write a replacement skill.",
            "For any repair_request, use a two-stage proposal: design first, code second. The design must name the memory fields, how observations update them, and how action choice reads them.",
            "Do not oscillate left/right in place. Do not repeatedly return forward into a blocked cell.",
            "The source must be a complete Python module containing:",
            "def act(obs, memory, api):",
            "  ...",
            "  return api.actions['forward']  # or another valid action",
            "Return JSON only: {\"name\": ..., \"reason\": ..., \"design\": ..., \"source\": ...}",
            "",
        ]
    if payload.get("repair_request"):
        compiler = payload["repair_request"].get("compiler_feedback")
        if compiler:
            lines.extend(compiler_feedback_prompt(compiler))
            lines.append("")
        lines.extend(reflection_gate_prompt(payload["repair_request"]))
        lines.append("")
    lines.append(
            json.dumps(payload, indent=2),
    )
    return "\n".join(lines)


def compiler_feedback_prompt(compiler: dict[str, Any]) -> list[str]:
    lines = [
        "PRIMARY INSTRUCTION: STATIC COMPILER FEEDBACK",
        "- Your previous proposal was rejected before execution.",
        "- You must fix the static validation errors exactly; do not produce another top-level memory/reflex policy.",
        "- Failed checks:",
    ]
    for check in compiler.get("failed_static_checks", []):
        lines.append(f"  * {check}")
    lines.extend(
        [
            "- Start source with this exact opening block:",
            "def act(obs, memory, api):",
            "    a = api.actions",
            "    np = api.np",
            "    if 'learned_structure' not in memory:",
            "        memory['learned_structure'] = {'observations': [], 'outcomes': [], 'hypotheses': {}, 'subgoals': [], 'control_state': {}}",
            "    learned = memory['learned_structure']",
            "- Add any persistent fields under learned, for example learned['control_state']['turn_dir'], not memory['turn_dir'].",
            "- Before returning, read from learned in the action-selection branch.",
            "- If you reuse ideas from the rejected proposal, move them inside learned_structure.",
        ]
    )
    return lines


def reflection_gate_prompt(repair_request: dict[str, Any]) -> list[str]:
    contract = repair_request.get("acceptance_contract", {})
    failed_class = repair_request.get("current_skill_self_audit", {}).get("capability_class", "unknown")
    lines = [
        "CAPABILITY GATE FOR THIS REFLECTION:",
        f"- The failed skill capability class is: {failed_class}.",
        "- The replacement must be a strictly higher capability class on this ladder:",
        "  front_cell_reflex -> current_view_reactive -> persistent_structure -> planner_or_search.",
            "- If structural escalation is active, a reactive policy will be rejected even if it is cleaner.",
            "- Two-stage protocol: first write design, then write source. The source must implement the design, not a different simpler reflex.",
            "- The design must specify: persistent memory fields, observation update rules, outcome update rules, and action-selection rules that read remembered state.",
            "- Minimal required source scaffold, with task-specific contents left for you to design:",
            "  def act(obs, memory, api):",
            "      a = api.actions",
            "      if 'learned_structure' not in memory:",
            "          memory['learned_structure'] = {...}",
            "      learned = memory['learned_structure']",
            "      # 1. update learned from body observation helpers",
            "      # 2. update learned from action outcomes and body feedback",
            "      # 3. choose action by reading learned, not only current front/current view",
            "      return a[...]",
            "- If structural escalation is active, the source will be rejected unless it literally initializes memory['learned_structure'].",
            "- Put persistent task state inside learned_structure. Do not create only top-level memory keys like memory['has_item'], memory['turn_dir'], memory['seen_objects'], or memory['blocked_count'].",
            "- You may store fields such as possession, encountered objects, blocked outcomes, visited hints, subgoals, or exploration mode, but they must be children of learned_structure.",
            "- A valid pattern is: learned = memory['learned_structure']; learned['observations'] = ...; learned['outcomes'] = ...; learned['hypotheses'] = ...; then return an action based on learned.",
            "- When the contract requires learned_structure, start your source from this valid Python template and fill in the general update/choice logic:",
            "  def act(obs, memory, api):",
            "      a = api.actions",
            "      np = api.np",
            "      if 'learned_structure' not in memory:",
            "          memory['learned_structure'] = {",
            "              'observations': [],",
            "              'outcomes': [],",
            "              'hypotheses': {},",
            "              'subgoals': [],",
            "              'control_state': {},",
            "          }",
            "      learned = memory['learned_structure']",
            "      cells = api.egocentric_cells(obs)",
            "      front = api.front_object(obs)",
            "      # update learned from cells/front and body feedback",
            "      # return a primitive action chosen from learned state",
            "      return a['forward']",
            "- Do not replace this template with separate top-level memory fields. Add your fields inside learned.",
    ]
    if contract.get("required_static_checks"):
        lines.append("- Your source must pass these static checks:")
        for check in contract["required_static_checks"]:
            lines.append(f"  * {check}")
    lines.extend(
        [
            "- Environment-general learned_structure examples:",
            "  * observations: body-relative cells/objects/states seen over time",
            "  * action outcomes: which actions were blocked, useful, or repeated",
            "  * task facts: objects encountered, interactions attempted, subgoals inferred",
            "  * persistent hypotheses about what must be revisited, avoided, collected, opened, toggled, or reached",
            "  * spatial/topological hints if useful, but no hidden map or oracle",
            "- The replacement should update learned_structure every turn from obs/api helpers before choosing an action.",
            "- Distinguish current perception from learned state: a useful observation can leave the view but still guide future action.",
            "- The returned action should depend on learned_structure, not only front_object/current visible cells.",
            "- If the trace shows repeated failure from local reaction, preserve more evidence rather than adding another local rule.",
            "- Do not write a DoorKey-specific script; write a reusable MiniGrid skill.",
        ]
    )
    return lines


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


def load_env_file(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")
