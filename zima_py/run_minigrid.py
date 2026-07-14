from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from .llm import OpenAICompatiblePythonSkillWriter
from .jepa_world_model import LatentActionWorldModel
from .full_jepa_world_model import FullJepaWorldModel
from .minigrid_adapter import MissingMiniGridError, MiniGridSpec, encode_observation, make_minigrid, normalize_reset, normalize_step
from .skill_runtime import MiniGridSkillApi, SkillProgram, dump_trace, trace_payload, update_body_memory, write_skill


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Python/NumPy MiniGrid Zima Blue harness")
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--think-after", type=int, default=12)
    parser.add_argument("--repair-after", type=int, default=18)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--quality-retries", type=int, default=0)
    parser.add_argument("--no-llm", action="store_true", help="Run random base actor only.")
    parser.add_argument(
        "--load-skill",
        action="append",
        default=[],
        help="Load an existing Python skill module before the run. Can be passed more than once.",
    )
    parser.add_argument("--skill-out", default="generated_skills/minigrid_skill.py")
    parser.add_argument("--trace-out", default="artifacts/minigrid_trace.json")
    parser.add_argument("--jepa", action="store_true", help="Train a NumPy action-conditioned latent world model and expose its prediction errors to the skill writer.")
    parser.add_argument("--full-jepa", action="store_true", help="Train learned context/EMA-target encoders and a multi-step action-conditioned predictor with PyTorch.")
    parser.add_argument("--jepa-warmup", type=int, default=32)
    parser.add_argument("--jepa-batch-size", type=int, default=32)
    parser.add_argument("--jepa-horizon", type=int, default=3)
    parser.add_argument("--jepa-train-steps", type=int, default=1, help="Gradient updates attempted after each environment transition.")
    parser.add_argument("--jepa-checkpoint", default=None, help="Optional full-JEPA checkpoint to load if present and save at the end.")
    args = parser.parse_args()

    try:
        result = run(args)
    except MissingMiniGridError as exc:
        print(exc)
        return 2

    return 0 if result else 1


def run(args) -> bool:
    env = make_minigrid(MiniGridSpec(env_id=args.env, seed=args.seed, max_steps=args.max_steps))
    obs, info = normalize_reset(env.reset(seed=args.seed))
    rng = random.Random(args.seed)
    memory: dict = {"blocked_or_low_reward_steps": 0, "skills_written": []}
    if getattr(args, "full_jepa", False):
        world_model = FullJepaWorldModel(
            env.action_space.n,
            batch_size=getattr(args, "jepa_batch_size", 32),
            horizon=getattr(args, "jepa_horizon", 3),
            warmup=getattr(args, "jepa_warmup", 32),
            seed=args.seed,
        )
        checkpoint = getattr(args, "jepa_checkpoint", None)
        if checkpoint and (ROOT / checkpoint).exists():
            world_model.load(ROOT / checkpoint)
    else:
        world_model = LatentActionWorldModel(env.action_space.n) if getattr(args, "jepa", False) else None
    program = SkillProgram()
    for skill_path in args.load_skill:
        resolved = (ROOT / skill_path).resolve()
        program.add_skill_file(resolved, reason="preloaded example skill")
        memory["skills_written"].append({"name": resolved.stem, "path": str(resolved), "reason": "preloaded"})
    trace: list[dict] = []
    writer = None if args.no_llm else OpenAICompatiblePythonSkillWriter()
    thinker_writes = 0
    last_skill_write_step = -1
    if program.records:
        last_skill_write_step = 0

    for step in range(args.max_steps):
        encoded = encode_observation(obs)
        api = MiniGridSkillApi(env, memory)
        action = program.act(encoded, memory, api)
        source = "skill"
        if action is None:
            action = rng.randrange(env.action_space.n)
            source = "base-random"

        next_obs, reward, terminated, truncated, info = normalize_step(env.step(action))
        update_body_memory(memory, obs, next_obs, env, step, source, action, reward, terminated, truncated)
        if world_model is not None:
            encoded_next = encode_observation(next_obs)
            action_names = {value: name for name, value in api.actions.items()}
            observe_kwargs = {
                "step": step,
                "action_name": action_names.get(int(action), str(int(action))),
                "blocked": bool(memory.get("last_action_blocked", False)),
                "reward": float(reward),
            }
            if isinstance(world_model, FullJepaWorldModel):
                observe_kwargs.update(
                    terminal=bool(terminated or truncated),
                    train_steps=getattr(args, "jepa_train_steps", 1),
                )
            world_event = world_model.observe(encoded, int(action), encoded_next, **observe_kwargs)
            memory["world_model_evidence"] = world_model.evidence()
        else:
            world_event = None
        event = {
            "step": step,
            "source": source,
            "action": int(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "mission": encoded.get("mission", ""),
        }
        if world_event is not None:
            event["world_model"] = world_event
        trace.append(event)

        if reward <= 0 and source == "base-random":
            memory["blocked_or_low_reward_steps"] += 1

        if (
            writer
            and not program.records
            and memory["blocked_or_low_reward_steps"] >= args.think_after
        ):
            payload = trace_payload(next_obs, env, memory, trace)
            proposal = writer.propose(payload)
            record = write_skill(ROOT / args.skill_out, proposal.source)
            record.reason = proposal.reason
            program.add(record)
            thinker_writes += 1
            last_skill_write_step = step
            memory["skills_written"].append(
                {"name": proposal.name, "path": str(record.path), "reason": proposal.reason, "design": proposal.design}
            )
            trace.append({"step": step, "source": "thinker", "event": "wrote_skill", "proposal": memory["skills_written"][-1]})

        if (
            writer
            and program.records
            and reflection_attempt_count(memory) < args.max_repairs
            and should_repair_skill(trace, last_skill_write_step, args.repair_after)
        ):
            repair_context = build_repair_context(program, trace, memory, last_skill_write_step)
            proposal = writer.propose(trace_payload(next_obs, env, memory, trace, repair_context))
            proposal_source, normalization = normalize_persistent_state_namespace(proposal.source)
            rejection = proposal_rejection(proposal_source, repair_context)
            rejected_sources = []
            for quality_retry in range(args.quality_retries):
                if rejection is None:
                    break
                rejected_sources.append(rejection_record(proposal, rejection, quality_retry + 1, proposal_source, normalization))
                retry_context = {
                    **repair_context,
                    "rejected_proposal": rejected_sources[-1],
                    "compiler_feedback": compiler_feedback(rejection),
                    "instruction": repair_context["instruction"] + " " + compiler_repair_instruction(rejection),
                }
                proposal = writer.propose(trace_payload(next_obs, env, memory, trace, retry_context))
                proposal_source, normalization = normalize_persistent_state_namespace(proposal.source)
                rejection = proposal_rejection(proposal_source, repair_context)
            if rejection is not None:
                rejected_sources.append(rejection_record(proposal, rejection, len(rejected_sources) + 1, proposal_source, normalization))
                trace.append(
                    {
                        "step": step,
                        "source": "thinker",
                        "event": "rejected_reflection_proposals",
                        "rejection": rejection,
                        "rejected_proposals": rejected_sources,
                    }
                )
                memory.setdefault("reflection_rejections", []).append(rejection)
                last_skill_write_step = step
                continue
            record = write_skill(ROOT / args.skill_out, proposal_source)
            record.reason = proposal.reason
            program.clear()
            program.add(record)
            thinker_writes += 1
            last_skill_write_step = step
            memory["skills_written"].append(
                {
                    "name": proposal.name,
                    "path": str(record.path),
                    "reason": proposal.reason,
                    "design": proposal.design,
                    "normalization": normalization,
                    "reflection": True,
                }
            )
            trace.append({"step": step, "source": "thinker", "event": "reflected_and_rewrote_skill", "proposal": memory["skills_written"][-1]})

        obs = next_obs
        if terminated or truncated:
            save_world_model(world_model, getattr(args, "jepa_checkpoint", None))
            dump_trace(ROOT / args.trace_out, trace)
            print_summary(args.env, terminated, truncated, step + 1, trace, memory)
            env.close()
            return bool(terminated)

    save_world_model(world_model, getattr(args, "jepa_checkpoint", None))
    dump_trace(ROOT / args.trace_out, trace)
    print_summary(args.env, False, True, args.max_steps, trace, memory)
    env.close()
    return False


def save_world_model(world_model, checkpoint: str | None) -> None:
    if checkpoint and isinstance(world_model, FullJepaWorldModel):
        world_model.save(ROOT / checkpoint)


def print_summary(env_id: str, terminated: bool, truncated: bool, steps: int, trace: list[dict], memory: dict) -> None:
    print(f"env={env_id}")
    print(f"status={'solved' if terminated else 'truncated' if truncated else 'stopped'} steps={steps}")
    print(f"skills={memory.get('skills_written', [])}")
    print("last events:")
    for event in trace[-8:]:
        print(event)


def should_repair_skill(trace: list[dict], last_skill_write_step: int, repair_after: int) -> bool:
    if last_skill_write_step < 0:
        return False
    skill_events = [event for event in trace if event.get("source") == "skill" and event.get("step", -1) > last_skill_write_step]
    if len(skill_events) < repair_after:
        return False
    recent = skill_events[-repair_after:]
    if any(event.get("reward", 0) > 0 or event.get("terminated") for event in recent):
        return False
    actions = [event.get("action") for event in recent]
    interaction_actions = {3, 5}
    if not any(action in interaction_actions for action in actions):
        return True
    if len(set(actions[-8:])) <= 1:
        return True
    turn_actions = {0, 1}
    if len(actions) >= 8 and all(action in turn_actions for action in actions[-8:]):
        return True
    return False


def build_repair_context(program: SkillProgram, trace: list[dict], memory: dict, last_skill_write_step: int) -> dict:
    current = program.records[-1] if program.records else None
    current_source = ""
    if current and current.path.exists():
        current_source = current.path.read_text(encoding="utf-8")
    skill_events = [event for event in trace if event.get("source") == "skill" and event.get("step", -1) > last_skill_write_step]
    reflection_count_value = reflection_attempt_count(memory)
    return {
        "reason": "Reflection loop: the current skill appears stuck, incomplete, or unproductive and must be studied before replacement.",
        "reflection_attempt": reflection_count_value + 1,
        "current_skill": {
            "name": current.name if current else None,
            "path": str(current.path) if current else None,
            "source": current_source,
        },
        "current_skill_self_audit": skill_self_audit(current_source),
        "failure_trace": skill_events[-64:],
        "observed_failure_patterns": {
            "recent_action_names": memory.get("recent_action_names", [])[-24:],
            "blocked_forward_steps": memory.get("blocked_forward_steps", 0),
            "last_action_blocked": memory.get("last_action_blocked", False),
            "seen_object_counts": memory.get("seen_object_counts", {}),
            "seen_interesting_objects": memory.get("seen_interesting_objects", [])[-20:],
        },
        "reflection_questions": [
            "What assumption did the current skill make that the trace contradicts?",
            "What information has the body actually learned that the current skill failed to use?",
            "Which useful facts may no longer be visible but should still influence future actions?",
            "Which interaction outcomes should change behavior the next time a similar situation appears?",
            "Is a local reflex enough, or does the replacement need a more persistent internal mechanism?",
            "How will the replacement distinguish current perception from learned state?",
            "What state should memory preserve so the body improves after each observation?",
            "How will the replacement avoid finite loops or repeated unproductive behavior?",
        ],
        "escalation_rule": escalation_rule(reflection_count_value),
        "acceptance_contract": acceptance_contract(reflection_count_value),
        "proposal_refinement_guidance": proposal_refinement_guidance(reflection_count_value),
        "instruction": (
            "Write a replacement skill after reflecting on the failure. "
            "Do not merely flip a turn direction or add another one-screen reflex unless the trace supports that. "
            "If the current skill is only a front-cell reflex and it failed, the replacement must add a qualitatively new capability that uses more of the body's observations or memory. "
            "If a current-view reactive explorer has already failed, abandon pure reactive exploration and introduce a persistent internal representation of what the body has learned over time. "
            "That representation should help choose future actions from remembered observations, interaction outcomes, and hypotheses, not just store counters. "
            "Useful facts may disappear from current view; the replacement should still use them when choosing later primitive actions. "
            "Use a two-stage proposal: the design must name the persistent memory fields, how body observations update them, how action outcomes update them, and how action selection reads them. "
            "Then implement that exact design in source. "
            "If the escalation_rule says an internal data structure is now required, the replacement source must actually create and update that structure in memory and use it to choose actions. "
            "If acceptance_contract is present, the replacement must satisfy every required_static_check in it. "
            "Invent the simplest reusable mechanism that can use body-only observations and memory to make progress. "
            "Any loop or search you write must be finite."
        ),
    }


def reflection_count(memory: dict) -> int:
    return sum(1 for item in memory.get("skills_written", []) if item.get("reflection"))


def reflection_attempt_count(memory: dict) -> int:
    return reflection_count(memory) + len(memory.get("reflection_rejections", []))


def escalation_rule(prior_reflections: int) -> dict:
    if prior_reflections < 2:
        return {
            "level": "reflect",
            "require_new_internal_data_structure": False,
            "message": "Reflect on the failed behavior and add the simplest missing capability.",
        }
    return {
        "level": "structural_change_required",
        "require_new_internal_data_structure": True,
        "message": (
            "At least two reflected skills have failed. A reactive policy is no longer acceptable. "
            "The next skill must define a persistent internal data structure named memory['learned_structure'], update it from every observation, "
            "and use it to choose actions from accumulated experience rather than only the current view. "
            "This rule is environment-general: the structure may store any task-relevant learned facts, topology, object history, action outcomes, hypotheses, or subgoals."
        ),
        "forbidden_replacements": [
            "only changing turn direction",
            "only checking api.front_object(obs)",
            "only reacting to currently visible cells",
            "only adding counters without a learned world/task representation",
        ],
    }


def acceptance_contract(prior_reflections: int) -> dict:
    if prior_reflections < 2:
        return {
            "capability_ladder": [
                "front_cell_reflex",
                "current_view_reactive",
                "persistent_structure",
                "planner_or_search",
            ],
            "required_static_checks": [
                "replacement capability class must be higher than failed skill capability class",
            ],
        }
    return {
        "capability_ladder": [
            "front_cell_reflex",
            "current_view_reactive",
            "persistent_structure",
            "planner_or_search",
        ],
        "required_static_checks": [
            "source contains the literal key learned_structure",
            "source initializes memory['learned_structure'] if absent",
            "source updates learned_structure from api.egocentric_cells(obs), api.visible_cells(obs), api.visible_objects(obs), or api.front_object(obs)",
            "source reads learned_structure later when selecting the returned action",
            "source is not merely counters, turn direction, or last action memory",
            "persistent task state is stored inside memory['learned_structure'] rather than only as separate top-level memory keys",
            "proposal design names persistent memory fields, observation updates, outcome updates, and action-selection rules",
        ],
        "note": (
            "Do not solve a specific named MiniGrid environment. Build a reusable learned representation from body-only observations. "
            "The representation should separate current perception from accumulated learned state and use learned facts after they leave view."
        ),
    }


def proposal_refinement_guidance(prior_reflections: int) -> dict:
    guidance = {
        "general_goal": "Refine the failed proposal into a higher-capability reusable MiniGrid skill.",
        "do_not": [
            "only flip turn direction",
            "only add counters",
            "only react to the front cell",
            "only react to currently visible cells",
            "hard-code a solution path or named environment",
        ],
        "do": [
            "store learned facts in memory across turns",
            "update learned facts from the body's partial observations",
            "separate current perception from accumulated learned state",
            "remember useful facts even after they leave the current view",
            "record interaction outcomes and use them when similar situations recur",
            "form task-agnostic hypotheses about objects, obstacles, interactions, subgoals, and terminal cues",
            "use the learned facts when choosing the returned primitive action",
            "keep all loops/search finite",
        ],
    }
    if prior_reflections >= 2:
        guidance["required_memory_key"] = "learned_structure"
        guidance["minimum_shape"] = {
            "initialize": "if 'learned_structure' not in memory: memory['learned_structure'] = {...}",
            "update": "use api.egocentric_cells(obs), api.visible_cells(obs), api.visible_objects(obs), or api.front_object(obs) to update learned_structure",
            "outcomes": "fold memory['recent_actions'], memory['last_action_blocked'], or related body feedback into learned_structure",
            "choose": "read learned_structure before returning api.actions[...]",
            "state_location": "persistent task state must be stored as children of memory['learned_structure'], not only as separate top-level memory keys",
        }
        guidance["acceptable_learned_structure_fields"] = [
            "observations",
            "outcomes",
            "hypotheses",
            "encountered",
            "subgoals",
            "exploration_state",
            "control_state",
        ]
        guidance["invalid_pattern"] = (
            "Do not only create memory['has_item'], memory['turn_dir'], memory['seen_objects'], "
            "or memory['blocked_count']; those ideas may be used only if nested inside memory['learned_structure']."
        )
        guidance["minimal_scaffold"] = [
            "def act(obs, memory, api):",
            "    a = api.actions",
            "    np = api.np",
            "    if 'learned_structure' not in memory:",
            "        memory['learned_structure'] = {'observations': [], 'outcomes': [], 'hypotheses': {}, 'subgoals': [], 'control_state': {}}",
            "    learned = memory['learned_structure']",
            "    cells = api.egocentric_cells(obs)",
            "    front = api.front_object(obs)",
            "    # update learned from body observations",
            "    # update learned from action outcomes/body feedback",
            "    # choose by reading learned, not only current perception",
            "    return a[...]",
        ]
    return guidance


def normalize_persistent_state_namespace(source: str) -> tuple[str, dict]:
    if "learned_structure" in source:
        return source, {"applied": False, "reason": "already_uses_learned_structure"}

    excluded = {
        "recent_actions",
        "recent_action_values",
        "recent_action_names",
        "last_action_blocked",
        "blocked_forward_steps",
        "blocked_or_low_reward_steps",
        "skills_written",
        "reflection_rejections",
        "mission",
        "step_count",
    }
    keys = re.findall(r"memory\['([^']+)'\]", source)
    keys.extend(re.findall(r"memory\.setdefault\('([^']+)'", source))
    keys.extend(re.findall(r"memory\.get\('([^']+)'", source))
    prefixed = [key for key in dict.fromkeys(keys) if re.match(r"^[A-Za-z0-9]+_.+", key) and key not in excluded]
    prefixes = {}
    for key in prefixed:
        prefix = key.split("_", 1)[0] + "_"
        prefixes.setdefault(prefix, []).append(key)
    prefix, prefix_keys = max(prefixes.items(), key=lambda item: len(item[1]), default=("", []))
    if len(prefix_keys) >= 3:
        normalized = inject_learned_structure_init(source)
        for key in sorted(prefix_keys, key=len, reverse=True):
            field = key[len(prefix):]
            normalized = normalized.replace(
                f"memory.setdefault('{key}',", f"memory['learned_structure'].setdefault('{field}',"
            )
            normalized = normalized.replace(
                f'memory.setdefault("{key}",', f'memory["learned_structure"].setdefault("{field}",'
            )
            normalized = normalized.replace(
                f"memory.get('{key}',", f"memory['learned_structure'].get('{field}',"
            )
            normalized = normalized.replace(
                f'memory.get("{key}",', f'memory["learned_structure"].get("{field}",'
            )
            normalized = normalized.replace(f"memory['{key}']", f"memory['learned_structure']['{field}']")
            normalized = normalized.replace(f'memory["{key}"]', f'memory["learned_structure"]["{field}"]')
            normalized = normalized.replace(f"'{key}' not in memory", f"'{field}' not in memory['learned_structure']")
            normalized = normalized.replace(f'"{key}" not in memory', f'"{field}" not in memory["learned_structure"]')
        return normalized, {"applied": True, "from_prefix": prefix, "to": "learned_structure", "fields": prefix_keys}

    candidates = [
        key
        for key in dict.fromkeys(keys)
        if key not in excluded and (
            key.startswith("_")
            or any(token in key.lower() for token in ["skill", "state", "map", "belief", "policy", "model", "memory", "kdg"])
        )
    ]
    if len(candidates) != 1:
        return source, {"applied": False, "reason": "no_single_private_state_key", "candidates": candidates}

    key = candidates[0]
    normalized = source.replace(f"memory['{key}']", "memory['learned_structure']")
    normalized = normalized.replace(f"memory.setdefault('{key}',", "memory.setdefault('learned_structure',")
    normalized = normalized.replace(f"memory.get('{key}',", "memory.get('learned_structure',")
    normalized = normalized.replace(f'memory["{key}"]', 'memory["learned_structure"]')
    normalized = normalized.replace(f'memory.setdefault("{key}",', 'memory.setdefault("learned_structure",')
    normalized = normalized.replace(f'memory.get("{key}",', 'memory.get("learned_structure",')
    normalized = normalized.replace(f"'{key}' not in memory", "'learned_structure' not in memory")
    normalized = normalized.replace(f'"{key}" not in memory', '"learned_structure" not in memory')
    return normalized, {"applied": True, "from": key, "to": "learned_structure"}


def inject_learned_structure_init(source: str) -> str:
    init = "    if 'learned_structure' not in memory:\n        memory['learned_structure'] = {}\n"
    if "def act(obs, memory, api):" not in source or "memory['learned_structure']" in source:
        return source
    lines = source.splitlines()
    insert_at = None
    for index, line in enumerate(lines):
        if "api.np" in line:
            insert_at = index + 1
            break
    if insert_at is None:
        for index, line in enumerate(lines):
            if line.startswith("def act("):
                insert_at = index + 1
                break
    if insert_at is None:
        return source
    return "\n".join(lines[:insert_at] + init.rstrip("\n").splitlines() + lines[insert_at:])


def proposal_rejection(source: str, repair_context: dict) -> dict | None:
    current_audit = repair_context.get("current_skill_self_audit", {})
    proposed_audit = skill_self_audit(source)
    failed_class = current_audit.get("capability_class", "unknown")
    proposed_class = proposed_audit.get("capability_class", "unknown")
    failed_rank = capability_rank(failed_class)
    proposed_rank = capability_rank(proposed_class)
    rule = repair_context.get("escalation_rule", {})
    if rule.get("require_new_internal_data_structure") and not proposed_audit.get("passes_structural_contract"):
        return {
            "reason": (
                "Structural escalation is active: proposal failed the learned_structure acceptance contract."
            ),
            "failed_static_checks": failed_static_checks(proposed_audit),
            "failed_capability_class": failed_class,
            "proposed_capability_class": proposed_class,
            "current_audit": current_audit,
            "proposed_audit": proposed_audit,
        }
    if failed_rank >= 0 and proposed_rank <= failed_rank:
        return {
            "reason": (
                "Capability-class rejection: the proposed skill is not a higher-capability class than the failed skill."
            ),
            "failed_static_checks": failed_static_checks(proposed_audit),
            "failed_capability_class": failed_class,
            "proposed_capability_class": proposed_class,
            "current_audit": current_audit,
            "proposed_audit": proposed_audit,
        }
    return None


def failed_static_checks(audit: dict) -> list[str]:
    checks = []
    if not audit.get("uses_learned_structure"):
        checks.append("source does not contain the literal key learned_structure")
    if not audit.get("initializes_learned_structure"):
        checks.append("source does not initialize memory['learned_structure'] if absent")
    if not audit.get("updates_from_observation"):
        checks.append("source does not update learned_structure from body observation helpers")
    if not audit.get("reads_for_decision"):
        checks.append("source does not read learned_structure when selecting the returned action")
    if audit.get("capability_class") in {"front_cell_reflex", "current_view_reactive"}:
        checks.append("source is still a local reflex/current-view policy rather than persistent learned state")
    return checks


def compiler_feedback(rejection: dict) -> dict:
    return {
        "kind": "static_validation_error",
        "message": "Your previous source did not satisfy the learned_structure contract.",
        "failed_static_checks": rejection.get("failed_static_checks", []),
        "required_opening_block": [
            "def act(obs, memory, api):",
            "    a = api.actions",
            "    np = api.np",
            "    if 'learned_structure' not in memory:",
            "        memory['learned_structure'] = {'observations': [], 'outcomes': [], 'hypotheses': {}, 'subgoals': [], 'control_state': {}}",
            "    learned = memory['learned_structure']",
        ],
        "repair_rule": (
            "Rewrite the source by starting with required_opening_block exactly, then put all persistent task state "
            "inside learned. Preserve useful task-agnostic ideas, but do not use separate top-level memory fields as the main state."
        ),
    }


def compiler_repair_instruction(rejection: dict) -> str:
    checks = "; ".join(rejection.get("failed_static_checks", [])) or "unknown static check failed"
    return (
        "STATIC VALIDATION ERROR. Your previous replacement was rejected before execution. "
        f"Failed checks: {checks}. "
        "Now rewrite the entire source. Start def act with this exact block: "
        "a = api.actions; np = api.np; "
        "if 'learned_structure' not in memory: memory['learned_structure'] = {'observations': [], 'outcomes': [], 'hypotheses': {}, 'subgoals': [], 'control_state': {}}; "
        "learned = memory['learned_structure']. "
        "All persistent state must be stored under learned. "
        "The returned action must be chosen after reading learned, not only front/current visible cells. "
        "Return JSON with name, reason, design, source."
    )


def rejection_record(proposal, rejection: dict, attempt: int, source: str | None = None, normalization: dict | None = None) -> dict:
    return {
        "reason": rejection["reason"],
        "failed_static_checks": rejection.get("failed_static_checks", []),
        "proposal_reason": proposal.reason,
        "proposal_design": proposal.design,
        "normalization": normalization or {"applied": False},
        "quality_retry": attempt,
        "failed_capability_class": rejection["failed_capability_class"],
        "proposed_capability_class": rejection["proposed_capability_class"],
        "source_audit": rejection["proposed_audit"],
        "source": source or proposal.source,
        "original_source": proposal.source,
    }


def capability_rank(name: str) -> int:
    return {
        "front_cell_reflex": 0,
        "current_view_reactive": 1,
        "persistent_structure": 2,
        "planner_or_search": 3,
    }.get(name, -1)


def skill_self_audit(source: str) -> dict:
    compact = source or ""
    uses_learned_structure = "'learned_structure'" in compact or '"learned_structure"' in compact
    learned_structure_mentions = compact.count("learned_structure")
    learned_aliases = re.findall(r"(\w+)\s*=\s*memory\[['\"]learned_structure['\"]\]", compact)
    learned_aliases.extend(
        re.findall(r"(\w+)\s*=\s*memory\.setdefault\(['\"]learned_structure['\"]", compact)
    )
    learned_aliases = list(dict.fromkeys(learned_aliases))
    aliases_learned_structure = bool(learned_aliases)
    learned_alias_mentions = sum(
        compact.count(f"{alias}[") + compact.count(f"{alias}.get(") + compact.count(f"{alias}.setdefault(")
        for alias in learned_aliases
    )
    initializes_learned_structure = (
        "if 'learned_structure' not in memory" in compact
        or 'if "learned_structure" not in memory' in compact
        or "setdefault('learned_structure'" in compact
        or 'setdefault("learned_structure"' in compact
    )
    observation_api_mentions = [
        token
        for token in ["egocentric_cells", "visible_cells", "visible_objects", "front_object"]
        if token in compact
    ]
    updates_from_observation = uses_learned_structure and bool(observation_api_mentions) and any(
        token in compact for token in ["append(", "update(", ".add(", "=", "setdefault("]
    )
    returns_action = any(token in compact for token in ["return a[", "return A[", "api.actions["])
    reads_for_decision = uses_learned_structure and returns_action and (
        learned_structure_mentions >= 4 or (aliases_learned_structure and learned_alias_mentions >= 2)
    )
    uses_plan_or_search = any(token in compact for token in ["'plan'", '"plan"', "path", "queue", "deque", "search"])
    uses_persistent_structure = any(
        token in compact
        for token in [
            "'map'",
            '"map"',
            "'memory_map'",
            '"memory_map"',
            "'world'",
            '"world"',
            "'graph'",
            '"graph"',
            "'visited'",
            '"visited"',
            "'landmarks'",
            '"landmarks"',
            "'belief'",
            '"belief"',
            "'learned_structure'",
            '"learned_structure"',
        ]
    )
    uses_current_view = "egocentric_cells" in compact or "visible_cells" in compact or "visible_objects" in compact
    uses_front = "front_object" in compact
    passes_structural_contract = (
        uses_learned_structure
        and initializes_learned_structure
        and updates_from_observation
        and reads_for_decision
    )
    if uses_plan_or_search and passes_structural_contract:
        capability_class = "planner_or_search"
    elif passes_structural_contract:
        capability_class = "persistent_structure"
    elif uses_current_view:
        capability_class = "current_view_reactive"
    elif uses_front:
        capability_class = "front_cell_reflex"
    else:
        capability_class = "unknown"
    return {
        "capability_class": capability_class,
        "uses_front_object": uses_front,
        "uses_visible_objects": "visible_objects" in compact,
        "uses_visible_cells": "visible_cells" in compact,
        "uses_egocentric_cells": "egocentric_cells" in compact,
        "uses_memory_map_name": "'map'" in compact or '"map"' in compact,
        "uses_path_or_plan_name": "'plan'" in compact or '"plan"' in compact or "path" in compact,
        "uses_learned_structure": uses_learned_structure,
        "learned_structure_mentions": learned_structure_mentions,
        "aliases_learned_structure": aliases_learned_structure,
        "learned_aliases": learned_aliases,
        "learned_alias_mentions": learned_alias_mentions,
        "initializes_learned_structure": initializes_learned_structure,
        "observation_api_mentions": observation_api_mentions,
        "updates_from_observation": updates_from_observation,
        "reads_for_decision": reads_for_decision,
        "passes_structural_contract": passes_structural_contract,
        "uses_persistent_structure": uses_persistent_structure,
        "likely_current_view_reactive_only": (
            capability_class == "current_view_reactive"
        ),
        "likely_front_cell_reflex_only": (
            capability_class == "front_cell_reflex"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
