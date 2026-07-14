"""A tiny exploration skill that cycles turn/forward actions.

Useful as a minimal loaded-skill example when you want to verify the harness can
import Python skills and call `act(obs, memory, api)` each MiniGrid step.
"""


def act(obs, memory, api):
    step = int(memory.get("explore_then_forward_step", 0))
    memory["explore_then_forward_step"] = step + 1

    if step % 5 == 0:
        return api.actions["right"]
    return api.actions["forward"]
