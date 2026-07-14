# MiniGrid Examples

These examples use the official Farama Foundation MiniGrid package:

<https://github.com/Farama-Foundation/Minigrid>

The base actor is intentionally weak: if no skill returns an action, it samples
a random MiniGrid action. The thinking layer can later write a Python skill file
that defines:

```python
def act(obs, memory, api):
    return api.actions["forward"]
```

`obs["image"]` is a NumPy array. `api.np` is NumPy. Object/color constants are
available as `api.object_to_idx` and `api.color_to_idx`.

## 1. Random Baseline

```bash
uv run zima-minigrid --no-llm --env MiniGrid-Empty-5x5-v0 --max-steps 40
```

This shows the base impulse without any thinking layer. It should stumble around
and usually fail unless it gets lucky.

## 2. Load A Simple Reflex Skill

```bash
uv run zima-minigrid ^
  --no-llm ^
  --env MiniGrid-Empty-5x5-v0 ^
  --load-skill examples/minigrid_skills/forward_reflex.py ^
  --max-steps 40
```

The skill is intentionally tiny: move forward, turn right when blocked, pick up a
key if one is directly ahead, and toggle a door if one is directly ahead.

## 3. DoorKey With The LLM Skill Writer

```bash
set AGENT_SKILL_API_KEY=...
uv run zima-minigrid --env MiniGrid-DoorKey-8x8-v0 --think-after 12 --max-steps 128
```

The run starts with random action. After enough low-reward/blocked wandering,
the thinking layer receives a compact observation summary plus recent trace and
writes `generated_skills/minigrid_skill.py`.

## 4. Harder Farama Environments

```bash
uv run zima-minigrid --env MiniGrid-LavaCrossingS9N1-v0 --think-after 12 --max-steps 160
uv run zima-minigrid --env MiniGrid-Dynamic-Obstacles-8x8-v0 --think-after 12 --max-steps 160
```

These are better benchmark candidates once the harness is running locally.
