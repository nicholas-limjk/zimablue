# LinkedIn Draft: MiniGrid Skill Reflection

In the science fiction novel Zima Blue and Other Stories - a simple cleaning robot (won't go into too much detail because of spoilers) is progressively upgraded, eventually gaining sentience.

Through its various upgrades the base simple impulse however persists. This has been bouncing around my head for awhile, alongside a bunch of things I like from evolutionary computing, Sakana AI's work on open-ended / self-improving systems, and David Ha's older work around world models and creative agent design.

So I made a small toy version in MiniGrid.

The agent starts with a simple impulse: reach the goal. It begins with a weak policy, fails, and then a reflection loop asks GPT-5.5 to write a new Python skill for the next run.

The important bit: the model is not controlling the agent step by step. It writes code. The body then executes that code turn by turn using only partial egocentric observations, its facing direction, the object in front of it, recent action outcomes, and memory.

A generated skill looks roughly like this:

```python
def act(obs, memory, api):
    actions = api.actions
    state = memory.setdefault("learned_structure", {})
    state.setdefault("map", {})
    state.setdefault("objects", {"key": {}, "door": {}, "goal": {}})

    cells = api.egocentric_cells(obs)
    front = api.front_object(obs)

    # remember what the body can currently see
    # update hypotheses about keys, doors, goals, walls
    # choose one primitive MiniGrid action

    if front["object"] == "key":
        return actions["pickup"]
    if front["object"] == "door" and state.get("has_key"):
        return actions["toggle"]
    if front["object"] == "goal":
        return actions["forward"]

    return choose_next_action_from_memory(state, actions)
```

In the demo, an early reflected skill learns DoorKey behaviour: find key, open door, reach goal. Useful, but narrow. It fails when the environment is just asking for navigation.

After curriculum reflection, the skill becomes more general: build a discovered map, explore frontiers, track terminal cues, and use finite search. It then transfers to unseen MiniGrid environments like MultiRoom, FourRooms, and SimpleCrossing.

It still fails on harder tasks with more complicated locked / blocked dependencies. But that is the point: the failure trajectory can be fed back into the loop and used to write a better skill.

Right now the skills are code because code is inspectable. Eventually I want the reflection step to choose what should be learned instead: maybe past frames, action history, a remembered map, object inventory, or a small neural policy trained from selected inputs.

The question I keep coming back to:

What if an agent is less like one fixed policy, and more like a growing library of skills written after its own failures?

Video demo: `artifacts/minigrid-nav-sequence-labelled.mp4`
