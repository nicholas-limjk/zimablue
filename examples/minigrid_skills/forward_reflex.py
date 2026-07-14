"""A deliberately small MiniGrid reflex skill.

This is not meant to solve every task. It is a sanity-check skill for the Python
harness: pick up keys, toggle doors, move forward when possible, and turn right
when blocked.
"""


def act(obs, memory, api):
    image = obs["image"]
    objects = image[:, :, 0]
    directions = image[:, :, 2] if image.shape[-1] > 2 else api.np.zeros_like(objects)

    agent_idx = api.object_to_idx.get("agent")
    wall_idx = api.object_to_idx.get("wall")
    lava_idx = api.object_to_idx.get("lava")
    door_idx = api.object_to_idx.get("door")
    key_idx = api.object_to_idx.get("key")

    positions = api.np.argwhere(objects == agent_idx)
    if len(positions) == 0:
        return api.actions["right"]

    y, x = positions[0]
    direction = int(directions[y, x]) if directions is not None else int(obs.get("direction", 0))
    delta_by_dir = {
        0: (0, 1),   # east
        1: (1, 0),   # south
        2: (0, -1),  # west
        3: (-1, 0),  # north
    }
    dy, dx = delta_by_dir.get(direction, (0, 1))
    fy, fx = y + dy, x + dx

    if fy < 0 or fx < 0 or fy >= objects.shape[0] or fx >= objects.shape[1]:
        return api.actions["right"]

    front = int(objects[fy, fx])
    if front == key_idx:
        return api.actions["pickup"]
    if front == door_idx:
        return api.actions["toggle"]
    if front in {wall_idx, lava_idx}:
        return api.actions["right"]
    return api.actions["forward"]
