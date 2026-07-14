export function createMiniGridDoorKeyOptions({
  size = 8,
  turnLimit = 35,
  key = { x: 1, y: 5 },
  agent = { x: 1, y: 1 },
  gateY = 3,
  target = { x: 6, y: 6 }
} = {}) {
  const wallX = Math.floor(size / 2);
  const rocks = [];

  for (let y = 0; y < size; y += 1) {
    if (y !== gateY) {
      rocks.push({ x: wallX, y });
    }
  }

  return {
    goal: 1,
    width: size,
    height: size,
    turnLimit,
    agent,
    target,
    fruits: [],
    bushes: [],
    trees: [],
    keys: [key],
    gate: { x: wallX, y: gateY, open: false },
    grove: { x: 0, y: size - 1, cycles: 0 },
    rocks,
    water: []
  };
}

export function describeMiniGridDoorKey() {
  return {
    id: "MiniGrid-DoorKey-style",
    family: "MiniGrid",
    task: "Use the key to open the locked door, then reach the target.",
    requiredCapabilities: [
      "pathfind through rooms",
      "collect key",
      "open locked door",
      "reach target"
    ]
  };
}
