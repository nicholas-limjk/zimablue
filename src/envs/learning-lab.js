export function createLearningLabOptions() {
  return {
    goal: 1,
    width: 22,
    height: 12,
    turnLimit: 76,
    agent: { x: 1, y: 1 },
    target: { x: 21, y: 6 },
    fruits: [],
    bushes: [],
    trees: [],
    keys: [{ x: 2, y: 10 }],
    gate: { x: 6, y: 5, open: false },
    logicGate: { x: 16, y: 6, open: false, activated: [] },
    logicRule: {
      clue: "Open the final seal by activating the prime-valued plates whose total is 40, in ascending value order.",
      targetSum: 40
    },
    clues: [
      {
        id: "prime-tablet",
        x: 15,
        y: 6,
        text: "The seal listens to prime plates. Make forty, smallest first."
      }
    ],
    switches: [
      { id: "A", x: 11, y: 10, value: 21, kind: "composite" },
      { id: "B", x: 12, y: 4, value: 17, kind: "prime" },
      { id: "C", x: 12, y: 8, value: 23, kind: "prime" }
    ],
    grove: { x: 0, y: 11, cycles: 0 },
    rocks: [
      ...verticalWall(6, 0, 11, 5),
      ...verticalWall(16, 0, 11, 6),
      ...points([
        [3, 2],
        [3, 3],
        [3, 7],
        [3, 8],
        [8, 1],
        [9, 1],
        [10, 1],
        [11, 1],
        [9, 2],
        [9, 3],
        [9, 9],
        [10, 9],
        [11, 9],
        [12, 9],
        [13, 9],
        [14, 9],
        [14, 2],
        [14, 3],
        [12, 10],
        [13, 10]
      ])
    ],
    wind: [
      ...rectPoints(8, 3, 8, 6),
      ...rectPoints(17, 4, 4, 5)
    ],
    water: []
  };
}

export function describeLearningLab() {
  return {
    id: "MiniGrid-LearningLab-style",
    family: "MiniGrid-inspired",
    task:
      "Open the locked door, then train a movement controller to cross an unstable dynamics field and reach the goal.",
    requiredCapabilities: [
      "invent route search",
      "collect key",
      "open locked door",
      "train movement controller",
      "reach target"
    ]
  };
}

function verticalWall(x, fromY, toY, openingY) {
  const points = [];
  for (let y = fromY; y <= toY; y += 1) {
    if (y !== openingY) points.push({ x, y });
  }
  return points;
}

function rectPoints(x, y, width, height) {
  const points = [];
  for (let dx = 0; dx < width; dx += 1) {
    for (let dy = 0; dy < height; dy += 1) {
      points.push({ x: x + dx, y: y + dy });
    }
  }
  return points;
}

function points(coordinates) {
  return coordinates.map(([x, y]) => ({ x, y }));
}
