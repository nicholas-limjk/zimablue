const baseBehaviors = [
  {
    name: "body:reach-target-here",
    kind: "body",
    priority: 0,
    code: "if target is underfoot: stop",
    run(state) {
      if (!sameCell(state.target, state.agent)) return { ok: false, reason: "target not here" };
      state.targetReached = true;
      return { ok: true, message: "The base body reached the target." };
    }
  },
  {
    name: "body:random-step",
    kind: "body",
    priority: 100,
    code: "try one random adjacent move",
    run(state) {
      return randomStep(state);
    }
  }
];

const SKILL_LIBRARY_KEY = "zima-blue-agent-skill-library-v7-random-maze";
const STEP_INTERVAL_MS = new URLSearchParams(location.search).has("record") ? 120 : 650;

const thinkerPatches = [
  {
    name: "maze:solve-route",
    kind: "self-written",
    action: "solveMaze",
    priority: 10,
    reason: "The body hit maze walls. Write a graph-search solver over the maze map.",
    code: "read api.maze(); search passable cells; api.moveTo(next)",
    run(state) {
      return solveMazeFallback(state);
    }
  }
];

const mazeRows = [
  "#######################",
  "#S..#.....#.....#.....#",
  "###.#.###.#.###.#.###.#",
  "#...#.#...#...#...#...#",
  "#.###.#.#####.#####.#.#",
  "#.....#.....#.....#.#.#",
  "#.#########.#.###.#.#.#",
  "#...#.....#.#...#.#.#.#",
  "###.#.###.#.###.#.#.#.#",
  "#...#.#...#...#...#.#.#",
  "#.###.#.#####.#####.#.#",
  "#.....#.............#T#",
  "#######################"
];

const mazeLayout = parseMaze(mazeRows);

const initial = {
  width: mazeLayout.width,
  height: mazeLayout.height,
  goal: 1,
  fruitCount: 0,
  targetReached: false,
  agent: mazeLayout.agent,
  target: mazeLayout.target,
  fruits: [],
  bushes: [],
  trees: [],
  keys: [],
  gate: { x: -1, y: -1, open: true },
  logicGate: { x: -1, y: -1, open: true, activated: [] },
  logicRule: null,
  clues: [],
  switches: [],
  grove: { x: -1, y: -1, cycles: 0 },
  turnLimit: 68,
  rocks: mazeLayout.rocks,
  windZones: [],
  wind: [],
  water: [],
  inventory: { key: false, bridge: false, neuralController: false, mazeSolver: false },
  memory: {
    lockedGateObserved: false,
    logicGateObserved: false,
    mazeObserved: false,
    randomSeed: 7,
    mechanismAttempts: [],
    movementSamples: [],
    cluesRead: []
  },
  tick: 0,
  failed: false,
  stuck: false,
  done: false,
  running: true,
  thinking: false,
  thinkingAction: null,
  lastBehavior: null,
  lastPatch: null,
  program: [],
  trace: []
};

const elements = {
  gridWorld: document.querySelector("#gridWorld"),
  fruitMetric: document.querySelector("#fruitMetric"),
  agentMetric: document.querySelector("#agentMetric"),
  stateMetric: document.querySelector("#stateMetric"),
  agentCore: document.querySelector("#agentCore"),
  impulse: document.querySelector("#impulse"),
  equationValue: document.querySelector("#equationValue"),
  realizations: document.querySelector("#realizations"),
  program: document.querySelector("#program"),
  trace: document.querySelector("#trace"),
  stepButton: document.querySelector("#stepButton"),
  runButton: document.querySelector("#runButton"),
  resetButton: document.querySelector("#resetButton"),
  clearSkillsButton: document.querySelector("#clearSkillsButton")
};

let state = cloneInitial();
let timer = null;
let stepping = false;

function cloneInitial() {
  return {
    ...initial,
    agent: { ...initial.agent },
    target: { ...initial.target },
    fruits: initial.fruits.map(clonePoint),
    bushes: initial.bushes.map((bush) => ({ ...bush })),
    trees: initial.trees.map((tree) => ({ ...tree })),
    keys: initial.keys.map(clonePoint),
    gate: { ...initial.gate },
    logicGate: { ...initial.logicGate, activated: [...initial.logicGate.activated] },
    logicRule: initial.logicRule ? { ...initial.logicRule } : null,
    clues: initial.clues.map((clue) => ({ ...clue })),
    switches: initial.switches.map((item) => ({ ...item })),
    grove: { ...initial.grove },
    rocks: initial.rocks.map(clonePoint),
    wind: initial.wind.map(clonePoint),
    windZones: initial.windZones.map((zone) => ({
      ...zone,
      points: zone.points.map(clonePoint)
    })),
    water: initial.water.map(clonePoint),
    inventory: { ...initial.inventory },
    memory: {
      ...initial.memory,
      mechanismAttempts: initial.memory.mechanismAttempts.map((attempt) => ({ ...attempt })),
      movementSamples: initial.memory.movementSamples.map(cloneMovementSample),
      cluesRead: initial.memory.cluesRead.map((clue) => ({ ...clue }))
    },
    program: buildInitialProgram(),
    trace: []
  };
}

function buildInitialProgram() {
  const program = [
    ...loadSkillLibrary().map(compileRememberedSkill).filter(Boolean),
    ...baseBehaviors.map((behavior) => ({ ...behavior }))
  ];

  return program.sort((left, right) => left.priority - right.priority);
}

function loadSkillLibrary() {
  try {
    const parsed = JSON.parse(localStorage.getItem(SKILL_LIBRARY_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveSkillSpec(spec) {
  if (!spec || typeof spec !== "object") return;

  const existing = loadSkillLibrary();
  if (existing.some((item) => item.action === spec.action || item.name === spec.name)) return;

  const storable = {
    name: spec.name,
    action: spec.action,
    reason: spec.reason,
    code: spec.code,
    source: spec.source,
    priority: spec.priority,
    plan: spec.plan
  };
  localStorage.setItem(SKILL_LIBRARY_KEY, JSON.stringify([...existing, storable]));
}

function rememberDemonstratedSkill(behavior) {
  if (behavior.kind !== "agent-written" || !behavior.spec) return;
  saveSkillSpec(behavior.spec);
}

function compileRememberedSkill(spec) {
  if (!spec || typeof spec !== "object") return null;
  const template = thinkerPatches.find((patch) => patch.action === spec.action);
  if (!template) return null;
  const run = compileAgentSkillSource(spec, template);
  if (!run) return null;

  return {
    ...template,
    name: spec.name,
    kind: "remembered",
    reason: typeof spec.reason === "string" ? spec.reason : template.reason,
    code: typeof spec.code === "string" ? spec.code : template.code,
    source: typeof spec.source === "string" ? spec.source : template.source,
    priority: Number.isInteger(spec.priority) ? spec.priority : template.priority,
    plan: spec.plan && typeof spec.plan === "object" ? spec.plan : template.plan,
    run
  };
}

function impulse() {
  return state.targetReached ? 0 : 1;
}

async function step() {
  if (state.done || stepping) return;
  stepping = true;

  try {
    const currentImpulse = impulse();
    state.lastBehavior = null;
    state.lastPatch = null;

    if (currentImpulse === 0) {
      addTrace("complete", null, "The base impulse became 0. Rest.");
      state.done = true;
      stop();
      render();
      return;
    }

    const outcome = runProgram();
    const reflection = await think(outcome);

    addTrace(outcome.status, outcome.behavior, outcome.message, reflection, outcome.failures);
    state.tick += 1;

    if (outcome.status === "no-action" && !reflection.patched && !hasRecoverableFailure(outcome)) {
      state.stuck = true;
      state.done = true;
      stop();
    }

    if (!state.done && impulse() === 1 && state.tick >= state.turnLimit) {
      state.failed = true;
      state.done = true;
      addTrace("failed", null, "The world clock expired before the target was reached.");
      stop();
    }

    render();
  } finally {
    stepping = false;
  }
}

function runProgram() {
  const failures = [];

  for (const behavior of state.program) {
    if (behavior.disabled) continue;
    const result = behavior.run(state, behavior);
    if (result.ok) {
      rememberDemonstratedSkill(behavior, result);
      state.lastBehavior = behavior.name;
      return {
        status: "acted",
        behavior: behavior.name,
        message: result.message,
        failures
      };
    }
    if (behavior.action === "executeProcedure" && result.reason === "mechanism rejected procedure") {
      behavior.disabled = true;
    }
    failures.push(`${behavior.name}: ${result.reason}`);
  }

  return {
    status: "no-action",
    behavior: null,
    message: "Current behavior code could not satisfy reach target.",
    failures
  };
}

async function think(outcome) {
  const usefulActions = usefulActionsNow(outcome);
  if (usefulActions.length === 0) return { patched: false, message: "No patch." };

  state.thinking = true;
  state.thinkingAction = usefulActions[0];
  render();

  const thinkingOutcome = spendThinkingTurn();
  addTrace("thinking", null, thinkingOutcome.message, null, thinkingOutcome.failures);
  if (state.failed) {
    state.thinking = false;
    state.thinkingAction = null;
    return { patched: false, message: "Thinking consumed the last available turn." };
  }

  let agentPatch;
  try {
    agentPatch = await proposeAgenticPatch(outcome, usefulActions);
  } finally {
    state.thinking = false;
    state.thinkingAction = null;
  }

  const patch = agentPatch.patch;

  if (!patch) {
    return {
      patched: false,
      message: agentPatch.error ?? "The skill-writing agent did not produce executable source.",
      source: agentPatch.source,
      spec: agentPatch.spec,
      error: agentPatch.error
    };
  }

  state.program.push({ ...patch });
  if (agentPatch.source === "agent") {
    state.program = state.program.map((behavior) =>
      behavior.name === patch.name ? { ...behavior, spec: agentPatch.spec } : behavior
    );
  }
  state.program.sort((left, right) => left.priority - right.priority);
  state.lastPatch = patch.name;
  return {
    patched: true,
    behavior: patch.name,
    message: agentPatch.source === "agent" ? `${patch.reason} [agent-written]` : patch.reason,
    source: agentPatch.source,
    spec: agentPatch.spec,
    error: agentPatch.error
  };
}

function spendThinkingTurn() {
  const failures = [];
  let message = `Thinking about ${state.thinkingAction}. The world clock advanced.`;

  state.tick += 1;
  if (!state.done && impulse() === 1 && state.tick >= state.turnLimit) {
    state.failed = true;
    state.done = true;
    stop();
  }

  return { message, failures };
}

function addTrace(status, behavior, message, reflection = null, failures = []) {
  state.trace.push({
    tick: state.tick,
    status,
    behavior,
    message,
    reflection,
    failures,
    movementSamples: state.memory.movementSamples.length,
    mechanismAttempts: state.memory.mechanismAttempts.map((attempt) => ({ ...attempt })),
    cluesRead: state.memory.cluesRead.map((clue) => ({ ...clue })),
    impulse: impulse()
  });
}

function render() {
  renderWorld();
  renderMind();
  renderRealizations();
  renderProgram();
  renderTrace();
}

function renderWorld() {
  elements.gridWorld.style.setProperty("--cols", state.width);
  elements.gridWorld.style.setProperty("--rows", state.height);
  elements.gridWorld.innerHTML = "";

  for (let y = 0; y < state.height; y += 1) {
    for (let x = 0; x < state.width; x += 1) {
      const point = { x, y };
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.x = x;
      cell.dataset.y = y;

      markTerrain(cell, point);

      const isTarget = sameCell(state.target, point);
      const bush = state.bushes.find((candidate) => sameCell(candidate, point));
      const tree = state.trees.find((candidate) => sameCell(candidate, point));
      const hasKey = state.keys.some((key) => sameCell(key, point));
      const clue = state.clues.find((candidate) => sameCell(candidate, point));
      const plate = state.switches.find((candidate) => sameCell(candidate, point));

      if (isTarget) {
        cell.classList.add("target-cell");
        if (state.targetReached) cell.classList.add("target-reached-cell");
        cell.dataset.count = 1;
      }

      if (bush) {
        cell.classList.add("bush-cell");
        cell.dataset.count = bush.fruit;
        cell.textContent = "B";
      }

      if (tree) {
        cell.classList.add("tree-cell");
        cell.dataset.count = tree.fruit;
        cell.textContent = "T";
      }

      if (hasKey) {
        cell.classList.add("key-cell");
        cell.textContent = "K";
      }

      if (clue) {
        cell.classList.add(clueRead(clue.id) ? "clue-read-cell" : "clue-cell");
        cell.textContent = "?";
        cell.title = clueRead(clue.id) ? clue.text : "Unread clue";
      }

      if (plate) {
        cell.classList.add(state.logicGate.activated.includes(plate.id) ? "switch-active-cell" : "switch-cell");
        cell.textContent = "";
        const switchIcon = document.createElement("span");
        switchIcon.className = "switch-icon";
        switchIcon.textContent = plate.id;
        const switchValue = document.createElement("span");
        switchValue.className = "switch-value";
        switchValue.textContent = String(plate.value);
        cell.append(switchIcon, switchValue);
      }

      if (sameCell(state.gate, point)) {
        cell.classList.add(state.gate.open ? "gate-open-cell" : "gate-cell");
        cell.textContent = state.gate.open ? "O" : "G";
      }

      if (sameCell(state.logicGate, point)) {
        cell.classList.add(state.logicGate.open ? "logic-open-cell" : "logic-cell");
        cell.textContent = "";
        const logicIcon = document.createElement("span");
        logicIcon.className = "logic-icon";
        logicIcon.textContent = state.logicGate.open ? "OPEN" : "LOCK";
        cell.append(logicIcon);
      }

      if (sameCell(state.grove, point)) {
        cell.classList.add("grove-cell");
        cell.dataset.count = state.grove.cycles;
        cell.textContent = "M";
      }

      if (isTarget) {
        const target = document.createElement("span");
        target.className = "target-beacon";
        target.textContent = "*";
        cell.append(target);
      }

      if (sameCell(state.agent, point)) {
        cell.classList.add("has-agent");
        const agent = document.createElement("span");
        agent.className = "agent-dot";
        agent.textContent = "A";
        cell.append(agent);
      }

      elements.gridWorld.append(cell);
    }
  }

  const inventory = [
    state.program.some((behavior) => behavior.action === "solveMaze") ? "maze-solver" : null,
    state.inventory.bridge ? "bridge" : null
  ].filter(Boolean);
  const rememberedCount = state.program.filter((behavior) => behavior.kind === "remembered").length;

  elements.fruitMetric.textContent = `${state.targetReached ? 1 : 0}/1`;
  elements.agentMetric.textContent = inventory.length
    ? `${inventory.join("+")} / skills:${rememberedCount}`
    : `(${state.agent.x},${state.agent.y}) / skills:${rememberedCount}`;
  elements.stateMetric.textContent = state.failed
    ? `Failed ${state.tick}/${state.turnLimit}`
    : state.stuck
      ? `Stuck ${state.tick}/${state.turnLimit}`
    : state.done
      ? `Resting ${state.tick}/${state.turnLimit}`
      : state.thinking
        ? `Thinking ${state.tick}/${state.turnLimit}`
        : `${state.tick}/${state.turnLimit}`;
}

function markTerrain(cell, point) {
  if (state.rocks.some((rock) => sameCell(rock, point))) {
    cell.classList.add("rock-cell");
  }

  if (state.water.some((water) => sameCell(water, point))) {
    cell.classList.add(state.inventory.bridge ? "bridged-cell" : "water-cell");
    cell.textContent = state.inventory.bridge ? "=" : "W";
  }
}

function renderMind() {
  const currentImpulse = impulse();
  elements.impulse.textContent = String(currentImpulse);
  elements.equationValue.textContent = currentImpulse === 1 ? "true" : "false";
  elements.agentCore.classList.toggle("rest", currentImpulse === 0);
  elements.agentCore.classList.toggle("failed", state.failed);
  elements.agentCore.classList.toggle("thinking", state.thinking);
  elements.agentCore.title = state.thinking
    ? `Thinking about ${state.thinkingAction ?? "a new skill"}`
    : "Primitive impulse";
  elements.agentCore.classList.remove("pulse");
  requestAnimationFrame(() => elements.agentCore.classList.add("pulse"));
  elements.runButton.textContent = state.running ? "Pause" : "Run";
}

function renderProgram() {
  elements.program.innerHTML = "";

  for (const behavior of state.program) {
    const row = document.createElement("article");
    row.className = "code-line";
    row.classList.toggle("active", state.lastBehavior === behavior.name);
    row.classList.toggle("patched", state.lastPatch === behavior.name);
    row.classList.toggle("disabled", Boolean(behavior.disabled));
    row.innerHTML = `
      <div>
        <strong>${behavior.name}</strong>
        <span>${behavior.kind}</span>
      </div>
      <code>${behavior.code}</code>
    `;
    elements.program.append(row);
  }
}

function renderRealizations() {
  elements.realizations.innerHTML = "";

  for (const item of currentRealizations()) {
    const row = document.createElement("li");
    row.className = item.active ? "realized" : "pending";
    row.innerHTML = `
      <span class="realization-icon">${item.active ? "!" : "..."}</span>
      <div>
        <strong>${item.title}</strong>
        <span>${item.detail}</span>
      </div>
    `;
    elements.realizations.append(row);
  }
}

function currentRealizations() {
  const thinkingTurns = state.trace.filter((entry) => entry.status === "thinking").length;
  const rememberedCount = state.program.filter((behavior) => behavior.kind === "remembered").length;
  const mazeSkill = state.program.find((behavior) => behavior.action === "solveMaze");
  const wallFailures = state.trace.filter((entry) =>
    entry.failures?.some((failure) => failure.includes("rock blocked") || failure.includes("edge of world"))
  ).length;
  const mazeSteps = state.trace.filter((entry) => entry.behavior === mazeSkill?.name).length;

  return [
    {
      title: "Skills persist across rounds",
      detail: rememberedCount > 0
        ? `${rememberedCount} remembered skill(s) loaded before this run.`
        : "No remembered skills yet; this is a cold run.",
      active: rememberedCount > 0
    },
    {
      title: "Target is not underfoot",
      detail: "The base impulse keeps firing until the agent reaches the beacon.",
      active: !state.targetReached
    },
    {
      title: "The body has no maze concept",
      detail: wallFailures > 0
        ? `${wallFailures} blocked step(s) made the corridor structure matter.`
        : "The body only tries random adjacent moves until it collides with the maze.",
      active: wallFailures > 0
    },
    {
      title: "Thinking is not outside the world",
      detail: thinkingTurns > 0
        ? `${thinkingTurns} thinking turn(s) consumed clock time before the solver could act.`
        : "No thinking turn has been spent yet.",
      active: thinkingTurns > 0
    },
    {
      title: "The world is a maze",
      detail: mazeSkill
        ? "The LLM wrote a solver that reads the maze map and searches passable cells."
        : "Pending until the thinker infers that wall failures call for graph search.",
      active: Boolean(mazeSkill)
    },
    {
      title: "The solver acts one move per turn",
      detail: mazeSteps > 0
        ? `${mazeSteps} maze-solver step(s) executed under the clock.`
        : "A complete route plan still has to be cashed out as adjacent moves.",
      active: mazeSteps > 0
    },
    {
      title: "Reach the beacon before timeout",
      detail: state.targetReached
        ? `Solved at t=${state.tick}/${state.turnLimit}.`
        : `Clock: ${state.tick}/${state.turnLimit}.`,
      active: state.targetReached
    }
  ];
}

function renderTrace() {
  elements.trace.innerHTML = "";

  for (const entry of state.trace.slice(-24)) {
    const item = document.createElement("li");
    item.className = entry.reflection?.patched ? "patched" : entry.status;
    item.innerHTML = `
      <strong>t=${entry.tick} impulse=${entry.impulse}</strong>
      ${entry.behavior ? ` via ${entry.behavior}` : ""}
      <br>${entry.message}
      ${
        entry.failures?.length
          ? `<br><span>failed against: ${summarizeFailure(entry.failures)}</span>`
          : ""
      }
      ${
        entry.reflection?.patched
          ? `<br><strong>thinker patched:</strong> ${entry.reflection.behavior}`
          : ""
      }
      ${
        entry.reflection?.spec
          ? `<pre class="llm-output">${escapeHtml(JSON.stringify(entry.reflection.spec, null, 2))}</pre>`
          : ""
      }
      ${
        entry.reflection?.behavior === "learner:movement-controller" ||
        entry.reflection?.spec?.action === "trainMovementController"
          ? `<br><span>diagnosis: ${entry.movementSamples} intended/actual mismatch samples</span>`
          : ""
      }
      ${
        entry.mechanismAttempts?.length
          ? `<br><span>mechanism evidence: ${summarizeMechanismAttempts(entry.mechanismAttempts)}</span>`
          : ""
      }
      ${
        entry.cluesRead?.length
          ? `<br><span>clue memory: ${entry.cluesRead.at(-1).text}</span>`
          : ""
      }
    `;
    elements.trace.append(item);
  }
}

function summarizeMechanismAttempts(attempts) {
  return attempts
    .slice(-2)
    .map((attempt) => `${attempt.result} [${attempt.proposed.join("->")}]`)
    .join(", ");
}

function summarizeFailure(failures) {
  return (
    failures.find((failure) => failure.startsWith("nav:route-search")) ??
    failures.find((failure) => failure.includes("movement-controller")) ??
    failures[0]
  );
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function start() {
  if (timer || state.done) return;
  state.running = true;
  timer = setInterval(step, STEP_INTERVAL_MS);
  render();
}

function stop() {
  clearInterval(timer);
  timer = null;
  state.running = false;
}

function move(targetState, dx, dy, message) {
  const next = { x: targetState.agent.x + dx, y: targetState.agent.y + dy };
  const blocked = blockedReason(targetState, next);
  if (blocked) return { ok: false, reason: blocked };

  const actual = applyMovementDynamics(targetState, targetState.agent, next);
  targetState.agent = actual;

  if (!sameCell(actual, next)) {
    return { ok: false, reason: "movement model mismatch" };
  }

  return { ok: true, message };
}

function randomStep(targetState) {
  const directions = [
    { dx: 1, dy: 0, label: "east" },
    { dx: -1, dy: 0, label: "west" },
    { dx: 0, dy: 1, label: "south" },
    { dx: 0, dy: -1, label: "north" }
  ];
  targetState.memory.randomSeed = nextRandomSeed(targetState.memory.randomSeed);
  const direction = directions[(targetState.memory.randomSeed >>> 16) % directions.length];
  return move(
    targetState,
    direction.dx,
    direction.dy,
    `The base body randomly tried ${direction.label}.`
  );
}

function nextRandomSeed(seed) {
  return (Math.imul(seed ?? 7, 1664525) + 1013904223) >>> 0;
}

function seekTarget(targetState) {
  const hadLockedGateObservation = targetState.memory.lockedGateObserved;
  const hadLogicGateObservation = targetState.memory.logicGateObserved;
  const path = pathTo(targetState, targetState.target);
  if (path.length === 0) {
    if (
      (!hadLogicGateObservation && targetState.memory.logicGateObserved) ||
      (targetState.memory.logicGateObserved && !targetState.logicGate.open)
    ) {
      return { ok: false, reason: "logic gate locked" };
    }

    if (
      (!hadLockedGateObservation && targetState.memory.lockedGateObserved) ||
      (targetState.memory.lockedGateObserved && !targetState.gate.open)
    ) {
      return { ok: false, reason: "gate locked" };
    }

    return { ok: false, reason: "target unreachable" };
  }
  if (path.length < 2) return { ok: false, reason: "already there" };

  const intended = path[1];
  const actual = applyMovementDynamics(targetState, targetState.agent, intended);
  targetState.agent = actual;

  if (!sameCell(actual, intended)) {
    return { ok: false, reason: "movement model mismatch" };
  }

  return { ok: true, message: "The self-written route-search skill moved toward the target." };
}

function solveMazeFallback(targetState) {
  const path = pathTo(targetState, targetState.target);
  if (path.length < 2) return { ok: false, reason: path.length === 1 ? "already there" : "maze path not found" };
  return moveToMazeCell(targetState, path[1]);
}

function moveToMazeCell(targetState, point) {
  if (!point || !Number.isInteger(point.x) || !Number.isInteger(point.y)) {
    return { ok: false, reason: "maze solver returned invalid point" };
  }

  const intended = { x: point.x, y: point.y };
  if (distance(targetState.agent, intended) !== 1) {
    return { ok: false, reason: "maze solver must move to an adjacent cell" };
  }

  const blocked = blockedReason(targetState, intended);
  if (blocked) return { ok: false, reason: blocked };

  targetState.agent = intended;
  targetState.inventory.mazeSolver = true;
  return { ok: true, message: "The LLM-written maze solver advanced one cell along its computed route." };
}

function mazeSnapshot(targetState) {
  return {
    width: targetState.width,
    height: targetState.height,
    agent: clonePoint(targetState.agent),
    target: clonePoint(targetState.target),
    walls: targetState.rocks.map(clonePoint)
  };
}

function harvestBushOrApproach(targetState) {
  const target = nearestReachable(targetState, targetState.bushes.filter((bush) => bush.fruit > 0));
  if (!target) return { ok: false, reason: "no reachable bush" };

  if (distance(targetState.agent, target) <= 1) {
    const dropped = Math.min(2, target.fruit);
    target.fruit -= dropped;
    for (let index = 0; index < dropped; index += 1) targetState.fruits.push({ x: target.x, y: target.y });
    return { ok: true, message: "The self-written bush skill shook berries loose." };
  }

  return stepAlongPath(targetState, target, "The bush skill moved toward berry cover.");
}

function collectKeyOrApproach(targetState) {
  if (targetState.inventory.key) return { ok: false, reason: "key already held" };
  const target = nearestReachable(targetState, targetState.keys);
  if (!target) return { ok: false, reason: "no reachable key" };

  if (sameCell(targetState.agent, target)) {
    targetState.inventory.key = true;
    targetState.keys = targetState.keys.filter((key) => !sameCell(key, target));
    return { ok: true, message: "The self-written key skill picked up a gate key." };
  }

  return stepAlongPath(targetState, target, "The key skill moved toward the gate key.");
}

function executeProcedure(targetState, behavior) {
  if (targetState.logicGate.open) return { ok: false, reason: "mechanism already open" };
  if (!targetState.memory.logicGateObserved) return { ok: false, reason: "mechanism not observed" };
  if (targetState.memory.mechanismAttempts.length === 0) return { ok: false, reason: "mechanism not probed" };
  if (targetState.memory.cluesRead.length === 0) return { ok: false, reason: "clue not read" };

  const proposed = behavior?.plan?.switches;
  if (!Array.isArray(proposed) || proposed.length === 0) {
    return { ok: false, reason: "missing switch procedure" };
  }

  const nextSwitch = proposed.find((id) => !targetState.logicGate.activated.includes(id));
  if (!nextSwitch) return { ok: false, reason: "mechanism unsolved" };

  const plate = targetState.switches.find((candidate) => candidate.id === nextSwitch);
  if (!plate) return { ok: false, reason: "missing switch" };

  if (sameCell(targetState.agent, plate)) {
    targetState.logicGate.activated.push(plate.id);

    if (targetState.logicGate.activated.length >= proposed.length) {
      if (!sameSequence(proposed, mechanismSolution(targetState))) {
        targetState.logicGate.activated = [];
        targetState.memory.mechanismAttempts.push({
          tick: targetState.tick,
          proposed: [...proposed],
          result: "rejected"
        });
        return { ok: false, reason: "mechanism rejected procedure" };
      }

      targetState.logicGate.open = true;
      return { ok: true, message: "The agent executed its inferred switch procedure and opened the logic gate." };
    }

    return { ok: true, message: `The mechanism procedure activated switch ${plate.id}.` };
  }

  return stepAlongPath(targetState, plate, "The mechanism procedure moved toward its next switch.");
}

function readClueOrApproach(targetState) {
  if (targetState.logicGate.open) return { ok: false, reason: "mechanism already open" };
  if (!targetState.memory.logicGateObserved) return { ok: false, reason: "mechanism not observed" };
  if (targetState.memory.mechanismAttempts.length === 0) return { ok: false, reason: "mechanism not probed" };
  const unread = targetState.clues.filter((clue) => !clueRead(clue.id, targetState));
  if (unread.length === 0) return { ok: false, reason: "clue already read" };

  const clue = nearestReachable(targetState, unread);
  if (!clue) return { ok: false, reason: "no reachable clue" };

  if (sameCell(targetState.agent, clue)) {
    targetState.memory.cluesRead.push({
      id: clue.id,
      text: clue.text,
      tick: targetState.tick
    });
    return { ok: true, message: `The agent read the clue tablet: "${clue.text}"` };
  }

  return stepAlongPath(targetState, clue, "The clue-search skill moved toward a written rule.");
}

function probeMechanism(targetState) {
  if (targetState.logicGate.open) return { ok: false, reason: "mechanism already open" };
  if (!targetState.memory.logicGateObserved) return { ok: false, reason: "mechanism not observed" };
  if (targetState.memory.mechanismAttempts.length > 0) return { ok: false, reason: "mechanism probe already recorded" };

  const plate = nearestReachable(targetState, targetState.switches);
  if (!plate) return { ok: false, reason: "no reachable switch" };

  if (sameCell(targetState.agent, plate)) {
    const proposed = [plate.id];
    targetState.logicGate.activated = [];
    targetState.memory.mechanismAttempts.push({
      tick: targetState.tick,
      proposed,
      result: "rejected"
    });
    return {
      ok: true,
      message: `The agent tried switch ${plate.id} alone; the mechanism rejected the incomplete procedure.`
    };
  }

  return stepAlongPath(targetState, plate, "The probe moved toward a visible switch to test the mechanism.");
}

function openGateOrApproach(targetState) {
  if (targetState.gate.open) return { ok: false, reason: "gate already open" };
  if (!targetState.inventory.key) return { ok: false, reason: "no key" };

  if (distance(targetState.agent, targetState.gate) <= 1) {
    targetState.gate.open = true;
    return { ok: true, message: "The self-written gate skill opened the locked passage." };
  }

  return stepAdjacentTo(targetState, targetState.gate, "The gate skill moved toward the lock.");
}

function buildBridgeOrApproach(targetState) {
  if (targetState.inventory.bridge) return { ok: false, reason: "bridge already built" };
  const target = targetState.water[1] ?? targetState.water[0];
  if (!target) return { ok: false, reason: "no water" };

  if (distance(targetState.agent, target) <= 1) {
    targetState.inventory.bridge = true;
    return { ok: true, message: "The self-written bridge skill made water passable." };
  }

  return stepAdjacentTo(targetState, target, "The bridge skill moved toward the stream.");
}

function trainMovementController(targetState) {
  if (targetState.inventory.neuralController) return { ok: false, reason: "controller already trained" };
  if (targetState.wind.length === 0) return { ok: false, reason: "no unstable field" };
  if (!hasMovementDiagnosis()) return { ok: false, reason: "insufficient movement evidence" };
  targetState.inventory.neuralController = true;
  return { ok: true, message: "The agent fit a tiny controller from its own failed movement samples." };
}

function shakeTreeOrApproach(targetState) {
  const target = nearestReachable(targetState, targetState.trees.filter((tree) => tree.fruit > 0));
  if (!target) return { ok: false, reason: "no reachable tree" };

  if (distance(targetState.agent, target) <= 1) {
    const dropped = Math.min(3, target.fruit);
    target.fruit -= dropped;
    for (let index = 0; index < dropped; index += 1) targetState.fruits.push({ x: target.x - index, y: target.y });
    return { ok: true, message: "The self-written tree skill shook down orchard fruit." };
  }

  return stepAlongPath(targetState, target, "The tree skill moved toward the orchard.");
}

function askGrove(targetState) {
  if (targetState.grove.cycles <= 0) return { ok: false, reason: "grove empty" };
  targetState.grove.cycles -= 1;
  targetState.fruits.push(
    { x: targetState.grove.x, y: targetState.grove.y },
    { x: targetState.grove.x + 1, y: targetState.grove.y },
    { x: targetState.grove.x + 2, y: targetState.grove.y },
    { x: targetState.grove.x + 3, y: targetState.grove.y }
  );
  return { ok: true, message: "The self-written MCP adapter grew late fruit." };
}

function stepAlongPath(targetState, target, message) {
  const path = pathTo(targetState, target);
  if (path.length < 2) return { ok: false, reason: "already there" };

  const intended = path[1];
  const actual = applyMovementDynamics(targetState, targetState.agent, intended);
  targetState.agent = actual;

  if (!sameCell(actual, intended)) {
    return { ok: false, reason: "movement model mismatch" };
  }

  return { ok: true, message };
}

function stepAdjacentTo(targetState, target, message) {
  const adjacent = neighbors(targetState, target)
    .map((point) => ({ point, path: pathTo(targetState, point) }))
    .filter((candidate) => candidate.path.length > 0)
    .sort((left, right) => left.path.length - right.path.length)[0];

  if (!adjacent) return { ok: false, reason: "no adjacent path" };
  if (adjacent.path.length < 2) return { ok: false, reason: "already there" };

  const intended = adjacent.path[1];
  const actual = applyMovementDynamics(targetState, targetState.agent, intended);
  targetState.agent = actual;

  if (!sameCell(actual, intended)) {
    return { ok: false, reason: "movement model mismatch" };
  }

  return { ok: true, message };
}

function nearestReachable(targetState, points) {
  return points
    .map((point) => ({ point, path: pathTo(targetState, point) }))
    .filter((candidate) => candidate.path.length > 0)
    .sort((left, right) => left.path.length - right.path.length)[0]?.point;
}

function pathTo(targetState, target) {
  const queue = [[targetState.agent]];
  const seen = new Set([keyOf(targetState.agent)]);

  while (queue.length > 0) {
    const path = queue.shift();
    const current = path.at(-1);
    if (sameCell(current, target)) return path;

    for (const next of neighbors(targetState, current)) {
      const key = keyOf(next);
      if (seen.has(key)) continue;
      seen.add(key);
      queue.push([...path, next]);
    }
  }

  return [];
}

function neighbors(targetState, point) {
  return [
    { x: point.x + 1, y: point.y },
    { x: point.x - 1, y: point.y },
    { x: point.x, y: point.y + 1 },
    { x: point.x, y: point.y - 1 }
  ].filter((candidate) => !blockedReason(targetState, candidate));
}

function blockedReason(targetState, point) {
  if (point.x < 0 || point.y < 0 || point.x >= targetState.width || point.y >= targetState.height) {
    return "edge of world";
  }
  if (targetState.rocks.some((rock) => sameCell(rock, point))) {
    targetState.memory.mazeObserved = true;
    return "rock blocked";
  }
  if (!targetState.gate.open && sameCell(targetState.gate, point)) {
    targetState.memory.lockedGateObserved = true;
    return "gate locked";
  }
  if (!targetState.logicGate.open && sameCell(targetState.logicGate, point)) {
    targetState.memory.logicGateObserved = true;
    return "logic gate locked";
  }
  if (!targetState.inventory.bridge && targetState.water.some((water) => sameCell(water, point))) {
    return "water blocked";
  }
  return null;
}

function needsBushSkill() {
  return state.fruits.length === 0 && state.bushes.some((bush) => bush.fruit > 0);
}

function needsKeySkill() {
  return state.memory.lockedGateObserved && !state.inventory.key && state.keys.length > 0;
}

function needsGateSkill() {
  return state.memory.lockedGateObserved && state.inventory.key && !state.gate.open;
}

function needsMechanismSkill() {
  return (
    state.memory.logicGateObserved &&
    state.memory.mechanismAttempts.some((attempt) => attempt.result === "rejected") &&
    state.memory.cluesRead.length > 0 &&
    !state.logicGate.open
  );
}

function needsMechanismProbeSkill() {
  return state.memory.logicGateObserved && !state.logicGate.open && state.memory.mechanismAttempts.length === 0;
}

function needsMechanismClueSkill() {
  return (
    state.memory.logicGateObserved &&
    state.memory.mechanismAttempts.some((attempt) => attempt.result === "rejected") &&
    state.memory.cluesRead.length === 0 &&
    !state.logicGate.open
  );
}

function needsBridgeSkill() {
  return state.water.length > 0 && state.gate.open && !state.inventory.bridge;
}

function needsTreeSkill() {
  return state.inventory.bridge && state.trees.some((tree) => tree.fruit > 0);
}

function needsGroveSkill() {
  return (
    state.grove.cycles > 0 &&
    !state.targetReached &&
    state.fruits.length === 0 &&
    !needsBushSkill() &&
    !needsKeySkill() &&
    !needsGateSkill() &&
    !needsBridgeSkill() &&
    !needsTreeSkill()
  );
}

function usefulActionsNow(outcome) {
  if (!hasAction("solveMaze") && outcome.status === "no-action" && mazeFailureCount(outcome) >= 2) {
    return ["solveMaze"];
  }
  if (!hasAction("harvestBushOrApproach") && needsBushSkill()) return ["harvestBushOrApproach"];
  if (!hasAction("collectKeyOrApproach") && needsKeySkill()) return ["collectKeyOrApproach"];
  if (!hasAction("openGateOrApproach") && needsGateSkill()) return ["openGateOrApproach"];

  if (
    !hasAction("trainMovementController") &&
    outcomeHasFailureReason(outcome, "movement model mismatch") &&
    hasMovementDiagnosis()
  ) {
    return ["trainMovementController"];
  }

  if (!hasAction("probeMechanism") && needsMechanismProbeSkill()) {
    return ["probeMechanism"];
  }

  if (!hasAction("readClueOrApproach") && needsMechanismClueSkill()) {
    return ["readClueOrApproach"];
  }

  if (!hasAction("executeProcedure") && needsMechanismSkill()) {
    return ["executeProcedure"];
  }

  if (!hasAction("buildBridgeOrApproach") && needsBridgeSkill()) return ["buildBridgeOrApproach"];
  if (!hasAction("shakeTreeOrApproach") && needsTreeSkill()) return ["shakeTreeOrApproach"];
  if (!hasAction("askGrove") && needsGroveSkill()) return ["askGrove"];
  return [];
}

function hasMazeFailure(outcome) {
  return outcome.failures?.some((failure) =>
    failure.includes("rock blocked") ||
    failure.includes("edge of world") ||
    failure.includes("target not here")
  ) ?? false;
}

function mazeFailureCount(outcome) {
  const previous = state.trace.filter((entry) => hasMazeFailure(entry)).length;
  return previous + (hasMazeFailure(outcome) ? 1 : 0);
}

async function proposeAgenticPatch(outcome, usefulActions) {
  try {
    const response = await fetch("/api/propose-skill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildSkillContext(outcome, usefulActions))
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      return { source: "fallback", patch: null, error: payload.error ?? "Skill writer unavailable." };
    }

    const spec = normalizeGeneratedSkillSpec(payload.spec);
    const patch = compileSkillSpec(spec, usefulActions);
    return patch
      ? { source: "agent", patch, spec }
      : { source: "fallback", patch: null, spec, error: "Agent returned an unusable skill." };
  } catch (error) {
    return {
      source: "fallback",
      patch: null,
      error: error instanceof Error ? error.message : "Skill writer unavailable."
    };
  }
}

function buildSkillContext(outcome, usefulActions) {
  return {
    impulse: "reach_target",
    brief: WORLD_BRIEF,
    targetReached: state.targetReached,
    world: {
      ...observedWorldForSkillContext()
    },
    dynamicsEvidence: state.memory.movementSamples.map(cloneMovementSample),
    existingProgram: state.program.map((behavior) => ({
      name: behavior.name,
      kind: behavior.kind,
      action: behavior.action,
      code: behavior.code
    })),
    existingActions: state.program.map((behavior) => behavior.action).filter(Boolean),
    usefulActions,
    latestOutcome: outcome,
    recentTrace: state.trace.slice(-6).map((entry) => ({
      tick: entry.tick,
      status: entry.status,
      behavior: entry.behavior,
      failures: entry.failures,
      movementSamples: entry.movementSamples
    })),
    allowedActions: thinkerPatches.map((patch) => patch.action),
    outputSchema: {
      name: "namespaced id, e.g. maze:bfs-solver",
      action: "one allowed action",
      reason: "why this skill is useful now",
      code: "short human-readable pseudo-code only; do not put executable JavaScript here",
      source: "required executable JavaScript function body. For solveMaze, call api.maze(), compute a route yourself, then return api.moveTo(nextPoint).",
      priority: "optional integer 1-99",
      plan: "optional structured notes"
    }
  };
}

function observedWorldForSkillContext() {
  const visibleRadius = 3;
  const localCells = [];

  for (let y = state.agent.y - visibleRadius; y <= state.agent.y + visibleRadius; y += 1) {
    for (let x = state.agent.x - visibleRadius; x <= state.agent.x + visibleRadius; x += 1) {
      const point = { x, y };
      if (x < 0 || y < 0 || x >= state.width || y >= state.height) continue;
      const tags = [];
      if (sameCell(state.target, point)) tags.push("target");
      if (state.rocks.some((rock) => sameCell(rock, point))) tags.push("rock");
      if (state.water.some((water) => sameCell(water, point))) tags.push("water");
      if (tags.length > 0) localCells.push({ ...point, tags });
    }
  }

  return {
    width: state.width,
    height: state.height,
    agent: clonePoint(state.agent),
    target: clonePoint(state.target),
    targetHint: "the beacon is visible, but the body has no built-in maze planning",
    visibleRadius,
    localCells,
    mazeApi: "Generated source may call api.maze() to inspect width, height, agent, target, and walls; it must compute the route itself and call api.moveTo(nextPoint).",
    inventory: { ...state.inventory },
    memory: {
      mazeObserved: state.memory.mazeObserved,
      movementSamples: state.memory.movementSamples.map(cloneMovementSample)
    }
  };
}

const WORLD_BRIEF = [
  "The world is a grid maze with one beacon target and many rock walls.",
  "The base body only knows how to test target-underfoot and try one random adjacent move.",
  "After random movement repeatedly collides with walls, infer that the world requires maze solving rather than more wandering.",
  "For solveMaze, generated source must call api.maze(), build its own graph search over passable cells, and return api.moveTo(nextPoint).",
  "The api.moveTo call accepts only one adjacent cell, so the solver must execute one route step per world turn.",
  "Thinking consumes a clock tick, and the target must be reached before the strict turn limit."
];

function compileSkillSpec(spec, usefulActions) {
  if (!spec || typeof spec !== "object") return null;
  if (typeof spec.name !== "string" || !/^[a-z]+:[a-z0-9-]+$/.test(spec.name)) return null;
  if (hasBehavior(spec.name)) return null;
  if (hasAction(spec.action)) return null;
  if (!usefulActions.includes(spec.action)) return null;

  const template = thinkerPatches.find((patch) => patch.action === spec.action);
  if (!template) return null;
  const run = compileAgentSkillSource(spec, template);
  if (!run) return null;

  return {
    ...template,
    name: spec.name,
    kind: "agent-written",
    reason: typeof spec.reason === "string" ? spec.reason : template.reason,
    code: typeof spec.code === "string" ? spec.code : template.code,
    source: typeof spec.source === "string" ? spec.source : template.source,
    plan: spec.plan && typeof spec.plan === "object" ? spec.plan : template.plan,
    priority: Number.isInteger(spec.priority) ? spec.priority : template.priority,
    run
  };
}

function normalizeGeneratedSkillSpec(spec) {
  if (!spec || typeof spec !== "object") return spec;
  const normalized = { ...spec };

  if (
    typeof normalized.source !== "string" &&
    typeof normalized.code === "string" &&
    /\bapi\.(maze|moveTo|fail)\b/.test(normalized.code)
  ) {
    normalized.source = normalized.code;
    normalized.code = "execute generated maze graph search and move one adjacent step";
  }

  return normalized;
}

function compileAgentSkillSource(spec, template) {
  if (typeof spec.source !== "string") return null;
  const source = spec.source.trim();
  if (!source || source.length > 2600) return null;
  if (containsUnsafeSkillSource(source)) return null;
  if (spec.action === "solveMaze" && /\bseekTarget\s*\(/.test(source)) return null;

  try {
    const fn = new Function("api", `"use strict";\n${source}`);
    return (targetState, behavior) => {
      const api = createSkillApi(targetState, behavior);
      const result = fn(api);
      return normalizeSkillResult(result);
    };
  } catch {
    return null;
  }
}

function containsUnsafeSkillSource(source) {
  return /\b(window|document|globalThis|Function|eval|fetch|XMLHttpRequest|WebSocket|import|constructor|__proto__|prototype|localStorage|sessionStorage|indexedDB|process|require)\b/.test(source);
}

function createSkillApi(targetState, behavior) {
  return Object.freeze({
    seekTarget: () => seekTarget(targetState),
    maze: () => mazeSnapshot(targetState),
    moveTo: (point) => moveToMazeCell(targetState, point),
    harvestBushOrApproach: () => harvestBushOrApproach(targetState),
    collectKeyOrApproach: () => collectKeyOrApproach(targetState),
    openGateOrApproach: () => openGateOrApproach(targetState),
    trainMovementController: () => trainMovementController(targetState),
    probeMechanism: () => probeMechanism(targetState),
    readClueOrApproach: () => readClueOrApproach(targetState),
    executeProcedure: () => executeProcedure(targetState, behavior),
    buildBridgeOrApproach: () => buildBridgeOrApproach(targetState),
    shakeTreeOrApproach: () => shakeTreeOrApproach(targetState),
    askGrove: () => askGrove(targetState),
    fail: (reason = "skill source returned no action") => ({ ok: false, reason: String(reason) }),
    state: () => ({
      agent: clonePoint(targetState.agent),
      inventory: { ...targetState.inventory },
      memory: {
        lockedGateObserved: targetState.memory.lockedGateObserved,
        logicGateObserved: targetState.memory.logicGateObserved,
        mazeObserved: targetState.memory.mazeObserved,
        mechanismAttempts: targetState.memory.mechanismAttempts.map((attempt) => ({ ...attempt })),
        cluesRead: targetState.memory.cluesRead.map((clue) => ({ ...clue })),
        movementSamples: targetState.memory.movementSamples.map(cloneMovementSample)
      }
    })
  });
}

function normalizeSkillResult(result) {
  if (result && typeof result === "object" && typeof result.ok === "boolean") return result;
  return { ok: false, reason: "skill source returned invalid result" };
}

function chooseLocalPatch(outcome) {
  const patch =
    !hasBehavior("nav:route-search") && outcome.status === "acted"
      ? patchNamed("nav:route-search")
      : !hasBehavior("tool:harvest-bush") && needsBushSkill()
        ? patchNamed("tool:harvest-bush")
        : !hasBehavior("tool:collect-key") && needsKeySkill()
          ? patchNamed("tool:collect-key")
          : !hasBehavior("tool:open-gate") && needsGateSkill()
            ? patchNamed("tool:open-gate")
            : !hasAction("trainMovementController") &&
                outcomeHasFailureReason(outcome, "movement model mismatch") &&
                hasMovementDiagnosis()
              ? patchNamed("learner:movement-controller")
              : !hasBehavior("tool:build-bridge") && needsBridgeSkill()
                ? patchNamed("tool:build-bridge")
                : !hasBehavior("tool:shake-tree") && needsTreeSkill()
                  ? patchNamed("tool:shake-tree")
                  : !hasBehavior("mcp:ask-grove") && needsGroveSkill()
                    ? patchNamed("mcp:ask-grove")
                    : null;

  return patch ? { ...patch } : null;
}

function hasBehavior(name) {
  return state.program.some((behavior) => behavior.name === name);
}

function hasAction(action) {
  return state.program.some((behavior) => behavior.action === action && !behavior.disabled);
}

function outcomeHasFailure(outcome, prefix) {
  return outcome.failures?.some((failure) => failure.startsWith(prefix)) ?? false;
}

function outcomeHasFailureReason(outcome, reason) {
  return outcome.failures?.some((failure) => failure.endsWith(`: ${reason}`)) ?? false;
}

function hasRecoverableFailure(outcome) {
  return outcome.failures?.some((failure) =>
    failure.includes("movement model mismatch") ||
    failure.includes("insufficient movement evidence") ||
    failure.includes("rock blocked") ||
    failure.includes("edge of world")
  ) ?? false;
}

function applyMovementDynamics(targetState, from, intended) {
  if (
    targetState.inventory.neuralController ||
    !targetState.wind.some((wind) => sameCell(wind, intended))
  ) {
    return intended;
  }

  const actual = firstOpen(targetState, driftCandidates(targetState, from, intended));

  targetState.memory.movementSamples.push({
    from: clonePoint(from),
    intended: clonePoint(intended),
    actual: clonePoint(actual),
    tick: targetState.tick
  });

  return actual;
}

function firstOpen(targetState, candidates) {
  return candidates.find((candidate) => !blockedReason(targetState, candidate)) ?? clonePoint(targetState.agent);
}

function driftCandidates(targetState, from, intended) {
  const pattern = (intended.x + intended.y + targetState.tick) % 4;
  const drifts = [
    [
      { x: intended.x, y: intended.y + 1 },
      { x: from.x, y: from.y + 1 }
    ],
    [
      { x: intended.x - 1, y: intended.y },
      { x: from.x - 1, y: from.y }
    ],
    [
      { x: intended.x, y: intended.y - 1 },
      { x: from.x, y: from.y - 1 }
    ],
    [
      { x: intended.x + 1, y: intended.y },
      { x: from.x + 1, y: from.y }
    ]
  ][pattern];

  return [...drifts, from];
}

function hasMovementDiagnosis() {
  return state.memory.movementSamples.filter((sample) => !sameCell(sample.intended, sample.actual)).length >= 3;
}

function patchNamed(name) {
  return thinkerPatches.find((patch) => patch.name === name);
}

function clonePoint(point) {
  return { x: point.x, y: point.y };
}

function parseMaze(rows) {
  const rocks = [];
  let agent = null;
  let target = null;

  rows.forEach((row, y) => {
    [...row].forEach((cell, x) => {
      if (cell === "#") rocks.push({ x, y });
      if (cell === "S") agent = { x, y };
      if (cell === "T") target = { x, y };
    });
  });

  return {
    width: rows[0].length,
    height: rows.length,
    agent,
    target,
    rocks
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

function cloneMovementSample(sample) {
  return {
    from: clonePoint(sample.from),
    intended: clonePoint(sample.intended),
    actual: clonePoint(sample.actual),
    tick: sample.tick ?? sample.turn,
    passive: Boolean(sample.passive)
  };
}

function mechanismSolution(targetState) {
  const target = targetState.logicRule?.targetSum ?? 0;
  return targetState.switches
    .filter((item) => item.kind === "prime")
    .sort((left, right) => left.value - right.value)
    .reduce(
      (accumulator, item) => {
        if (accumulator.total >= target) return accumulator;
        return {
          total: accumulator.total + item.value,
          ids: [...accumulator.ids, item.id]
        };
      },
      { total: 0, ids: [] }
    ).ids;
}

function sameSequence(left, right) {
  return left.length === right.length && left.every((item, index) => item === right[index]);
}

function clueRead(id, targetState = state) {
  return targetState.memory.cluesRead.some((clue) => clue.id === id);
}

function keyOf(point) {
  return `${point.x},${point.y}`;
}

function distance(left, right) {
  return Math.abs(left.x - right.x) + Math.abs(left.y - right.y);
}

function sameCell(left, right) {
  return left.x === right.x && left.y === right.y;
}

elements.stepButton.addEventListener("click", () => {
  stop();
  step();
});

elements.runButton.addEventListener("click", () => {
  if (state.running) {
    stop();
    render();
  } else {
    start();
  }
});

elements.resetButton.addEventListener("click", () => {
  stop();
  state = cloneInitial();
  start();
});

elements.clearSkillsButton.addEventListener("click", () => {
  stop();
  localStorage.removeItem(SKILL_LIBRARY_KEY);
  state = cloneInitial();
  start();
});

render();
start();
