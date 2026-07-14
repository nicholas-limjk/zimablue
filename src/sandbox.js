export class FruitSandbox {
  constructor({
    goal = 10,
    width = 10,
    height = 7,
    agent = { x: 0, y: 0 },
    target = null,
    fruits = [{ x: 1, y: 0 }],
    bushes = [{ x: 3, y: 2, fruit: 2 }],
    trees = [{ x: 8, y: 5, fruit: 3 }],
    keys = [{ x: 5, y: 1 }],
    gate = { x: 6, y: 2, open: false },
    logicGate = null,
    switches = [],
    logicRule = null,
    clues = [],
    grove = { x: 1, y: 6, cycles: 1 },
    turnLimit = 50,
    wind = [],
    rocks = [
      { x: 2, y: 0 },
      { x: 2, y: 2 },
      { x: 5, y: 3 },
      { x: 7, y: 3 }
    ],
    water = [
      { x: 4, y: 3 },
      { x: 4, y: 4 },
      { x: 4, y: 5 }
    ]
  } = {}) {
    this.state = {
      goal,
      width,
      height,
      fruitCount: 0,
      target: target ? clonePoint(target) : clonePoint(fruits[0] ?? { x: 1, y: 0 }),
      targetReached: false,
      agent: { ...agent },
      fruits: fruits.map(clonePoint),
      bushes: bushes.map((bush) => ({ ...bush })),
      trees: trees.map((tree) => ({ ...tree })),
      keys: keys.map(clonePoint),
      gate: { ...gate },
      logicGate: logicGate ? { activated: [], ...logicGate } : null,
      switches: switches.map((item) => ({ ...item })),
      logicRule: logicRule ? { ...logicRule } : null,
      clues: clues.map((clue) => ({ ...clue })),
      grove: { ...grove },
      turn: 0,
      turnLimit,
      failed: false,
      rocks: rocks.map(clonePoint),
      water: water.map(clonePoint),
      wind: wind.map(clonePoint),
      inventory: {
        key: false,
        bridge: false,
        neuralController: false
      },
      memory: {
        lockedGateObserved: false,
        logicGateObserved: false,
        mechanismAttempts: [],
        movementSamples: [],
        cluesRead: []
      }
    };
  }

  observe() {
    return {
      targetReached: this.state.targetReached,
      target: { ...this.state.target },
      turn: this.state.turn,
      turnLimit: this.state.turnLimit,
      failed: this.state.failed
    };
  }

  advanceTurn() {
    if (this.state.failed || this.state.targetReached) {
      return;
    }

    this.state.turn += 1;

    if (this.state.turn >= this.state.turnLimit && !this.state.targetReached) {
      this.state.failed = true;
    }
  }

  spendThinkingTurn() {
    if (this.state.failed || this.state.targetReached) {
      return { ok: true, event: "No thinking cost after termination." };
    }

    const failures = [];
    let event = "Thinking consumed one world turn.";

    if (
      !this.state.inventory.neuralController &&
      this.state.wind.some((wind) => this.#sameCell(wind, this.state.agent))
    ) {
      const from = clonePoint(this.state.agent);
      const actual = this.#firstOpen(this.#driftCandidates(from, from));
      this.state.agent = actual;
      this.state.memory.movementSamples.push({
        from,
        intended: clonePoint(from),
        actual: clonePoint(actual),
        turn: this.state.turn,
        passive: true
      });

      if (!this.#sameCell(actual, from)) {
        failures.push("thinking:movement_model_mismatch");
        event = "Thinking inside unstable dynamics let the field push the agent.";
      }
    }

    this.advanceTurn();
    return { ok: failures.length === 0, event, failures };
  }

  failed() {
    return this.state.failed;
  }

  reachTargetHere() {
    if (!this.#sameCell(this.state.agent, this.state.target)) {
      return { ok: false, reason: "target_not_here" };
    }

    this.state.targetReached = true;
    this.state.fruitCount = 1;
    return { ok: true, event: "The base body reached the target." };
  }

  gatherHere() {
    return this.reachTargetHere();
  }

  move(dx, dy) {
    const next = {
      x: this.state.agent.x + dx,
      y: this.state.agent.y + dy
    };
    const blocked = this.#blockedReason(next);

    if (blocked) return { ok: false, reason: blocked };

    const actual = this.#applyMovementDynamics(this.state.agent, next);
    this.state.agent = actual;

    if (!this.#sameCell(actual, next)) {
      return { ok: false, reason: "movement_model_mismatch" };
    }

    return { ok: true, event: `The base body moved to (${next.x}, ${next.y}).` };
  }

  seekTarget() {
    const hadLockedGateObservation = this.state.memory.lockedGateObserved;
    const hadLogicGateObservation = this.state.memory.logicGateObserved;
    const path = this.#pathTo(this.state.target);
    if (path.length === 0) {
      if (
        (!hadLogicGateObservation && this.state.memory.logicGateObserved) ||
        (this.state.memory.logicGateObserved && this.state.logicGate && !this.state.logicGate.open)
      ) {
        return { ok: false, reason: "logic_gate_locked" };
      }

      if (
        (!hadLockedGateObservation && this.state.memory.lockedGateObserved) ||
        (this.state.memory.lockedGateObserved && !this.state.gate.open)
      ) {
        return { ok: false, reason: "gate_locked" };
      }

      return { ok: false, reason: "target_unreachable" };
    }
    if (path.length < 2) return { ok: false, reason: "already_there" };

    const intended = path[1];
    const actual = this.#applyMovementDynamics(this.state.agent, intended);
    this.state.agent = actual;

    if (!this.#sameCell(actual, intended)) {
      return { ok: false, reason: "movement_model_mismatch" };
    }

    return { ok: true, event: "The self-written route-search skill moved toward the target." };
  }

  mazeSnapshot() {
    return {
      width: this.state.width,
      height: this.state.height,
      agent: clonePoint(this.state.agent),
      target: clonePoint(this.state.target),
      walls: this.state.rocks.map(clonePoint)
    };
  }

  moveTo(point) {
    if (!point || !Number.isInteger(point.x) || !Number.isInteger(point.y)) {
      return { ok: false, reason: "invalid_maze_point" };
    }

    const intended = { x: point.x, y: point.y };
    if (this.#distance(this.state.agent, intended) !== 1) {
      return { ok: false, reason: "maze_move_not_adjacent" };
    }

    const blocked = this.#blockedReason(intended);
    if (blocked) return { ok: false, reason: blocked };

    this.state.agent = intended;
    return { ok: true, event: "The LLM-written maze solver advanced one cell along its computed route." };
  }

  seekNearestFruit() {
    return this.seekTarget();
  }

  harvestBushOrApproach() {
    const bush = this.#nearestReachable(this.state.bushes.filter((candidate) => candidate.fruit > 0));
    if (!bush) return { ok: false, reason: "no_reachable_bush" };

    if (this.#distance(this.state.agent, bush) <= 1) {
      const dropped = Math.min(2, bush.fruit);
      bush.fruit -= dropped;
      for (let index = 0; index < dropped; index += 1) {
        this.state.fruits.push({ x: bush.x, y: bush.y });
      }
      return { ok: true, event: "The self-written bush skill shook berries loose." };
    }

    return this.#stepAlongPath(bush, "The bush skill moved toward berry cover.");
  }

  collectKeyOrApproach() {
    if (this.state.inventory.key) return { ok: false, reason: "key_already_held" };

    const key = this.#nearestReachable(this.state.keys);
    if (!key) return { ok: false, reason: "no_reachable_key" };

    if (this.#sameCell(this.state.agent, key)) {
      this.state.inventory.key = true;
      this.state.keys = this.state.keys.filter((candidate) => !this.#sameCell(candidate, key));
      return { ok: true, event: "The self-written key skill picked up a gate key." };
    }

    return this.#stepAlongPath(key, "The key skill moved toward the gate key.");
  }

  executeProcedure(spec = {}) {
    if (!this.state.logicGate) return { ok: false, reason: "no_mechanism" };
    if (this.state.logicGate.open) return { ok: false, reason: "mechanism_already_open" };
    if (!this.state.memory.logicGateObserved) {
      return { ok: false, reason: "mechanism_not_observed" };
    }
    if (this.state.memory.mechanismAttempts.length === 0) {
      return { ok: false, reason: "mechanism_not_probed" };
    }
    if (this.state.memory.cluesRead.length === 0) {
      return { ok: false, reason: "clue_not_read" };
    }

    const proposed = spec.plan?.switches;
    if (!Array.isArray(proposed) || proposed.length === 0) {
      return { ok: false, reason: "missing_switch_procedure" };
    }

    const solution = this.#mechanismSolution();
    const nextSwitch = proposed.find((id) => !this.state.logicGate.activated.includes(id));
    if (!nextSwitch) return { ok: false, reason: "mechanism_unsolved" };

    const plate = this.state.switches.find((candidate) => candidate.id === nextSwitch);
    if (!plate) return { ok: false, reason: "missing_switch" };

    if (this.#sameCell(this.state.agent, plate)) {
      this.state.logicGate.activated.push(plate.id);
      if (this.state.logicGate.activated.length >= proposed.length) {
        if (!sameSequence(proposed, solution)) {
          this.state.logicGate.activated = [];
          this.state.memory.mechanismAttempts.push({
            turn: this.state.turn,
            proposed: [...proposed],
            result: "rejected"
          });
          return { ok: false, reason: "mechanism_rejected_procedure" };
        }

        this.state.logicGate.open = true;
        return { ok: true, event: "The agent executed its inferred switch procedure and opened the logic gate." };
      }

      return { ok: true, event: `The mechanism skill activated switch ${plate.id}.` };
    }

    return this.#stepAlongPath(plate, "The mechanism skill moved toward the next inferred switch.");
  }

  readClueOrApproach() {
    if (!this.state.logicGate) return { ok: false, reason: "no_mechanism" };
    if (this.state.logicGate.open) return { ok: false, reason: "mechanism_already_open" };
    if (!this.state.memory.logicGateObserved) {
      return { ok: false, reason: "mechanism_not_observed" };
    }
    if (this.state.memory.mechanismAttempts.length === 0) {
      return { ok: false, reason: "mechanism_not_probed" };
    }

    const clue = this.#nearestReachable(
      this.state.clues.filter((candidate) => !this.#clueRead(candidate.id))
    );
    if (!clue) return { ok: false, reason: "no_reachable_clue" };

    if (this.#sameCell(this.state.agent, clue)) {
      this.state.memory.cluesRead.push({
        id: clue.id,
        text: clue.text,
        turn: this.state.turn
      });
      return { ok: true, event: `The agent read the clue tablet: "${clue.text}"` };
    }

    return this.#stepAlongPath(clue, "The clue-search skill moved toward a written rule.");
  }

  probeMechanism() {
    if (!this.state.logicGate) return { ok: false, reason: "no_mechanism" };
    if (this.state.logicGate.open) return { ok: false, reason: "mechanism_already_open" };
    if (!this.state.memory.logicGateObserved) {
      return { ok: false, reason: "mechanism_not_observed" };
    }
    if (this.state.memory.mechanismAttempts.length > 0) {
      return { ok: false, reason: "mechanism_probe_already_recorded" };
    }

    const plate = this.#nearestReachable(this.state.switches);
    if (!plate) return { ok: false, reason: "no_reachable_switch" };

    if (this.#sameCell(this.state.agent, plate)) {
      const proposed = [plate.id];
      this.state.logicGate.activated = [];
      this.state.memory.mechanismAttempts.push({
        turn: this.state.turn,
        proposed,
        result: "rejected"
      });
      return {
        ok: true,
        event: `The agent tried switch ${plate.id} alone; the mechanism rejected the incomplete procedure.`
      };
    }

    return this.#stepAlongPath(plate, "The probe moved toward a visible switch to test the mechanism.");
  }

  openGateOrApproach() {
    if (this.state.gate.open) return { ok: false, reason: "gate_already_open" };
    if (!this.state.inventory.key) return { ok: false, reason: "no_key" };

    if (this.#distance(this.state.agent, this.state.gate) <= 1) {
      this.state.gate.open = true;
      return { ok: true, event: "The self-written gate skill opened the locked passage." };
    }

    return this.#stepAdjacentTo(this.state.gate, "The gate skill moved toward the lock.");
  }

  buildBridgeOrApproach() {
    if (this.state.inventory.bridge) return { ok: false, reason: "bridge_already_built" };

    const bridgeSite = this.state.water[1] ?? this.state.water[0];
    if (!bridgeSite) return { ok: false, reason: "no_water" };

    if (this.#distance(this.state.agent, bridgeSite) <= 1) {
      this.state.inventory.bridge = true;
      return { ok: true, event: "The self-written bridge skill made water passable." };
    }

    return this.#stepAdjacentTo(bridgeSite, "The bridge skill moved toward the stream.");
  }

  shakeTreeOrApproach() {
    const tree = this.#nearestReachable(this.state.trees.filter((candidate) => candidate.fruit > 0));
    if (!tree) return { ok: false, reason: "no_reachable_tree" };

    if (this.#distance(this.state.agent, tree) <= 1) {
      const dropped = Math.min(3, tree.fruit);
      tree.fruit -= dropped;
      for (let index = 0; index < dropped; index += 1) {
        this.state.fruits.push({ x: tree.x - index, y: tree.y });
      }
      return { ok: true, event: "The self-written tree skill shook down orchard fruit." };
    }

    return this.#stepAlongPath(tree, "The tree skill moved toward the orchard.");
  }

  askGrove() {
    if (this.state.grove.cycles <= 0) return { ok: false, reason: "grove_empty" };

    this.state.grove.cycles -= 1;
    this.state.fruits.push(
      { x: this.state.grove.x, y: this.state.grove.y },
      { x: this.state.grove.x + 1, y: this.state.grove.y },
      { x: this.state.grove.x + 2, y: this.state.grove.y },
      { x: this.state.grove.x + 3, y: this.state.grove.y }
    );

    return { ok: true, event: "The self-written MCP adapter asked the grove for late fruit." };
  }

  trainMovementController() {
    if (this.state.inventory.neuralController) {
      return { ok: false, reason: "controller_already_trained" };
    }

    if (this.state.wind.length === 0) {
      return { ok: false, reason: "no_unstable_field" };
    }

    if (!this.#hasEnoughMovementEvidence()) {
      return { ok: false, reason: "insufficient_movement_evidence" };
    }

    this.state.inventory.neuralController = true;

    return {
      ok: true,
      event:
        "The agent trained a tiny movement controller from its own transition evidence; unstable cells are now passable."
    };
  }

  needsBushSkill() {
    return this.state.fruits.length === 0 && this.state.bushes.some((bush) => bush.fruit > 0);
  }

  needsKeySkill() {
    return (
      this.state.memory.lockedGateObserved &&
      !this.state.inventory.key &&
      this.state.keys.length > 0
    );
  }

  needsGateSkill() {
    return (
      this.state.memory.lockedGateObserved &&
      this.state.inventory.key &&
      !this.state.gate.open
    );
  }

  needsMechanismSkill() {
    return (
      this.state.memory.logicGateObserved &&
      this.state.memory.mechanismAttempts.some((attempt) => attempt.result === "rejected") &&
      this.state.memory.cluesRead.length > 0 &&
      this.state.logicGate &&
      !this.state.logicGate.open
    );
  }

  needsMechanismClueSkill() {
    return (
      this.state.memory.logicGateObserved &&
      this.state.memory.mechanismAttempts.some((attempt) => attempt.result === "rejected") &&
      this.state.memory.cluesRead.length === 0 &&
      this.state.logicGate &&
      !this.state.logicGate.open
    );
  }

  needsMechanismProbeSkill() {
    return (
      this.state.memory.logicGateObserved &&
      this.state.logicGate &&
      !this.state.logicGate.open &&
      this.state.memory.mechanismAttempts.length === 0
    );
  }

  needsBridgeSkill() {
    return this.state.water.length > 0 && this.state.gate.open && !this.state.inventory.bridge;
  }

  needsTreeSkill() {
    return this.state.inventory.bridge && this.state.trees.some((tree) => tree.fruit > 0);
  }

  needsGroveSkill() {
    return (
      this.state.grove.cycles > 0 &&
      !this.state.targetReached &&
      this.state.fruits.length === 0 &&
      !this.needsBushSkill() &&
      !this.needsKeySkill() &&
      !this.needsGateSkill() &&
      !this.needsBridgeSkill() &&
      !this.needsTreeSkill()
    );
  }

  snapshot() {
    return {
      ...this.state,
      agent: { ...this.state.agent },
      target: { ...this.state.target },
      targetReached: this.state.targetReached,
      fruits: this.state.fruits.map(clonePoint),
      bushes: this.state.bushes.map((bush) => ({ ...bush })),
      trees: this.state.trees.map((tree) => ({ ...tree })),
      keys: this.state.keys.map(clonePoint),
      gate: { ...this.state.gate },
      logicGate: this.state.logicGate
        ? { ...this.state.logicGate, activated: [...this.state.logicGate.activated] }
        : null,
      switches: this.state.switches.map((item) => ({ ...item })),
      logicRule: this.state.logicRule ? { ...this.state.logicRule } : null,
      clues: this.state.clues.map((clue) => ({ ...clue })),
      grove: { ...this.state.grove },
      turn: this.state.turn,
      turnLimit: this.state.turnLimit,
      failed: this.state.failed,
      rocks: this.state.rocks.map(clonePoint),
      water: this.state.water.map(clonePoint),
      wind: this.state.wind.map(clonePoint),
      inventory: { ...this.state.inventory },
      memory: {
        ...this.state.memory,
        mechanismAttempts: this.state.memory.mechanismAttempts.map((attempt) => ({ ...attempt })),
        movementSamples: this.state.memory.movementSamples.map((sample) => ({
          from: clonePoint(sample.from),
          intended: clonePoint(sample.intended),
          actual: clonePoint(sample.actual),
          turn: sample.turn,
          passive: Boolean(sample.passive)
        })),
        cluesRead: this.state.memory.cluesRead.map((clue) => ({ ...clue }))
      }
    };
  }

  hasMovementDiagnosis() {
    return this.#hasEnoughMovementEvidence();
  }

  #stepAlongPath(target, event) {
    const path = this.#pathTo(target);
    if (path.length < 2) return { ok: false, reason: "already_there" };

    const intended = path[1];
    const actual = this.#applyMovementDynamics(this.state.agent, intended);
    this.state.agent = actual;

    if (!this.#sameCell(actual, intended)) {
      return { ok: false, reason: "movement_model_mismatch" };
    }

    return { ok: true, event };
  }

  #stepAdjacentTo(target, event) {
    const adjacent = this.#neighbors(target)
      .map((point) => ({ point, path: this.#pathTo(point) }))
      .filter((candidate) => candidate.path.length > 0)
      .sort((left, right) => left.path.length - right.path.length)[0];

    if (!adjacent) return { ok: false, reason: "no_adjacent_path" };
    if (adjacent.path.length < 2) return { ok: false, reason: "already_there" };

    const intended = adjacent.path[1];
    const actual = this.#applyMovementDynamics(this.state.agent, intended);
    this.state.agent = actual;

    if (!this.#sameCell(actual, intended)) {
      return { ok: false, reason: "movement_model_mismatch" };
    }

    return { ok: true, event };
  }

  #nearestReachable(points) {
    return points
      .map((point) => ({ point, path: this.#pathTo(point) }))
      .filter((candidate) => candidate.path.length > 0)
      .sort((left, right) => left.path.length - right.path.length)[0]?.point;
  }

  #pathTo(target) {
    const start = this.state.agent;
    const queue = [[start]];
    const seen = new Set([keyOf(start)]);

    while (queue.length > 0) {
      const path = queue.shift();
      const current = path.at(-1);

      if (this.#sameCell(current, target)) return path;

      for (const next of this.#neighbors(current)) {
        const key = keyOf(next);
        if (seen.has(key)) continue;
        seen.add(key);
        queue.push([...path, next]);
      }
    }

    return [];
  }

  #neighbors(point) {
    return [
      { x: point.x + 1, y: point.y },
      { x: point.x - 1, y: point.y },
      { x: point.x, y: point.y + 1 },
      { x: point.x, y: point.y - 1 }
    ].filter((candidate) => !this.#blockedReason(candidate));
  }

  #blockedReason(point) {
    if (!this.#inside(point)) return "edge_of_world";
    if (this.state.rocks.some((rock) => this.#sameCell(rock, point))) return "rock_blocked";
    if (!this.state.gate.open && this.#sameCell(this.state.gate, point)) {
      this.state.memory.lockedGateObserved = true;
      return "gate_locked";
    }
    if (
      this.state.logicGate &&
      !this.state.logicGate.open &&
      this.#sameCell(this.state.logicGate, point)
    ) {
      this.state.memory.logicGateObserved = true;
      return "logic_gate_locked";
    }
    if (!this.state.inventory.bridge && this.state.water.some((water) => this.#sameCell(water, point))) {
      return "water_blocked";
    }
    return null;
  }

  #applyMovementDynamics(from, intended) {
    if (
      this.state.inventory.neuralController ||
      !this.state.wind.some((wind) => this.#sameCell(wind, intended))
    ) {
      return intended;
    }

    const actual = this.#firstOpen(this.#driftCandidates(from, intended));

    this.state.memory.movementSamples.push({
      from: clonePoint(from),
      intended: clonePoint(intended),
      actual: clonePoint(actual),
      turn: this.state.turn
    });

    return actual;
  }

  #firstOpen(candidates) {
    return candidates.find((candidate) => !this.#blockedReason(candidate)) ?? clonePoint(this.state.agent);
  }

  #driftCandidates(from, intended) {
    const pattern = (intended.x + intended.y + this.state.turn) % 4;
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

  #hasEnoughMovementEvidence() {
    return this.state.memory.movementSamples.filter(
      (sample) => !this.#sameCell(sample.intended, sample.actual)
    ).length >= 3;
  }

  #mechanismSolution() {
    const target = this.state.logicRule?.targetSum ?? 0;
    const candidates = this.state.switches
      .filter((item) => item.kind === "prime")
      .sort((left, right) => left.value - right.value);
    const solution = [];
    let total = 0;

    for (const item of candidates) {
      if (total + item.value <= target) {
        solution.push(item.id);
        total += item.value;
      }
    }

    return total === target ? solution : candidates.map((item) => item.id);
  }

  #clueRead(id) {
    return this.state.memory.cluesRead.some((clue) => clue.id === id);
  }

  #distance(left, right) {
    return Math.abs(left.x - right.x) + Math.abs(left.y - right.y);
  }

  #inside(point) {
    return point.x >= 0 && point.y >= 0 && point.x < this.state.width && point.y < this.state.height;
  }

  #sameCell(left, right) {
    return left.x === right.x && left.y === right.y;
  }
}

function clonePoint(point) {
  return { x: point.x, y: point.y };
}

function keyOf(point) {
  return `${point.x},${point.y}`;
}

function sameSequence(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
