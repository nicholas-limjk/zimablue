export class BehaviorProgram {
  #behaviors = [];
  #patches = [];

  constructor(initialBehaviors = []) {
    for (const behavior of initialBehaviors) {
      this.add(behavior, "base body");
    }
  }

  add(behavior, reason) {
    if (this.#behaviors.some((existing) => existing.name === behavior.name)) return false;

    this.#behaviors.push(behavior);
    this.#behaviors.sort((left, right) => left.priority - right.priority);
    this.#patches.push({
      name: behavior.name,
      code: behavior.code,
      reason
    });

    return true;
  }

  run(sandbox) {
    const failures = [];

    for (const behavior of this.#behaviors) {
      const result = behavior.run(sandbox);

      if (result.ok) {
        return {
          status: "acted",
          behavior: behavior.name,
          message: result.event,
          failures
        };
      }

      failures.push(`${behavior.name}:${result.reason}`);
    }

    return {
      status: "no_action",
      behavior: null,
      message: "No behavior in the current program helped.",
      failures
    };
  }

  has(name) {
    return this.#behaviors.some((behavior) => behavior.name === name);
  }

  hasAction(action) {
    return this.#behaviors.some((behavior) => behavior.action === action);
  }

  snapshot() {
    return this.#behaviors.map((behavior) => ({
      name: behavior.name,
      kind: behavior.kind,
      action: behavior.action,
      code: behavior.code
    }));
  }

  patches() {
    return [...this.#patches];
  }
}

export function createBaseBehaviors() {
  return [
    {
      name: "body:reach-target-here",
      kind: "body",
      priority: 0,
      code: "if target is underfoot: stop",
      run(sandbox) {
        return sandbox.reachTargetHere();
      }
    },
    {
      name: "body:move-east",
      kind: "body",
      priority: 100,
      code: "move x + 1",
      run(sandbox) {
        return sandbox.move(1, 0);
      }
    },
    {
      name: "body:move-south",
      kind: "body",
      priority: 110,
      code: "move y + 1",
      run(sandbox) {
        return sandbox.move(0, 1);
      }
    }
  ];
}

export function createThinkerPatches() {
  return [
    {
      name: "nav:route-search",
      kind: "self-written",
      action: "seekTarget",
      priority: 10,
      reason: "The target is not underfoot, so the agent invents route search.",
      code: "build a local route search; discover blockers such as locked gates and unstable cells",
      run(sandbox) {
        return sandbox.seekTarget();
      }
    },
    {
      name: "tool:harvest-bush",
      kind: "self-written",
      action: "harvestBushOrApproach",
      priority: 20,
      reason: "Loose fruit ran out, so the agent writes a bush harvester.",
      code: "move beside berry bush; shake berries into loose fruit",
      run(sandbox) {
        return sandbox.harvestBushOrApproach();
      }
    },
    {
      name: "tool:collect-key",
      kind: "self-written",
      action: "collectKeyOrApproach",
      priority: 30,
      reason: "A target route touched a locked gate, so the agent writes key collection.",
      code: "path to key; pick it up into inventory",
      run(sandbox) {
        return sandbox.collectKeyOrApproach();
      }
    },
    {
      name: "tool:open-gate",
      kind: "self-written",
      action: "openGateOrApproach",
      priority: 40,
      reason: "The key exists now, so the agent writes a gate-opening routine.",
      code: "stand beside locked gate; spend key; mark gate open",
      run(sandbox) {
        return sandbox.openGateOrApproach();
      }
    },
    {
      name: "learner:movement-controller",
      kind: "self-written",
      action: "trainMovementController",
      priority: 45,
      reason: "Repeated transition samples disagree with the movement model, so the agent trains a new controller from scratch.",
      code: "compare intended vs actual moves; learn a movement policy from scratch",
      run(sandbox) {
        return sandbox.trainMovementController();
      }
    },
    {
      name: "tool:build-bridge",
      kind: "self-written",
      action: "buildBridgeOrApproach",
      priority: 50,
      reason: "Water blocks the orchard route, so the agent writes bridge building.",
      code: "stand beside stream; set bridge=true; water becomes passable",
      run(sandbox) {
        return sandbox.buildBridgeOrApproach();
      }
    },
    {
      name: "tool:shake-tree",
      kind: "self-written",
      action: "shakeTreeOrApproach",
      priority: 60,
      reason: "The orchard is reachable, so the agent writes a tree-shaking skill.",
      code: "move beside fruit tree; shake orchard fruit loose",
      run(sandbox) {
        return sandbox.shakeTreeOrApproach();
      }
    },
    {
      name: "mcp:ask-grove",
      kind: "self-written",
      action: "askGrove",
      priority: 70,
      reason: "The local world is nearly spent, so the agent writes a grove adapter.",
      code: "call quiet_grove.grow_fruit(); add four loose fruits",
      run(sandbox) {
        return sandbox.askGrove();
      }
    }
  ];
}

export class ThinkingAgent {
  constructor(patches = createThinkerPatches()) {
    this.patches = patches;
  }

  reflect({ program, sandbox, outcome }) {
    const nextPatch = this.#choosePatch({ program, sandbox, outcome });

    if (!nextPatch) return { patched: false, message: "No useful patch remains." };

    program.add(nextPatch, nextPatch.reason);

    return {
      patched: true,
      behavior: nextPatch.name,
      message: nextPatch.reason
    };
  }

  #choosePatch({ program, sandbox, outcome }) {
    if (!program.has("nav:route-search") && outcome.status === "acted") {
      return this.#patch("nav:route-search");
    }

    if (!program.has("tool:harvest-bush") && sandbox.needsBushSkill()) {
      return this.#patch("tool:harvest-bush");
    }

    if (!program.has("tool:collect-key") && sandbox.needsKeySkill()) {
      return this.#patch("tool:collect-key");
    }

    if (!program.has("tool:open-gate") && sandbox.needsGateSkill()) {
      return this.#patch("tool:open-gate");
    }

    if (
      !program.hasAction("trainMovementController") &&
      outcomeHasFailureReason(outcome, "movement_model_mismatch") &&
      sandbox.hasMovementDiagnosis()
    ) {
      return this.#patch("learner:movement-controller");
    }

    if (!program.has("tool:build-bridge") && sandbox.needsBridgeSkill()) {
      return this.#patch("tool:build-bridge");
    }

    if (!program.has("tool:shake-tree") && sandbox.needsTreeSkill()) {
      return this.#patch("tool:shake-tree");
    }

    if (!program.has("mcp:ask-grove") && sandbox.needsGroveSkill()) {
      return this.#patch("mcp:ask-grove");
    }

    return null;
  }

  #patch(name) {
    return this.patches.find((patch) => patch.name === name);
  }
}

function outcomeHasFailure(outcome, prefix) {
  return outcome.failures?.some((failure) => failure.startsWith(prefix)) ?? false;
}

function outcomeHasFailureReason(outcome, reason) {
  return outcome.failures?.some((failure) => failure.endsWith(`:${reason}`)) ?? false;
}
