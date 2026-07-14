export const SKILL_ACTIONS = {
  solveMaze: {
    priority: 10,
    code: "inspect api.maze(); run graph search; move one adjacent step with api.moveTo(next)",
    run(sandbox) {
      return sandbox.seekTarget();
    }
  },
  seekTarget: {
    priority: 10,
    code: "invent a route-search policy to the target; discover blockers such as locked gates",
    run(sandbox) {
      return sandbox.seekTarget();
    }
  },
  harvestBushOrApproach: {
    priority: 20,
    code: "move beside a berry bush; shake berries into loose fruit",
    run(sandbox) {
      return sandbox.harvestBushOrApproach();
    }
  },
  collectKeyOrApproach: {
    priority: 30,
    code: "path to a key; pick it up into inventory",
    run(sandbox) {
      return sandbox.collectKeyOrApproach();
    }
  },
  openGateOrApproach: {
    priority: 40,
    code: "stand beside a locked gate; open it when holding a key",
    run(sandbox) {
      return sandbox.openGateOrApproach();
    }
  },
  trainMovementController: {
    priority: 45,
    code: "train a fresh movement controller from intended-vs-actual transition samples",
    run(sandbox) {
      return sandbox.trainMovementController();
    }
  },
  probeMechanism: {
    priority: 46,
    code: "try a simple mechanism interaction to collect failure evidence",
    run(sandbox) {
      return sandbox.probeMechanism();
    }
  },
  readClueOrApproach: {
    priority: 47,
    code: "search for a written clue and read it into memory",
    run(sandbox) {
      return sandbox.readClueOrApproach();
    }
  },
  executeProcedure: {
    priority: 48,
    code: "execute the switch procedure inferred from a discovered clue",
    run(sandbox, spec) {
      return sandbox.executeProcedure(spec);
    }
  },
  buildBridgeOrApproach: {
    priority: 50,
    code: "stand beside water; build a bridge to make water passable",
    run(sandbox) {
      return sandbox.buildBridgeOrApproach();
    }
  },
  shakeTreeOrApproach: {
    priority: 60,
    code: "move beside an orchard tree; shake fruit loose",
    run(sandbox) {
      return sandbox.shakeTreeOrApproach();
    }
  },
  askGrove: {
    priority: 70,
    code: "call the quiet grove adapter to grow late fruit",
    run(sandbox) {
      return sandbox.askGrove();
    }
  }
};

export function actionNames() {
  return Object.keys(SKILL_ACTIONS);
}

export function compileSkillSpec(spec) {
  spec = normalizeGeneratedSkillSpec(spec);
  const errors = validateSkillSpec(spec);
  if (errors.length > 0) {
    return { ok: false, errors };
  }

  const action = SKILL_ACTIONS[spec.action];
  const runSource = compileGeneratedSkillSource(spec.source);
  if (!runSource.ok) {
    return { ok: false, errors: runSource.errors };
  }

  return {
    ok: true,
    behavior: {
      name: spec.name,
      kind: "agent-written",
      action: spec.action,
      priority: spec.priority ?? action.priority,
      reason: spec.reason,
      code: spec.code ?? action.code,
      source: spec.source,
      plan: spec.plan,
      run(sandbox) {
        return runSource.run(createSkillApi(sandbox, spec));
      }
    }
  };
}

function normalizeGeneratedSkillSpec(spec) {
  if (!spec || typeof spec !== "object" || Array.isArray(spec)) return spec;
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

export function validateSkillSpec(spec) {
  const errors = [];

  if (!spec || typeof spec !== "object" || Array.isArray(spec)) {
    return ["Skill spec must be a JSON object."];
  }

  if (!validName(spec.name)) {
    errors.push("name must be a stable namespaced id like nav:route-search.");
  }

  if (!actionNames().includes(spec.action)) {
    errors.push(`action must be one of: ${actionNames().join(", ")}.`);
  }

  if (spec.reason !== undefined && typeof spec.reason !== "string") {
    errors.push("reason must be a string when provided.");
  }

  if (spec.code !== undefined && typeof spec.code !== "string") {
    errors.push("code must be a string when provided.");
  }

  if (typeof spec.source !== "string" || spec.source.trim().length === 0) {
    errors.push("source must be a JavaScript function body string.");
  } else if (spec.source.length > 2600) {
    errors.push("source must be 2600 characters or fewer.");
  } else if (containsUnsafeSkillSource(spec.source)) {
    errors.push("source may only use the provided api object.");
  } else if (spec.action === "solveMaze" && /\bseekTarget\s*\(/.test(spec.source)) {
    errors.push("solveMaze source must implement its own graph search instead of calling api.seekTarget().");
  }

  if (spec.plan !== undefined && (typeof spec.plan !== "object" || Array.isArray(spec.plan))) {
    errors.push("plan must be an object when provided.");
  }

  if (
    spec.priority !== undefined &&
    (!Number.isInteger(spec.priority) || spec.priority < 1 || spec.priority > 99)
  ) {
    errors.push("priority must be an integer from 1 to 99 when provided.");
  }

  return errors;
}

function validName(name) {
  return typeof name === "string" && /^[a-z]+:[a-z0-9-]+$/.test(name);
}

function compileGeneratedSkillSource(source) {
  if (typeof source !== "string" || !source.trim() || containsUnsafeSkillSource(source)) {
    return { ok: false, errors: ["source may only use the provided api object."] };
  }

  try {
    const fn = new Function("api", `"use strict";\n${source}`);
    return {
      ok: true,
      run(api) {
        const result = fn(api);
        if (result && typeof result === "object" && typeof result.ok === "boolean") {
          return result;
        }
        return { ok: false, reason: "generated_skill_returned_invalid_result" };
      }
    };
  } catch (error) {
    return {
      ok: false,
      errors: [error instanceof Error ? error.message : "source could not be compiled."]
    };
  }
}

function containsUnsafeSkillSource(source) {
  return /\b(window|document|globalThis|Function|eval|fetch|XMLHttpRequest|WebSocket|import|constructor|__proto__|prototype|localStorage|sessionStorage|indexedDB|process|require)\b/.test(source);
}

function createSkillApi(sandbox, spec) {
  return Object.freeze({
    maze: () => sandbox.mazeSnapshot(),
    moveTo: (point) => sandbox.moveTo(point),
    seekTarget: () => sandbox.seekTarget(),
    harvestBushOrApproach: () => sandbox.harvestBushOrApproach(),
    collectKeyOrApproach: () => sandbox.collectKeyOrApproach(),
    openGateOrApproach: () => sandbox.openGateOrApproach(),
    trainMovementController: () => sandbox.trainMovementController(),
    probeMechanism: () => sandbox.probeMechanism(),
    readClueOrApproach: () => sandbox.readClueOrApproach(),
    executeProcedure: () => sandbox.executeProcedure(spec),
    buildBridgeOrApproach: () => sandbox.buildBridgeOrApproach(),
    shakeTreeOrApproach: () => sandbox.shakeTreeOrApproach(),
    askGrove: () => sandbox.askGrove(),
    fail: (reason = "generated_skill_failed") => ({ ok: false, reason: String(reason) }),
    state: () => sandbox.snapshot()
  });
}
