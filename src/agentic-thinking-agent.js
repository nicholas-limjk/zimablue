import { actionNames, compileSkillSpec } from "./skill-actions.js";

export class AgenticThinkingAgent {
  constructor({ provider, maxAttempts = 2, fallback = null } = {}) {
    if (!provider) {
      throw new Error("AgenticThinkingAgent requires a provider with proposeSkill(context).");
    }

    this.provider = provider;
    this.maxAttempts = maxAttempts;
    this.fallback = fallback;
  }

  async reflect({ program, sandbox, outcome, trace = [] }) {
    const usefulActions = usefulActionsNow({ program, sandbox, outcome });
    if (usefulActions.length === 0) {
      return {
        patched: false,
        message: "No mutation pressure this turn.",
        source: "agent"
      };
    }

    const context = buildSkillContext({ program, sandbox, outcome, trace, usefulActions });
    let lastErrors = [];

    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      debugLog(`requesting skill attempt ${attempt}`);
      const spec = await this.provider.proposeSkill({
        ...context,
        attempt,
        validationErrors: lastErrors
      });
      debugLog(`received skill attempt ${attempt}: ${spec?.name ?? "invalid"}`);

      const compiled = compileSkillSpec(spec);
      if (!compiled.ok) {
        lastErrors = compiled.errors;
        continue;
      }

      if (program.has(compiled.behavior.name)) {
        lastErrors = [`${compiled.behavior.name} already exists in the behavior program.`];
        continue;
      }

      if (program.hasAction(compiled.behavior.action)) {
        lastErrors = [
          `${compiled.behavior.action} is already present under another skill name. Choose a new action.`
        ];
        continue;
      }

      if (!usefulActions.includes(compiled.behavior.action)) {
        lastErrors = [
          `Current mutation pressure only allows: ${usefulActions.join(", ")}. ` +
            `Choose one of those actions.`
        ];
        continue;
      }

      program.add(compiled.behavior, compiled.behavior.reason);

      return {
        patched: true,
        behavior: compiled.behavior.name,
        message: compiled.behavior.reason,
        source: "agent",
        spec
      };
    }

    if (this.fallback) {
      const fallbackReflection = this.fallback.reflect({ program, sandbox, outcome });
      return {
        ...fallbackReflection,
        source: fallbackReflection.patched ? "fallback" : "agent",
        validationErrors: lastErrors
      };
    }

    return {
      patched: false,
      message: "The skill-writing agent did not produce a valid new skill.",
      source: "agent",
      validationErrors: lastErrors
    };
  }
}

export class OpenAICompatibleSkillProvider {
  constructor({
    apiKey = process.env.AGENT_SKILL_API_KEY,
    endpoint = process.env.AGENT_SKILL_ENDPOINT,
    model = process.env.AGENT_SKILL_MODEL ?? "gpt-4.1-mini",
    timeoutMs = Number.parseInt(process.env.AGENT_SKILL_TIMEOUT_MS ?? "20000", 10),
    fetchImpl = globalThis.fetch
  } = {}) {
    if (!apiKey) throw new Error("Missing AGENT_SKILL_API_KEY.");
    if (!endpoint) throw new Error("Missing AGENT_SKILL_ENDPOINT.");
    if (!fetchImpl) throw new Error("No fetch implementation is available.");

    this.apiKey = apiKey;
    this.endpoint = endpoint;
    this.model = model;
    this.timeoutMs = timeoutMs;
    this.fetch = fetchImpl;
  }

  async proposeSkill(context) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

    const response = await this.fetch(this.endpoint, {
      method: "POST",
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: this.model,
        messages: [
          {
            role: "system",
            content:
              "You write one new JSON skill for a toy target-reaching agent. " +
              "Return only JSON. No markdown."
          },
          {
            role: "user",
            content: buildPrompt(context)
          }
        ],
        temperature: 0.2
      })
    }).finally(() => clearTimeout(timeout));

    if (!response.ok) {
      throw new Error(`Skill provider failed with HTTP ${response.status}.`);
    }

    const payload = await response.json();
    const text = payload.choices?.[0]?.message?.content;

    if (typeof text !== "string") {
      throw new Error("Skill provider response did not contain choices[0].message.content.");
    }

    return parseJsonObject(text);
  }

}

export class ScriptedSkillProvider {
  constructor(specs) {
    this.specs = [...specs];
  }

  async proposeSkill(context = {}) {
    const index = this.specs.findIndex((spec) => context.usefulActions?.includes(spec.action));
    if (index < 0) return this.specs.shift() ?? null;
    return this.specs.splice(index, 1)[0];
  }
}

export function buildSkillContext({ program, sandbox, outcome, trace = [], usefulActions = actionNames() }) {
  const snapshot = sandbox.snapshot();

  return {
    impulse: "reach_target",
    brief: WORLD_BRIEF,
    targetReached: snapshot.targetReached,
    world: observedWorldForSkillContext(snapshot),
    dynamicsEvidence: snapshot.memory.movementSamples,
    existingProgram: program.snapshot(),
    existingActions: program.snapshot().map((behavior) => behavior.action).filter(Boolean),
    usefulActions,
    latestOutcome: outcome,
    recentTrace: trace.slice(-6).map((entry) => ({
      tick: entry.tick,
      status: entry.status,
      behavior: entry.behavior,
      failures: entry.failures
    })),
    allowedActions: actionNames(),
    outputSchema: {
      name: "namespaced id, e.g. tool:collect-key",
      action: "one allowed action",
      reason: "why this skill is useful now",
      code: "short human-readable pseudo-code only; do not put executable JavaScript here",
      source: "required executable JavaScript function body. For solveMaze, call api.maze(), compute a route yourself, then return api.moveTo(nextPoint).",
      plan: "optional structured notes",
      priority: "optional integer 1-99"
    }
  };
}

export function buildPrompt(context) {
  return [
    "The base impulse is fixed: reach_target.",
    "The base body is weak. It can only run its current behavior program.",
    "Environment brief:",
    ...WORLD_BRIEF.map((line) => `- ${line}`),
    "",
    "Choose exactly one new skill that helps the next turns.",
    "If the useful action is solveMaze, write the maze-solving algorithm in source; do not delegate to api.seekTarget().",
    "",
    "Rules:",
    "- Return one JSON object only.",
    "- Do not choose a skill name already in existingProgram.",
    "- Do not choose an action already in existingActions, even with a new name.",
    "- action must be one of allowedActions.",
    "- For this turn, action must be one of usefulActions.",
    "- source is required. It must be a JavaScript function body that uses only the provided api object and returns the api result.",
    "- Put executable JavaScript only in source. The code field is just a short prose summary.",
    "- For solveMaze source, use api.maze() to read {width,height,agent,target,walls}, run BFS/A*/DFS over passable cells, and return api.moveTo(path[1]).",
    "- Do not use api.seekTarget() for solveMaze; the point is that the generated source writes the maze algorithm itself.",
    "- Prefer the smallest useful next capability.",
    "",
    JSON.stringify(context, null, 2)
  ].join("\n");
}

export const WORLD_BRIEF = [
  "The current browser world is a grid maze with one beacon target and many rock walls.",
  "The base body only knows how to test target-underfoot and try one random adjacent move.",
  "After random movement repeatedly collides with walls, infer that the world requires maze solving rather than more wandering.",
  "For solveMaze, generated source must call api.maze(), build its own graph search over passable cells, and return api.moveTo(nextPoint).",
  "The api.moveTo call accepts only one adjacent cell, so the solver must execute one route step per world turn.",
  "Thinking consumes a clock tick, and the target must be reached before the strict turn limit."
];

function observedWorldForSkillContext(snapshot) {
  const visibleRadius = 3;
  const localCells = [];
  const hasMechanismRejection = snapshot.memory.mechanismAttempts?.some(
    (attempt) => attempt.result === "rejected"
  );

  for (let y = snapshot.agent.y - visibleRadius; y <= snapshot.agent.y + visibleRadius; y += 1) {
    for (let x = snapshot.agent.x - visibleRadius; x <= snapshot.agent.x + visibleRadius; x += 1) {
      if (x < 0 || y < 0 || x >= snapshot.width || y >= snapshot.height) continue;

      const point = { x, y };
      const tags = [];
      if (sameCell(snapshot.target, point)) tags.push("target");
      if (snapshot.rocks.some((rock) => sameCell(rock, point))) tags.push("rock");
      if (snapshot.water.some((water) => sameCell(water, point))) tags.push("water");
      if (snapshot.wind.some((wind) => sameCell(wind, point))) tags.push("unstable-dynamics");
      if (snapshot.keys.some((key) => sameCell(key, point))) tags.push("key");
      const clue = snapshot.clues.find((candidate) => sameCell(candidate, point));
      if (clue) {
        const read = snapshot.memory.cluesRead?.some((item) => item.id === clue.id);
        tags.push(read ? `clue:${clue.id}:read` : `clue:${clue.id}:unread`);
      }
      if (sameCell(snapshot.gate, point)) tags.push(snapshot.gate.open ? "open-gate" : "locked-gate");
      if (snapshot.logicGate && sameCell(snapshot.logicGate, point)) {
        tags.push(snapshot.logicGate.open ? "open-seal" : "sealed-mechanism");
      }
      const plate = snapshot.switches.find((candidate) => sameCell(candidate, point));
      if (plate) tags.push(`switch:${plate.id}:value-${plate.value}:${plate.kind}`);
      if (tags.length > 0) localCells.push({ ...point, tags });
    }
  }

  return {
    width: snapshot.width,
    height: snapshot.height,
    agent: snapshot.agent,
    targetHint: "the target is visible in the world, but the full obstacle map is not provided to the skill writer",
    visibleRadius,
    localCells,
    knownKeys: snapshot.memory.lockedGateObserved ? snapshot.keys : [],
    gate: snapshot.memory.lockedGateObserved || snapshot.gate.open ? snapshot.gate : "unobserved",
    logicGate: snapshot.memory.logicGateObserved ? snapshot.logicGate : "unobserved",
    switches: hasMechanismRejection ? snapshot.switches : "unobserved until a mechanism interaction fails",
    cluesRead: snapshot.memory.cluesRead ?? [],
    logicRule: (snapshot.memory.cluesRead?.length ?? 0) > 0
      ? { ...snapshot.logicRule, clue: snapshot.memory.cluesRead.at(-1).text }
      : "unobserved until the written clue is found and read",
    inventory: snapshot.inventory,
    memory: snapshot.memory
  };
}

function sameCell(left, right) {
  return left && right && left.x === right.x && left.y === right.y;
}

function parseJsonObject(text) {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  const json = fenced ? fenced[1] : trimmed;
  return JSON.parse(json);
}

function usefulActionsNow({ program, sandbox, outcome }) {
  if (!program.hasAction("seekTarget") && outcome.status === "acted") {
    return ["seekTarget"];
  }

  if (!program.hasAction("harvestBushOrApproach") && sandbox.needsBushSkill()) {
    return ["harvestBushOrApproach"];
  }

  if (!program.hasAction("collectKeyOrApproach") && sandbox.needsKeySkill()) {
    return ["collectKeyOrApproach"];
  }

  if (!program.hasAction("openGateOrApproach") && sandbox.needsGateSkill()) {
    return ["openGateOrApproach"];
  }

  if (
    !program.hasAction("trainMovementController") &&
    outcomeHasFailureReason(outcome, "movement_model_mismatch") &&
    sandbox.hasMovementDiagnosis()
  ) {
    return ["trainMovementController"];
  }

  if (!program.hasAction("probeMechanism") && sandbox.needsMechanismProbeSkill()) {
    return ["probeMechanism"];
  }

  if (!program.hasAction("readClueOrApproach") && sandbox.needsMechanismClueSkill()) {
    return ["readClueOrApproach"];
  }

  if (!program.hasAction("executeProcedure") && sandbox.needsMechanismSkill()) {
    return ["executeProcedure"];
  }

  if (!program.hasAction("buildBridgeOrApproach") && sandbox.needsBridgeSkill()) {
    return ["buildBridgeOrApproach"];
  }

  if (!program.hasAction("shakeTreeOrApproach") && sandbox.needsTreeSkill()) {
    return ["shakeTreeOrApproach"];
  }

  if (!program.hasAction("askGrove") && sandbox.needsGroveSkill()) {
    return ["askGrove"];
  }

  return [];
}

function outcomeHasFailure(outcome, prefix) {
  return outcome.failures?.some((failure) => failure.startsWith(prefix)) ?? false;
}

function outcomeHasFailureReason(outcome, reason) {
  return outcome.failures?.some((failure) => failure.endsWith(`:${reason}`)) ?? false;
}

function debugLog(message) {
  if (process.env.AGENT_SKILL_DEBUG === "1") {
    console.error(`[agentic-thinker] ${message}`);
  }
}
