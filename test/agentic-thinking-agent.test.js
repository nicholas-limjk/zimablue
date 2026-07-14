import test from "node:test";
import assert from "node:assert/strict";

import { PrimitiveTargetAgent } from "../src/agent.js";
import {
  AgenticThinkingAgent,
  ScriptedSkillProvider
} from "../src/agentic-thinking-agent.js";
import { AsyncRuntime } from "../src/async-runtime.js";
import { BehaviorProgram, createBaseBehaviors } from "../src/behavior.js";
import { createLearningLabOptions } from "../src/envs/learning-lab.js";
import { FruitSandbox } from "../src/sandbox.js";
import { compileSkillSpec } from "../src/skill-actions.js";

const generatedSpecs = [
  {
    name: "nav:route-to-target",
    action: "seekTarget",
    reason: "The target is not underfoot, so add route search.",
    code: "Search possible routes to the target, recording blockers.",
    source: "return api.seekTarget();"
  },
  {
    name: "tool:take-key",
    action: "collectKeyOrApproach",
    reason: "A gate key is visible.",
    code: "Move to the key and pick it up.",
    source: "return api.collectKeyOrApproach();"
  },
  {
    name: "tool:unlock-door",
    action: "openGateOrApproach",
    reason: "The key is held and the gate is shut.",
    code: "Move beside the gate and open it.",
    source: "return api.openGateOrApproach();"
  },
  {
    name: "learner:fit-dynamics",
    action: "trainMovementController",
    reason: "Repeated intended-vs-actual transitions disagree.",
    code: "Train a fresh movement controller from dynamics evidence.",
    source: "return api.trainMovementController();"
  },
  {
    name: "logic:probe-mechanism",
    action: "probeMechanism",
    reason: "The route reached a sealed mechanism, so first try a simple interaction and observe rejection.",
    code: "Try one visible switch alone and record whether the mechanism rejects it.",
    source: "return api.probeMechanism();"
  },
  {
    name: "logic:read-tablet",
    action: "readClueOrApproach",
    reason: "The mechanism rejected a blind probe, so search for the written rule.",
    code: "Move to the clue tablet and read the puzzle rule into memory.",
    source: "return api.readClueOrApproach();"
  },
  {
    name: "logic:infer-switches",
    action: "executeProcedure",
    reason: "The mechanism rejected a partial probe, and the clue tablet has been read.",
    code: "Infer the switch sequence from the clue, then execute that sequence.",
    source: "return api.executeProcedure();",
    plan: { switches: ["B", "C"] }
  }
];

test("compiled generated skills must use a whitelisted action", () => {
  const compiled = compileSkillSpec({
    name: "tool:invent-danger",
    action: "evalArbitraryCode",
    reason: "Try something unsafe."
  });

  assert.equal(compiled.ok, false);
  assert.match(compiled.errors.join(" "), /action must be one of/);
});

test("the async runtime can use provider-generated skills", async () => {
  const runtime = new AsyncRuntime({
    agent: new PrimitiveTargetAgent(),
    program: new BehaviorProgram(createBaseBehaviors()),
    thinker: new AgenticThinkingAgent({
      provider: new ScriptedSkillProvider(generatedSpecs)
    }),
    sandbox: new FruitSandbox(createLearningLabOptions())
  });

  const result = await runtime.run();

  assert.equal(result.status, "complete");
  assert.equal(result.sandbox.targetReached, true);
  assert.equal(result.sandbox.logicGate.open, true);
  assert.equal(
    result.trace.some((entry) => entry.reflection?.source === "agent"),
    true
  );
  assert.deepEqual(
    result.program.map((behavior) => behavior.name),
    [
      "body:reach-target-here",
      "nav:route-to-target",
      "tool:take-key",
      "tool:unlock-door",
      "learner:fit-dynamics",
      "logic:probe-mechanism",
      "logic:read-tablet",
      "logic:infer-switches",
      "body:move-east",
      "body:move-south"
    ]
  );

  const lockedObservation = result.trace.findIndex(
    (entry) => entry.sandbox.memory.lockedGateObserved
  );
  const keyPatch = result.trace.findIndex(
    (entry) => entry.reflection?.behavior === "tool:take-key"
  );
  assert.ok(lockedObservation >= 0);
  assert.ok(keyPatch >= lockedObservation);
  assert.equal(
    result.trace[keyPatch].failures.some((failure) =>
      failure.endsWith(":gate_locked")
    ),
    true
  );

  const movementDiagnosis = result.trace.findIndex(
    (entry) => entry.sandbox.memory.movementSamples.length >= 3
  );
  const controllerPatch = result.trace.findIndex(
    (entry) => entry.reflection?.behavior === "learner:fit-dynamics"
  );
  assert.ok(movementDiagnosis >= 0);
  assert.ok(controllerPatch >= movementDiagnosis);
  assert.equal(result.trace[controllerPatch].status, "no_action");
  assert.equal(result.trace[controllerPatch].reflection.source, "agent");
  assert.equal(
    result.trace[controllerPatch].failures.some((failure) =>
      failure.endsWith(":movement_model_mismatch")
    ),
    true
  );

  const logicObservation = result.trace.findIndex(
    (entry) => entry.sandbox.memory.logicGateObserved
  );
  const mechanismPatch = result.trace.findIndex(
    (entry) => entry.reflection?.behavior === "logic:infer-switches"
  );
  const cluePatch = result.trace.findIndex(
    (entry) => entry.reflection?.behavior === "logic:read-tablet"
  );
  const clueRead = result.trace.findIndex(
    (entry) => entry.sandbox.memory.cluesRead?.length > 0
  );
  const mechanismProbe = result.trace.findIndex(
    (entry) => entry.reflection?.behavior === "logic:probe-mechanism"
  );
  const mechanismRejection = result.trace.findIndex(
    (entry) => entry.sandbox.memory.mechanismAttempts?.some((attempt) => attempt.result === "rejected")
  );
  assert.ok(logicObservation >= 0);
  assert.ok(mechanismProbe >= logicObservation);
  assert.ok(mechanismRejection >= mechanismProbe);
  assert.ok(cluePatch >= mechanismRejection);
  assert.ok(clueRead >= cluePatch);
  assert.ok(mechanismPatch >= clueRead);
  assert.equal(result.trace[mechanismPatch].reflection.source, "agent");
});
