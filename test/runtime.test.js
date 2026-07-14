import test from "node:test";
import assert from "node:assert/strict";

import { PrimitiveTargetAgent } from "../src/agent.js";
import { BehaviorProgram, createBaseBehaviors, ThinkingAgent } from "../src/behavior.js";
import { createLearningLabOptions } from "../src/envs/learning-lab.js";
import { createMiniGridDoorKeyOptions } from "../src/envs/minigrid-doorkey.js";
import { Runtime } from "../src/runtime.js";
import { FruitSandbox } from "../src/sandbox.js";

function makeRuntime(options = {}) {
  return new Runtime({
    agent: new PrimitiveTargetAgent(),
    program: new BehaviorProgram(createBaseBehaviors()),
    thinker: new ThinkingAgent(),
    sandbox: new FruitSandbox(options)
  });
}

test("the primitive agent only emits the base target impulse", () => {
  const agent = new PrimitiveTargetAgent();

  assert.equal(agent.decide({ targetReached: false }), 1);
  assert.equal(agent.decide({ targetReached: true }), 0);
});

test("the runtime reaches the goal by letting the thinker patch behavior", () => {
  const result = makeRuntime().run();

  assert.equal(result.status, "complete");
  assert.equal(result.sandbox.targetReached, true);
  assert.deepEqual(
    result.program.map((behavior) => behavior.name),
    [
      "body:reach-target-here",
      "nav:route-search",
      "body:move-east",
      "body:move-south"
    ]
  );
  assert.equal(
    result.trace.some((entry) => entry.reflection?.patched),
    true
  );
});

test("the runtime gets stuck when the thinker runs out of useful patches", () => {
  const result = makeRuntime({
    width: 1,
    height: 1,
    agent: { x: 0, y: 0 },
    target: { x: 1, y: 0 },
    fruits: [],
    bushes: [],
    trees: [],
    keys: [],
    grove: { x: 1, y: 6, cycles: 0 }
  }).run();

  assert.equal(result.status, "stuck");
  assert.equal(result.sandbox.targetReached, false);
});

test("the world fails when the turn limit expires before the goal", () => {
  const result = makeRuntime({
    turnLimit: 1
  }).run();

  assert.equal(result.status, "failed");
  assert.equal(result.sandbox.failed, true);
  assert.equal(result.sandbox.turn, 1);
});

test("the mutation loop solves a MiniGrid DoorKey-style world", () => {
  const result = makeRuntime(createMiniGridDoorKeyOptions()).run();

  assert.equal(result.status, "complete");
  assert.equal(result.sandbox.targetReached, true);
  assert.equal(result.sandbox.inventory.key, true);
  assert.equal(result.sandbox.gate.open, true);

  const lockedObservation = result.trace.findIndex(
    (entry) => entry.sandbox.memory.lockedGateObserved
  );
  const keyPatch = result.trace.findIndex(
    (entry) => entry.reflection?.behavior === "tool:collect-key"
  );
  assert.ok(lockedObservation >= 0);
  assert.ok(keyPatch >= lockedObservation);
  assert.equal(
    result.trace[keyPatch].failures.some((failure) =>
      failure.startsWith("nav:route-search:gate_locked")
    ),
    true
  );
});

test("the local mutation loop cannot solve the learning lab mechanism without an agent-authored procedure", () => {
  const result = makeRuntime(createLearningLabOptions()).run();

  assert.equal(result.status, "stuck");
  assert.equal(result.sandbox.targetReached, false);
  assert.equal(result.sandbox.inventory.key, true);
  assert.equal(result.sandbox.gate.open, true);
  assert.equal(result.sandbox.logicGate.open, false);
  assert.equal(
    result.program.some((behavior) => behavior.action === "executeProcedure"),
    false
  );

  const logicObservation = result.trace.findIndex(
    (entry) => entry.sandbox.memory.logicGateObserved
  );
  assert.equal(
    result.trace.some((entry) =>
      entry.failures?.some((failure) => failure.startsWith("nav:route-search:logic_gate_locked"))
    ),
    true
  );
  assert.ok(logicObservation >= 0);
});

test("the movement controller cannot train before enough movement evidence is observed", () => {
  const sandbox = new FruitSandbox(createLearningLabOptions());

  assert.deepEqual(sandbox.trainMovementController(), {
    ok: false,
    reason: "insufficient_movement_evidence"
  });
});
