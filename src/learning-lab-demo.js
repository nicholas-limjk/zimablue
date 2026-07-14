import { PrimitiveTargetAgent } from "./agent.js";
import { BehaviorProgram, createBaseBehaviors, ThinkingAgent } from "./behavior.js";
import { createLearningLabOptions, describeLearningLab } from "./envs/learning-lab.js";
import { Runtime } from "./runtime.js";
import { FruitSandbox } from "./sandbox.js";

const env = describeLearningLab();
const runtime = new Runtime({
  agent: new PrimitiveTargetAgent(),
  program: new BehaviorProgram(createBaseBehaviors()),
  thinker: new ThinkingAgent(),
  sandbox: new FruitSandbox(createLearningLabOptions())
});

const result = runtime.run();

console.log(`${env.id}`);
console.log("=".repeat(env.id.length));
console.log(env.task);
console.log("");

for (const entry of result.trace) {
  const behavior = entry.behavior ? ` via ${entry.behavior}` : "";
  console.log(`t=${entry.tick} impulse=${entry.impulse} status=${entry.status}${behavior}`);

  if (entry.reflection?.patched) {
    console.log(`  thinker patch: ${entry.reflection.behavior}`);
  }

  console.log(
    `  target=${entry.sandbox.targetReached ? 1 : 0}/1 ` +
      `agent=(${entry.sandbox.agent.x},${entry.sandbox.agent.y}) ` +
      `key=${entry.sandbox.inventory.key} gate=${entry.sandbox.gate.open} ` +
      `controller=${entry.sandbox.inventory.neuralController} ` +
      `time=${entry.sandbox.turn}/${entry.sandbox.turnLimit}`
  );
}

console.log("");
console.log(`Final status: ${result.status}`);
console.log("Final behavior program:");
for (const behavior of result.program) {
  console.log(`- ${behavior.name}: ${behavior.code}`);
}
