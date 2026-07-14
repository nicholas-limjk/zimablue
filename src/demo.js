import { PrimitiveTargetAgent } from "./agent.js";
import { BehaviorProgram, createBaseBehaviors, ThinkingAgent } from "./behavior.js";
import { Runtime } from "./runtime.js";
import { FruitSandbox } from "./sandbox.js";

const runtime = new Runtime({
  agent: new PrimitiveTargetAgent(),
  program: new BehaviorProgram(createBaseBehaviors()),
  thinker: new ThinkingAgent(),
  sandbox: new FruitSandbox()
});

const result = runtime.run();

console.log("Zima Blue Toy Agent");
console.log("====================");
console.log("Base impulse: targetReached ? rest : reach_target");
console.log("Base body: reach target here, move east, move south");
console.log("");

for (const entry of result.trace) {
  const behavior = entry.behavior ? ` via ${entry.behavior}` : "";
  console.log(
    `t=${entry.tick} impulse=${entry.impulse} intent=${entry.intent} ` +
      `status=${entry.status}${behavior}`
  );
  console.log(`  ${entry.message}`);

  if (entry.reflection?.patched) {
    console.log(`  thinker patch: ${entry.reflection.behavior}`);
    console.log(`  why: ${entry.reflection.message}`);
  }

  console.log(
    `  target=${entry.sandbox.targetReached ? 1 : 0}/1 ` +
      `agent=(${entry.sandbox.agent.x},${entry.sandbox.agent.y}) ` +
      `loose=${entry.sandbox.fruits.length} ` +
      `bush=${sumFruit(entry.sandbox.bushes)} tree=${sumFruit(entry.sandbox.trees)} ` +
      `key=${entry.sandbox.inventory.key} gate=${entry.sandbox.gate.open} ` +
      `bridge=${entry.sandbox.inventory.bridge} grove=${entry.sandbox.grove.cycles}`
  );
}

console.log("");
console.log(`Final status: ${result.status}`);
console.log("Final behavior program:");
for (const behavior of result.program) {
  console.log(`- ${behavior.name}: ${behavior.code}`);
}

function sumFruit(items) {
  return items.reduce((total, item) => total + item.fruit, 0);
}
