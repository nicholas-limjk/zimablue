import { PrimitiveTargetAgent } from "./agent.js";
import { AgenticThinkingAgent, OpenAICompatibleSkillProvider } from "./agentic-thinking-agent.js";
import { BehaviorProgram, createBaseBehaviors, ThinkingAgent } from "./behavior.js";
import { AsyncRuntime } from "./async-runtime.js";
import { createLearningLabOptions } from "./envs/learning-lab.js";
import { FruitSandbox } from "./sandbox.js";

const thinker = new AgenticThinkingAgent({
  provider: new OpenAICompatibleSkillProvider(),
  fallback: new ThinkingAgent()
});

const runtime = new AsyncRuntime({
  agent: new PrimitiveTargetAgent(),
  program: new BehaviorProgram(createBaseBehaviors()),
  thinker,
  sandbox: new FruitSandbox(createLearningLabOptions())
});

const result = await runtime.run();

console.log(`Final status: ${result.status}`);
console.log(`Target: ${result.sandbox.targetReached ? 1 : 0}/1`);
console.log("Generated program:");
for (const behavior of result.program) {
  console.log(`- ${behavior.name}: ${behavior.code}`);
}
console.log("Dynamics evidence:");
for (const sample of result.sandbox.memory.movementSamples) {
  console.log(
    `- from (${sample.from.x},${sample.from.y}) intended (${sample.intended.x},${sample.intended.y}) ` +
      `actual (${sample.actual.x},${sample.actual.y})`
  );
}
