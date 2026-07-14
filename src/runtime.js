const IMPULSES = new Map([
  [0, "rest"],
  [1, "reach_target"]
]);

export class Runtime {
  constructor({ agent, program, thinker, sandbox, maxTicks = 80 }) {
    this.agent = agent;
    this.program = program;
    this.thinker = thinker;
    this.sandbox = sandbox;
    this.maxTicks = maxTicks;
    this.trace = [];
  }

  run() {
    for (let tick = 0; tick < this.maxTicks; tick += 1) {
      const observation = this.sandbox.observe();
      const impulse = this.agent.decide(observation);
      const intent = IMPULSES.get(impulse) ?? "unknown";

      if (intent === "rest") {
        this.#record(tick, impulse, intent, {
          status: "complete",
          message: "The base impulse is quiet. Goal reached."
        });
        return this.#result("complete");
      }

      const outcome = this.program.run(this.sandbox);
      const reflection = this.thinker.reflect({
        program: this.program,
        sandbox: this.sandbox,
        outcome
      });

      this.#record(tick, impulse, intent, outcome, reflection);

      if (outcome.status === "no_action" && !reflection.patched) {
        return this.#result("stuck");
      }

      if (reflection.patched) {
        const thinkingOutcome = this.sandbox.spendThinkingTurn();
        this.#record(tick, impulse, intent, {
          status: "thinking",
          message: thinkingOutcome.event,
          failures: thinkingOutcome.failures ?? []
        });

        if (this.sandbox.failed()) {
          this.#record(tick + 1, impulse, intent, {
            status: "failed",
            message: "The world clock expired while the thinker was extending behavior."
          });
          return this.#result("failed");
        }
      }

      this.sandbox.advanceTurn();

      if (this.sandbox.failed()) {
        this.#record(tick + 1, impulse, intent, {
          status: "failed",
          message: "The world clock expired before the target was reached."
        });
        return this.#result("failed");
      }
    }

    return this.#result("timeout");
  }

  #record(tick, impulse, intent, outcome, reflection = null) {
    this.trace.push({
      tick,
      impulse,
      intent,
      status: outcome.status,
      behavior: outcome.behavior,
      message: outcome.message,
      failures: outcome.failures ?? [],
      reflection,
      sandbox: this.sandbox.snapshot(),
      program: this.program.snapshot()
    });
  }

  #result(status) {
    return {
      status,
      sandbox: this.sandbox.snapshot(),
      program: this.program.snapshot(),
      patches: this.program.patches(),
      trace: this.trace
    };
  }
}
