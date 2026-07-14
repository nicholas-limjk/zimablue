export class PrimitiveTargetAgent {
  decide(observation) {
    return observation.targetReached ? 0 : 1;
  }
}

export const PrimitiveFruitAgent = PrimitiveTargetAgent;
