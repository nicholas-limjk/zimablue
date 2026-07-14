from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def observation_embedding(obs: dict[str, Any]) -> np.ndarray:
    """Encode only the body's partial observation into a stable latent target."""
    image = np.asarray(obs["image"], dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("MiniGrid observation image must have shape [width, height, >=3].")

    # MiniGrid channels are categorical. Scaling preserves changes without asking
    # the predictor to reproduce pixels or access hidden environment state.
    object_channel = image[..., 0].reshape(-1) / 10.0
    color_channel = image[..., 1].reshape(-1) / 5.0
    state_channel = image[..., 2].reshape(-1) / 2.0
    direction = np.zeros(4, dtype=np.float32)
    direction_value = int(obs.get("direction", -1))
    if 0 <= direction_value < 4:
        direction[direction_value] = 1.0
    latent = np.concatenate((object_channel, color_channel, state_channel, direction))
    norm = float(np.linalg.norm(latent))
    return latent / max(norm, 1e-6)


@dataclass
class ActionDynamics:
    matrix: np.ndarray
    bias: np.ndarray
    updates: int = 0
    error_ema: float = 0.0


@dataclass
class LatentActionWorldModel:
    """Small NumPy JEPA-style model for testing the skill-writer interaction.

    The observation embedding is deliberately stable in this first experiment;
    the trained parameters are one latent transition and bias per action.
    """

    action_count: int
    learning_rate: float = 0.05
    max_gradient_norm: float = 5.0
    dynamics: dict[int, ActionDynamics] = field(default_factory=dict)
    recent_surprises: list[dict[str, Any]] = field(default_factory=list)
    total_updates: int = 0

    def observe(
        self,
        previous_obs: dict[str, Any],
        action: int,
        next_obs: dict[str, Any],
        *,
        step: int,
        action_name: str,
        blocked: bool,
        reward: float,
    ) -> dict[str, Any]:
        context = observation_embedding(previous_obs)
        target = observation_embedding(next_obs)
        model = self._dynamics_for(int(action), context.size)
        prediction = model.matrix @ context + model.bias
        residual = prediction - target
        surprise = float(np.mean(residual * residual))

        # Online SGD on ||P_a(z_t) - stop_gradient(z_{t+1})||^2.
        grad_prediction = (2.0 / residual.size) * residual
        grad_norm = float(np.linalg.norm(grad_prediction))
        if grad_norm > self.max_gradient_norm:
            grad_prediction *= self.max_gradient_norm / grad_norm
        model.matrix -= self.learning_rate * np.outer(grad_prediction, context)
        model.bias -= self.learning_rate * grad_prediction
        model.updates += 1
        model.error_ema = surprise if model.updates == 1 else 0.9 * model.error_ema + 0.1 * surprise
        self.total_updates += 1

        event = {
            "step": int(step),
            "action": action_name,
            "action_value": int(action),
            "prediction_error": round(surprise, 8),
            "action_error_ema": round(model.error_ema, 8),
            "action_samples": model.updates,
            "blocked": bool(blocked),
            "reward": float(reward),
        }
        self.recent_surprises.append(event)
        del self.recent_surprises[:-64]
        return event

    def evidence(self, limit: int = 8) -> dict[str, Any]:
        per_action = []
        for action, model in sorted(self.dynamics.items()):
            per_action.append(
                {
                    "action_value": action,
                    "samples": model.updates,
                    "prediction_error_ema": round(model.error_ema, 8),
                }
            )
        surprising = sorted(self.recent_surprises, key=lambda item: item["prediction_error"], reverse=True)[:limit]
        return {
            "kind": "jepa_lite_action_conditioned_latent_dynamics",
            "trained_parameters": "one affine latent transition matrix and bias per primitive action",
            "fixed_component": "body-only observation embedding (the encoder is not learned in this first experiment)",
            "total_transition_updates": self.total_updates,
            "per_action": per_action,
            "highest_recent_prediction_errors": surprising,
            "interpretation": (
                "High error means the learned action dynamics did not predict the observed next latent state. "
                "Recurring high error suggests missing world knowledge; low error with poor reward suggests a policy/skill failure."
            ),
        }

    def _dynamics_for(self, action: int, latent_size: int) -> ActionDynamics:
        if action not in self.dynamics:
            self.dynamics[action] = ActionDynamics(
                matrix=np.eye(latent_size, dtype=np.float32),
                bias=np.zeros(latent_size, dtype=np.float32),
            )
        return self.dynamics[action]
