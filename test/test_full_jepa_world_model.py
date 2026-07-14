from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from zima_py.full_jepa_world_model import FullJepaWorldModel


def observation(value: int, direction: int = 0) -> dict:
    image = np.zeros((7, 7, 3), dtype=np.int64)
    image[..., 0] = value
    image[3, 3, 1] = value % 6
    return {"image": image, "direction": direction, "mission": "reach goal"}


class FullJepaWorldModelTests(unittest.TestCase):
    def make_model(self) -> FullJepaWorldModel:
        return FullJepaWorldModel(
            7,
            latent_dim=16,
            batch_size=2,
            horizon=2,
            warmup=3,
            seed=4,
        )

    def populate(self, model: FullJepaWorldModel, count: int = 6) -> None:
        for step in range(count):
            model.observe(
                observation(1 + step % 2, step % 4),
                step % 3,
                observation(1 + (step + 1) % 2, (step + 1) % 4),
                step=step,
                action_name=str(step % 3),
                blocked=False,
                reward=0.0,
                terminal=False,
                train_steps=1,
            )

    def test_training_updates_context_but_target_has_no_gradients(self) -> None:
        model = self.make_model()
        before = [parameter.detach().clone() for parameter in model.context_encoder.parameters()]
        self.populate(model)
        self.assertGreater(model.total_train_steps, 0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.context_encoder.parameters())))
        self.assertTrue(all(parameter.grad is None for parameter in model.target_encoder.parameters()))

    def test_evidence_reports_full_model_maturity(self) -> None:
        model = self.make_model()
        self.populate(model)
        evidence = model.evidence()
        self.assertEqual(evidence["kind"], "full_jepa_action_conditioned_world_model")
        self.assertEqual(evidence["maturity"], "training")
        self.assertIsNotNone(evidence["latest_train_loss"])
        self.assertEqual(evidence["prediction_horizon"], 2)

    def test_checkpoint_round_trip(self) -> None:
        model = self.make_model()
        self.populate(model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model.save(checkpoint)
            restored = self.make_model()
            restored.load(checkpoint)
        self.assertEqual(restored.total_train_steps, model.total_train_steps)
        for expected, actual in zip(model.context_encoder.parameters(), restored.context_encoder.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
