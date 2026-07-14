from __future__ import annotations

import unittest

import numpy as np

from zima_py.jepa_world_model import LatentActionWorldModel, observation_embedding


def observation(object_value: int, direction: int = 0) -> dict:
    image = np.zeros((7, 7, 3), dtype=np.int64)
    image[:, :, 0] = object_value
    return {"image": image, "direction": direction, "mission": "reach the goal"}


class LatentActionWorldModelTests(unittest.TestCase):
    def test_embedding_is_normalized_and_direction_sensitive(self) -> None:
        north = observation_embedding(observation(2, direction=0))
        east = observation_embedding(observation(2, direction=1))
        self.assertAlmostEqual(float(np.linalg.norm(north)), 1.0, places=6)
        self.assertFalse(np.array_equal(north, east))

    def test_repeated_transition_reduces_prediction_error(self) -> None:
        model = LatentActionWorldModel(action_count=7, learning_rate=0.5)
        before = observation(1, direction=0)
        after = observation(3, direction=1)
        errors = []
        for step in range(30):
            event = model.observe(
                before,
                2,
                after,
                step=step,
                action_name="forward",
                blocked=False,
                reward=0.0,
            )
            errors.append(event["prediction_error"])
        self.assertLess(errors[-1], errors[0])

    def test_evidence_is_compact_and_action_conditioned(self) -> None:
        model = LatentActionWorldModel(action_count=7)
        model.observe(observation(1), 0, observation(1, 1), step=0, action_name="left", blocked=False, reward=0.0)
        model.observe(observation(1), 2, observation(2), step=1, action_name="forward", blocked=True, reward=0.0)
        evidence = model.evidence(limit=1)
        self.assertEqual(evidence["total_transition_updates"], 2)
        self.assertEqual({row["action_value"] for row in evidence["per_action"]}, {0, 2})
        self.assertEqual(len(evidence["highest_recent_prediction_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
