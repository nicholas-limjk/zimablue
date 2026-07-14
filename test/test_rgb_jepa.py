from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from zima_py.evolve_rgb_neat import apply_rgb_style
from zima_py.rgb_jepa import RgbJepaAgent, TemporalRgbJepaAgent


class RgbJepaTests(unittest.TestCase):
    def test_cyclic_rgb_style_changes_color_but_not_geometry(self) -> None:
        image = np.asarray([[[10, 20, 30], [40, 40, 40]]], dtype=np.uint8)
        shifted = apply_rgb_style(image, "cyclic")
        np.testing.assert_array_equal(shifted[0, 0], [20, 30, 10])
        np.testing.assert_array_equal(shifted[0, 1], image[0, 1])

    def test_spatial_features_have_expected_shape(self) -> None:
        agent = RgbJepaAgent(7, batch_size=2, warmup=2, seed=3)
        features = agent.features(np.zeros((56, 56, 3), dtype=np.uint8))
        self.assertEqual(features.shape, (128,))
        self.assertTrue(np.isfinite(features).all())

    def test_training_and_checkpoint_round_trip(self) -> None:
        agent = RgbJepaAgent(7, batch_size=2, warmup=2, seed=5)
        first = np.zeros((56, 56, 3), dtype=np.uint8)
        second = first.copy()
        second[16:32, 16:32, 0] = 255
        agent.observe(first, 2, second)
        agent.observe(second, 0, first)
        self.assertIsNotNone(agent.train_step())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.pt"
            agent.save(path)
            restored = RgbJepaAgent(7, batch_size=2, warmup=2, seed=8)
            restored.load(path)
            np.testing.assert_allclose(agent.features(second), restored.features(second), atol=1e-7)

    def test_temporal_jepa_uses_four_frame_context_and_round_trips(self) -> None:
        agent = TemporalRgbJepaAgent(7, context_length=4, batch_size=2, warmup=2, seed=11)
        frames = np.zeros((4, 56, 56, 3), dtype=np.uint8)
        frames[1, 8:16, :, 0] = 64
        frames[2, 16:24, :, 0] = 128
        frames[3, 24:32, :, 0] = 255
        next_image = np.roll(frames[-1], 4, axis=0)
        agent.observe(frames, [-1, 2, 2], 0, next_image)
        agent.observe(frames[::-1].copy(), [1, 1, 2], 2, frames[0])
        self.assertIsNotNone(agent.train_step())
        features = agent.features(frames, [-1, 2, 2])
        self.assertEqual(features.shape, (128,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temporal.pt"
            agent.save(path)
            restored = TemporalRgbJepaAgent(7, context_length=4, batch_size=2, warmup=2, seed=12)
            restored.load(path)
            np.testing.assert_allclose(features, restored.features(frames, [-1, 2, 2]), atol=1e-7)

    def test_temporal_jepa_trains_color_augmented_multi_horizon_rollout(self) -> None:
        agent = TemporalRgbJepaAgent(
            7,
            context_length=4,
            horizons=(1, 2),
            color_augmentation=True,
            batch_size=2,
            warmup=2,
            seed=17,
        )
        frames = np.zeros((4, 56, 56, 3), dtype=np.uint8)
        frames[-1, 16:32, 16:32] = [240, 80, 20]
        futures = np.stack((np.roll(frames[-1], 2, axis=0), np.roll(frames[-1], 4, axis=0)))
        agent.observe_sequence(frames, [-1, 2, 2], [2, 1], futures)
        agent.observe_sequence(frames[::-1].copy(), [1, 0, 2], [0, 2], futures[::-1].copy())
        metrics = agent.train_step()
        self.assertIsNotNone(metrics)
        self.assertIn("consistency_loss", metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_channel_canonicalization_is_exactly_permutation_invariant(self) -> None:
        agent = TemporalRgbJepaAgent(7, channel_permutation_invariant=True, seed=23)
        rng = np.random.default_rng(23)
        frames = rng.integers(0, 256, size=(4, 56, 56, 3), dtype=np.uint8)
        standard = agent.features(frames, [-1, 0, 2])
        cyclic = agent.features(frames[..., [1, 2, 0]], [-1, 0, 2])
        np.testing.assert_allclose(standard, cyclic, atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
