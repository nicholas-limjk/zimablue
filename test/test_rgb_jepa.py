from __future__ import annotations

import tempfile
import unittest
import itertools
import random
from pathlib import Path

import numpy as np
import torch

from zima_py.evolve_jepa_neat import neat_config_for_inputs
from zima_py.evolve_jepa_frontier_ga import LinearFrontierNetwork, PARAMETER_COUNT
from zima_py.evolve_jepa_mpc_neat import (
    FRONTIER_FEATURE_COUNT,
    MODE_FEATURE_COUNT,
    geometry_balanced_sample,
    lava_layout_geometry,
    fixed_sequence_energy,
    frontier_selector_features,
    mode_selector_features,
    remap_genome_innovations,
    safe_required_action,
    selected_frontier,
    select_frontier_action,
    select_mode_action,
)
from zima_py.evaluate_jepa_mpc import outcome_sequence_features
from zima_py.evaluate_temporal_jepa_rollouts import (
    build_branched_samples,
    summarize_rows,
    transition_category,
)
from zima_py.episodic_latent_memory import EpisodicLatentMemory
from zima_py.episodic_latent_graph import EpisodicLatentGraph
from zima_py.evolve_rgb_neat import (
    active_belief_lexicase_scores,
    apply_rgb_style,
    manhattan,
    parse_horizons,
)
from zima_py.jepa_belief import (
    JepaBeliefHead,
    TinyRecursiveBeliefHead,
    load_belief_head,
)
from zima_py.jepa_evidence import JepaEvidenceMapHead
from zima_py.jepa_rollout_outcome import JepaRolloutOutcomeHead
from zima_py.minigrid_adapter import MiniGridSpec, make_minigrid
from zima_py.rgb_jepa import RgbJepaAgent, TemporalRgbJepaAgent
from zima_py.select_jepa_mpc_restart import selection_key, validation_metrics


class RgbJepaTests(unittest.TestCase):
    def test_matched_branches_expand_to_each_requested_horizon(self) -> None:
        frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
        rows = []
        for action in (0, 1, 2):
            rows.append(
                {
                    "frames": frames,
                    "previous_actions": np.asarray([-1, 2, 1], dtype=np.int64),
                    "future_actions": np.asarray([action, 2, 2, 2], dtype=np.int64),
                    "future_images": np.stack(
                        [np.full((16, 16, 3), action + step, dtype=np.uint8) for step in range(4)]
                    ),
                    "category": "survived_4",
                    "first_action": action,
                }
            )

        samples = build_branched_samples(rows, horizons=(1, 2, 4))

        self.assertEqual(len(samples), 9)
        self.assertEqual({sample.horizon for sample in samples}, {1, 2, 4})
        self.assertEqual(
            {sample.category for sample in samples},
            {"survived_4:action=0", "survived_4:action=1", "survived_4:action=2"},
        )
        horizon_four = next(
            sample for sample in samples if sample.horizon == 4 and sample.category.endswith("=2")
        )
        np.testing.assert_array_equal(horizon_four.future_actions, [2, 2, 2, 2])
        self.assertEqual(int(horizon_four.target_image[0, 0, 0]), 5)

    def test_reactive_racing_distance_uses_finish_progress(self) -> None:
        self.assertEqual(manhattan((1, 1), (4, 4)), 6)
        self.assertEqual(manhattan((3, 4), (4, 4)), 1)

    def test_active_belief_lexicase_exposes_concurrent_cases(self) -> None:
        outcomes = [
            {
                "fitness": 10.0,
                "solved": True,
                "steps": 20,
                "unique_observations": 18,
                "unique_positions": 15,
                "blocked_steps": 0,
                "belief_cumulative_reduction": 0.8,
            },
            {
                "fitness": -0.5,
                "solved": False,
                "steps": 100,
                "unique_observations": 4,
                "unique_positions": 1,
                "blocked_steps": 20,
                "belief_cumulative_reduction": 0.1,
            },
        ]
        scores = active_belief_lexicase_scores(outcomes, 192)
        self.assertEqual(len(scores), 8)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[3])
        self.assertGreater(scores[4], scores[5])
        self.assertGreater(scores[6], scores[7])

    def test_recurrent_belief_head_steps_and_round_trips(self) -> None:
        belief = JepaBeliefHead(latent_size=128, action_count=7, hidden_size=16)
        latent = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        first, hidden = belief.step(latent, -1)
        second, _ = belief.step(latent, 2, hidden)
        self.assertEqual(first.shape, (28,))
        self.assertEqual(second.shape, (28,))
        self.assertFalse(np.allclose(first, second))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "belief.pt"
            belief.save(path, metadata={"privileged_labels_training_only": True})
            restored = JepaBeliefHead.load(path)
            replay, _ = restored.step(latent, -1)
            np.testing.assert_allclose(first, replay, atol=1e-7)
            self.assertTrue(
                JepaBeliefHead.checkpoint_metadata(path)["privileged_labels_training_only"]
            )
        uncertainty_belief = JepaBeliefHead(
            latent_size=128,
            action_count=7,
            hidden_size=16,
            exploration_size=4,
        )
        uncertainty_features, _ = uncertainty_belief.step(latent, -1)
        self.assertEqual(uncertainty_features.shape, (32,))

    def test_tiny_recursive_belief_refines_and_round_trips(self) -> None:
        latent = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        recursive = TinyRecursiveBeliefHead(
            latent_size=128,
            action_count=7,
            hidden_size=64,
            exploration_size=4,
            recursion_depth=6,
            refinement_size=96,
        )
        features, state = recursive.step(latent, -1)
        continued, _ = recursive.step(latent, 2, state)
        self.assertEqual(features.shape, (80,))
        self.assertFalse(np.allclose(features, continued))
        parameter_count = sum(parameter.numel() for parameter in recursive.parameters())
        self.assertGreater(parameter_count, 39_000)
        self.assertLess(parameter_count, 43_000)

        depth_one = TinyRecursiveBeliefHead(
            latent_size=128,
            action_count=7,
            hidden_size=64,
            exploration_size=4,
            recursion_depth=1,
            refinement_size=96,
        )
        depth_one.load_state_dict(recursive.state_dict())
        shallow, _ = depth_one.step(latent, -1)
        self.assertFalse(np.allclose(features, shallow))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recursive-belief.pt"
            recursive.save(path, metadata={"belief_architecture": "tiny-recursive"})
            restored = load_belief_head(path)
            self.assertIsInstance(restored, TinyRecursiveBeliefHead)
            replay, _ = restored.step(latent, -1)
            np.testing.assert_allclose(features, replay, atol=1e-7)

    def test_jepa_evidence_map_decodes_and_persists_observed_evidence(self) -> None:
        evidence = JepaEvidenceMapHead(latent_size=128, action_count=7)
        evidence.set_calibration(
            0.8,
            torch.as_tensor((0.5, 0.2, -0.1, -0.3, -0.3)),
            torch.as_tensor((0.6, 0.3, 0.08, 0.02)),
        )
        latent = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        logits = evidence.terrain_logits(torch.as_tensor(latent).reshape(1, -1))
        self.assertEqual(tuple(logits.shape), (1, 7, 7, 5))
        features, state = evidence.step(latent, -1)
        self.assertEqual(features.shape, (25,))
        self.assertGreater(len(state.evidence), 0)
        next_features, next_state = evidence.step(latent * 0.9, 0, state)
        self.assertEqual(next_state.direction, 3)
        self.assertFalse(np.allclose(features, next_features))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence-map.pt"
            evidence.save(path, metadata={"belief_architecture": "evidence_map"})
            restored = load_belief_head(path)
            self.assertIsInstance(restored, JepaEvidenceMapHead)
            self.assertAlmostEqual(float(restored.calibration_temperature), 0.8, places=6)
            replay, _ = restored.step(latent, -1)
            np.testing.assert_allclose(features, replay, atol=1e-7)

    def test_feedforward_neat_config_is_explicit_and_has_requested_inputs(self) -> None:
        config = neat_config_for_inputs(139, lexicase=True, feed_forward=True)
        text = config.read_text(encoding="utf-8")
        self.assertIn("num_inputs              = 139", text)
        self.assertIn("feed_forward            = True", text)
        self.assertIn("[LexicaseReproduction]", text)

    def test_neat_config_can_create_scalar_sequence_scorer(self) -> None:
        config = neat_config_for_inputs(
            28,
            lexicase=True,
            feed_forward=True,
            output_count=1,
        )
        text = config.read_text(encoding="utf-8")
        self.assertIn("num_inputs              = 28", text)
        self.assertIn("num_outputs             = 1", text)

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
        self.assertIn("absorbing_loss", metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_temporal_jepa_applies_absorbing_loss_after_terminal_step(self) -> None:
        agent = TemporalRgbJepaAgent(
            3,
            context_length=4,
            horizons=(1, 2, 4),
            absorbing_weight=1.0,
            batch_size=2,
            warmup=2,
            seed=23,
        )
        frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        terminal = frames[-1].copy()
        terminal[8:24, 8:24, 0] = 255
        future_images = np.stack([terminal] * 4)
        for action in (0, 2):
            agent.observe_sequence(
                frames,
                [-1, 2, 1],
                [action, 1, 2, 0],
                future_images,
                terminal_step=0,
            )

        metrics = agent.train_step()

        self.assertIsNotNone(metrics)
        self.assertGreater(metrics["absorbing_loss"], 0.0)

    def test_temporal_counterfactual_features_are_spatial_and_action_sensitive(self) -> None:
        agent = TemporalRgbJepaAgent(7, context_length=4, seed=29)
        rng = np.random.default_rng(29)
        frames = rng.integers(0, 256, size=(4, 56, 56, 3), dtype=np.uint8)
        horizons = (1, 2, 4)
        first = agent.features_with_counterfactuals(frames, [0, 2, 1], horizons)
        replay = agent.features_with_counterfactuals(frames, [0, 2, 1], horizons)
        expected = agent.feature_size + agent.counterfactual_feature_size(horizons)
        self.assertEqual(first.shape, (expected,))
        self.assertEqual(expected, 425)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, replay, atol=1e-7)

        per_action = len(horizons) * agent.counterfactual_features_per_query
        imagined = first[agent.feature_size :]
        self.assertFalse(np.allclose(imagined[:per_action], imagined[per_action : 2 * per_action]))

        counterfactual_only = agent.features_with_counterfactuals(
            frames,
            [0, 2, 1],
            horizons,
            include_context=False,
        )
        relative = agent.features_with_counterfactuals(
            frames,
            [0, 2, 1],
            horizons,
            include_context=False,
            relative=True,
        )
        self.assertEqual(counterfactual_only.shape, (297,))
        self.assertEqual(relative.shape, (297,))
        np.testing.assert_allclose(relative.reshape(3, -1).mean(axis=0), 0.0, atol=1e-7)

    def test_counterfactual_horizon_parser_is_canonical(self) -> None:
        self.assertEqual(parse_horizons("4,1,2,2"), (1, 2, 4))
        self.assertEqual(parse_horizons(""), ())

    def test_predictor_reset_preserves_encoder_and_changes_imagined_futures(self) -> None:
        agent = TemporalRgbJepaAgent(7, context_length=4, seed=31)
        rng = np.random.default_rng(31)
        frames = rng.integers(0, 256, size=(4, 56, 56, 3), dtype=np.uint8)
        before = agent.features_with_counterfactuals(frames, [0, 1, 2], (1, 2))
        agent.reset_predictor(97)
        after = agent.features_with_counterfactuals(frames, [0, 1, 2], (1, 2))
        np.testing.assert_allclose(before[: agent.feature_size], after[: agent.feature_size], atol=1e-7)
        self.assertFalse(np.allclose(before[agent.feature_size :], after[agent.feature_size :]))

    def test_temporal_context_tensor_preserves_spatial_shape(self) -> None:
        agent = TemporalRgbJepaAgent(7, context_length=4, seed=37)
        frames = np.zeros((4, 56, 56, 3), dtype=np.uint8)
        latent = agent.encode_context(frames, [-1, 0, 2])
        self.assertEqual(tuple(latent.shape), (1, 8, 4, 4))
        frame_latent = agent.encode_frame(frames[-1])
        self.assertEqual(tuple(frame_latent.shape), (1, 8, 4, 4))

    def test_rollout_outcome_head_round_trips(self) -> None:
        head = JepaRolloutOutcomeHead(latent_size=128, max_horizon=4, hidden_size=32)
        latent = torch.zeros(3, 8, 4, 4)
        output = head(latent, torch.as_tensor([0, 2, 4]))
        self.assertEqual(tuple(output["route_logits"].shape), (3, 3))
        self.assertEqual(tuple(output["collision_logits"].shape), (3, 3))
        self.assertEqual(tuple(output["goal"].shape), (3, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcome.pt"
            head.save(path, metadata={"imagined_latents": True})
            restored = JepaRolloutOutcomeHead.load(path)
            replay = restored(latent, torch.as_tensor([0, 2, 4]))
            np.testing.assert_allclose(
                output["goal"].detach().numpy(), replay["goal"].detach().numpy(), atol=1e-7
            )

    def test_sequence_features_preserve_ordered_rollout_outcomes(self) -> None:
        agent = TemporalRgbJepaAgent(7, context_length=4, seed=41)
        head = JepaRolloutOutcomeHead(latent_size=128, max_horizon=4, hidden_size=32)
        frames = np.zeros((4, 56, 56, 3), dtype=np.uint8)
        latent = agent.encode_context(frames, [-1, 0, 2])
        sequences = torch.as_tensor(list(itertools.product((0, 1, 2), repeat=4)))
        features = outcome_sequence_features(agent, head, latent, sequences, 0.0)
        self.assertEqual(features.shape, (81, 28))
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(np.allclose(features[0], features[-1]))
        memory = EpisodicLatentMemory(capacity=8)
        memory.add(latent)
        memory_features = outcome_sequence_features(
            agent, head, latent, sequences, 0.0, memory
        )
        self.assertEqual(memory_features.shape, (81, 37))
        graph = EpisodicLatentGraph(capacity=8)
        graph.observe(latent)
        graph_features = outcome_sequence_features(
            agent, head, latent, sequences, 0.0, None, graph
        )
        self.assertEqual(graph_features.shape, (81, 36))

    def test_rollout_transition_categories_distinguish_lava_and_motion(self) -> None:
        from types import SimpleNamespace

        terrain = np.zeros((7, 7), dtype=np.int64)
        terrain[5, 3] = 3
        current = SimpleNamespace(terrain=terrain)
        self.assertEqual(
            transition_category(current, SimpleNamespace(previous_action=2, moved_last=0.0)),
            "lava_visible",
        )
        terrain.fill(0)
        self.assertEqual(
            transition_category(current, SimpleNamespace(previous_action=2, moved_last=1.0)),
            "free_forward",
        )
        self.assertEqual(
            transition_category(current, SimpleNamespace(previous_action=0, moved_last=0.0)),
            "turn",
        )

    def test_rollout_metric_summary_stratifies_models_and_horizons(self) -> None:
        rows = [
            {"model": "trained", "horizon": 1, "category": "turn", "smooth_l1": 0.1, "cosine": 0.9, "l2": 0.2},
            {"model": "random", "horizon": 1, "category": "turn", "smooth_l1": 0.3, "cosine": 0.7, "l2": 0.4},
        ]
        report = summarize_rows(rows)
        self.assertAlmostEqual(report["overall"]["trained"]["smooth_l1"], 0.1)
        self.assertEqual(report["by_horizon"]["1"]["random"]["count"], 1)

    def test_episodic_latent_memory_tracks_novelty_recency_and_density(self) -> None:
        memory = EpisodicLatentMemory(capacity=4, recent_size=2, similarity_threshold=0.9)
        first = torch.zeros(1, 8, 4, 4)
        first[0, 0, 0, 0] = 1.0
        novel = memory.features(first)
        self.assertEqual(tuple(novel.shape), (1, 5))
        self.assertAlmostEqual(float(novel[0, 0]), 1.0, places=6)
        memory.add(first)
        familiar = memory.features(first)
        self.assertLess(float(familiar[0, 0]), 1e-6)
        self.assertGreater(float(familiar[0, 1]), 0.99)
        self.assertGreater(float(familiar[0, 3]), 0.99)
        second = torch.zeros_like(first)
        second[0, 1, 0, 0] = 1.0
        self.assertGreater(float(memory.features(second)[0, 0]), 0.9)
        memory.add(second)
        self.assertEqual(len(memory), 2)
        self.assertEqual(len(memory.signature()), 16)

    def test_episodic_latent_graph_records_action_edges_and_frontiers(self) -> None:
        graph = EpisodicLatentGraph(capacity=8, match_threshold=0.95)
        first = torch.zeros(1, 8, 4, 4)
        first[0, 0, 0, 0] = 1.0
        second = torch.zeros_like(first)
        second[0, 1, 0, 0] = 1.0
        first_node = graph.observe(first)
        second_node = graph.observe(second, previous_action=2)
        self.assertNotEqual(first_node, second_node)
        self.assertEqual(graph.edges[(first_node, 2, second_node)], 1)
        self.assertIn(2, graph.nodes[first_node].tried_actions)
        candidates = torch.cat((first, second), dim=0)
        features = graph.candidate_features(candidates, torch.as_tensor([0, 2]))
        self.assertEqual(tuple(features.shape), (2, 8))
        self.assertGreater(float(features[0, 6]), 0.5)
        self.assertEqual(len(graph.signature()), 16)

    def test_episodic_latent_graph_plans_to_nearest_reachable_frontier(self) -> None:
        graph = EpisodicLatentGraph(capacity=8, match_threshold=0.95)
        latents = []
        for channel in range(3):
            latent = torch.zeros(1, 8, 4, 4)
            latent[0, channel, 0, 0] = 1.0
            latents.append(latent)
        first = graph.observe(latents[0])
        second = graph.observe(latents[1], previous_action=2)
        third = graph.observe(latents[2], previous_action=1)
        graph.observe(latents[0], previous_action=0)
        self.assertEqual(graph.current_node, first)
        self.assertEqual(graph.current_untried_actions(), (0, 1))
        self.assertEqual(graph.current_untried_actions((2,)), ())
        self.assertEqual(graph.frontier_plan(), (2, 1, second))
        self.assertEqual(graph.frontier_plan((2,)), (2, 1, second))
        self.assertEqual(graph.path_to_node(second), (2,))
        graph.nodes[second].tried_actions.add(2)
        self.assertEqual(selected_frontier(graph, "any-action"), (2, 1, second))
        self.assertEqual(selected_frontier(graph, "forward-first"), (2, 2, third))
        graph.nodes[second].visits = 5
        graph.nodes[second].last_seen = 3
        graph.nodes[third].visits = 1
        graph.nodes[third].last_seen = 0
        self.assertEqual(graph.frontier_plan(strategy="nearest"), (2, 1, second))
        self.assertEqual(graph.frontier_plan(strategy="oldest"), (2, 2, third))
        self.assertEqual(graph.frontier_plan(strategy="least-visited"), (2, 2, third))

    def test_mode_selector_can_probe_without_overriding_fixed_safety(self) -> None:
        class ProbeNetwork:
            def activate(self, inputs):
                self.inputs = inputs
                return (0.0, 0.0, 1.0)

        graph = EpisodicLatentGraph(capacity=8)
        latent = torch.zeros(1, 8, 4, 4)
        latent[0, 0, 0, 0] = 1.0
        graph.observe(latent)
        sequences = np.asarray(list(itertools.product((0, 1, 2), repeat=4)))
        features = np.zeros((len(sequences), 36), dtype=np.float64)
        for step in range(4):
            features[:, 5 * step] = 0.5
        features[sequences[:, 0] == 0, 18] = 1.0
        scores = fixed_sequence_energy(features, 4, sequences)
        network = ProbeNetwork()
        selector_inputs = mode_selector_features(features, scores, graph, 0.25)
        self.assertEqual(selector_inputs.shape, (MODE_FEATURE_COUNT,))
        action, mode, target = select_mode_action(
            network, features, sequences, graph, 0.25, safety_margin=3.0
        )
        self.assertEqual(action, 2)
        self.assertEqual(mode, "probe")
        self.assertIsNone(target)
        self.assertEqual(len(network.inputs), MODE_FEATURE_COUNT)

        committed = safe_required_action(features, sequences, action, safety_margin=3.0)
        self.assertEqual(committed, (action, "return"))

    def test_frontier_selector_exposes_three_proposals_and_probe(self) -> None:
        class ProbeNetwork:
            def activate(self, inputs):
                self.inputs = inputs
                return (0.0, 0.0, 0.0, 0.0, 1.0)

        graph = EpisodicLatentGraph(capacity=8)
        latent = torch.zeros(1, 8, 4, 4)
        latent[0, 0, 0, 0] = 1.0
        graph.observe(latent)
        sequences = np.asarray(list(itertools.product((0, 1, 2), repeat=4)))
        features = np.zeros((len(sequences), 36), dtype=np.float64)
        for step in range(4):
            features[:, 5 * step] = 0.5
        features[sequences[:, 0] == 0, 18] = 1.0
        scores = fixed_sequence_energy(features, 4, sequences)
        selector_features = frontier_selector_features(
            features, scores, graph, 0.25
        )
        self.assertEqual(selector_features.shape, (FRONTIER_FEATURE_COUNT,))
        network = ProbeNetwork()
        action, mode, target, strategy = select_frontier_action(
            network, features, sequences, graph, 0.25, safety_margin=3.0
        )
        self.assertEqual(action, 2)
        self.assertEqual(mode, "probe")
        self.assertIsNone(target)
        self.assertIsNone(strategy)
        self.assertEqual(len(network.inputs), FRONTIER_FEATURE_COUNT)

    def test_fixed_topology_frontier_network_has_expected_shape(self) -> None:
        network = LinearFrontierNetwork(np.zeros(PARAMETER_COUNT))
        outputs = network.activate(np.ones(FRONTIER_FEATURE_COUNT))
        self.assertEqual(PARAMETER_COUNT, 125)
        self.assertEqual(len(outputs), 5)
        np.testing.assert_allclose(outputs, 0.0)

    def test_lava_geometry_and_balanced_sampling(self) -> None:
        env = make_minigrid(MiniGridSpec("MiniGrid-LavaCrossingS9N1-v0", 0, 192))
        try:
            self.assertEqual(lava_layout_geometry(env, 52), ("horizontal", 4, 1))
            self.assertEqual(lava_layout_geometry(env, 57), ("vertical", 4, 3))
        finally:
            env.close()
        groups = {"horizontal:gap=1": [1, 2], "vertical:gap=3": [3, 4]}
        sample = geometry_balanced_sample(groups, 2, random.Random(7))
        self.assertEqual(len(sample), 2)
        self.assertEqual(len(set(sample)), 2)
        self.assertTrue(any(seed in {1, 2} for seed in sample))
        self.assertTrue(any(seed in {3, 4} for seed in sample))

    def test_restart_selection_prefers_validation_geometry_breadth(self) -> None:
        env = make_minigrid(MiniGridSpec("MiniGrid-LavaCrossingS9N1-v0", 0, 192))
        report = {
            "candidate_selection_evaluation": {
                "runs": [
                    {"seed": 52, "solved": True, "fitness": 10.0},
                    {"seed": 60, "solved": True, "fitness": 10.0},
                    {"seed": 57, "solved": True, "fitness": 10.0},
                ]
            }
        }
        try:
            metrics = validation_metrics(report, env)
        finally:
            env.close()
        self.assertEqual(metrics["validation_solved"], 3)
        self.assertEqual(metrics["validation_geometry_count"], 2)
        broad = metrics
        narrow = {
            "validation_geometry_count": 1,
            "validation_solved": 10,
            "validation_mean_fitness": 10.0,
        }
        self.assertGreater(selection_key(broad), selection_key(narrow))

    def test_warm_start_remaps_innovations_by_connection_structure(self) -> None:
        from types import SimpleNamespace

        class Tracker:
            def get_innovation_number(self, source, target, mutation_type):
                self.request = (source, target, mutation_type)
                return 11

        reference = SimpleNamespace(
            connections={(-1, 0): SimpleNamespace(innovation=7)}
        )
        genome = SimpleNamespace(
            connections={
                (-1, 0): SimpleNamespace(innovation=2),
                (-2, 1): SimpleNamespace(innovation=3),
            }
        )
        tracker = Tracker()
        remap_genome_innovations(genome, [reference], tracker)
        self.assertEqual(genome.connections[(-1, 0)].innovation, 7)
        self.assertEqual(genome.connections[(-2, 1)].innovation, 11)
        self.assertEqual(tracker.request, (-2, 1, "warm_start_connection"))

    def test_fixed_sequence_energy_prefers_safe_progress(self) -> None:
        features = np.zeros((2, 28), dtype=np.float64)
        for step in range(4):
            offset = 5 * step
            features[:, offset] = 0.5
        features[0, 1] = 0.05
        features[0, 17] = 0.05
        features[0, 18] = 0.1
        features[0, 25] = 1.0
        features[1, 1] = 0.9
        features[1, 17] = 0.2
        features[1, 18] = -0.1
        features[1, 23] = 1.0
        sequences = np.asarray(((2, 2, 2, 2), (0, 0, 0, 0)))
        energy = fixed_sequence_energy(features, 4, sequences)
        self.assertGreater(energy[0], energy[1])

    def test_channel_canonicalization_is_exactly_permutation_invariant(self) -> None:
        agent = TemporalRgbJepaAgent(7, channel_permutation_invariant=True, seed=23)
        rng = np.random.default_rng(23)
        frames = rng.integers(0, 256, size=(4, 56, 56, 3), dtype=np.uint8)
        standard = agent.features(frames, [-1, 0, 2])
        cyclic = agent.features(frames[..., [1, 2, 0]], [-1, 0, 2])
        np.testing.assert_allclose(standard, cyclic, atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
