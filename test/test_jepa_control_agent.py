from __future__ import annotations

import unittest
import random
from types import SimpleNamespace

import numpy as np
import torch

from zima_py.jepa_control_agent import ControlTransition, JepaControlAgent, belief_observation
from zima_py.evolve_jepa_neat import aggregate_layout_fitness, raw_body_features
from zima_py.embodied_map import EmbodiedMap
from zima_py.embodied_map import FULL_FEATURE_SIZE
from zima_py.evolution_replay import BalancedEvolutionReplay
from zima_py.lexicase_reproduction import epsilon_lexicase_select
from zima_py.map_elites_archive import MiniGridMapElites
from zima_py.minimal_jepa import MinimalJepaAgent
from zima_py.train_jepa_minigrid import epsilon_at, shaped_reward, update_carrying, useful_actions


def obs(value: int, direction: int = 0) -> dict:
    image = np.zeros((7, 7, 3), dtype=np.int64)
    image[..., 0] = value
    return {"image": image, "direction": direction}


class JepaControlAgentTests(unittest.TestCase):
    def test_control_training_updates_q_and_world_model(self) -> None:
        agent = JepaControlAgent(7, latent_dim=16, batch_size=2, warmup=2, seed=3)
        before_q = [parameter.detach().clone() for parameter in agent.q_head.parameters()]
        for index in range(4):
            agent.observe(
                ControlTransition(
                    state=belief_observation(obs(1, index % 4), index % 3, False),
                    action=index % 3,
                    reward=1.0 if index == 3 else -0.001,
                    next_state=belief_observation(obs(2, (index + 1) % 4), index % 3, index == 3),
                    terminal=index == 3,
                )
            )
        metrics = agent.train_step()
        self.assertIsNotNone(metrics)
        self.assertIn("jepa_loss", metrics)
        self.assertIn("q_loss", metrics)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before_q, agent.q_head.parameters())))

    def test_action_selection_respects_body_affordance_mask(self) -> None:
        agent = JepaControlAgent(7, latent_dim=16, batch_size=2, warmup=2, seed=3)
        state = belief_observation(obs(1), -1, False)
        for _ in range(20):
            self.assertIn(agent.act(state, epsilon=1.0, valid_actions=[0, 2]), {0, 2})

    def test_greedy_action_uses_latent_model_planning(self) -> None:
        agent = JepaControlAgent(7, latent_dim=16, batch_size=2, warmup=2, seed=3)
        state = belief_observation(obs(1), -1, False)
        action = agent.act(state, epsilon=0.0, valid_actions=[0, 1, 2])
        self.assertIn(action, {0, 1, 2})

    def test_world_features_have_stable_neat_interface(self) -> None:
        agent = JepaControlAgent(7, latent_dim=16, batch_size=2, warmup=2, seed=3)
        features = agent.world_features(belief_observation(obs(1), -1, False))
        self.assertEqual(features.shape, (16 + 7 + 7,))
        self.assertTrue(np.isfinite(features).all())

    def test_minimal_jepa_has_small_prediction_only_interface(self) -> None:
        agent = MinimalJepaAgent(7, latent_dim=16, batch_size=2, warmup=2, seed=3)
        for index in range(4):
            agent.observe(
                ControlTransition(
                    state=belief_observation(obs(1, index % 4), index % 3, False),
                    action=index % 3,
                    reward=0.0,
                    next_state=belief_observation(obs(2, (index + 1) % 4), index % 3, False),
                    terminal=False,
                )
            )
        metrics = agent.train_step()
        features = agent.world_features(belief_observation(obs(1), -1, False))
        parameter_count = sum(
            parameter.numel()
            for module in (agent.encoder, agent.target_encoder, agent.predictor)
            for parameter in module.parameters()
        )
        self.assertIsNotNone(metrics)
        self.assertEqual(features.shape, (23,))
        self.assertTrue(np.isfinite(features).all())
        self.assertLess(parameter_count, 20_000)

    def test_raw_neat_interface_contains_no_jepa_features(self) -> None:
        state = belief_observation(obs(1), -1, False)
        features = raw_body_features(state, 7)
        self.assertEqual(features.shape, (160,))
        self.assertTrue(np.isfinite(features).all())

    def test_epsilon_lexicase_prefers_cross_seed_competence(self) -> None:
        members = [
            (1, SimpleNamespace(lexicase_scores=[10.0, 0.0])),
            (2, SimpleNamespace(lexicase_scores=[0.0, 10.0])),
            (3, SimpleNamespace(lexicase_scores=[8.0, 8.0])),
        ]
        random.seed(4)
        selected = [epsilon_lexicase_select(members)[0] for _ in range(20)]
        self.assertEqual(set(selected), {3})

    def test_map_elites_keeps_best_genome_per_behavior(self) -> None:
        archive = MiniGridMapElites()
        weak = SimpleNamespace(key=1)
        strong = SimpleNamespace(key=2)
        outcomes = [
            {
                "collected": True,
                "opened_door": False,
                "solved": False,
                "unique_observations": 12,
                "blocked_steps": 1,
            }
        ]
        self.assertTrue(archive.update(weak, outcomes, fitness=1.0, generation=0))
        self.assertFalse(archive.update(strong, outcomes, fitness=0.5, generation=1))
        self.assertTrue(archive.update(strong, outcomes, fitness=2.0, generation=2))
        self.assertEqual(archive.elites()[0].genome.key, 2)
        self.assertEqual(archive.elites()[0].generation, 2)

    def test_body_rules_use_only_observed_interaction_outcomes(self) -> None:
        actions = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
        self.assertEqual(useful_actions(actions, {"object": "key"}), [0, 1, 2, 3])
        self.assertTrue(update_carrying(False, 3, actions, {"object": "key"}, {"object": "empty"}))
        reward = shaped_reward(
            0.0,
            3,
            actions,
            obs(1),
            obs(2),
            False,
            True,
            {"object": "key"},
            {"object": "empty"},
        )
        self.assertGreater(reward, 0.0)
        repeated = shaped_reward(
            0.0,
            5,
            actions,
            obs(1),
            obs(2),
            True,
            True,
            {"object": "door", "state_name": "closed"},
            {"object": "door", "state_name": "open"},
            award_door=False,
        )
        self.assertLess(repeated, 0.0)

    def test_epsilon_schedule_reaches_floor(self) -> None:
        self.assertEqual(epsilon_at(0, 1.0, 0.05, 100), 1.0)
        self.assertAlmostEqual(epsilon_at(100, 1.0, 0.05, 100), 0.05)
        self.assertAlmostEqual(epsilon_at(200, 1.0, 0.05, 100), 0.05)

    def test_multi_layout_fitness_penalizes_specialists(self) -> None:
        specialist = aggregate_layout_fitness(
            [{"fitness": 13.0, "solved": True}, {"fitness": 0.0, "solved": False}]
        )
        consistent = aggregate_layout_fitness(
            [{"fitness": 7.0, "solved": False}, {"fitness": 7.0, "solved": False}]
        )
        self.assertGreater(consistent, specialist)

    def test_embodied_map_dead_reckons_and_remembers_objects(self) -> None:
        body_map = EmbodiedMap(direction=0)
        body_map.observe([{"object": "key", "forward": 2, "lateral": 1}], direction=0)
        self.assertEqual(body_map.remembered["key"], (2, 1))
        body_map.advance("forward", blocked=False, next_direction=0)
        features = body_map.features(carrying=False, door_opened=False)
        self.assertEqual(features.shape, (15,))
        self.assertAlmostEqual(features[4], 1.0 / 8.0)
        body_map.advance("forward", blocked=True, next_direction=0)
        self.assertEqual((body_map.x, body_map.y), (1, 0))

    def test_embodied_map_persists_observed_lava_and_frontiers(self) -> None:
        body_map = EmbodiedMap(direction=0)
        body_map.observe(
            [
                {"object": "empty", "forward": 1, "lateral": 0},
                {"object": "lava", "forward": 2, "lateral": 0},
                {"object": "wall", "forward": 1, "lateral": 1},
            ],
            direction=0,
        )
        self.assertEqual(body_map.terrain[(2, 0)], "lava")
        self.assertEqual(body_map.terrain[(1, 1)], "wall")
        self.assertIsNotNone(body_map.nearest_frontier())
        features = body_map.features(False, False, include_terrain=True)
        self.assertEqual(features.shape, (FULL_FEATURE_SIZE,))
        self.assertTrue(np.isfinite(features).all())

    def test_evolution_replay_balances_priority_transitions(self) -> None:
        collector = BalancedEvolutionReplay(capacity=20, seed=4)
        transition = ControlTransition(
            state=belief_observation(obs(1), -1, False),
            action=2,
            reward=-0.001,
            next_state=belief_observation(obs(2), 2, False),
            terminal=False,
        )
        for _ in range(10):
            collector.add(transition, priority=True)
            collector.add(transition, priority=False)
        agent = JepaControlAgent(7, latent_dim=16, batch_size=2, warmup=2, seed=3)
        stats = collector.feed(agent, 8)
        self.assertEqual(stats["fed"], 8)
        self.assertGreaterEqual(stats["priority_available"], 4)
        self.assertEqual(len(agent.replay), 8)


if __name__ == "__main__":
    unittest.main()
