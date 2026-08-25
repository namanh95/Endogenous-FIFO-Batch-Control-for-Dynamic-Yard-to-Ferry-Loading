from dataclasses import replace
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import pandas as pd

import hierarchical_batch_mpc as model


class BatchMPCTests(unittest.TestCase):
    def test_every_generated_action_is_a_fifo_prefix_and_feasible_after_execution(self):
        cfg = replace(model.FerryConfig(), balance_ratio=1.0)
        positions = model.make_positions(cfg)
        waiting = model.generate_scenario(11, "balanced", 7.0, horizon=1)[0]
        policy = model.controllers()[0]
        actions = model.enumerate_actions(
            waiting, model.empty_committed(), positions, cfg, policy.batch_sizes,
            epoch=0, previous_yard_lane=None, controller=policy,
        )
        self.assertTrue(actions)
        action = actions[0]
        lane = waiting[waiting.yard_lane == action["yard_lane"]].sort_values("yard_position")
        self.assertEqual(tuple(lane.head(action["batch_size"]).veh_id), action["vehicle_ids"])
        _, committed = model.apply_action(
            waiting, model.empty_committed(), action, positions, 0, 1
        )
        self.assertTrue(model.audit_plan(committed, positions, cfg)["feasible"])

    def test_source_order_is_preserved_in_receiving_slots(self):
        cfg = replace(model.FerryConfig(), balance_ratio=1.0)
        positions = model.make_positions(cfg)
        waiting = model.generate_scenario(23, "car_dense", 8.0, horizon=1)[0]
        policy = model.controllers()[0]
        action, _ = model.choose_action(
            waiting, model.empty_committed(), positions, cfg, policy, 0, None
        )
        self.assertIsNotNone(action)
        _, committed = model.apply_action(
            waiting, model.empty_committed(), action, positions, 0, 1
        )
        self.assertEqual(
            committed.sort_values("slot").veh_id.astype(int).tolist(),
            list(action["vehicle_ids"]),
        )

    def test_paired_simulation_uses_identical_arrival_path(self):
        waves = model.generate_scenario(31, "heavy", 6.0)
        arrival_count = sum(len(w) for w in waves)
        for policy in model.controllers():
            metrics, _, _ = model.simulate(waves, policy)
            self.assertEqual(metrics["arrivals"], arrival_count)
            self.assertEqual(metrics["loaded"] + metrics["deferred"], arrival_count)
            self.assertTrue(metrics["feasible"])

    def test_fixed_batch_cannot_create_a_partial_batch(self):
        policy = model.ControllerConfig("Fixed-4", (4,))
        cfg = replace(model.FerryConfig(), balance_ratio=1.0)
        positions = model.make_positions(cfg)
        waiting = model.generate_scenario(41, "balanced", 0.0, horizon=1)[0].head(3)
        actions = model.enumerate_actions(
            waiting, model.empty_committed(), positions, cfg, policy.batch_sizes,
            0, None, policy,
        )
        self.assertEqual(actions, [])

    def test_two_step_choice_minimizes_second_cost_plus_terminal_value(self):
        policy = model.ControllerConfig("test", (1,), lookahead=2, beam_width=2)
        first = [
            {"name": "A", "immediate_cost": 0.0, "batch_size": 1,
             "yard_lane": 1, "pos_id": 0, "vehicle_ids": (1,), "duration": 2.0},
            {"name": "B", "immediate_cost": 1.0, "batch_size": 1,
             "yard_lane": 2, "pos_id": 0, "vehicle_ids": (2,), "duration": 2.0},
        ]
        seconds = {
            "after_A": [
                {"name": "A_bad", "immediate_cost": 0.0, "batch_size": 1,
                 "yard_lane": 1, "pos_id": 0, "vehicle_ids": (3,), "duration": 2.0},
                {"name": "A_good", "immediate_cost": 1.0, "batch_size": 1,
                 "yard_lane": 1, "pos_id": 0, "vehicle_ids": (4,), "duration": 2.0},
            ],
            "after_B": [
                {"name": "B_next", "immediate_cost": 2.0, "batch_size": 1,
                 "yard_lane": 2, "pos_id": 0, "vehicle_ids": (5,), "duration": 2.0},
            ],
        }

        def fake_enumerate(waiting, *args, **kwargs):
            return first if waiting == "initial" else seconds[waiting]

        def fake_apply(waiting, committed, action, *args, **kwargs):
            return f"after_{action['name']}" if waiting == "initial" else action["name"], committed

        terminal = {"A_bad": 100.0, "A_good": 0.0, "B_next": 0.0}
        with patch.object(model, "enumerate_actions", side_effect=fake_enumerate), \
             patch.object(model, "apply_action", side_effect=fake_apply), \
             patch.object(model, "_backlog_cost", side_effect=lambda state, epoch: terminal.get(state, 0.0)):
            action, _ = model.choose_action(
                "initial", None, None, None, policy, 0, None, remaining_time=10.0
            )
        self.assertEqual(action["name"], "A")

    def test_every_policy_respects_the_same_execution_time_budget(self):
        waves = model.generate_scenario(57, "balanced", 8.0, horizon=3)
        for policy in model.controllers():
            metrics, _, actions = model.simulate(waves, policy, epoch_time_budget=14.0)
            self.assertTrue(metrics["all_intermediate_feasible"])
            if len(actions):
                used = actions.groupby("epoch").duration.sum()
                self.assertTrue(bool((used <= 14.0 + 1e-9).all()))


if __name__ == "__main__":
    unittest.main()
