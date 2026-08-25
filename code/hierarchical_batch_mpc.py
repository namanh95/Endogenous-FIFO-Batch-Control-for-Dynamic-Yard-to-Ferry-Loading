"""State-aware hierarchical batch MPC for dynamic yard-to-ferry loading.

The controller acts on an observed terminal state.  An action selects one yard
lane, an accessible FIFO prefix, and one receiving ferry lane/deck.  Vehicles
retain their source order when appended to the receiving lane.  Every
enumerated action is checked against geometry, residual mass, ordered
discharge, slot continuity, and a transverse-moment envelope before it may be
executed.

This module is independent of the neural Top-K assignment model in the related
RINAM submission.  It studies execution control and endogenous batching.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence
import argparse
import json
import time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FerryConfig:
    decks: int = 2
    lanes_per_deck: int = 8
    lane_length: float = 46.0
    lane_width: float = 2.80
    lane_heights: tuple[float, ...] = (4.50, 2.70)
    lane_mass: float = 82.0
    deck_mass: float = 520.0
    max_slots: int = 10
    balance_ratio: float = 0.22


@dataclass(frozen=True)
class ControllerConfig:
    name: str
    batch_sizes: tuple[int, ...]
    lookahead: int = 1
    beam_width: int = 12
    setup_cost: float = 5.0
    switch_cost: float = 1.5
    load_reward: float = 12.0
    waiting_weight: float = 1.0
    trim_weight: float = 0.8
    imbalance_weight: float = 2.0
    terminal_backlog_weight: float = 1.5
    selection_rule: str = "cost"


VEHICLES = {
    "car": (4.5, 1.82, 1.55, 1.55),
    "suv": (4.9, 1.92, 1.78, 2.10),
    "van": (5.6, 2.05, 2.45, 3.20),
    "rigid_truck": (9.0, 2.45, 3.60, 12.0),
    "coach": (12.0, 2.50, 3.70, 17.0),
    "artic": (16.2, 2.50, 4.00, 29.0),
}

REGIME_PROBS = {
    "balanced": np.array([0.40, 0.18, 0.14, 0.10, 0.06, 0.12]),
    "heavy": np.array([0.25, 0.12, 0.13, 0.16, 0.09, 0.25]),
    "car_dense": np.array([0.58, 0.20, 0.10, 0.05, 0.02, 0.05]),
}

COMMITTED_COLUMNS = [
    "veh_id", "class", "length", "width", "height", "mass", "destination",
    "priority", "arrival_epoch", "yard_lane", "yard_position", "yard_sequence",
    "pos_id", "deck", "ferry_lane",
    "slot", "batch_id", "commit_epoch",
]


def make_positions(cfg: FerryConfig) -> pd.DataFrame:
    rows = []
    y_values = np.linspace(-1.0, 1.0, cfg.lanes_per_deck)
    for deck in range(cfg.decks):
        for lane in range(cfg.lanes_per_deck):
            rows.append({
                "pos_id": deck * cfg.lanes_per_deck + lane,
                "deck": deck,
                "ferry_lane": lane + 1,
                "length_cap": cfg.lane_length,
                "width_cap": cfg.lane_width,
                "height_cap": cfg.lane_heights[deck],
                "mass_cap": cfg.lane_mass,
                "y_coord": float(y_values[lane]),
            })
    return pd.DataFrame(rows)


def empty_waiting() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "veh_id", "class", "length", "width", "height", "mass", "destination",
        "priority", "arrival_epoch", "yard_lane", "yard_position", "yard_sequence",
    ])


def empty_committed() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMITTED_COLUMNS)


def normalize_queues(waiting: pd.DataFrame) -> pd.DataFrame:
    if waiting.empty:
        return waiting.copy()
    out = waiting.sort_values(
        ["yard_lane", "arrival_epoch", "yard_position", "veh_id"]
    ).reset_index(drop=True)
    out["yard_position"] = out.groupby("yard_lane").cumcount() + 1
    return out


def generate_scenario(
    seed: int,
    regime: str,
    arrival_rate: float,
    horizon: int = 8,
) -> list[pd.DataFrame]:
    """Generate a complete exogenous path for paired policy comparisons."""
    if regime not in REGIME_PROBS:
        raise ValueError(f"unknown regime: {regime}")
    rng = np.random.default_rng(seed)
    classes = list(VEHICLES)
    probabilities = REGIME_PROBS[regime] / REGIME_PROBS[regime].sum()
    waves = []
    next_id = 0
    for epoch in range(horizon):
        count = int(rng.poisson(arrival_rate))
        if epoch == 0:
            count += 8
        rows = []
        for local in range(count):
            vehicle_class = str(rng.choice(classes, p=probabilities))
            length, width, height, mass = VEHICLES[vehicle_class]
            destination = int(rng.integers(1, 4))
            is_heavy = vehicle_class in {"rigid_truck", "coach", "artic"}
            # Destination-segmented staging lanes avoid manufacturing discharge
            # inversions while retaining heterogeneous geometry within a queue.
            yard_lane = 2 * (destination - 1) + (1 if is_heavy else 2)
            rows.append({
                "veh_id": next_id,
                "class": vehicle_class,
                "length": float(max(3.0, rng.normal(length, 0.03 * length))),
                "width": float(max(1.5, rng.normal(width, 0.015 * width))),
                "height": float(max(1.3, rng.normal(height, 0.02 * height))),
                "mass": float(max(0.8, rng.normal(mass, 0.05 * mass))),
                "destination": destination,
                "priority": int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])),
                "arrival_epoch": epoch,
                "yard_lane": yard_lane,
                "yard_position": local + 1,
            })
            next_id += 1
        waves.append(pd.DataFrame(rows) if rows else empty_waiting())
    all_vehicles = pd.concat([wave for wave in waves if len(wave)], ignore_index=True)
    if len(all_vehicles):
        ordered = all_vehicles.sort_values(
            ["yard_lane", "arrival_epoch", "yard_position", "veh_id"]
        ).copy()
        ordered["yard_sequence"] = ordered.groupby("yard_lane").cumcount() + 1
        sequence = ordered.set_index("veh_id")["yard_sequence"].to_dict()
        for index, wave in enumerate(waves):
            wave = wave.copy()
            wave["yard_sequence"] = wave.veh_id.map(sequence).astype(int) if len(wave) else pd.Series(dtype=int)
            waves[index] = wave
    return waves


def action_duration(batch_size: int, setup_time: float = 1.0, per_vehicle_time: float = 1.0) -> float:
    """Physical execution time used to give every policy an equal epoch budget."""
    return float(setup_time + per_vehicle_time * int(batch_size))


def _usage(committed: pd.DataFrame, positions: pd.DataFrame):
    length = {int(p): 0.0 for p in positions.pos_id}
    mass = {int(p): 0.0 for p in positions.pos_id}
    count = {int(p): 0 for p in positions.pos_id}
    deck_mass = {int(d): 0.0 for d in positions.deck.unique()}
    total_mass = 0.0
    moment = 0.0
    pidx = positions.set_index("pos_id")
    for _, row in committed.iterrows():
        pid = int(row.pos_id)
        length[pid] += float(row.length)
        mass[pid] += float(row.mass)
        count[pid] += 1
        deck_mass[int(row.deck)] += float(row.mass)
        total_mass += float(row.mass)
        moment += float(row.mass) * float(pidx.loc[pid, "y_coord"])
    return length, mass, count, deck_mass, total_mass, moment


def _compatible(vehicle: pd.Series, position: pd.Series, cfg: FerryConfig) -> bool:
    if float(vehicle.width) > float(position.width_cap) + 1e-9:
        return False
    if float(vehicle.height) > float(position.height_cap) + 1e-9:
        return False
    if float(vehicle.mass) > float(position.mass_cap) + 1e-9:
        return False
    if vehicle["class"] in {"rigid_truck", "coach", "artic"} and int(position.deck) == cfg.decks - 1:
        return False
    return True


def enumerate_actions(
    waiting: pd.DataFrame,
    committed: pd.DataFrame,
    positions: pd.DataFrame,
    cfg: FerryConfig,
    batch_sizes: Iterable[int],
    epoch: int,
    previous_yard_lane: Optional[int],
    controller: ControllerConfig,
    remaining_time: float = np.inf,
    setup_time: float = 1.0,
    per_vehicle_time: float = 1.0,
) -> list[dict]:
    waiting = normalize_queues(waiting)
    if waiting.empty:
        return []
    pidx = positions.set_index("pos_id")
    length, mass, count, deck_mass, total_mass, moment = _usage(committed, positions)
    actions = []
    allowed_sizes = sorted(set(int(x) for x in batch_sizes if int(x) > 0))
    for yard_lane, queue in waiting.groupby("yard_lane", sort=True):
        queue = queue.sort_values("yard_position")
        for batch_size in allowed_sizes:
            duration = action_duration(batch_size, setup_time, per_vehicle_time)
            if duration > remaining_time + 1e-9:
                continue
            if batch_size > len(queue):
                continue
            prefix = queue.head(batch_size).copy()
            for _, position in positions.iterrows():
                pid = int(position.pos_id)
                if count[pid] + batch_size > cfg.max_slots:
                    continue
                if any(not _compatible(row, position, cfg) for _, row in prefix.iterrows()):
                    continue
                if length[pid] + float(prefix.length.sum()) > float(position.length_cap) + 1e-9:
                    continue
                if mass[pid] + float(prefix.mass.sum()) > float(position.mass_cap) + 1e-9:
                    continue
                deck = int(position.deck)
                if deck_mass[deck] + float(prefix.mass.sum()) > cfg.deck_mass + 1e-9:
                    continue
                destinations = prefix.destination.astype(int).tolist()
                if destinations != sorted(destinations):
                    continue
                fixed = committed[committed.pos_id.astype(int) == pid]
                if len(fixed) and int(fixed.sort_values("slot").iloc[-1].destination) > destinations[0]:
                    continue
                added_mass = float(prefix.mass.sum())
                new_total = total_mass + added_mass
                new_moment = moment + added_mass * float(position.y_coord)
                if new_total > 0 and abs(new_moment) > cfg.balance_ratio * new_total + 1e-9:
                    continue
                ages = epoch - prefix.arrival_epoch.to_numpy(float) + 1.0
                pressure = float(np.sum(ages * prefix.priority.to_numpy(float)))
                remaining_length = float(position.length_cap) - length[pid] - float(prefix.length.sum())
                trim = remaining_length / max(1.0, float(position.length_cap))
                imbalance = abs(new_moment) / max(1.0, cfg.balance_ratio * new_total)
                switched = previous_yard_lane is not None and int(yard_lane) != int(previous_yard_lane)
                cost = (
                    controller.setup_cost
                    + controller.switch_cost * float(switched)
                    - controller.load_reward * float(batch_size)
                    - controller.waiting_weight * pressure
                    + controller.trim_weight * trim
                    + controller.imbalance_weight * imbalance
                )
                actions.append({
                    "yard_lane": int(yard_lane),
                    "batch_size": int(batch_size),
                    "pos_id": pid,
                    "deck": deck,
                    "ferry_lane": int(position.ferry_lane),
                    "vehicle_ids": tuple(prefix.veh_id.astype(int)),
                    "immediate_cost": float(cost),
                    "duration": duration,
                })
    return actions


def apply_action(
    waiting: pd.DataFrame,
    committed: pd.DataFrame,
    action: dict,
    positions: pd.DataFrame,
    epoch: int,
    batch_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pidx = positions.set_index("pos_id")
    pid = int(action["pos_id"])
    fixed_count = int((committed.pos_id.astype(int) == pid).sum()) if len(committed) else 0
    selected = waiting.set_index("veh_id").loc[list(action["vehicle_ids"])].reset_index()
    selected = selected.sort_values("yard_position").reset_index(drop=True)
    rows = []
    for offset, row in selected.iterrows():
        rows.append({
            **{key: row[key] for key in [
                "veh_id", "class", "length", "width", "height", "mass",
                "destination", "priority", "arrival_epoch", "yard_lane",
                "yard_position",
                "yard_sequence",
            ]},
            "pos_id": pid,
            "deck": int(pidx.loc[pid, "deck"]),
            "ferry_lane": int(pidx.loc[pid, "ferry_lane"]),
            "slot": fixed_count + offset,
            "batch_id": int(batch_id),
            "commit_epoch": int(epoch),
        })
    added = pd.DataFrame(rows)[COMMITTED_COLUMNS]
    updated_committed = added.reset_index(drop=True) if committed.empty else pd.concat(
        [committed, added], ignore_index=True
    )
    moved = set(int(x) for x in action["vehicle_ids"])
    updated_waiting = waiting[~waiting.veh_id.astype(int).isin(moved)].copy()
    return normalize_queues(updated_waiting), updated_committed


def _backlog_cost(waiting: pd.DataFrame, epoch: int) -> float:
    if waiting.empty:
        return 0.0
    ages = epoch - waiting.arrival_epoch.to_numpy(float) + 1.0
    return float(np.sum((ages ** 2) * waiting.priority.to_numpy(float)))


def choose_action(
    waiting: pd.DataFrame,
    committed: pd.DataFrame,
    positions: pd.DataFrame,
    cfg: FerryConfig,
    controller: ControllerConfig,
    epoch: int,
    previous_yard_lane: Optional[int],
    remaining_time: float = np.inf,
    setup_time: float = 1.0,
    per_vehicle_time: float = 1.0,
) -> tuple[Optional[dict], int]:
    first = enumerate_actions(
        waiting, committed, positions, cfg, controller.batch_sizes, epoch,
        previous_yard_lane, controller, remaining_time, setup_time, per_vehicle_time,
    )
    if not first:
        return None, 0
    if controller.selection_rule == "largest":
        first.sort(key=lambda a: (-a["batch_size"], a["immediate_cost"], a["yard_lane"], a["pos_id"]))
    elif controller.selection_rule == "fixed_cleanup":
        preferred = max(controller.batch_sizes)
        first.sort(key=lambda a: (a["batch_size"] != preferred, -a["batch_size"], a["immediate_cost"], a["yard_lane"], a["pos_id"]))
    else:
        first.sort(key=lambda a: (a["immediate_cost"], -a["batch_size"], a["yard_lane"], a["pos_id"]))
    if controller.lookahead <= 1:
        return first[0], len(first)
    shortlisted = first[: controller.beam_width]
    best_action = None
    best_value = np.inf
    evaluations = len(first)
    for rank, action in enumerate(shortlisted):
        w2, c2 = apply_action(waiting, committed, action, positions, epoch, -1)
        remaining_after_first = remaining_time - float(action["duration"])
        second = enumerate_actions(
            w2, c2, positions, cfg, controller.batch_sizes, epoch,
            action["yard_lane"], controller, remaining_after_first,
            setup_time, per_vehicle_time,
        )
        evaluations += len(second)
        if second:
            continuation_values = []
            for second_rank, second_action in enumerate(second):
                w3_candidate, _ = apply_action(
                    w2, c2, second_action, positions, epoch, -2
                )
                continuation_values.append((
                    second_action["immediate_cost"]
                    + controller.terminal_backlog_weight * _backlog_cost(w3_candidate, epoch)
                    + 1e-10 * second_rank,
                    w3_candidate,
                ))
            continuation, w3 = min(continuation_values, key=lambda item: item[0])
        else:
            w3 = w2
            continuation = controller.terminal_backlog_weight * _backlog_cost(w3, epoch)
        value = action["immediate_cost"] + continuation + 1e-8 * rank
        if value < best_value:
            best_value = value
            best_action = dict(action)
            best_action["predictive_value"] = float(value)
    return best_action, evaluations


def audit_plan(committed: pd.DataFrame, positions: pd.DataFrame, cfg: FerryConfig) -> dict:
    if committed.empty:
        return {
            "feasible": True, "max_capacity_violation": 0.0,
            "discharge_inversions": 0, "fifo_violations": 0,
            "duplicate_slots": 0, "no_gap_violations": 0,
            "balance_violation": 0.0,
            "compatibility_violations": 0,
        }
    pidx = positions.set_index("pos_id")
    max_capacity = 0.0
    inversions = 0
    no_gaps = 0
    compatibility_violations = 0
    for pid, group in committed.groupby("pos_id"):
        pid = int(pid)
        max_capacity = max(
            max_capacity,
            max(0.0, float(group.length.sum()) - float(pidx.loc[pid, "length_cap"])),
            max(0.0, float(group.mass.sum()) - float(pidx.loc[pid, "mass_cap"])),
        )
        ordered = group.sort_values("slot")
        slots = ordered.slot.astype(int).tolist()
        no_gaps += int(slots != list(range(len(slots))))
        destinations = ordered.destination.astype(int).tolist()
        inversions += sum(a > b for a, b in zip(destinations, destinations[1:]))
        position = pidx.loc[pid]
        compatibility_violations += sum(
            not _compatible(row, position, cfg) for _, row in group.iterrows()
        )
    for deck, group in committed.groupby("deck"):
        max_capacity = max(max_capacity, max(0.0, float(group.mass.sum()) - cfg.deck_mass))
    fifo_violations = 0
    for _, group in committed.groupby("yard_lane"):
        ordered = group.sort_values(["commit_epoch", "batch_id", "slot"])
        sequence = ordered.yard_sequence.astype(int).tolist()
        fifo_violations += sum(a >= b for a, b in zip(sequence, sequence[1:]))
    total_mass = float(committed.mass.sum())
    moment = sum(
        float(row.mass) * float(pidx.loc[int(row.pos_id), "y_coord"])
        for _, row in committed.iterrows()
    )
    balance = max(0.0, abs(moment) - cfg.balance_ratio * total_mass)
    duplicates = int(committed.duplicated(["pos_id", "slot"]).sum())
    feasible = (
        max_capacity <= 1e-8 and inversions == 0 and fifo_violations == 0
        and duplicates == 0 and no_gaps == 0 and balance <= 1e-8
        and compatibility_violations == 0
    )
    return {
        "feasible": bool(feasible),
        "max_capacity_violation": float(max_capacity),
        "discharge_inversions": int(inversions),
        "fifo_violations": int(fifo_violations),
        "duplicate_slots": int(duplicates),
        "no_gap_violations": int(no_gaps),
        "balance_violation": float(balance),
        "compatibility_violations": int(compatibility_violations),
    }


def simulate(
    waves: Sequence[pd.DataFrame],
    controller: ControllerConfig,
    cfg: FerryConfig = FerryConfig(),
    epoch_time_budget: float = 14.0,
    setup_time: float = 1.0,
    per_vehicle_time: float = 1.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    positions = make_positions(cfg)
    waiting = empty_waiting()
    committed = empty_committed()
    previous_yard = None
    actions = []
    trajectory = []
    waiting_area = 0.0
    evaluations = 0
    all_intermediate_feasible = True
    started = time.perf_counter()
    for epoch, wave in enumerate(waves):
        if len(wave):
            waiting = normalize_queues(
                wave.copy() if waiting.empty else pd.concat([waiting, wave], ignore_index=True)
            )
        arrivals = len(wave)
        remaining_time = float(epoch_time_budget)
        while remaining_time >= setup_time + per_vehicle_time - 1e-9:
            action, count = choose_action(
                waiting, committed, positions, cfg, controller, epoch, previous_yard,
                remaining_time, setup_time, per_vehicle_time,
            )
            evaluations += count
            if action is None:
                break
            batch_id = len(actions) + 1
            waiting, committed = apply_action(
                waiting, committed, action, positions, epoch, batch_id
            )
            previous_yard = int(action["yard_lane"])
            remaining_time -= float(action["duration"])
            actions.append({
                **action, "epoch": epoch, "batch_id": batch_id,
                "remaining_time_after": remaining_time,
            })
            all_intermediate_feasible = all_intermediate_feasible and audit_plan(
                committed, positions, cfg
            )["feasible"]
        waiting_area += _backlog_cost(waiting, epoch)
        trajectory.append({
            "epoch": epoch,
            "arrivals": arrivals,
            "loaded_cumulative": len(committed),
            "waiting": len(waiting),
            "weighted_backlog": _backlog_cost(waiting, epoch),
        })
    runtime = time.perf_counter() - started
    audit = audit_plan(committed, positions, cfg)
    action_frame = pd.DataFrame(actions)
    switches = 0
    if len(action_frame) > 1:
        switches = int((action_frame.yard_lane.to_numpy()[1:] != action_frame.yard_lane.to_numpy()[:-1]).sum())
    total_length = float(committed.length.sum()) if len(committed) else 0.0
    metrics = {
        "policy": controller.name,
        "arrivals": int(sum(len(w) for w in waves)),
        "loaded": int(len(committed)),
        "deferred": int(len(waiting)),
        "weighted_wait_area": float(waiting_area),
        "batches": int(len(action_frame)),
        "mean_batch_size": float(action_frame.batch_size.mean()) if len(action_frame) else 0.0,
        "yard_switches": switches,
        "runtime_s": float(runtime),
        "candidate_evaluations": int(evaluations),
        "all_intermediate_feasible": bool(all_intermediate_feasible),
        "length_utilization": total_length / (cfg.decks * cfg.lanes_per_deck * cfg.lane_length),
        **audit,
    }
    return metrics, pd.DataFrame(trajectory), action_frame


def controllers() -> list[ControllerConfig]:
    return [
        ControllerConfig("HBC-MPC", tuple(range(1, 9)), lookahead=2, beam_width=4),
        ControllerConfig("Myopic-Endogenous", tuple(range(1, 9)), lookahead=1),
        ControllerConfig("Largest-Feasible", tuple(range(1, 9)), lookahead=1, selection_rule="largest"),
        ControllerConfig("Fixed-4", (4,), lookahead=1),
        ControllerConfig("Fixed-4-Cleanup", (1, 2, 3, 4), lookahead=1, selection_rule="fixed_cleanup"),
        ControllerConfig("Fixed-6", (6,), lookahead=1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--regime", choices=sorted(REGIME_PROBS), default="balanced")
    parser.add_argument("--arrival-rate", type=float, default=8.0)
    parser.add_argument("--output", default="results_demo")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    waves = generate_scenario(args.seed, args.regime, args.arrival_rate)
    rows = []
    for policy in controllers():
        metrics, trajectory, actions = simulate(waves, policy)
        rows.append(metrics)
        trajectory.to_csv(output / f"trajectory_{policy.name}.csv", index=False)
        actions.to_csv(output / f"actions_{policy.name}.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    (output / "metadata.json").write_text(json.dumps({
        "seed": args.seed, "regime": args.regime, "arrival_rate": args.arrival_rate,
    }, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
