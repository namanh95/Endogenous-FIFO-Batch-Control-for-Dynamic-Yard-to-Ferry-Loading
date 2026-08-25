"""Paired computational study for the hierarchical batch-MPC manuscript."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hierarchical_batch_mpc as model


REGIMES = ("balanced", "heavy", "car_dense")
INTENSITIES = {"low": 6.0, "medium": 8.0, "high": 10.0}
SEEDS = tuple(101 * index for index in range(1, 21))
POLICY_ORDER = [
    "HBC-MPC", "Myopic-Endogenous", "Largest-Feasible",
    "Fixed-4-Cleanup", "Fixed-4", "Fixed-6",
]


def run_scenario(task: tuple[str, str, float, int]):
    regime, intensity, rate, seed = task
    waves = model.generate_scenario(seed, regime, rate)
    rows = []
    trajectories = {}
    actions = {}
    for controller in model.controllers():
        metrics, trajectory, action_table = model.simulate(waves, controller)
        rows.append({
            "regime": regime,
            "intensity": intensity,
            "arrival_rate": rate,
            "seed": seed,
            **metrics,
        })
        if regime == "balanced" and intensity == "medium" and seed == 101:
            trajectories[controller.name] = trajectory
            actions[controller.name] = action_table
    return rows, trajectories, actions


def bootstrap_ci(values: np.ndarray, seed: int = 7301) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "loaded", "deferred", "weighted_wait_area", "mean_batch_size",
        "yard_switches", "runtime_s", "candidate_evaluations", "length_utilization",
    ]
    rows = []
    for keys, group in raw.groupby(["policy", "regime", "intensity"], sort=False):
        row = dict(zip(["policy", "regime", "intensity"], keys))
        row["n"] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(float)
            lo, hi = bootstrap_ci(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(raw: pd.DataFrame) -> pd.DataFrame:
    index = ["regime", "intensity", "arrival_rate", "seed"]
    reference = raw[raw.policy == "HBC-MPC"].set_index(index)
    rows = []
    for policy in POLICY_ORDER[1:]:
        comparison = raw[raw.policy == policy].set_index(index)
        joined = reference[["deferred", "weighted_wait_area", "loaded"]].join(
            comparison[["deferred", "weighted_wait_area", "loaded"]],
            lsuffix="_hbc", rsuffix="_baseline", how="inner",
        )
        for metric in ["deferred", "weighted_wait_area", "loaded"]:
            # Positive values mean HBC-MPC is better: fewer deferrals/waiting,
            # or more loaded vehicles.
            if metric == "loaded":
                diff = joined[f"{metric}_hbc"] - joined[f"{metric}_baseline"]
            else:
                diff = joined[f"{metric}_baseline"] - joined[f"{metric}_hbc"]
            diff_frame = diff.rename("improvement").reset_index()
            lo, hi = stratified_paired_bootstrap(diff_frame, seed=9100 + len(rows))
            rows.append({
                "baseline": policy,
                "metric": metric,
                "n_pairs": len(diff),
                "mean_improvement": float(diff.mean()),
                "median_improvement": float(diff.median()),
                "ci_low": lo,
                "ci_high": hi,
                "hbc_better_pairs": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "hbc_worse_pairs": int((diff < 0).sum()),
            })
    return pd.DataFrame(rows)


def stratified_paired_bootstrap(differences: pd.DataFrame, seed: int = 9100) -> tuple[float, float]:
    """Resample seeds within each regime-intensity cell, then weight cells equally."""
    rng = np.random.default_rng(seed)
    cells = [group.improvement.to_numpy(float) for _, group in differences.groupby(
        ["regime", "intensity"], sort=True
    )]
    estimates = np.empty(5000)
    for draw in range(len(estimates)):
        estimates[draw] = np.mean([
            rng.choice(values, size=len(values), replace=True).mean() for values in cells
        ])
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def paired_differences_by_cell(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference = raw[raw.policy == "HBC-MPC"].set_index(
        ["regime", "intensity", "seed"]
    )
    for policy in POLICY_ORDER[1:]:
        baseline = raw[raw.policy == policy].set_index(["regime", "intensity", "seed"])
        joined = reference[["deferred", "weighted_wait_area", "loaded"]].join(
            baseline[["deferred", "weighted_wait_area", "loaded"]],
            lsuffix="_hbc", rsuffix="_baseline",
        )
        for (regime, intensity), group in joined.groupby(level=[0, 1]):
            for metric in ["deferred", "weighted_wait_area", "loaded"]:
                if metric == "loaded":
                    diff = group[f"{metric}_hbc"] - group[f"{metric}_baseline"]
                else:
                    diff = group[f"{metric}_baseline"] - group[f"{metric}_hbc"]
                lo, hi = bootstrap_ci(diff.to_numpy(float), seed=12000 + len(rows))
                rows.append({
                    "baseline": policy, "regime": regime, "intensity": intensity,
                    "metric": metric, "n_pairs": len(diff),
                    "mean_improvement": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                    "hbc_better_pairs": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()), "hbc_worse_pairs": int((diff < 0).sum()),
                })
    return pd.DataFrame(rows)


def create_figures(raw: pd.DataFrame, representative: dict[str, pd.DataFrame], output: Path):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
    })
    colors = {
        "HBC-MPC": "#1768AC", "Myopic-Endogenous": "#4E9F3D",
        "Largest-Feasible": "#6F4E7C", "Fixed-4-Cleanup": "#1B998B",
        "Fixed-4": "#E09F3E", "Fixed-6": "#9E2A2B",
    }

    overall = raw.groupby("policy", sort=False).agg(
        deferred=("deferred", "mean"),
        waiting=("weighted_wait_area", "mean"),
        runtime=("runtime_s", "mean"),
    ).reindex(POLICY_ORDER)
    scenario_means = raw.groupby(["policy", "regime", "intensity"])[
        ["deferred", "weighted_wait_area"]
    ].mean().reset_index()
    cell_sd = scenario_means.groupby("policy")[["deferred", "weighted_wait_area"]].std(ddof=1)
    cell_se = cell_sd / np.sqrt(len(REGIMES) * len(INTENSITIES))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    y = np.arange(len(overall))
    axes[0].barh(y, overall.deferred, xerr=1.96 * cell_se.reindex(POLICY_ORDER).deferred,
                 color=[colors[x] for x in overall.index], capsize=3)
    axes[0].set_yticks(y, overall.index)
    axes[0].set_xlabel("Mean deferred vehicles")
    axes[0].set_title("Sailing-cutoff deferral")
    axes[1].barh(y, overall.waiting, xerr=1.96 * cell_se.reindex(POLICY_ORDER).weighted_wait_area,
                 color=[colors[x] for x in overall.index], capsize=3)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].set_xlabel("Mean weighted waiting area")
    axes[1].set_title("Cumulative congestion")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "Figure_policy_comparison.png", bbox_inches="tight")
    fig.savefig(output / "Figure_policy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fixed = raw[raw.policy == "Fixed-4"].set_index(["regime", "intensity", "seed"])
    hbc = raw[raw.policy == "HBC-MPC"].set_index(["regime", "intensity", "seed"])
    improvement = (fixed.deferred - hbc.deferred).groupby(["regime", "intensity"]).mean()
    matrix = np.array([
        [improvement.loc[(regime, intensity)] for intensity in INTENSITIES]
        for regime in REGIMES
    ])
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(INTENSITIES)), [x.title() for x in INTENSITIES])
    ax.set_yticks(range(len(REGIMES)), [x.replace("_", " ").title() for x in REGIMES])
    ax.set_xlabel("Arrival intensity")
    ax.set_ylabel("Fleet regime")
    ax.set_title("Mean deferrals avoided by HBC-MPC relative to Fixed-4")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Vehicles per sailing")
    fig.tight_layout()
    fig.savefig(output / "Figure_regime_sensitivity.png", bbox_inches="tight")
    fig.savefig(output / "Figure_regime_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    line_styles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1))]
    markers = ["o", "s", "^", "D", "v", "P"]
    for policy, linestyle, marker in zip(POLICY_ORDER, line_styles, markers):
        trajectory = representative[policy]
        ax.plot(trajectory.epoch + 1, trajectory.waiting, marker=marker, linestyle=linestyle,
                linewidth=2, label=policy, color=colors[policy])
    ax.set_xlabel("Decision epoch")
    ax.set_ylabel("Vehicles waiting after control")
    ax.set_title("Representative paired path: balanced, medium arrivals, seed 101")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "Figure_representative_trajectory.png", bbox_inches="tight")
    fig.savefig(output / "Figure_representative_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (regime, intensity, rate, seed)
        for regime in REGIMES
        for intensity, rate in INTENSITIES.items()
        for seed in SEEDS
    ]
    all_rows = []
    representative = {}
    representative_actions = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_scenario, task): task for task in tasks}
        for future in as_completed(futures):
            rows, trajectories, actions = future.result()
            all_rows.extend(rows)
            representative.update(trajectories)
            representative_actions.update(actions)
            print("completed", futures[future], flush=True)
    raw = pd.DataFrame(all_rows).sort_values(
        ["regime", "intensity", "seed", "policy"]
    ).reset_index(drop=True)
    raw.to_csv(output / "experiment_raw.csv", index=False)
    summary = aggregate(raw)
    summary.to_csv(output / "experiment_summary.csv", index=False)
    paired = paired_differences(raw)
    paired.to_csv(output / "paired_improvements.csv", index=False)
    paired_cell = paired_differences_by_cell(raw)
    paired_cell.to_csv(output / "paired_improvements_by_cell.csv", index=False)
    for policy, trajectory in representative.items():
        trajectory.to_csv(output / f"representative_trajectory_{policy}.csv", index=False)
    for policy, actions in representative_actions.items():
        actions.to_csv(output / f"representative_actions_{policy}.csv", index=False)
    create_figures(raw, representative, output)
    metadata = {
        "seeds": list(SEEDS), "regimes": list(REGIMES),
        "arrival_rates": INTENSITIES, "horizon": 8,
        "epoch_time_budget": 14.0, "setup_time": 1.0, "per_vehicle_time": 1.0,
        "policies": POLICY_ORDER, "paired_scenarios": len(tasks),
    }
    (output / "experiment_design.json").write_text(json.dumps(metadata, indent=2))
    if not bool(raw.feasible.all()):
        raise RuntimeError("modeled feasibility audit failed")
    print("\nOverall means")
    print(raw.groupby("policy")[["loaded", "deferred", "weighted_wait_area", "runtime_s"]].mean())
    print("\nPaired improvements")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
