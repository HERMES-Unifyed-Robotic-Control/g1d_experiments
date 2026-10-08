#!/usr/bin/env python3
"""Plot a recorded pi0.5 rollout against the LeRobot action distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd


NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
    "left_gripper",
    "right_gripper",
]


def load_dataset_bands(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    counts = []
    # Exact p01/p99 are expensive without scanning all parquet files. The stats
    # band is a fast deployment diagnostic: mean +/- 2.33 std approximates p01/p99.
    for line in open(dataset / "meta/episodes_stats.jsonl"):
        row = json.loads(line)
        stats = row["stats"]["action"]
        mean = np.array(stats["mean"], dtype=float)
        std = np.array(stats["std"], dtype=float)
        count = int(stats["count"][0])
        mins.append(mean - 2.33 * std)
        maxs.append(mean + 2.33 * std)
        counts.append(count)
    counts = np.array(counts, dtype=float)
    p01 = np.average(np.stack(mins), axis=0, weights=counts)
    p99 = np.average(np.stack(maxs), axis=0, weights=counts)
    delta_p99 = np.array([0.0180, 0.0161, 0.0175, 0.0293, 0.0411, 0.0212, 0.0202, 0.0308,
                          0.0273, 0.0260, 0.0502, 0.0275, 0.0317, 0.0276, 0.1799, 0.1978])
    return p01, p99, delta_p99


def columns(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_{i:02d}" for i in range(16)]].to_numpy(dtype=float)


def plot_grid(path: Path, t: np.ndarray, y: np.ndarray, ref_p01: np.ndarray, ref_p99: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.fill_between(t, ref_p01[i], ref_p99[i], color="0.88", label="dataset p01-p99")
        ax.plot(t, y[:, i], lw=1.2, label="rollout")
        ax.set_title(f"{i:02d} {NAMES[i]}", fontsize=9)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--dataset", default="/mnt/data/yangky/test/datasets/g1d_food_sorting_123_lerobot")
    args = parser.parse_args()

    record_dir = Path(args.record_dir)
    dataset = Path(args.dataset)
    out = record_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    rollout = pd.read_csv(record_dir / "rollout_actions.csv")
    t = rollout["time_s"].to_numpy(dtype=float)
    policy_action = columns(rollout, "policy_action")
    cmd_action = columns(rollout, "cmd_action")
    robot_state = columns(rollout, "robot_state_before")
    p01, p99, d99 = load_dataset_bands(dataset)

    plot_grid(out / "policy_action_vs_dataset.png", t, policy_action, p01, p99, "Policy action vs dataset action p01-p99")
    plot_grid(out / "cmd_action_vs_dataset.png", t, cmd_action, p01, p99, "Executed command action vs dataset action p01-p99")
    plot_grid(out / "robot_state_vs_dataset_action.png", t, robot_state, p01, p99, "Robot state before command vs dataset action p01-p99")

    delta_policy = np.vstack([np.zeros((1, 16)), np.diff(policy_action, axis=0)])
    delta_cmd = np.vstack([np.zeros((1, 16)), np.diff(cmd_action, axis=0)])
    zero = -d99
    plot_grid(out / "policy_delta_vs_dataset_delta.png", t, delta_policy, zero, d99, "Policy action delta vs dataset |delta| p99 band")
    plot_grid(out / "cmd_delta_vs_dataset_delta.png", t, delta_cmd, zero, d99, "Executed command delta vs dataset |delta| p99 band")

    summary = []
    summary.append("# Rollout Plot Summary\n\n")
    summary.append(f"- record_dir: `{record_dir}`\n")
    summary.append(f"- steps: `{len(rollout)}`\n")
    summary.append(f"- duration_s: `{t[-1] - t[0]:.3f}`\n")
    summary.append("- plots:\n")
    for png in sorted(out.glob("*.png")):
        summary.append(f"  - `{png}`\n")
    (record_dir / "PLOT_SUMMARY.md").write_text("".join(summary))
    print(record_dir / "PLOT_SUMMARY.md")


if __name__ == "__main__":
    main()
