#!/usr/bin/env python3
"""Compare a pi0.5 rollout with demo actions using subtask/time alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTION_NAMES = [
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "L_grip",
    "R_grip",
]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resample(arr: np.ndarray, bins: int) -> np.ndarray:
    if len(arr) == 0:
        return np.full((bins, arr.shape[1]), np.nan)
    if len(arr) == 1:
        return np.repeat(arr, bins, axis=0)
    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, bins)
    return np.stack([np.interp(dst, src, arr[:, i]) for i in range(arr.shape[1])], axis=1)


def _subtask_slices(label: dict, n_frames: int) -> list[tuple[str, int, int]]:
    slices = []
    for item in sorted(label["subtasks"], key=lambda x: x["index"]):
        start = max(0, min(n_frames - 1, int(item["start_frame"])))
        end = max(start + 1, min(n_frames, int(item["end_frame"]) + 1))
        slices.append((item["object"], start, end))
    return slices


def _load_demo_segments(dataset: Path, labels_dir: Path, episode_indices: list[int], bins: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    all_eps = _read_jsonl(dataset / "meta" / "episodes.jsonl")
    by_index = {int(ep["episode_index"]): ep for ep in all_eps}
    names: list[str] | None = None
    per_episode = []
    durations = []
    for ep_idx in episode_indices:
        ep = by_index[ep_idx]
        parquet = dataset / ep.get("data_path", f"data/chunk-000/episode_{ep_idx:06d}.parquet")
        label = json.loads((labels_dir / f"g1d_food_sorting_123_episode_{ep_idx:06d}.json").read_text())
        df = pd.read_parquet(parquet, columns=["action"])
        actions = np.stack(df["action"].to_numpy()).astype(float)
        slices = _subtask_slices(label, len(actions))
        if names is None:
            names = [name for name, _, _ in slices]
        segs = []
        frac_edges = [0.0]
        for name, start, end in slices:
            segs.append(_resample(actions[start:end], bins))
            frac_edges.append(end / len(actions))
        per_episode.append(np.stack(segs, axis=0))
        durations.append(np.diff(frac_edges))
    if names is None:
        raise RuntimeError("No demo episodes found")
    return names, np.stack(per_episode, axis=0), np.asarray(durations)


def _load_rollout_segments(csv_path: Path, prefix: str, fractions: np.ndarray, bins: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    cols = [f"{prefix}_{i:02d}" for i in range(16)]
    arr = df[cols].to_numpy(dtype=float)
    edges = np.rint(np.concatenate([[0.0], np.cumsum(fractions)]) * len(arr)).astype(int)
    edges[0] = 0
    edges[-1] = len(arr)
    segs = []
    for start, end in zip(edges[:-1], edges[1:]):
        segs.append(_resample(arr[start:max(start + 1, end)], bins))
    return np.stack(segs, axis=0)


def _plot_grid(out: Path, title: str, names: list[str], demo_q10: np.ndarray, demo_mean: np.ndarray, demo_q90: np.ndarray, rollout: np.ndarray, ylabel: str) -> None:
    n_sub, bins, n_dim = rollout.shape
    x = np.concatenate([np.linspace(i, i + 1, bins, endpoint=False) for i in range(n_sub)])
    fig, axes = plt.subplots(4, 4, figsize=(20, 13), sharex=True)
    fig.suptitle(title)
    for dim, ax in enumerate(axes.flat):
        demo_mean_flat = demo_mean[:, :, dim].reshape(-1)
        demo_q10_flat = demo_q10[:, :, dim].reshape(-1)
        demo_q90_flat = demo_q90[:, :, dim].reshape(-1)
        rollout_flat = rollout[:, :, dim].reshape(-1)
        ax.fill_between(x, demo_q10_flat, demo_q90_flat, color="#7aa6c2", alpha=0.25, linewidth=0)
        ax.plot(x, demo_mean_flat, color="#1f77b4", linewidth=1.4, label="demo mean")
        ax.plot(x, rollout_flat, color="#d62728", linewidth=1.0, label="rollout")
        ax.set_title(ACTION_NAMES[dim], fontsize=10)
        ax.grid(alpha=0.25)
        if dim % 4 == 0:
            ax.set_ylabel(ylabel)
        for boundary in range(1, n_sub):
            ax.axvline(boundary, color="black", alpha=0.18, linewidth=0.8)
    axes.flat[0].legend(loc="best", fontsize=8)
    for i, name in enumerate(names):
        axes[-1, 0].text(i + 0.5, -0.35, name, ha="center", va="top", transform=axes[-1, 0].get_xaxis_transform(), fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("normalized subtask time")
        ax.set_xlim(0, n_sub)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/mnt/data/yangky/test/datasets/g1d_food_sorting_123_lerobot"))
    parser.add_argument("--labels-dir", type=Path, default=Path("/mnt/data/yangky/test/xr_teleoperate_g1d/annotation_food_sorting_123/labels"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    csv_path = args.record_dir / "rollout_actions.csv"
    out_dir = args.record_dir / "subtask_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = _read_jsonl(args.dataset / "meta" / "episodes.jsonl")
    episode_indices = [int(ep["episode_index"]) for ep in episodes if int(ep.get("task_index", -1)) == args.task_index]
    names, demos, durations = _load_demo_segments(args.dataset, args.labels_dir, episode_indices, args.bins)
    fractions = np.median(durations, axis=0)
    fractions = fractions / fractions.sum()

    demo_mean = np.nanmean(demos, axis=0)
    demo_q10 = np.nanquantile(demos, 0.10, axis=0)
    demo_q90 = np.nanquantile(demos, 0.90, axis=0)

    outputs = {}
    for prefix in ["policy_action", "cmd_action", "robot_state_after"]:
        rollout = _load_rollout_segments(csv_path, prefix, fractions, args.bins)
        err = rollout - demo_mean
        mae_by_sub = np.nanmean(np.abs(err), axis=(1, 2))
        mae_by_dim = np.nanmean(np.abs(err), axis=(0, 1))
        _plot_grid(
            out_dir / f"{prefix}_vs_demo_subtask_aligned.png",
            f"{prefix} vs demo action, task_index={args.task_index}",
            names,
            demo_q10,
            demo_mean,
            demo_q90,
            rollout,
            "action/state value",
        )
        zero = np.zeros_like(demo_mean)
        err_q = np.zeros_like(demo_mean)
        _plot_grid(
            out_dir / f"{prefix}_minus_demo_mean_subtask_aligned.png",
            f"{prefix} - demo mean, task_index={args.task_index}",
            names,
            err_q,
            zero,
            err_q,
            err,
            "delta",
        )
        outputs[prefix] = {
            "mae_by_subtask": {name: float(value) for name, value in zip(names, mae_by_sub)},
            "mae_by_dim": {name: float(value) for name, value in zip(ACTION_NAMES, mae_by_dim)},
        }

    summary = {
        "record_dir": str(args.record_dir),
        "dataset": str(args.dataset),
        "task_index": args.task_index,
        "demo_episodes": len(episode_indices),
        "subtasks": names,
        "median_subtask_fractions": {name: float(value) for name, value in zip(names, fractions)},
        "metrics": outputs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = ["# Subtask-aligned rollout analysis", ""]
    md.append(f"- rollout: `{args.record_dir}`")
    md.append(f"- demo dataset: `{args.dataset}`")
    md.append(f"- task_index: `{args.task_index}`")
    md.append(f"- demo episodes: {len(episode_indices)}")
    md.append("")
    md.append("## Median subtask fractions")
    for name, value in zip(names, fractions):
        md.append(f"- {name}: {value:.3f}")
    md.append("")
    md.append("## Mean absolute error vs demo mean")
    for prefix, values in outputs.items():
        md.append(f"### {prefix}")
        for name, value in values["mae_by_subtask"].items():
            md.append(f"- {name}: {value:.4f}")
    (out_dir / "SUBTASK_ALIGNED_SUMMARY.md").write_text("\n".join(md) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
