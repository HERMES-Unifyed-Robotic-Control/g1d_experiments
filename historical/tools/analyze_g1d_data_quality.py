#!/usr/bin/env python3
"""Read-only G1-D data distribution and quality audit.

This script never edits a dataset.  It creates a review manifest instead of
dropping episodes so that a human can inspect every proposed exclusion.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


VIEWS = {
    "egocentric": "head",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parquet_arrays(root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(path, columns=["observation.state", "action"])
    return (
        np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
        np.asarray(table["action"].to_pylist(), dtype=np.float32),
    )


def frame(path: Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, image = cap.read()
    cap.release()
    return image if ok else None


def image_metrics(image: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(hsv[..., 2].mean()),
        "saturation": float(hsv[..., 1].mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "blue": float(image[..., 0].mean()),
        "green": float(image[..., 1].mean()),
        "red": float(image[..., 2].mean()),
    }


def policy_resolution(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the client geometry closely enough for comparable image metrics."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    top = (size - resized.shape[0]) // 2
    bottom = size - resized.shape[0] - top
    left = (size - resized.shape[1]) // 2
    right = size - resized.shape[1] - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def robust_limits(values: pd.Series) -> tuple[float, float]:
    q1, q3 = values.quantile([0.25, 0.75])
    spread = float(q3 - q1)
    return max(0.0, float(q1 - 1.5 * spread)), float(q3 + 1.5 * spread)


def load_rollout(path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if path is None or not path.exists():
        return None, None, None
    rows = pd.read_csv(path)
    def matrix(prefix: str) -> np.ndarray:
        return rows[[f"{prefix}_{i:02d}" for i in range(16)]].to_numpy(dtype=float)
    return matrix("obs_state"), matrix("policy_action"), matrix("cmd_action")


def plot_joint_distributions(
    output: Path,
    names: list[str],
    train: np.ndarray,
    online: np.ndarray | None,
    title: str,
) -> list[dict]:
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), constrained_layout=True)
    result = []
    for dim, ax in enumerate(axes.flat):
        values = train[:, dim]
        lo, hi = np.quantile(values, [0.01, 0.99])
        pad = max(0.05, 0.08 * (hi - lo))
        bins = np.linspace(lo - pad, hi + pad, 36)
        ax.hist(values, bins=bins, density=True, alpha=0.62, label="train", color="#377eb8")
        row = {"dimension": names[dim], "train_q01": float(lo), "train_q99": float(hi)}
        if online is not None and len(online):
            ax.hist(online[:, dim], bins=bins, density=True, alpha=0.52, label="real rollout", color="#e6550d")
            outside = np.mean((online[:, dim] < lo) | (online[:, dim] > hi))
            iqr = max(float(np.subtract(*np.quantile(values, [0.75, 0.25]))), 1e-5)
            sorted_train = np.sort(values)
            # Quantile matching gives a stable, dependency-free shift estimate.
            q = np.linspace(0.01, 0.99, 99)
            w1_iqr = float(np.mean(np.abs(np.quantile(values, q) - np.quantile(online[:, dim], q))) / iqr)
            row.update({"rollout_outside_train_q01_q99": float(outside), "quantile_shift_iqr": w1_iqr})
        ax.set_title(names[dim], fontsize=9)
        ax.grid(alpha=0.18)
        result.append(row)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return result


def plot_quality(output: Path, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for task, group in metrics.groupby("task_index"):
        axes[0].scatter(group["duration_s"], group["max_action_step"], s=18, alpha=0.65, label=f"task {task}")
        axes[1].scatter(group["duration_s"], group["action_path_length"], s=18, alpha=0.65, label=f"task {task}")
    review = metrics[metrics["requires_review"]]
    for ax, column in zip(axes, ["max_action_step", "action_path_length"]):
        ax.scatter(review["duration_s"], review[column], facecolors="none", edgecolors="crimson", s=70, linewidths=1.4, label="review candidate")
        ax.set_xlabel("segment duration (s)")
        ax.set_ylabel(column.replace("_", " ") + " (rad)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Episode-level mechanical quality candidates (review only)", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_image_metrics(output: Path, train_metrics: pd.DataFrame, rollout_metrics: pd.DataFrame) -> list[dict]:
    fields = ["brightness", "saturation", "sharpness", "red", "green", "blue"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    summary = []
    for field, ax in zip(fields, axes.flat):
        for view, group in train_metrics.groupby("view"):
            ax.hist(group[field], bins=28, density=True, alpha=0.42, label=f"train {view}")
        if not rollout_metrics.empty:
            for view, group in rollout_metrics.groupby("view"):
                ax.axvline(group[field].median(), color={"head":"#e6550d", "left_wrist":"#31a354", "right_wrist":"#756bb1"}[view], linewidth=2, linestyle="--", label=f"rollout {view} median")
        ax.set_title(field)
        ax.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Image statistics: training frames and real rollout medians", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    for view, group in train_metrics.groupby("view"):
        row = {"view": view}
        online = rollout_metrics[rollout_metrics["view"] == view]
        for field in fields:
            median, std = float(group[field].median()), max(float(group[field].std()), 1e-5)
            row[f"train_{field}_median"] = median
            if not online.empty:
                row[f"rollout_{field}_median"] = float(online[field].median())
                row[f"rollout_{field}_median_z"] = float((online[field].median() - median) / std)
        summary.append(row)
    return summary


def contact_sheet(output: Path, samples: dict[str, list[np.ndarray]], rollout: dict[str, list[np.ndarray]]) -> None:
    rows = []
    for view in ("head", "left_wrist", "right_wrist"):
        train = samples.get(view, [])[:3]
        real = rollout.get(view, [])[:3]
        tiles = []
        for title, source in (("train", train), ("rollout", real)):
            for image in source:
                resized = cv2.resize(image, (224, 168))
                cv2.putText(resized, f"{view} {title}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(resized)
        if tiles:
            while len(tiles) < 6:
                tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.concatenate(tiles, axis=1))
    if rows:
        cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rollout-csv", type=Path)
    parser.add_argument("--rollout-image-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-episodes-per-task", type=int, default=30)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((args.train_root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    episodes = read_jsonl(args.train_root / "meta" / "episodes.jsonl")
    source_rows = {int(row["episode_index"]): row for row in read_jsonl(args.source_root / "meta" / "episodes.jsonl")}
    episode_metrics, states, actions = [], [], []
    for row in episodes:
        ep = int(row["episode_index"])
        state, action = parquet_arrays(args.train_root, ep)
        states.append(state)
        actions.append(action)
        state_step = np.abs(np.diff(state, axis=0)) if len(state) > 1 else np.zeros((1, 16))
        action_step = np.abs(np.diff(action, axis=0)) if len(action) > 1 else np.zeros((1, 16))
        episode_metrics.append({
            "episode_index": ep,
            "source_episode_index": int(row.get("source_episode_index", -1)),
            "task_index": int(row["task_index"]),
            "subtask": row.get("subtask_object", "unknown"),
            "length": int(len(state)),
            "duration_s": float(len(state) / 30.0),
            "finite": bool(np.isfinite(state).all() and np.isfinite(action).all()),
            "max_state_step": float(state_step.max()),
            "max_action_step": float(action_step.max()),
            "action_path_length": float(np.abs(np.diff(action, axis=0)).sum()),
            "left_gripper_closed_fraction": float(np.mean(action[:, 14] < 2.0)),
            "right_gripper_closed_fraction": float(np.mean(action[:, 15] < 2.0)),
        })
    state_all, action_all = np.concatenate(states), np.concatenate(actions)
    metrics = pd.DataFrame(episode_metrics)
    metrics["requires_review"] = False
    metrics["review_reasons"] = ""
    for task, indices in metrics.groupby("task_index").groups.items():
        subset = metrics.loc[indices]
        low_duration, high_duration = robust_limits(subset["duration_s"])
        max_jump = float(subset["max_action_step"].quantile(0.99))
        low_motion = float(subset["action_path_length"].quantile(0.01))
        mask = (
            (~subset["finite"])
            | (subset["length"] < 30)
            | (subset["duration_s"] < low_duration)
            | (subset["duration_s"] > high_duration)
            | (subset["max_action_step"] > max_jump)
            | (subset["action_path_length"] < low_motion)
        )
        reasons = []
        for _, r in subset.iterrows():
            parts = []
            if not r.finite: parts.append("nonfinite")
            if r.length < 30: parts.append("shorter_than_1s")
            if r.duration_s < low_duration or r.duration_s > high_duration: parts.append("duration_outlier")
            if r.max_action_step > max_jump: parts.append("action_step_outlier")
            if r.action_path_length < low_motion: parts.append("near_static_outlier")
            reasons.append(",".join(parts))
        metrics.loc[indices, "requires_review"] = mask.to_numpy()
        metrics.loc[indices, "review_reasons"] = reasons

    # Labels are an independent annotation artifact.  Verify that every raw
    # source episode has a completed, contiguous, non-overlapping annotation.
    label_rows = []
    for source_index, source in sorted(source_rows.items()):
        path = args.labels_root / f"g1d_food_sorting_123_episode_{source_index:06d}.json"
        issue = []
        if not path.exists():
            issue.append("missing_label")
            payload = {}
        else:
            payload = json.loads(path.read_text())
            subtasks = payload.get("subtasks", [])
            previous_end = None
            shared_boundary_frames = 0
            for part in subtasks:
                start, end = int(part.get("start_frame", -1)), int(part.get("end_frame", -1))
                if start < 0 or end < start:
                    issue.append("invalid_range")
                if previous_end is None and start != 0:
                    issue.append("does_not_start_at_zero")
                if previous_end is not None:
                    if start == previous_end:
                        shared_boundary_frames += 1
                    elif start > previous_end + 1:
                        issue.append("gap_between_subtasks")
                    elif start < previous_end:
                        issue.append("overlap_more_than_one_frame")
                previous_end = end
            if previous_end != int(source["length"]) - 1:
                issue.append("label_does_not_end_at_last_frame")
            if not payload.get("done", False):
                issue.append("label_not_done")
        label_rows.append({"source_episode_index": source_index, "length": int(source["length"]), "label_path": str(path), "shared_boundary_frames": shared_boundary_frames if path.exists() else 0, "issues": ",".join(issue)})
    labels = pd.DataFrame(label_rows)

    rollout_state, rollout_policy, rollout_cmd = load_rollout(args.rollout_csv)
    task0 = metrics[metrics.task_index == 0]
    task0_states, task0_actions = [], []
    for ep in task0.episode_index:
        state, action = parquet_arrays(args.train_root, int(ep))
        task0_states.append(state)
        task0_actions.append(action)
    state_shift = plot_joint_distributions(out / "state_train_vs_rollout.png", names, np.concatenate(task0_states), rollout_state, "Subtask 0 state distribution: training vs real rollout")
    action_shift = plot_joint_distributions(out / "action_train_vs_rollout.png", names, np.concatenate(task0_actions), rollout_cmd, "Subtask 0 command/action distribution: training vs real rollout")
    plot_quality(out / "episode_quality_scatter.png", metrics)

    # Uniform, deterministic image sample across subtasks.  This is enough to
    # expose camera/view changes without decoding every 30 Hz frame.
    rng = np.random.default_rng(20260927)
    train_images: dict[str, list[np.ndarray]] = defaultdict(list)
    image_rows = []
    for task, group in metrics.groupby("task_index"):
        selected = group.sample(min(args.sample_episodes_per_task, len(group)), random_state=20260927 + int(task))
        for ep in selected.episode_index:
            meta = episodes[int(ep)]
            midpoint = int(meta["length"]) // 2
            for video_view, view in VIEWS.items():
                video = args.train_root / "videos" / f"chunk-{int(ep) // 1000:03d}" / f"observation.images.{video_view}" / f"episode_{int(ep):06d}.mp4"
                image = frame(video, midpoint)
                if image is None:
                    continue
                image = policy_resolution(image)
                train_images[view].append(image)
                image_rows.append({"source": "train", "view": view, "task_index": int(task), "episode_index": int(ep), **image_metrics(image)})
    rollout_images: dict[str, list[np.ndarray]] = defaultdict(list)
    rollout_rows = []
    if args.rollout_image_dir and args.rollout_image_dir.exists():
        file_map = {"head": "input_base_224.mp4", "left_wrist": "input_left_wrist_224.mp4", "right_wrist": "input_right_wrist_224.mp4"}
        for view, name in file_map.items():
            video = args.rollout_image_dir / name
            cap = cv2.VideoCapture(str(video))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            for index in np.unique(np.linspace(0, max(0, count - 1), min(8, count), dtype=int)):
                image = frame(video, int(index))
                if image is None:
                    continue
                rollout_images[view].append(image)
                rollout_rows.append({"source": "rollout", "view": view, "frame": int(index), **image_metrics(image)})
    train_image_df, rollout_image_df = pd.DataFrame(image_rows), pd.DataFrame(rollout_rows)
    image_shift = plot_image_metrics(out / "image_metric_train_vs_rollout.png", train_image_df, rollout_image_df)
    contact_sheet(out / "image_train_vs_rollout_sheet.jpg", train_images, rollout_images)

    metrics.to_csv(out / "episode_quality_candidates.csv", index=False)
    labels.to_csv(out / "label_integrity.csv", index=False)
    pd.DataFrame(state_shift).to_csv(out / "state_shift_by_joint.csv", index=False)
    pd.DataFrame(action_shift).to_csv(out / "action_shift_by_joint.csv", index=False)
    pd.DataFrame(image_shift).to_csv(out / "image_shift_by_view.csv", index=False)
    summary = {
        "scope": {"train_root": str(args.train_root), "source_root": str(args.source_root), "rollout_csv": str(args.rollout_csv) if args.rollout_csv else None},
        "train": {"episodes": int(len(metrics)), "frames": int(len(state_all)), "tasks": metrics.groupby("task_index").size().to_dict()},
        "quality": {
            "hard_nonfinite": int((~metrics.finite).sum()),
            "review_candidates": int(metrics.requires_review.sum()),
            "label_integrity_issues": int(labels.issues.ne("").sum()),
            "label_shared_boundary_frames": int(labels.shared_boundary_frames.sum()),
        },
        "rollout": {"rows": int(0 if rollout_state is None else len(rollout_state))},
        "notes": [
            "Review candidates are not removed automatically.",
            "The source LeRobot dataset has no carried-through success/FAILED marker, so task success cannot be reconstructed from these files.",
            "State/action shift is reported against subtask 0 because the supplied rollout executes subtask 0.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = [
        "# G1-D data distribution and quality audit",
        "",
        f"- Training subtask episodes: {summary['train']['episodes']}; frames: {summary['train']['frames']}",
        f"- Mechanical review candidates: {summary['quality']['review_candidates']} (not removed)",
        f"- Annotation-integrity issues: {summary['quality']['label_integrity_issues']}; shared label boundary frames: {summary['quality']['label_shared_boundary_frames']}",
        f"- Real rollout rows compared: {summary['rollout']['rows']}",
        "",
        "## Files",
        "",
        "- `state_train_vs_rollout.png`, `action_train_vs_rollout.png`: per-joint distributions.",
        "- `image_metric_train_vs_rollout.png`, `image_train_vs_rollout_sheet.jpg`: three-camera visual shift checks.",
        "- `episode_quality_candidates.csv`: review-only mechanical quality manifest.",
        "- `label_integrity.csv`: source-label coverage validation.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
