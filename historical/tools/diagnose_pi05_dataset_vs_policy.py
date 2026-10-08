#!/usr/bin/env python3
"""Offline raw-policy diagnostic for the G1-D PI0.5 checkpoint.

This script intentionally does not apply client smoothing, delta limits, or robot commands.
It sends real dataset images/state to a diagnostic policy server and compares raw policy output
against the recorded action horizon.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import websockets.sync.client
from openpi_client import msgpack_numpy


NAMES = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw", "left_gripper", "right_gripper",
]
GROUPS = {"left_arm_7": slice(0, 7), "right_arm_7": slice(7, 14), "left_gripper": slice(14, 15), "right_gripper": slice(15, 16)}


@dataclass
class Sample:
    episode: int
    frame: int
    prompt: str
    state: np.ndarray
    actions: np.ndarray
    images: tuple[np.ndarray, np.ndarray, np.ndarray]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_stats(path: pathlib.Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(path.read_text())
    root = raw.get("norm_stats", raw)
    return {k: {n: np.asarray(v, dtype=np.float32) for n, v in stats.items()} for k, stats in root.items()}


def normalize(x: np.ndarray, stats: dict[str, np.ndarray], quantile: bool) -> np.ndarray:
    if quantile:
        q01, q99 = stats["q01"][: x.shape[-1]], stats["q99"][: x.shape[-1]]
        ill = np.abs(q99 - q01) < 1e-4 * (np.abs(q99) + np.abs(q01) + 1e-8)
        scale = np.where(ill, 1.0, q99 - q01)
        y = (x - q01) / (scale + 1e-6) * 2.0 - 1.0
        y = np.where(ill, 0.0, y)
        return np.clip(y, -1, 1).astype(np.float32)
    return ((x - stats["mean"][: x.shape[-1]]) / (stats["std"][: x.shape[-1]] + 1e-6)).astype(np.float32)


def normalize_unclipped(x: np.ndarray, stats: dict[str, np.ndarray], quantile: bool) -> np.ndarray:
    """Normalization without OpenPI's deployment clip, used only for round-trip QA."""
    if quantile:
        q01, q99 = stats["q01"][: x.shape[-1]], stats["q99"][: x.shape[-1]]
        return ((x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)
    return normalize(x, stats, False)


def unnormalize(x: np.ndarray, stats: dict[str, np.ndarray], quantile: bool) -> np.ndarray:
    if quantile:
        q01, q99 = stats["q01"][: x.shape[-1]], stats["q99"][: x.shape[-1]]
        return ((x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01).astype(np.float32)
    return (x * (stats["std"][: x.shape[-1]] + 1e-6) + stats["mean"][: x.shape[-1]]).astype(np.float32)


def read_frame(path: pathlib.Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read {path} frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_samples(dataset: pathlib.Path, count: int, seed: int, task_index: int | None = None) -> list[Sample]:
    tasks = {}
    for line in (dataset / "meta/tasks.jsonl").read_text().splitlines():
        item = json.loads(line)
        tasks[int(item["task_index"])] = item["task"]
    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    samples = []
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action", "frame_index", "episode_index", "task_index"])
        rows = table.to_pydict()
        n = len(rows["frame_index"])
        if n < 32:
            continue
        if task_index is not None and not np.any(np.asarray(rows["task_index"]) == task_index):
            continue
        valid = np.arange(0, n - 31)
        if task_index is not None:
            valid = valid[np.asarray(rows["task_index"])[valid] == task_index]
        if not len(valid):
            continue
        frame = int(rng.choice(valid))
        ep = int(rows["episode_index"][0])
        chunk = f"chunk-{ep // 1000:03d}"
        stem = f"episode_{ep:06d}.mp4"
        video_root = dataset / "videos" / chunk
        images = tuple(
            read_frame(video_root / stream / stem, frame)
            for stream in ("observation.images.egocentric", "observation.images.left_wrist", "observation.images.right_wrist")
        )
        gt = np.asarray(rows["action"][frame : frame + 32], dtype=np.float32)
        samples.append(Sample(ep, frame, tasks[int(rows["task_index"][frame])], np.asarray(rows["observation.state"][frame], dtype=np.float32), gt, images))
        if len(samples) >= count:
            break
    if len(samples) < count:
        raise RuntimeError(f"only found {len(samples)} usable samples, requested {count}")
    return samples


def stats_table(gt: np.ndarray, pred: np.ndarray) -> list[dict]:
    error = pred - gt
    result = []
    for d, name in enumerate(NAMES):
        result.append({
            "dim": d, "name": name,
            "gt_mean": float(gt[..., d].mean()), "gt_std": float(gt[..., d].std()),
            "gt_q01": float(np.quantile(gt[..., d], .01)), "gt_q99": float(np.quantile(gt[..., d], .99)),
            "pred_mean": float(pred[..., d].mean()), "pred_std": float(pred[..., d].std()),
            "pred_q01": float(np.quantile(pred[..., d], .01)), "pred_q99": float(np.quantile(pred[..., d], .99)),
            "error_mean": float(error[..., d].mean()), "error_std": float(error[..., d].std()),
            "mae": float(np.abs(error[..., d]).mean()), "rmse": float(np.sqrt(np.mean(error[..., d] ** 2))),
        })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("/mnt/data/yangky/test/datasets/g1d_food_sorting_123_subtasks_lerobot"))
    ap.add_argument("--norm-stats", type=pathlib.Path, default=None)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("/mnt/data/yangky/test/Psi0/.runs/openpi-05/g1d_food_sorting_pi05/g1d_food_sorting_123_subtasks_pi05_10k_bs8/10000"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9015)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--task-index", type=int, default=None, help="Only sample this task_index from meta/tasks.jsonl.")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_finetune/raw_policy_diagnostics_100"))
    args = ap.parse_args()
    args.norm_stats = args.norm_stats or args.dataset / "norm_stats.json"
    args.output.mkdir(parents=True, exist_ok=True)
    stats = load_stats(args.norm_stats)
    quantile = True  # g1d_food_sorting_pi05: ModelType.PI05 != ModelType.PI0
    samples = load_samples(args.dataset, args.count, args.seed, args.task_index)
    client = websockets.sync.client.connect(f"ws://{args.host}:{args.port}", compression=None, max_size=None, ping_interval=None, close_timeout=300)
    packer = msgpack_numpy.Packer()
    metadata = msgpack_numpy.unpackb(client.recv())
    raw_states, norm_states, gt_actions, norm_gt, raw_network, norm_actions, policy_actions = [], [], [], [], [], [], []
    records = []
    try:
        for i, s in enumerate(samples):
            raw_state = s.state.astype(np.float32)
            norm_state = normalize(raw_state, stats["state"], quantile)
            response = None
            obs = {"observation/base_image": s.images[0], "observation/left_wrist_image": s.images[1], "observation/right_wrist_image": s.images[2], "state": raw_state, "prompt": s.prompt}
            client.send(packer.pack(obs))
            response = msgpack_numpy.unpackb(client.recv(timeout=300))
            policy = np.asarray(response["actions"], dtype=np.float32)[..., :16]
            if "normalized_actions" in response:
                network = np.asarray(response["normalized_actions"], dtype=np.float32)[..., :16]
            else:
                # Compatibility with an already-running non-diagnostic server.
                # This is the exact inverse-transform reconstruction, not the
                # pre-unnormalization tensor emitted by diagnostic servers.
                network = normalize(np.asarray(response["actions"], dtype=np.float32)[..., :16], stats["actions"], quantile)
            gt = s.actions[:, :16]
            raw_states.append(raw_state); norm_states.append(norm_state); gt_actions.append(gt)
            norm_gt.append(normalize(gt, stats["actions"], quantile)); raw_network.append(network); norm_actions.append(network); policy_actions.append(policy)
            records.append({"sample": i, "episode": s.episode, "frame": s.frame, "prompt": s.prompt, "infer_ms": response.get("policy_timing", {}).get("infer_ms")})
            if (i + 1) % 10 == 0: print(f"inference {i + 1}/{len(samples)}")
    finally:
        client.close()
    raw_states, norm_states = np.asarray(raw_states), np.asarray(norm_states)
    gt_actions, norm_gt = np.asarray(gt_actions), np.asarray(norm_gt)
    raw_network, norm_actions, policy_actions = np.asarray(raw_network), np.asarray(norm_actions), np.asarray(policy_actions)
    state_unclipped = normalize_unclipped(raw_states, stats["state"], quantile)
    action_unclipped = normalize_unclipped(gt_actions, stats["actions"], quantile)
    roundtrip_state_exact = np.max(np.abs(unnormalize(state_unclipped, stats["state"], quantile) - raw_states))
    roundtrip_action_exact = np.max(np.abs(unnormalize(action_unclipped, stats["actions"], quantile) - gt_actions))
    roundtrip_state = np.max(np.abs(unnormalize(norm_states, stats["state"], quantile) - raw_states))
    roundtrip_action = np.max(np.abs(unnormalize(norm_gt, stats["actions"], quantile) - gt_actions))
    state_clipped_fraction = float(np.mean(np.abs(state_unclipped) > 1.0))
    action_clipped_fraction = float(np.mean(np.abs(action_unclipped) > 1.0))
    table = stats_table(gt_actions, policy_actions)
    (args.output / "metrics.json").write_text(json.dumps({"per_dimension": table, "normalization": {"mode": "q01_q99", "norm_stats": str(args.norm_stats), "norm_stats_sha256": sha256(args.norm_stats), "roundtrip_max_abs_state_after_openpi_clip": float(roundtrip_state), "roundtrip_max_abs_action_after_openpi_clip": float(roundtrip_action), "roundtrip_max_abs_state_without_clip": float(roundtrip_state_exact), "roundtrip_max_abs_action_without_clip": float(roundtrip_action_exact), "state_fraction_clipped_by_openpi": state_clipped_fraction, "action_fraction_clipped_by_openpi": action_clipped_fraction}, "server_metadata": metadata, "sample_count": len(samples), "checkpoint": str(args.checkpoint)}, indent=2))
    with (args.output / "per_dimension_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys()); writer.writeheader(); writer.writerows(table)
    group = {}
    for name, sl in GROUPS.items():
        e = policy_actions[..., sl] - gt_actions[..., sl]
        group[name] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.mean(e ** 2))), "error_mean": float(e.mean()), "error_std": float(e.std())}
    (args.output / "group_metrics.json").write_text(json.dumps(group, indent=2))
    np.savez_compressed(args.output / "samples.npz", raw_state=raw_states, normalized_state=norm_states, gt_action=gt_actions, gt_action_normalized=norm_gt, raw_network_action=raw_network, normalized_action=norm_actions, policy_action=policy_actions)
    fig, axes = plt.subplots(4, 4, figsize=(18, 14));
    for d, ax in enumerate(axes.ravel()):
        ax.hist(gt_actions[..., d].ravel(), bins=35, alpha=.55, density=True, label="GT")
        ax.hist(policy_actions[..., d].ravel(), bins=35, alpha=.55, density=True, label="prediction")
        ax.set_title(f"{d}: {NAMES[d]}"); ax.grid(alpha=.2)
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(args.output / "gt_vs_prediction_histograms_16d.png", dpi=160); plt.close(fig)
    err = policy_actions - gt_actions
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    horizons = np.arange(err.shape[1])
    axes[0].plot(horizons, np.abs(err).mean(axis=(0, 2)), label="MAE")
    axes[0].plot(horizons, np.sqrt(np.mean(err ** 2, axis=(0, 2))), label="RMSE")
    for h in (0, 7, 15, 31): axes[0].axvline(h, color="gray", ls="--", alpha=.4)
    for name, sl in GROUPS.items(): axes[1].plot(horizons, np.abs(err[..., sl]).mean(axis=(0, 2)), label=name)
    axes[0].set_title("Error versus action horizon"); axes[1].set_title("MAE by action group")
    axes[0].legend(); axes[1].legend(); axes[1].set_xlabel("horizon index"); fig.tight_layout(); fig.savefig(args.output / "error_vs_horizon.png", dpi=160); plt.close(fig)
    (args.output / "samples.jsonl").write_text("\n".join(json.dumps(x) for x in records) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(samples), "norm_mode": "q01_q99", "roundtrip_state_after_clip": float(roundtrip_state), "roundtrip_action_after_clip": float(roundtrip_action), "roundtrip_state_without_clip": float(roundtrip_state_exact), "roundtrip_action_without_clip": float(roundtrip_action_exact)}, indent=2))


if __name__ == "__main__":
    main()
