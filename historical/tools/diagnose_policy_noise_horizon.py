#!/usr/bin/env python3
"""Measure policy sampling noise at each action horizon for one fixed observation."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose_pi05_dataset_vs_policy import GROUPS, NAMES, load_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("/mnt/data/yangky/test/datasets/g1d_food_sorting_123_subtasks_lerobot"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9015)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_finetune/policy_noise_horizon_50"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sample = load_samples(args.dataset, 1, args.seed)[0]
    obs = {
        "observation/base_image": sample.images[0],
        "observation/left_wrist_image": sample.images[1],
        "observation/right_wrist_image": sample.images[2],
        "state": sample.state,
        "prompt": sample.prompt,
    }
    client = websockets.sync.client.connect(f"ws://{args.host}:{args.port}", compression=None, max_size=None, ping_interval=None, close_timeout=300)
    packer = msgpack_numpy.Packer()
    metadata = msgpack_numpy.unpackb(client.recv())
    raw = []
    normalized = []
    timings = []
    try:
        for i in range(args.samples):
            client.send(packer.pack(obs))
            result = msgpack_numpy.unpackb(client.recv(timeout=300))
            raw.append(np.asarray(result["actions"], dtype=np.float32)[..., :16])
            normalized.append(np.asarray(result.get("normalized_actions"), dtype=np.float32)[..., :16])
            timings.append(result.get("policy_timing", {}).get("infer_ms"))
    finally:
        client.close()
    raw = np.asarray(raw)
    normalized = np.asarray(normalized)
    raw_std = raw.std(axis=0)
    normalized_std = normalized.std(axis=0)
    groups = {}
    for name, sl in GROUPS.items():
        groups[name] = {
            "raw_policy_action_std_mean": raw_std[..., sl].mean(axis=-1).tolist(),
            "raw_policy_action_std_max": raw_std[..., sl].max(axis=-1).tolist(),
            "normalized_action_std_mean": normalized_std[..., sl].mean(axis=-1).tolist(),
            "normalized_action_std_max": normalized_std[..., sl].max(axis=-1).tolist(),
        }
    payload = {
        "sample_count": args.samples,
        "episode": sample.episode,
        "frame": sample.frame,
        "prompt": sample.prompt,
        "metadata": metadata,
        "raw_policy_action_std": raw_std.tolist(),
        "normalized_action_std": normalized_std.tolist(),
        "groups": groups,
        "mean_infer_ms": float(np.mean(timings)),
    }
    (args.output / "noise_horizon_metrics.json").write_text(json.dumps(payload, indent=2))
    with (args.output / "noise_horizon_per_dim.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon", "dim", "name", "raw_policy_std", "normalized_action_std"])
        for h in range(raw_std.shape[0]):
            for d, name in enumerate(NAMES):
                writer.writerow([h, d, name, float(raw_std[h, d]), float(normalized_std[h, d])])
    np.savez_compressed(args.output / "noise_samples.npz", raw_policy_action_samples=raw, normalized_action_samples=normalized)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    horizons = np.arange(raw_std.shape[0])
    for name, sl in GROUPS.items():
        axes[0].plot(horizons, raw_std[..., sl].mean(axis=-1), label=name)
        axes[1].plot(horizons, normalized_std[..., sl].mean(axis=-1), label=name)
    axes[0].set_title("Raw policy action sampling std vs horizon")
    axes[1].set_title("Normalized network action sampling std vs horizon")
    axes[1].set_xlabel("action horizon index")
    axes[0].set_ylabel("std in action units"); axes[1].set_ylabel("std in normalized units")
    axes[0].legend(); axes[1].legend(); axes[0].grid(alpha=.2); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.output / "noise_std_vs_horizon.png", dpi=160); plt.close(fig)
    print(json.dumps({"output": str(args.output), "samples": args.samples, "episode": sample.episode, "frame": sample.frame, "raw_h0_mean_std": float(raw_std[0].mean()), "raw_h31_mean_std": float(raw_std[-1].mean())}, indent=2))


if __name__ == "__main__":
    main()
