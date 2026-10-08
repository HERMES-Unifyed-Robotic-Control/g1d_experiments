#!/usr/bin/env python3
"""Convert G1-D EpisodeWriter data to a local LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FPS = 30
CAMERAS = {
    "observation.images.egocentric": "color_0",
    "observation.images.left_wrist": "color_2",
    "observation.images.right_wrist": "color_3",
}
JOINT_NAMES = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "left_gripper", "right_gripper",
]


def state_or_action(frame, root):
    record = frame[root]
    return np.asarray(
        record["left_arm"]["qpos"]
        + record["right_arm"]["qpos"]
        + record["left_ee"]["qpos"]
        + record["right_ee"]["qpos"],
        dtype=np.float32,
    )


def encode_video(episode, camera, destination, overwrite):
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    pattern = episode / "colors" / f"%06d_{camera}.jpg"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(FPS), "-i", str(pattern),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True)


def summary(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def video_feature():
    return {
        "dtype": "video", "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(FPS), "video.codec": "h264",
            "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def convert(source, output, prompt, max_episodes, overwrite, video_workers):
    episodes = sorted(source.glob("episode_*"))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise RuntimeError(f"No episode directories found in {source}")

    payloads = []
    for episode in episodes:
        payload = json.loads((episode / "data.json").read_text())
        frames = payload["data"]
        if not frames:
            raise RuntimeError(f"Empty episode: {episode}")
        payloads.append((episode, frames))

    output.mkdir(parents=True, exist_ok=True)
    meta = output / "meta"
    meta.mkdir(exist_ok=True)
    episode_rows = []
    episode_stats = []
    global_index = 0

    video_futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=video_workers) as executor:
        for episode_index, (episode, frames) in enumerate(payloads):
            states = [state_or_action(frame, "states") for frame in frames]
            actions = [state_or_action(frame, "actions") for frame in frames]
            count = len(frames)
            chunk = episode_index // 1000
            data_path = output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)

            table = pa.table({
                "observation.state": pa.array(states, type=pa.list_(pa.float32(), 16)),
                "action": pa.array(actions, type=pa.list_(pa.float32(), 16)),
                "timestamp": pa.array(np.arange(count, dtype=np.float32) / FPS),
                "frame_index": pa.array(np.arange(count), type=pa.int64()),
                "episode_index": pa.array(np.full(count, episode_index), type=pa.int64()),
                "index": pa.array(np.arange(global_index, global_index + count), type=pa.int64()),
                "task_index": pa.array(np.zeros(count), type=pa.int64()),
                "next.done": pa.array([False] * (count - 1) + [True]),
            })
            pq.write_table(table, data_path, compression="zstd")

            for video_key, raw_camera in CAMERAS.items():
                destination = output / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
                video_futures.append(executor.submit(encode_video, episode, raw_camera, destination, overwrite))

            episode_rows.append({
                "episode_index": episode_index, "tasks": [0], "length": count,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + count - 1,
                "robot_type": "g1d", "instruction": prompt,
            })
            episode_stats.append({
                "episode_index": episode_index,
                "stats": {
                    "observation.state": summary(states),
                    "action": summary(actions),
                    "timestamp": summary(np.arange(count, dtype=np.float32)[:, None] / FPS),
                },
            })
            global_index += count
            print(f"queued {episode.name}: {count} frames", flush=True)

        for completed, future in enumerate(concurrent.futures.as_completed(video_futures), 1):
            future.result()
            if completed % 10 == 0 or completed == len(video_futures):
                print(f"encoded videos: {completed}/{len(video_futures)}", flush=True)

    features = {
        **{key: video_feature() for key in CAMERAS},
        "observation.state": {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1", "robot_type": "g1d",
        "total_episodes": len(episode_rows), "total_frames": global_index,
        "total_tasks": 1, "total_videos": len(episode_rows) * len(CAMERAS),
        "total_chunks": math.ceil(len(episode_rows) / 1000), "chunks_size": 1000,
        "fps": FPS, "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    files = {
        "tasks.jsonl": [{"task_index": 0, "task": prompt}],
        "episodes.jsonl": episode_rows,
        "episodes_stats.jsonl": episode_stats,
    }
    for name, rows in files.items():
        (meta / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (meta / "g1d_schema.json").write_text(json.dumps({
        "raw_camera_mapping": CAMERAS,
        "dropped_training_camera": {"color_1": "second binocular head view"},
        "state_order": JOINT_NAMES, "action_order": JOINT_NAMES,
        "action_semantics": "absolute joint position targets in radians; Dex1 motor q, open approximately 5.4",
        "raw_source": str(source.resolve()),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default="Move the black cup, place fruit into it, then pour water from the white cup into the black cup.")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite-videos", action="store_true")
    parser.add_argument("--video-workers", type=int, default=8)
    args = parser.parse_args()
    convert(args.source, args.output, args.prompt, args.max_episodes, args.overwrite_videos, args.video_workers)


if __name__ == "__main__":
    main()
