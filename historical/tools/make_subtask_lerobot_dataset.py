#!/usr/bin/env python3
"""Build a subtask-sliced LeRobot dataset from labeled G1-D food-sorting episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FPS = 30
VIDEO_KEYS = (
    "observation.images.egocentric",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)

SHARED_PROMPTS = {
    "move_black_cup": "Move the black cup from the right side to the middle position.",
    "left_hand_get_white_cup": "Use the left hand to pick up the white cup on the left.",
    "right_hand_hold_black_cup": "Use the right hand to pick up and steady the black cup for pouring.",
    "pour_water": "Pour water from the white cup into the black cup, then return the robot to its home pose.",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def task_prompt(label_item: dict) -> str:
    name = label_item["object"]
    if name in SHARED_PROMPTS:
        return SHARED_PROMPTS[name]
    if name == "place_ingredient":
        return label_item["text_en"]
    raise ValueError(f"Unknown subtask object: {name}")


def ffmpeg_cut(src: Path, dst: Path, start_frame: int, frame_count: int, fps: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    start_s = start_frame / fps
    duration_s = frame_count / fps
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.9f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.9f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(command, check=True)


def build_dataset(source: Path, labels_dir: Path, output: Path, max_source_episodes: int | None, overwrite: bool, video_workers: int) -> None:
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    (output / "meta").mkdir(parents=True)

    info = json.loads((source / "meta" / "info.json").read_text())
    episodes = read_jsonl(source / "meta" / "episodes.jsonl")
    if max_source_episodes is not None:
        episodes = episodes[:max_source_episodes]
    source_tasks = {int(row["task_index"]): row["task"] for row in read_jsonl(source / "meta" / "tasks.jsonl")}

    task_to_index: dict[str, int] = {}
    task_rows: list[dict] = []
    episode_rows: list[dict] = []
    stats_rows: list[dict] = []
    segment_rows: list[dict] = []
    global_index = 0
    new_episode_index = 0
    video_jobs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=video_workers) as executor:
        for source_ep in episodes:
            source_index = int(source_ep["episode_index"])
            label_path = labels_dir / f"g1d_food_sorting_123_episode_{source_index:06d}.json"
            if not label_path.exists():
                continue
            label = json.loads(label_path.read_text())
            if not label.get("done") or not label.get("subtasks"):
                continue

            parquet_path = source / source_ep.get("data_path", f"data/chunk-000/episode_{source_index:06d}.parquet")
            df = pd.read_parquet(parquet_path)
            n_frames = len(df)
            source_task_index = int(source_ep.get("task_index", source_ep.get("tasks", [0])[0]))
            source_instruction = source_ep.get("instruction", source_tasks.get(source_task_index, ""))

            for item in sorted(label["subtasks"], key=lambda x: x["index"]):
                start = max(0, min(n_frames - 1, int(item["start_frame"])))
                end = max(start + 1, min(n_frames, int(item["end_frame"]) + 1))
                count = end - start
                if count < 8:
                    continue
                prompt = task_prompt(item)
                if prompt not in task_to_index:
                    task_to_index[prompt] = len(task_rows)
                    task_rows.append({"task_index": task_to_index[prompt], "task": prompt})
                task_index = task_to_index[prompt]

                segment = df.iloc[start:end].copy()
                segment["timestamp"] = np.arange(count, dtype=np.float32) / FPS
                segment["frame_index"] = np.arange(count, dtype=np.int64)
                segment["episode_index"] = np.full(count, new_episode_index, dtype=np.int64)
                segment["index"] = np.arange(global_index, global_index + count, dtype=np.int64)
                segment["task_index"] = np.full(count, task_index, dtype=np.int64)
                segment["next.done"] = [False] * (count - 1) + [True]

                chunk = new_episode_index // 1000
                rel_data_path = Path(f"data/chunk-{chunk:03d}/episode_{new_episode_index:06d}.parquet")
                data_path = output / rel_data_path
                data_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(segment, preserve_index=False), data_path, compression="zstd")

                states = np.stack(segment["observation.state"].to_numpy()).astype(np.float32)
                actions = np.stack(segment["action"].to_numpy()).astype(np.float32)
                timestamps = np.asarray(segment["timestamp"].to_numpy(), dtype=np.float32)[:, None]
                stats_rows.append(
                    {
                        "episode_index": new_episode_index,
                        "stats": {
                            "observation.state": stats(states),
                            "action": stats(actions),
                            "timestamp": stats(timestamps),
                        },
                    }
                )
                episode_rows.append(
                    {
                        "episode_index": new_episode_index,
                        "tasks": [task_index],
                        "length": count,
                        "dataset_from_index": global_index,
                        "dataset_to_index": global_index + count - 1,
                        "robot_type": "g1d",
                        "instruction": prompt,
                        "task_index": task_index,
                        "source_episode_index": source_index,
                        "source_task_index": source_task_index,
                        "source_instruction": source_instruction,
                        "subtask_index": int(item["index"]),
                        "subtask_object": item["object"],
                        "source_start_frame": start,
                        "source_end_frame_exclusive": end,
                    }
                )
                segment_rows.append(episode_rows[-1].copy())

                for video_key in VIDEO_KEYS:
                    src_video = source / "videos" / f"chunk-{source_index // 1000:03d}" / video_key / f"episode_{source_index:06d}.mp4"
                    dst_video = output / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{new_episode_index:06d}.mp4"
                    video_jobs.append(executor.submit(ffmpeg_cut, src_video, dst_video, start, count, FPS, overwrite))

                global_index += count
                new_episode_index += 1

        for completed, future in enumerate(concurrent.futures.as_completed(video_jobs), 1):
            future.result()
            if completed % 50 == 0 or completed == len(video_jobs):
                print(f"encoded segment videos: {completed}/{len(video_jobs)}", flush=True)

    info["total_episodes"] = len(episode_rows)
    info["total_frames"] = global_index
    info["total_tasks"] = len(task_rows)
    info["total_videos"] = len(episode_rows) * len(VIDEO_KEYS)
    info["total_chunks"] = math.ceil(max(1, len(episode_rows)) / int(info.get("chunks_size", 1000)))
    info["splits"] = {"train": f"0:{len(episode_rows)}"}
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
    write_jsonl(output / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", stats_rows)
    shutil.copy2(source / "meta" / "g1d_schema.json", output / "meta" / "g1d_schema.json")
    (output / "meta" / "subtask_segments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in segment_rows)
    )
    print(f"wrote {len(episode_rows)} subtask episodes, {global_index} frames, {len(task_rows)} prompts -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/mnt/data/yangky/test/datasets/g1d_food_sorting_123_lerobot"))
    parser.add_argument("--labels-dir", type=Path, default=Path("/mnt/data/yangky/test/xr_teleoperate_g1d/annotation_food_sorting_123/labels"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/yangky/test/datasets/g1d_food_sorting_123_subtasks_lerobot"))
    parser.add_argument("--max-source-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-workers", type=int, default=8)
    args = parser.parse_args()
    build_dataset(args.source, args.labels_dir, args.output, args.max_source_episodes, args.overwrite, args.video_workers)


if __name__ == "__main__":
    main()
