#!/usr/bin/env python3
"""Audit raw G1-D episodes before policy dataset conversion."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARM_NAMES = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
GRIPPER_NAMES = ["left_gripper", "right_gripper"]


def nested(obj, *keys, default=None):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def vector(frame, root, field):
    if field == "arm":
        left = nested(frame, root, "left_arm", "qpos", default=[])
        right = nested(frame, root, "right_arm", "qpos", default=[])
        return list(left) + list(right)
    if field == "gripper":
        left = nested(frame, root, "left_ee", "qpos", default=[])
        right = nested(frame, root, "right_ee", "qpos", default=[])
        return list(left) + list(right)
    return nested(frame, root, field, default=[])


def summarize(values, names=None):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    if arr.ndim == 1:
        arr = arr[:, None]
    finite = np.isfinite(arr)
    result = {
        "count": int(arr.shape[0]),
        "dimensions": int(arr.shape[1]),
        "nonfinite": int((~finite).sum()),
        "per_dimension": [],
    }
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        col = col[np.isfinite(col)]
        item = {"index": idx, "name": names[idx] if names and idx < len(names) else str(idx)}
        if col.size:
            item.update({
                "min": float(np.min(col)),
                "p01": float(np.percentile(col, 1)),
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "p50": float(np.percentile(col, 50)),
                "p99": float(np.percentile(col, 99)),
                "max": float(np.max(col)),
            })
        result["per_dimension"].append(item)
    return result


def scalar_summary(values):
    summary = summarize(values)
    if summary.get("count", 0) == 0:
        return summary
    return {"count": summary["count"], "nonfinite": summary["nonfinite"], **summary["per_dimension"][0]}


def append_if_shape(store, key, value, shape, bad_shapes):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        bad_shapes[key][str(tuple(arr.shape))] += 1
        return
    store[key].append(arr)


def audit(dataset: Path, decode_stride: int):
    episodes = sorted(path for path in dataset.iterdir() if path.is_dir() and path.name.startswith("episode_"))
    values = defaultdict(list)
    bad_shapes = defaultdict(Counter)
    field_missing = Counter()
    camera_ref_counts = Counter()
    camera_missing = Counter()
    camera_bad_name = Counter()
    decode_failures = Counter()
    episode_rows = []
    all_action_deltas = []
    all_state_deltas = []
    all_tracking_errors = []
    total_frames = 0

    try:
        from PIL import Image
    except ImportError:
        Image = None

    for episode_index, episode in enumerate(episodes):
        json_path = episode / "data.json"
        if not json_path.exists():
            episode_rows.append({"episode": episode.name, "error": "missing data.json"})
            continue
        try:
            payload = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            episode_rows.append({"episode": episode.name, "error": f"invalid data.json: {exc}"})
            continue

        frames = payload.get("data", [])
        frame_count = len(frames)
        total_frames += frame_count
        indices = []
        timestamps = []
        episode_actions = []
        episode_states = []
        episode_tracking = []
        episode_image_missing = 0

        for local_index, frame in enumerate(frames):
            idx = frame.get("idx")
            if isinstance(idx, int):
                indices.append(idx)
            else:
                field_missing["idx"] += 1

            timestamp = nested(frame, "states", "xr", "timestamp_ns")
            if isinstance(timestamp, (int, float)):
                timestamps.append(int(timestamp))
            else:
                field_missing["states.xr.timestamp_ns"] += 1

            state_arm = vector(frame, "states", "arm")
            action_arm = vector(frame, "actions", "arm")
            state_gripper = vector(frame, "states", "gripper")
            action_gripper = vector(frame, "actions", "gripper")
            append_if_shape(values, "state_arm", state_arm, (14,), bad_shapes)
            append_if_shape(values, "action_arm", action_arm, (14,), bad_shapes)
            append_if_shape(values, "state_gripper", state_gripper, (2,), bad_shapes)
            append_if_shape(values, "action_gripper", action_gripper, (2,), bad_shapes)

            if len(state_arm) == 14 and len(state_gripper) == 2:
                state = np.asarray(state_arm + state_gripper, dtype=np.float64)
                episode_states.append(state)
                values["state_16d"].append(state)
            if len(action_arm) == 14 and len(action_gripper) == 2:
                action = np.asarray(action_arm + action_gripper, dtype=np.float64)
                episode_actions.append(action)
                values["action_16d"].append(action)
            if len(state_arm) == len(action_arm) == 14 and len(state_gripper) == len(action_gripper) == 2:
                err = np.asarray(action_arm + action_gripper) - np.asarray(state_arm + state_gripper)
                episode_tracking.append(err)
                all_tracking_errors.append(err)

            scalar_fields = {
                "torso_height": nested(frame, "states", "torso", "height"),
                "torso_velocity_state": nested(frame, "states", "torso", "qvel"),
                "torso_velocity_action": nested(frame, "actions", "torso", "qvel"),
                "waist_pitch_state": nested(frame, "states", "waist", "qpos", default=[None])[0],
                "waist_pitch_action": nested(frame, "actions", "waist", "qpos", default=[None])[0],
                "operator_waist_pitch": nested(frame, "actions", "operator", "waist", "pitch_q"),
                "operator_left_closed_ratio": nested(frame, "actions", "operator", "dex1", "left_closed_ratio"),
                "operator_right_closed_ratio": nested(frame, "actions", "operator", "dex1", "right_closed_ratio"),
            }
            for key, value in scalar_fields.items():
                if isinstance(value, (int, float)):
                    values[key].append(float(value))
                else:
                    field_missing[key] += 1

            append_if_shape(values, "chassis_velocity_state", nested(frame, "states", "chassis", "qvel", default=[]), (2,), bad_shapes)
            append_if_shape(values, "chassis_velocity_action", nested(frame, "actions", "chassis", "qvel", default=[]), (2,), bad_shapes)
            append_if_shape(values, "body_qpos", nested(frame, "states", "body", "qpos", default=[]), (35,), bad_shapes)

            colors = frame.get("colors", {})
            if not isinstance(colors, dict):
                field_missing["colors"] += 1
                continue
            for camera, relative_path in colors.items():
                camera_ref_counts[camera] += 1
                image_path = episode / relative_path
                expected = f"{local_index:06d}_{camera}.jpg"
                if Path(relative_path).name != expected:
                    camera_bad_name[camera] += 1
                if not image_path.is_file():
                    camera_missing[camera] += 1
                    episode_image_missing += 1
                elif Image is not None and decode_stride > 0 and local_index % decode_stride == 0:
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                    except Exception:
                        decode_failures[camera] += 1

        if len(episode_actions) > 1:
            all_action_deltas.extend(np.diff(np.asarray(episode_actions), axis=0))
        if len(episode_states) > 1:
            all_state_deltas.extend(np.diff(np.asarray(episode_states), axis=0))

        dt_ms = np.diff(np.asarray(timestamps, dtype=np.float64)) / 1e6 if len(timestamps) > 1 else np.asarray([])
        positive_dt = dt_ms[dt_ms > 0]
        duration_s = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else None
        tracking = np.asarray(episode_tracking)
        episode_rows.append({
            "episode": episode.name,
            "frames": frame_count,
            "duration_s": duration_s,
            "median_hz": float(1000.0 / np.median(positive_dt)) if positive_dt.size else None,
            "dt_p99_ms": float(np.percentile(positive_dt, 99)) if positive_dt.size else None,
            "nonpositive_timestamps": int((dt_ms <= 0).sum()),
            "idx_contiguous": indices == list(range(frame_count)),
            "missing_images": episode_image_missing,
            "tracking_mae_16d": float(np.mean(np.abs(tracking))) if tracking.size else None,
        })

    action_16d = np.asarray(values["action_16d"])
    state_16d = np.asarray(values["state_16d"])
    chassis_state = np.asarray(values["chassis_velocity_state"])
    chassis_action = np.asarray(values["chassis_velocity_action"])
    waist_state = np.asarray(values["waist_pitch_state"])
    waist_action = np.asarray(values["waist_pitch_action"])
    torso_velocity = np.asarray(values["torso_velocity_action"])

    report = {
        "dataset": str(dataset.resolve()),
        "episode_count": len(episodes),
        "total_frames": total_frames,
        "total_image_refs": int(sum(camera_ref_counts.values())),
        "camera_ref_counts": dict(camera_ref_counts),
        "camera_missing": dict(camera_missing),
        "camera_bad_filename": dict(camera_bad_name),
        "camera_decode_failures_sampled": dict(decode_failures),
        "decode_stride": decode_stride,
        "field_missing": dict(field_missing),
        "bad_shapes": {key: dict(value) for key, value in bad_shapes.items()},
        "episodes": episode_rows,
        "episode_frames": scalar_summary([row["frames"] for row in episode_rows if "frames" in row]),
        "episode_duration_s": scalar_summary([row["duration_s"] for row in episode_rows if row.get("duration_s") is not None]),
        "frequency_hz": scalar_summary([row["median_hz"] for row in episode_rows if row.get("median_hz") is not None]),
        "signals": {
            "state_arm": summarize(values["state_arm"], ARM_NAMES),
            "action_arm": summarize(values["action_arm"], ARM_NAMES),
            "state_gripper": summarize(values["state_gripper"], GRIPPER_NAMES),
            "action_gripper": summarize(values["action_gripper"], GRIPPER_NAMES),
            "torso_height": scalar_summary(values["torso_height"]),
            "torso_velocity_state": scalar_summary(values["torso_velocity_state"]),
            "torso_velocity_action": scalar_summary(values["torso_velocity_action"]),
            "waist_pitch_state": scalar_summary(values["waist_pitch_state"]),
            "waist_pitch_action": scalar_summary(values["waist_pitch_action"]),
            "operator_waist_pitch": scalar_summary(values["operator_waist_pitch"]),
            "operator_closed_ratio": summarize(
                np.column_stack([values["operator_left_closed_ratio"], values["operator_right_closed_ratio"]]),
                ["left_closed_ratio", "right_closed_ratio"],
            ),
            "chassis_velocity_state": summarize(values["chassis_velocity_state"], ["vx", "vyaw"]),
            "chassis_velocity_action": summarize(values["chassis_velocity_action"], ["vx", "vyaw"]),
            "body_qpos": summarize(values["body_qpos"]),
            "tracking_error_action_minus_state_16d": summarize(all_tracking_errors, ARM_NAMES + GRIPPER_NAMES),
            "action_delta_per_frame_16d": summarize(all_action_deltas, ARM_NAMES + GRIPPER_NAMES),
            "state_delta_per_frame_16d": summarize(all_state_deltas, ARM_NAMES + GRIPPER_NAMES),
        },
        "derived": {
            "action_dimension": int(action_16d.shape[1]) if action_16d.ndim == 2 else None,
            "state_dimension": int(state_16d.shape[1]) if state_16d.ndim == 2 else None,
            "chassis_action_stationary_fraction_1e-4": float(np.mean(np.max(np.abs(chassis_action), axis=1) < 1e-4)) if chassis_action.size else None,
            "chassis_state_stationary_fraction_1e-3": float(np.mean(np.max(np.abs(chassis_state), axis=1) < 1e-3)) if chassis_state.size else None,
            "waist_action_stationary_fraction_1e-4": float(np.mean(np.abs(waist_action) < 1e-4)) if waist_action.size else None,
            "waist_state_stationary_fraction_1e-4": float(np.mean(np.abs(waist_state) < 1e-4)) if waist_state.size else None,
            "lift_action_stationary_fraction_1e-4": float(np.mean(np.abs(torso_velocity) < 1e-4)) if torso_velocity.size else None,
        },
    }
    return report


def f(value, digits=4):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown(report):
    derived = report["derived"]
    signals = report["signals"]
    dataset_name = Path(report["dataset"]).name
    lines = [
        f"# G1-D {dataset_name} 数据质检报告",
        "",
        f"- Episodes: {report['episode_count']}",
        f"- 控制帧: {report['total_frames']}",
        f"- 图像引用: {report['total_image_refs']} ({', '.join(f'{k}: {v}' for k, v in sorted(report['camera_ref_counts'].items()))})",
        f"- 每 episode 帧数: {report['episode_frames'].get('min')}–{report['episode_frames'].get('max')}，中位数 {f(report['episode_frames'].get('p50'), 1)}",
        f"- 每 episode 时长: {f(report['episode_duration_s'].get('min'), 1)}–{f(report['episode_duration_s'].get('max'), 1)} 秒，中位数 {f(report['episode_duration_s'].get('p50'), 1)} 秒",
        f"- 采样频率中位数: {f(report['frequency_hz'].get('p50'), 2)} Hz",
        "",
        "## 完整性",
        "",
        f"- 缺失图像: {sum(report['camera_missing'].values())}",
        f"- 图像命名不连续/不匹配: {sum(report['camera_bad_filename'].values())}",
        f"- 抽样 JPEG 解码失败: {sum(report['camera_decode_failures_sampled'].values())}",
        f"- 字段缺失: {sum(report['field_missing'].values())}",
        f"- 数组维度异常: {sum(sum(v.values()) for v in report['bad_shapes'].values())}",
        "",
        "## 动作接口",
        "",
        f"- Policy 动作候选维度: {derived['action_dimension']}D = 14D 双臂关节目标 + 2D Dex1 电机目标",
        f"- 底盘动作静止比例 (|v|<1e-4): {f(derived['chassis_action_stationary_fraction_1e-4'] * 100, 2)}%",
        f"- 腰部动作静止比例 (|q|<1e-4): {f(derived['waist_action_stationary_fraction_1e-4'] * 100, 2)}%",
        f"- 升降动作静止比例 (|vz|<1e-4): {f(derived['lift_action_stationary_fraction_1e-4'] * 100, 2)}%",
        "",
        "## 升降与腰部",
        "",
        f"- 升降高度: min={f(signals['torso_height'].get('min'))}, p01={f(signals['torso_height'].get('p01'))}, mean={f(signals['torso_height'].get('mean'))}, p99={f(signals['torso_height'].get('p99'))}, max={f(signals['torso_height'].get('max'))}",
        f"- 腰部 pitch 状态: min={f(signals['waist_pitch_state'].get('min'))}, mean={f(signals['waist_pitch_state'].get('mean'))}, max={f(signals['waist_pitch_state'].get('max'))}",
        f"- 腰部 pitch 动作: min={f(signals['waist_pitch_action'].get('min'))}, mean={f(signals['waist_pitch_action'].get('mean'))}, max={f(signals['waist_pitch_action'].get('max'))}",
        "",
        "## Episode 明细",
        "",
        "| episode | frames | duration_s | median_hz | dt_p99_ms | missing_images | tracking_mae_16d |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["episodes"]:
        if "error" in row:
            lines.append(f"| {row['episode']} | ERROR: {row['error']} ||||||")
        else:
            lines.append(
                f"| {row['episode']} | {row['frames']} | {f(row['duration_s'], 2)} | {f(row['median_hz'], 2)} | "
                f"{f(row['dt_p99_ms'], 2)} | {row['missing_images']} | {f(row['tracking_mae_16d'], 5)} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--decode-stride", type=int, default=200)
    args = parser.parse_args()
    report = audit(args.dataset, args.decode_stride)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.markdown_output.write_text(markdown(report))
    print(json.dumps({
        "episodes": report["episode_count"],
        "frames": report["total_frames"],
        "images": report["total_image_refs"],
        "action_dim": report["derived"]["action_dimension"],
        "json": str(args.json_output),
        "markdown": str(args.markdown_output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
