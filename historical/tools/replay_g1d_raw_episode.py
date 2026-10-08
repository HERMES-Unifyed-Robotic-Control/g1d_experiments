#!/usr/bin/env python3
"""Replay recorded G1-D raw episode actions on the robot controller."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
import sys
import time
import types
from multiprocessing import Array, Lock, Value
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "food_sorting"
TELEOP_SEARCH_ROOTS = [
    ROOT,
    Path.cwd(),
    Path.home() / "xr_teleoperate_g1d",
    Path.home() / "unitree_eai_environment" / "xr_teleoperate_g1d",
    Path.home() / "unitree_ear_environment" / "xr_teleoperate_g1d",
    Path("/mnt/data/yangky/test/xr_teleoperate_g1d"),
]
for candidate in TELEOP_SEARCH_ROOTS:
    if (candidate / "teleop").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    teleop_root = candidate / "teleop"
    if teleop_root.exists() and str(teleop_root) not in sys.path:
        sys.path.insert(0, str(teleop_root))

try:
    import logging_mp  # noqa: F401
except ModuleNotFoundError:
    logging_mp = types.ModuleType("logging_mp")
    for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        setattr(logging_mp, name, getattr(logging, name))
    logging_mp.getLogger = logging.getLogger
    logging_mp.basicConfig = logging.basicConfig
    sys.modules["logging_mp"] = logging_mp


def gripper_q_to_trigger_value(q: float) -> float:
    q = float(np.clip(q, 0.0, 5.4))
    return 5.0 + (q / 5.4) * 2.0


def validate_network_interface(name: str) -> None:
    interfaces = {iface[1] for iface in socket.if_nameindex()}
    if name not in interfaces:
        raise RuntimeError(f"network interface {name!r} not found; available={sorted(interfaces)}")
    operstate = Path("/sys/class/net") / name / "operstate"
    if operstate.exists() and operstate.read_text().strip() != "up":
        raise RuntimeError(f"network interface {name!r} is not UP")


def resolve_episode_path(name_or_path: str, raw_root: Path) -> Path:
    value = Path(name_or_path).expanduser()
    candidates = []
    if value.is_absolute():
        candidates.append(value)
    else:
        candidates.extend([raw_root / value, ROOT / value, Path.cwd() / value])
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "data.json":
            return candidate.parent
        if (candidate / "data.json").exists():
            return candidate
    raise FileNotFoundError(f"Cannot find episode data.json from {name_or_path!r}")


def action_from_frame(frame: dict) -> np.ndarray:
    actions = frame.get("actions") or frame.get("states")
    left_arm = actions["left_arm"]["qpos"]
    right_arm = actions["right_arm"]["qpos"]
    left_gripper = actions["left_ee"]["qpos"][0]
    right_gripper = actions["right_ee"]["qpos"][0]
    return np.asarray([*left_arm, *right_arm, left_gripper, right_gripper], dtype=np.float64)


def load_actions(episode_dir: Path) -> tuple[np.ndarray, float]:
    payload = json.loads((episode_dir / "data.json").read_text())
    fps = float(payload.get("info", {}).get("image", {}).get("fps", 30.0))
    frames = payload["data"]
    actions = np.stack([action_from_frame(frame) for frame in frames], axis=0)
    return actions, fps


def interpolate_move(controller, left_gripper_value, right_gripper_value, target: np.ndarray, seconds: float, fps: float) -> None:
    current = np.concatenate([controller.get_current_dual_arm_q(), controller.get_current_dual_gripper_q()])
    steps = max(1, int(seconds * fps))
    tau = np.zeros(14, dtype=np.float64)
    for idx in range(steps):
        alpha = (idx + 1) / steps
        cmd = current * (1.0 - alpha) + target * alpha
        controller.ctrl_dual_arm(cmd[:14], tau)
        with left_gripper_value.get_lock():
            left_gripper_value.value = gripper_q_to_trigger_value(cmd[14])
        with right_gripper_value.get_lock():
            right_gripper_value.value = gripper_q_to_trigger_value(cmd[15])
        time.sleep(1.0 / fps)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", help="Episode path/name, e.g. food_sorting_1/episode_0000 or /abs/path/episode_0000")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--fps", type=float, default=None, help="Override recorded fps.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--arm-velocity-limit", type=float, default=30.0)
    parser.add_argument("--move-to-first-seconds", type=float, default=3.0)
    parser.add_argument("--hold-final-seconds", type=float, default=2.0)
    parser.add_argument("--record-csv", type=Path, default=None)
    parser.add_argument("--no-go-home-at-end", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually send commands to G1-D.")
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    episode_dir = resolve_episode_path(args.episode, args.raw_root)
    actions, recorded_fps = load_actions(episode_dir)
    fps = float(args.fps or recorded_fps)
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    start = max(0, args.start)
    end = len(actions) if args.end is None else min(len(actions), args.end)
    actions = actions[start:end:max(1, args.stride)]
    if len(actions) == 0:
        raise RuntimeError("No actions selected for replay")

    print(f"[replay] episode={episode_dir}")
    print(f"[replay] selected_steps={len(actions)} recorded_fps={recorded_fps:.2f} replay_fps={fps:.2f} speed={args.speed:.2f}")
    print(f"[replay] first={np.array2string(actions[0], precision=3)}")
    print(f"[replay] last ={np.array2string(actions[-1], precision=3)}")
    if not args.execute:
        print("[dry-run] add --execute to control G1-D")
        return

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.robot_arm import G1_29_Arm_Internal_Dex1_Controller

    validate_network_interface(args.network_interface)
    left_gripper_value = Value("d", 5.0, lock=True)
    right_gripper_value = Value("d", 5.0, lock=True)
    motion_ready = Value("b", True, lock=True)
    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    controller = G1_29_Arm_Internal_Dex1_Controller(
        left_gripper_value,
        right_gripper_value,
        Lock(),
        Array("d", 2, lock=False),
        Array("d", 2, lock=False),
        simulation_mode=False,
        use_waist=False,
        xr_motion_data_ready_in=motion_ready,
    )
    controller.arm_velocity_limit = args.arm_velocity_limit
    tau = np.zeros(14, dtype=np.float64)
    csv_file = None
    writer = None
    if args.record_csv:
        args.record_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.record_csv.open("w", newline="")
        fields = ["step", "time_s"]
        for prefix in ("demo_action", "robot_state"):
            fields.extend(f"{prefix}_{i:02d}" for i in range(16))
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
    try:
        interpolate_move(controller, left_gripper_value, right_gripper_value, actions[0], args.move_to_first_seconds, fps)
        replay_dt = 1.0 / (fps * args.speed)
        start_time = time.time()
        for step, action in enumerate(actions):
            controller.ctrl_dual_arm(action[:14], tau)
            with left_gripper_value.get_lock():
                left_gripper_value.value = gripper_q_to_trigger_value(action[14])
            with right_gripper_value.get_lock():
                right_gripper_value.value = gripper_q_to_trigger_value(action[15])
            robot_state = np.concatenate([controller.get_current_dual_arm_q(), controller.get_current_dual_gripper_q()])
            if writer is not None:
                row = {"step": step, "time_s": time.time() - start_time}
                for i, value in enumerate(action):
                    row[f"demo_action_{i:02d}"] = float(value)
                for i, value in enumerate(robot_state):
                    row[f"robot_state_{i:02d}"] = float(value)
                writer.writerow(row)
            if step % max(1, int(fps)) == 0:
                err = np.max(np.abs(robot_state[:14] - action[:14]))
                print(f"[replay] step={step}/{len(actions)} max_arm_err={err:.4f} gripper={robot_state[14:16]}")
            time.sleep(replay_dt)
        time.sleep(max(0.0, args.hold_final_seconds))
    except KeyboardInterrupt:
        print("[replay] interrupted")
    finally:
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
            print(f"[record] wrote {args.record_csv}")
        if not args.no_go_home_at_end:
            print("[robot] returning arms/grippers to home")
            try:
                controller.ctrl_dual_arm_go_home()
                with left_gripper_value.get_lock():
                    left_gripper_value.value = gripper_q_to_trigger_value(2.7)
                with right_gripper_value.get_lock():
                    right_gripper_value.value = gripper_q_to_trigger_value(2.7)
                time.sleep(3.0)
            except Exception as exc:
                print(f"[robot] shutdown warning: {exc}")
        controller.running = False


if __name__ == "__main__":
    main()
