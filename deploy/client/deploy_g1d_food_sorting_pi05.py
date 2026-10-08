#!/usr/bin/env python3
"""Deploy the G1-D food-sorting pi0.5 checkpoint with a conservative 16D action bridge."""

from __future__ import annotations

import argparse
import csv
import logging
import json
import hashlib
import os
import signal
import shutil
import socket
import select
import subprocess
import sys
import threading
import termios
import tty
import time
import types
from dataclasses import dataclass
from multiprocessing import Array, Lock, Value
from pathlib import Path

import cv2
import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy, image_tools


ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = ROOT / "teleop"
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
    if candidate.name == "teleop" and candidate.exists() and str(candidate.parent) not in sys.path:
        sys.path.insert(0, str(candidate.parent))
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

try:
    from teleop.teleimager.src.teleimager.image_client import ImageClient  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import teleop. Run this script from the xr_teleoperate_g1d repo, "
        "or set XR_TELEOP_ROOT/PYTHONPATH to the repo containing the teleop/ directory."
    ) from exc


TASKS = {
    "chrysanthemum": "Move the black cup from the right to the middle, place chrysanthemum into it, then pour water from the white cup from the left into the black cup.",
    "red_dates": "Move the black cup from the right to the middle, place red dates into it, then pour water from the white cup from the left into the black cup.",
    "lemon_slices": "Move the black cup from the right to the middle, place lemon slices into it, then pour water from the white cup from the left into the black cup.",
}

INGREDIENTS = {
    "chrysanthemum": "chrysanthemum",
    "red_dates": "red dates",
    "lemon_slices": "lemon slices",
}

SUBTASK_FRACTIONS = (0.220, 0.355, 0.134, 0.089, 0.202)
SUBTASK_NAMES = (
    "move_black_cup",
    "place_ingredient",
    "left_hand_get_white_cup",
    "right_hand_hold_black_cup",
    "pour_water",
)

READY_ARM_Q = np.array(
    [
        0.31652737,
        -0.00931174,
        -0.03947604,
        0.37479466,
        0.06829810,
        -0.80679005,
        0.01634648,
        0.27863324,
        -0.01936651,
        0.07125820,
        0.27490613,
        0.08434498,
        -0.64983261,
        0.21690249,
    ],
    dtype=np.float64,
)
READY_GRIPPER_Q = np.array([5.32469273, 5.23435688], dtype=np.float64)
READY_TORSO_HEIGHT_M = 0.065
DEFAULT_HEIGHT_CONTROL_BINARY = "/home/unitree/unitree_sdk2/build/bin/g1d_height_control"
DEFAULT_HEIGHT_STOP_HELPER = "/usr/local/sbin/g1d_height_control_stop"
DEFAULT_WAIST_YAW_TARGET = 0.0
from robot_alignment import GravityCompensation, GravityAlignedController, RobotTelemetry, DEFAULT_URDF


def rgb224(frame_bgr: np.ndarray) -> np.ndarray:
    """Match OpenPI training preprocessing: resize without aspect distortion."""
    return image_tools.resize_with_pad(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), 224, 224)


def get_images(client: ImageClient) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = client.get_cam_config()
    head_bgr = left_bgr = right_bgr = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        head = client.get_head_frame()
        left = client.get_left_wrist_frame()
        right = client.get_right_wrist_frame()
        head_bgr = None if head is None else head.bgr
        left_bgr = None if left is None else left.bgr
        right_bgr = None if right is None else right.bgr
        if head_bgr is not None and left_bgr is not None and right_bgr is not None:
            break
        time.sleep(0.02)
    if head_bgr is None or left_bgr is None or right_bgr is None:
        raise RuntimeError("camera frame missing")
    if cfg["head_camera"].get("binocular", False):
        image_shape = cfg["head_camera"].get("image_shape") or [head_bgr.shape[0], head_bgr.shape[1]]
        head_bgr = head_bgr[:, : int(image_shape[1]) // 2]
    return rgb224(head_bgr), rgb224(left_bgr), rgb224(right_bgr)


def gripper_q_to_trigger_value(q: float) -> float:
    q = float(np.clip(q, 0.0, 5.4))
    return 5.0 + (q / 5.4) * 2.0


def clip_step(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    return current + np.clip(target - current, -max_delta, max_delta)


class ActionSmoother:
    """Keep absolute-qpos chunks continuous at inference boundaries."""

    def __init__(self, alpha: float = 0.35, max_dq: float = 0.08, max_gripper_dq: float = 0.30):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_dq = max(0.0, float(max_dq))
        self.max_gripper_dq = max(0.0, float(max_gripper_dq))
        self._last_arm = None
        self._last_gripper = None

    def apply(self, action: np.ndarray, current_arm: np.ndarray, current_gripper: np.ndarray) -> np.ndarray:
        arm_target = np.asarray(action[:14], dtype=np.float64)
        grip_target = np.asarray(action[14:16], dtype=np.float64)
        if self._last_arm is None:
            self._last_arm = np.asarray(current_arm, dtype=np.float64).copy()
        if self._last_gripper is None:
            self._last_gripper = np.asarray(current_gripper, dtype=np.float64).copy()
        arm_blended = self.alpha * arm_target + (1.0 - self.alpha) * self._last_arm
        grip_blended = self.alpha * grip_target + (1.0 - self.alpha) * self._last_gripper
        self._last_arm += np.clip(arm_blended - self._last_arm, -self.max_dq, self.max_dq)
        self._last_gripper += np.clip(
            grip_blended - self._last_gripper,
            -self.max_gripper_dq,
            self.max_gripper_dq,
        )
        return np.concatenate([self._last_arm, self._last_gripper])


def blend_chunk_boundary(
    actions: np.ndarray,
    anchor: np.ndarray | None,
    blend_steps: int,
) -> np.ndarray:
    """Join a newly inferred absolute-qpos chunk to the previous command.

    The policy emits absolute joint positions.  A new observation can therefore
    produce a first waypoint that is several steps away from the last waypoint
    still being executed.  Blend only the first few waypoints; the remainder of
    the policy trajectory is left untouched.
    """
    result = np.asarray(actions, dtype=np.float64).copy()
    if anchor is None or blend_steps <= 0 or result.size == 0:
        return result
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor.size != result.shape[1]:
        return result
    count = min(int(blend_steps), result.shape[0])
    for index in range(count):
        # Keep a small amount of the old command at the boundary, then hand
        # control back to the policy over the following waypoints.
        alpha = float(index + 1) / float(count + 1)
        result[index] = (1.0 - alpha) * anchor + alpha * result[index]
    return result


def move_to_ready_pose(
    controller,
    left_gripper_value,
    right_gripper_value,
    hold_seconds: float,
) -> None:
    print(
        "[robot] moving to dataset ready pose; set physical torso/lift height to "
        f"~{READY_TORSO_HEIGHT_M:.3f}m before rollout"
    )
    controller.ctrl_dual_arm(READY_ARM_Q, np.zeros(14, dtype=np.float64))
    left_trigger = gripper_q_to_trigger_value(READY_GRIPPER_Q[0])
    right_trigger = gripper_q_to_trigger_value(READY_GRIPPER_Q[1])
    end_time = time.time() + hold_seconds
    while time.time() < end_time:
        with left_gripper_value.get_lock():
            left_gripper_value.value = left_trigger
        with right_gripper_value.get_lock():
            right_gripper_value.value = right_trigger
        time.sleep(0.05)


class PolicyClient:
    def __init__(self, host: str, port: int):
        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            ping_interval=None,
            close_timeout=300,
        )
        self.metadata = msgpack_numpy.unpackb(self._ws.recv())

    def infer(self, obs: dict, timeout: float = 300.0) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv(timeout=timeout)
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


@dataclass
class PolicyChunk:
    actions: np.ndarray
    infer_ms: float
    obs_state: np.ndarray
    images: tuple[np.ndarray, np.ndarray, np.ndarray]
    prompt: str
    submitted_at: float
    ready_at: float
    observed_at: float
    observed_step: int | None


@dataclass
class PromptSchedule:
    mode: str
    task: str
    total_steps: int
    fractions: tuple[float, ...]
    whole_prompt: str
    schedule: str = "fixed"
    fixed_index: int = 0
    _last_index: int | None = None

    def switch_to_next(self, step: int) -> tuple[str, str, int]:
        """Advance one fixed subtask at a chunk boundary and log it."""
        self.fixed_index = min(self.fixed_index + 1, len(SUBTASK_NAMES) - 1)
        prompt, name, index = self.prompt_for_step(step)
        self._last_index = None
        self.maybe_log_switch(step, prompt, name, index)
        return prompt, name, index

    def prompt_for_step(self, step: int) -> tuple[str, str, int]:
        if self.mode != "subtask":
            return self.whole_prompt, "whole_task", -1
        if self.schedule == "fixed":
            return self._subtask_prompt(self.fixed_index), SUBTASK_NAMES[self.fixed_index], self.fixed_index
        clamped_step = max(0, min(int(step), max(0, self.total_steps - 1)))
        edges = np.cumsum(np.asarray(self.fractions, dtype=np.float64))
        if edges[-1] <= 0:
            edges = np.cumsum(np.asarray(SUBTASK_FRACTIONS, dtype=np.float64))
        edges = edges / edges[-1]
        frac = (clamped_step + 0.5) / max(1, self.total_steps)
        index = int(np.searchsorted(edges, frac, side="right"))
        index = max(0, min(index, len(SUBTASK_NAMES) - 1))
        return self._subtask_prompt(index), SUBTASK_NAMES[index], index

    def maybe_log_switch(self, step: int, prompt: str, name: str, index: int) -> None:
        if self.mode != "subtask" or index == self._last_index:
            return
        self._last_index = index
        print(f"[prompt] step={step} subtask={index}:{name} prompt={prompt}")

    def _subtask_prompt(self, index: int) -> str:
        ingredient = INGREDIENTS[self.task]
        prompts = (
            "Move the black cup from the right side to the middle position.",
            f"Place {ingredient} into the black cup.",
            "Use the left hand to pick up the white cup on the left.",
            "Use the right hand to pick up and steady the black cup for pouring.",
            "Pour water from the white cup into the black cup, then return the robot to its home pose.",
        )
        return prompts[index]


class NextSubtaskKey:
    """Read a raw terminal N key without blocking the 30 Hz control loop."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._count = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="g1d-next-subtask", daemon=True)
        self._thread.start()

    def pop(self) -> bool:
        with self._lock:
            if self._count <= 0:
                return False
            self._count -= 1
            return True

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (OSError, termios.error):
            return
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    key = sys.stdin.read(1).lower()
                    if key == "n":
                        with self._lock:
                            self._count += 1
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except (OSError, termios.error):
                pass


class AsyncPolicyPrefetcher:
    """Single-worker policy prefetcher.

    The websocket client is used only from the worker while async mode is active.
    Main thread executes the latest ready queue at 30Hz and launches the next
    request before the current queue is exhausted.
    """

    def __init__(self, policy: PolicyClient):
        self.policy = policy
        self._lock = threading.Lock()
        self._thread = None
        self._result = None
        self._error = None
        self._generation = 0
        self._closed = False

    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None

    def start(self, obs_or_factory, images=None, prompt: str = "", observation_step_factory=None) -> bool:
        with self._lock:
            if self._closed or self._thread is not None:
                return False
            generation = self._generation
            self._result = None
            self._error = None

        submitted_at = time.perf_counter()

        def worker() -> None:
            try:
                # Camera acquisition can block while waiting for a fresh frame.
                # Keep it off the 30 Hz execution thread as well as the policy
                # websocket call; otherwise the queue can still run empty even
                # when server inference itself is only ~85 ms.
                if callable(obs_or_factory):
                    obs, captured_images = obs_or_factory()
                else:
                    obs, captured_images = obs_or_factory, images
                observed_at = time.perf_counter()
                observed_step = int(observation_step_factory()) if observation_step_factory is not None else None
                obs_state = np.asarray(obs["state"], dtype=np.float64).copy()
                result = self.policy.infer(obs)
                actions = np.asarray(result["actions"], dtype=np.float64)
                if actions.shape != (32, 16) or not np.isfinite(actions).all():
                    raise RuntimeError(f"unexpected policy action shape: {actions.shape}")
                infer_ms = float(result.get("server_timing", {}).get("infer_ms", -1.0))
                chunk = PolicyChunk(
                    actions=actions,
                    infer_ms=infer_ms,
                    obs_state=obs_state,
                    images=captured_images,
                    prompt=prompt,
                    submitted_at=submitted_at,
                    ready_at=time.perf_counter(),
                    observed_at=observed_at,
                    observed_step=observed_step,
                )
                with self._lock:
                    if generation == self._generation and not self._closed:
                        self._result = chunk
                        self._error = None
            except Exception as exc:
                with self._lock:
                    if generation == self._generation and not self._closed:
                        self._result = None
                        self._error = exc
            finally:
                with self._lock:
                    if self._thread is threading.current_thread():
                        self._thread = None

        thread = threading.Thread(target=worker, daemon=True)
        with self._lock:
            if self._closed or self._thread is not None:
                return False
            self._thread = thread
            thread.start()
        return True

    def pop_result(self) -> PolicyChunk | None:
        with self._lock:
            result = self._result
            self._result = None
            return result

    def pop_error(self) -> Exception | None:
        with self._lock:
            error = self._error
            self._error = None
            return error

    def invalidate(self) -> None:
        with self._lock:
            self._generation += 1
            self._result = None
            self._error = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)


class RolloutRecorder:
    def __init__(self, record_dir: str | None, fps: float):
        self.enabled = bool(record_dir)
        self.dir = Path(record_dir).expanduser() if record_dir else None
        self.fps = fps
        self.csv_file = None
        self.writer = None
        self.video_writers = {}
        self.rows = []
        self.telemetry = None
        self.gravity = None
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / "rollout_actions.csv").exists():
            raise FileExistsError(f"Use a new record directory: {self.dir}")
        self.csv_file = open(self.dir / "rollout_actions.csv", "w", newline="")
        fields = ["step", "chunk_index", "chunk_step", "time_s", "infer_ms"]
        for prefix in ("obs_state", "policy_action", "cmd_action", "robot_state_before", "robot_state_after"):
            fields.extend(f"{prefix}_{i:02d}" for i in range(16))
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.writer.writeheader()

    def record_event(self, event: dict) -> None:
        if self.enabled:
            with (self.dir / "events.jsonl").open("a") as f:
                f.write(json.dumps({"wall_time": time.time(), **event}) + "\n")

    def _video_writer(self, name: str, frame: np.ndarray):
        if name not in self.video_writers:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writers[name] = cv2.VideoWriter(str(self.dir / f"{name}.mp4"), fourcc, max(1.0, self.fps / 32.0), (w, h))
        return self.video_writers[name]

    def record_images(self, chunk_index: int, base_rgb: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray) -> None:
        if not self.enabled:
            return
        frames = {
            "input_base_224": base_rgb,
            "input_left_wrist_224": left_rgb,
            "input_right_wrist_224": right_rgb,
            "input_contact_sheet_224": np.concatenate([base_rgb, left_rgb, right_rgb], axis=1),
        }
        for name, frame_rgb in frames.items():
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"chunk {chunk_index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            self._video_writer(name, frame_bgr).write(frame_bgr)
            if chunk_index == 0:
                cv2.imwrite(str(self.dir / f"{name}_first.jpg"), frame_bgr)

    def record_policy_chunk(self, chunk_index, actions, obs_state, prompt):
        if self.enabled:
            directory = self.dir / "raw_policy_chunks"
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory / f"chunk_{chunk_index:06d}.npz",
                                actions=np.asarray(actions), state=np.asarray(obs_state), prompt=np.array(prompt))

    def record_action(
        self,
        step: int,
        chunk_index: int,
        chunk_step: int,
        time_s: float,
        infer_ms: float,
        obs_state: np.ndarray,
        policy_action: np.ndarray,
        cmd_action: np.ndarray,
        robot_state_before: np.ndarray,
        robot_state_after: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        row = {
            "step": step,
            "chunk_index": chunk_index,
            "chunk_step": chunk_step,
            "time_s": time_s,
            "infer_ms": infer_ms,
        }
        arrays = {
            "obs_state": obs_state,
            "policy_action": policy_action,
            "cmd_action": cmd_action,
            "robot_state_before": robot_state_before,
            "robot_state_after": robot_state_after,
        }
        for prefix, arr in arrays.items():
            arr = np.asarray(arr, dtype=float).reshape(-1)
            for i in range(16):
                row[f"{prefix}_{i:02d}"] = float(arr[i]) if i < arr.size else np.nan
        self.writer.writerow(row)
        self.rows.append(row)
        if self.telemetry is not None:
            state = self.telemetry.snapshot()
            state.pop("arm_q", None)
            self.record_event({"type": "physical_state", "step": step, "time_s": time_s, **state,
                               "gravity_tau_nm": self.gravity.last_tau.tolist() if self.gravity is not None else [0.0]*14})

    def close(self) -> None:
        if not self.enabled:
            return
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
        for writer in self.video_writers.values():
            writer.release()
        if shutil.which("ffmpeg"):
            h264_dir = self.dir / "h264"
            h264_dir.mkdir(exist_ok=True)
            for mp4 in self.dir.glob("input_*_224.mp4"):
                out = h264_dir / f"{mp4.stem}_h264.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(mp4),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(out),
                    ],
                    check=False,
                )
        if self.rows:
            np.savez_compressed(self.dir / "rollout_actions.npz", rows=np.array(self.rows, dtype=object))
        print(f"[record] saved rollout logs to {self.dir}")


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=9014)
    parser.add_argument("--img-server-ip", default="192.168.10.104")
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--task", choices=sorted(TASKS), default="chrysanthemum")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-mode", choices=("whole", "subtask"), default="subtask")
    parser.add_argument("--subtask-schedule", choices=("fixed", "timed"), default="fixed")
    parser.add_argument("--subtask-index", type=int, choices=range(5), default=0)
    parser.add_argument("--profile", choices=("legacy", "official"), default="legacy")
    parser.add_argument("--profiles-file", type=Path, default=Path(__file__).with_name("g1d_profiles_baseline.json"))
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument("--gravity-urdf", default=DEFAULT_URDF)
    parser.add_argument("--height-tolerance-m", type=float, default=.003)
    parser.add_argument("--check-server-only", action="store_true")
    parser.add_argument("--disable-action-smoothing", action="store_true")
    parser.add_argument("--subtask-fractions", default=",".join(str(x) for x in SUBTASK_FRACTIONS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--execute-chunk-steps", type=int, default=8)
    parser.add_argument("--async-prefetch", action="store_true")
    parser.add_argument(
        "--prefetch-remaining-steps",
        type=int,
        default=8,
        help="Start the next camera+inference request when this many actions remain.",
    )
    parser.add_argument(
        "--interpolation-substeps",
        type=int,
        default=4,
        help="Linear q-target updates per 30 Hz policy step; 1 disables client-side interpolation.",
    )
    parser.add_argument(
        "--chunk-blend-steps",
        type=int,
        default=2,
        help="Number of waypoints blended from the previous chunk at each chunk boundary.",
    )
    parser.add_argument("--max-arm-delta", type=float, default=0.30)
    parser.add_argument("--max-gripper-delta", type=float, default=1.00)
    parser.add_argument("--arm-velocity-limit", type=float, default=30.0)
    parser.add_argument("--action-smooth-alpha", type=float, default=0.35)
    parser.add_argument("--action-max-dq", type=float, default=0.08)
    parser.add_argument("--go-home-before-start", action="store_true")
    parser.add_argument("--go-ready-before-start", action="store_true")
    parser.add_argument("--home-wait", type=float, default=1.0)
    parser.add_argument("--ready-wait", type=float, default=3.0)
    parser.add_argument(
        "--fixed-height-m",
        type=float,
        default=None,
        help="Run g1d_height_control at this torso height during real execution.",
    )
    parser.add_argument(
        "--height-control-binary",
        default=DEFAULT_HEIGHT_CONTROL_BINARY,
        help="Path to the continuous G1-D height controller on the robot.",
    )
    parser.add_argument(
        "--disable-height-lock",
        action="store_true",
        help="Do not start the physical torso height controller (debug only).",
    )
    parser.add_argument(
        "--waist-yaw-target",
        type=float,
        default=None,
        help="Fixed waist-yaw target in radians; official profile defaults to home (0).",
    )
    parser.add_argument(
        "--disable-waist-yaw-lock",
        action="store_true",
        help="Do not lock waist yaw; incompatible with the official fixed-camera profile.",
    )
    parser.add_argument(
        "--monitor-waist-yaw",
        action="store_true",
        help="Launch the read-only waist-yaw state/command monitor alongside execution.",
    )
    parser.add_argument("--initial-state", choices=("current", "home", "ready"), default="current")
    parser.add_argument("--gripper-open-only", action="store_true")
    parser.add_argument("--gripper-open-q", type=float, default=5.4)
    parser.add_argument("--hold-seconds", type=float, default=30.0)
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--execute", action="store_true", help="Actually send commands to G1-D.")
    return parser


def parse_subtask_fractions(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != len(SUBTASK_NAMES) or any(value <= 0 for value in values):
        raise ValueError(
            f"--subtask-fractions must contain {len(SUBTASK_NAMES)} positive comma-separated values; got {text!r}"
        )
    return values


def validate_network_interface(name: str) -> None:
    interfaces = {iface[1] for iface in socket.if_nameindex()}
    if name not in interfaces:
        raise RuntimeError(f"network interface {name!r} not found; available={sorted(interfaces)}")
    operstate = Path("/sys/class/net") / name / "operstate"
    if operstate.exists() and operstate.read_text().strip() != "up":
        raise RuntimeError(f"network interface {name!r} is not UP")


class FixedHeightLock:
    """Run the SDK height controller for the lifetime of a real rollout.

    The G1-D utility is a continuous PD controller.  The robot is configured
    with a narrowly scoped NOPASSWD sudoers rule for this exact command; no
    password is stored in this project or passed through the client.
    """

    def __init__(self, binary: str, network_interface: str, target_height_m: float):
        self.binary = Path(binary).expanduser()
        self.network_interface = network_interface
        self.target_height_m = float(target_height_m)
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if not self.binary.is_file():
            raise FileNotFoundError(f"height controller not found: {self.binary}")
        if not 0.0 <= self.target_height_m <= 0.20:
            raise ValueError(f"target height must be in [0, 0.20]m, got {self.target_height_m}")
        command = [str(self.binary), self.network_interface, f"{self.target_height_m:.3f}"]
        print(
            "[robot] requesting sudo once, then locking torso height at "
            f"{self.target_height_m:.3f}m: {' '.join(command)}",
            flush=True,
        )
        self.process = subprocess.Popen(
            ["sudo", "-n", *command],
            stdin=None,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )
        time.sleep(0.5)
        if self.process.poll() is not None:
            code = self.process.returncode
            self.process = None
            raise RuntimeError(
                "height controller exited immediately; verify the exact NOPASSWD "
                f"sudoers rule for {' '.join(command)} (exit={code})"
            )
        print(f"[robot] torso height lock active (pid={self.process.pid})", flush=True)

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except ProcessLookupError:
            pass
        except PermissionError:
            # sudo owns the height process, so a unitree process cannot signal
            # its process group directly.  Use the narrowly scoped root helper
            # when available; it only matches the exact official command.
            subprocess.run(["sudo", "-n", DEFAULT_HEIGHT_STOP_HELPER], check=False)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                print("[robot] warning: height controller did not stop after helper", flush=True)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)
        print("[robot] torso height lock stopped", flush=True)


def wait_for_waist_yaw(controller, target: float, timeout: float = 3.0, tolerance: float = 0.03) -> None:
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        current = float(controller.get_current_waist_yaw_q())
        if abs(current - target) <= tolerance:
            print(f"[robot] waist yaw locked at target={target:.3f} current={current:.3f}", flush=True)
            return
        time.sleep(0.02)
    current = float(controller.get_current_waist_yaw_q())
    print(
        f"[robot] warning: waist yaw did not reach target={target:.3f} "
        f"within {timeout:.1f}s (current={current:.3f})",
        flush=True,
    )


def build_observation(
    image_client: ImageClient,
    controller,
    initial_state: str,
    prompt: str,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    base, left, right = get_images(image_client)
    if controller is not None:
        if initial_state == "home":
            arm_state = np.zeros(14, dtype=np.float32)
        elif initial_state == "ready":
            arm_state = READY_ARM_Q.astype(np.float32)
        else:
            arm_state = controller.get_current_dual_arm_q().astype(np.float32)
        if initial_state == "ready":
            gripper_state = READY_GRIPPER_Q.astype(np.float32)
        else:
            gripper_state = controller.get_current_dual_gripper_q().astype(np.float32)
    else:
        arm_state = np.zeros(14, dtype=np.float32)
        gripper_state = np.array([2.7, 2.7], dtype=np.float32)
    obs = {
        "observation/base_image": base,
        "observation/left_wrist_image": left,
        "observation/right_wrist_image": right,
        "state": np.concatenate([arm_state, gripper_state]).astype(np.float32),
        "prompt": prompt,
    }
    return obs, (base, left, right)


def execute_policy_action(
    action: np.ndarray,
    controller,
    left_gripper_value,
    right_gripper_value,
    max_arm_delta: float,
    max_gripper_delta: float,
    obs_state: np.ndarray,
    action_smoother: ActionSmoother | None = None,
    interpolation_substeps: int = 1,
    control_period: float = 1.0 / 30.0,
    interpolation_start_action: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot_state_before = obs_state.copy()
    cmd_action = action.copy()
    if controller is not None:
        current_arm = controller.get_current_dual_arm_q()
        current_gripper = controller.get_current_dual_gripper_q()
        robot_state_before = np.concatenate([current_arm, current_gripper]).astype(np.float64)
        target_action = action
        if action_smoother is not None:
            target_action = action_smoother.apply(action, current_arm, current_gripper)
        arm_cmd = clip_step(target_action[:14], current_arm, max_arm_delta)
        gripper_cmd = clip_step(np.clip(target_action[14:16], 0.0, 5.4), current_gripper, max_gripper_delta)
        cmd_action = np.concatenate([arm_cmd, gripper_cmd]).astype(np.float64)

        # The policy is sampled at 30 Hz, while the arm controller publishes at
        # 250 Hz.  Updating q_target only once per policy waypoint creates a
        # visible staircase.  Send a short linear trajectory to the controller
        # during this policy period; the low-level 250 Hz loop then sees a
        # continuous target even when the policy chunk itself has a kink.
        substeps = max(1, int(interpolation_substeps))
        period = max(0.0, float(control_period))
        # Continue from the last SENT endpoint. Restarting from measured q each
        # policy period pulls the target backwards whenever the plant lags.
        # Measurements still bound both the anchor and endpoint, preserving the
        # existing maximum target-to-measurement delta safeguards.
        if interpolation_start_action is None:
            arm_start = np.asarray(current_arm, dtype=np.float64)
            gripper_start = np.asarray(current_gripper, dtype=np.float64)
        else:
            anchor = np.asarray(interpolation_start_action, dtype=np.float64)
            if anchor.shape != (16,) or not np.isfinite(anchor).all():
                raise ValueError("interpolation anchor must be finite 16D qpos")
            arm_start = clip_step(anchor[:14], current_arm, max_arm_delta)
            gripper_start = clip_step(np.clip(anchor[14:16], 0.0, 5.4), current_gripper, max_gripper_delta)
        deadline = time.perf_counter()
        for substep in range(substeps):
            fraction = float(substep + 1) / float(substeps)
            arm_waypoint = arm_start + fraction * (arm_cmd - arm_start)
            gripper_waypoint = gripper_start + fraction * (gripper_cmd - gripper_start)
            controller.ctrl_dual_arm(arm_waypoint, np.zeros(14, dtype=np.float64))
            with left_gripper_value.get_lock():
                left_gripper_value.value = gripper_q_to_trigger_value(gripper_waypoint[0])
            with right_gripper_value.get_lock():
                right_gripper_value.value = gripper_q_to_trigger_value(gripper_waypoint[1])
            if substep + 1 < substeps:
                deadline += period / float(substeps)
                time.sleep(max(0.0, deadline - time.perf_counter()))
        robot_state_after = np.concatenate(
            [controller.get_current_dual_arm_q(), controller.get_current_dual_gripper_q()]
        ).astype(np.float64)
    else:
        robot_state_after = cmd_action.copy()
    return cmd_action, robot_state_before, robot_state_after


def run_sync_rollout(
    args,
    prompt_schedule: PromptSchedule,
    policy: PolicyClient,
    image_client: ImageClient,
    controller,
    left_gripper_value,
    right_gripper_value,
    recorder: RolloutRecorder,
    next_key: NextSubtaskKey | None = None,
) -> None:
    step = 0
    chunk_index = 0
    rollout_start = time.time()
    action_smoother = None if args.disable_action_smoothing else ActionSmoother(args.action_smooth_alpha, args.action_max_dq)
    last_command_action: np.ndarray | None = None
    while step < args.steps:
        if next_key is not None and next_key.pop():
            if prompt_schedule.schedule == "fixed" and prompt_schedule.fixed_index < len(SUBTASK_NAMES) - 1:
                prompt, name, index = prompt_schedule.switch_to_next(step)
                print(f"[prompt] N pressed; switched to subtask={index}:{name} prompt={prompt}", flush=True)
                recorder.record_event({"type": "subtask_switch", "step": step, "subtask_index": index, "subtask": name, "prompt": prompt})
            else:
                print("[prompt] N pressed; already at final subtask", flush=True)
        prompt, prompt_name, prompt_index = prompt_schedule.prompt_for_step(step)
        prompt_schedule.maybe_log_switch(step, prompt, prompt_name, prompt_index)
        obs, images = build_observation(image_client, controller, args.initial_state, prompt)
        request_started = time.perf_counter()
        result = policy.infer(obs)
        roundtrip_ms = (time.perf_counter() - request_started) * 1000
        recorder.record_event({"type": "inference", "step": step, "prompt": prompt, "roundtrip_ms": roundtrip_ms})
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.shape != (32, 16) or not np.isfinite(actions).all():
            raise RuntimeError(f"unexpected policy action shape: {actions.shape}")
        infer_ms = float(result.get("server_timing", {}).get("infer_ms", -1.0))
        print(
            f"[policy] seq={actions.shape} infer_ms={infer_ms:.1f} "
            f"prompt={prompt_name} first_arm={np.array2string(actions[0, :14], precision=3)} "
            f"gripper={actions[0, 14:16]}"
        )
        obs_state = obs["state"].astype(np.float64)
        recorder.record_images(chunk_index, *images)
        recorder.record_policy_chunk(chunk_index, actions, obs_state, prompt)
        actions = blend_chunk_boundary(actions, last_command_action, args.chunk_blend_steps)
        execute_chunk_steps = min(max(1, args.execute_chunk_steps), actions.shape[0])
        for chunk_step, action in enumerate(actions[:execute_chunk_steps]):
            if step >= args.steps:
                break
            loop_start = time.perf_counter()
            cmd_action, robot_state_before, robot_state_after = execute_policy_action(
                action,
                controller,
                left_gripper_value,
                right_gripper_value,
                args.max_arm_delta,
                args.max_gripper_delta,
                obs_state,
                action_smoother,
                args.interpolation_substeps,
                1.0 / args.fps,
                last_command_action,
            )
            last_command_action = cmd_action.copy()
            recorder.record_action(
                step,
                chunk_index,
                chunk_step,
                time.time() - rollout_start,
                infer_ms if chunk_step == 0 else np.nan,
                obs_state,
                action,
                cmd_action,
                robot_state_before,
                robot_state_after,
            )
            step += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - loop_start)))
        chunk_index += 1


def run_async_rollout(
    args,
    prompt_schedule: PromptSchedule,
    policy: PolicyClient,
    image_client: ImageClient,
    controller,
    left_gripper_value,
    right_gripper_value,
    recorder: RolloutRecorder,
    next_key: NextSubtaskKey | None = None,
) -> None:
    prefetcher = AsyncPolicyPrefetcher(policy)
    action_queue: list[np.ndarray] = []
    active_obs_state = np.zeros(16, dtype=np.float64)
    active_infer_ms = np.nan
    active_chunk_index = -1
    pending_chunk: PolicyChunk | None = None
    step = 0
    chunk_step = 0
    chunk_index = 0
    rollout_start = time.time()
    action_smoother = None if args.disable_action_smoothing else ActionSmoother(args.action_smooth_alpha, args.action_max_dq)
    prefetch_remaining = max(0, int(args.prefetch_remaining_steps))
    last_command_action: np.ndarray | None = None

    def activate_chunk(ready: PolicyChunk, current_step: int) -> tuple[list[np.ndarray], int]:
        """Drop actions whose timestamps elapsed while the request was in flight."""
        # Align to actual executed waypoints after q/image observation capture.
        # Wall-clock age also includes camera acquisition, waiting and disk I/O;
        # holding one endpoint is not executing additional policy actions.
        now = time.perf_counter()
        observation_age_ms = max(0.0, (now - ready.observed_at) * 1000.0)
        if current_step == 0 and active_chunk_index < 0:
            stale_steps = 0
        elif ready.observed_step is not None:
            stale_steps = max(0, current_step - ready.observed_step)
        else:
            stale_steps = int(observation_age_ms * args.fps / 1000.0)
        recorder.record_policy_chunk(chunk_index, ready.actions, ready.obs_state, ready.prompt)
        recorder.record_event({
            "type": "chunk_activation", "chunk_index": chunk_index,
            "step": current_step, "observed_step": ready.observed_step,
            "skipped_steps": stale_steps, "observation_age_ms": observation_age_ms,
            "capture_ms": (ready.observed_at-ready.submitted_at)*1000.0,
            "capture_to_ready_ms": (ready.ready_at-ready.observed_at)*1000.0,
            "ready_wait_ms": (now-ready.ready_at)*1000.0,
            "prefetch_remaining_steps": prefetch_remaining,
            "discarded": stale_steps >= ready.actions.shape[0],
        })
        if stale_steps >= ready.actions.shape[0]:
            return [], stale_steps
        actions = ready.actions[stale_steps:]
        actions = blend_chunk_boundary(actions, last_command_action, args.chunk_blend_steps)
        execute_chunk_steps = min(max(1, args.execute_chunk_steps), actions.shape[0])
        print(
            f"[policy] chunk timing: inference_age_steps={stale_steps} "
            f"ready_at_step={current_step} remaining={actions.shape[0]}"
        )
        return list(actions[:execute_chunk_steps]), stale_steps

    def start_prefetch(start_step: int) -> bool:
        prompt, prompt_name, prompt_index = prompt_schedule.prompt_for_step(start_step)
        prompt_schedule.maybe_log_switch(start_step, prompt, prompt_name, prompt_index)
        return prefetcher.start(
            lambda: build_observation(image_client, controller, args.initial_state, prompt),
            prompt=prompt,
            observation_step_factory=lambda: step,
        )

    print(
        f"[policy] async prefetch enabled; execute_chunk_steps={args.execute_chunk_steps} "
        f"prefetch_remaining_steps={prefetch_remaining}"
    )
    start_prefetch(0)
    try:
        while step < args.steps:
            if next_key is not None and next_key.pop():
                if prompt_schedule.fixed_index < len(SUBTASK_NAMES)-1:
                    prompt, name, index = prompt_schedule.switch_to_next(step)
                    prefetcher.invalidate()
                    action_queue.clear()
                    pending_chunk = None
                    recorder.record_event({"type":"subtask_switch","step":step,"subtask_index":index,"subtask":name,"prompt":prompt})
                    print(f"[prompt] N pressed; async generation invalidated; subtask={index}:{name}",flush=True)
            ready = prefetcher.pop_result()
            if ready is not None:
                if action_queue:
                    pending_chunk = ready
                else:
                    action_queue, _ = activate_chunk(ready, step)
                    active_chunk_index = chunk_index
                    active_obs_state = ready.obs_state
                    active_infer_ms = ready.infer_ms
                    chunk_step = 0
                    recorder.record_images(active_chunk_index, *ready.images)
                    print(
                        f"[policy] chunk={active_chunk_index} seq={ready.actions.shape} infer_ms={ready.infer_ms:.1f} "
                        f"prompt={ready.prompt} first_arm={np.array2string(ready.actions[0, :14], precision=3)} "
                        f"gripper={ready.actions[0, 14:16]}"
                    )
                    chunk_index += 1
            error = prefetcher.pop_error()
            if error is not None:
                print(f"[policy] async inference failed; retrying: {error}")
                start_prefetch(step + len(action_queue))
            if (
                pending_chunk is None
                and
                len(action_queue) <= prefetch_remaining
                and not prefetcher.busy()
            ):
                try:
                    start_prefetch(step + len(action_queue))
                except Exception as exc:
                    print(f"[policy] async observation failed; retrying: {exc}")
            if not action_queue:
                time.sleep(0.002)
                continue
            loop_start = time.time()
            action = action_queue.pop(0)
            cmd_action, robot_state_before, robot_state_after = execute_policy_action(
                action,
                controller,
                left_gripper_value,
                right_gripper_value,
                args.max_arm_delta,
                args.max_gripper_delta,
                active_obs_state,
                action_smoother,
                args.interpolation_substeps,
                1.0 / args.fps,
                last_command_action,
            )
            last_command_action = cmd_action.copy()
            recorder.record_action(
                step,
                active_chunk_index,
                chunk_step,
                time.time() - rollout_start,
                active_infer_ms if chunk_step == 0 else np.nan,
                active_obs_state,
                action,
                cmd_action,
                robot_state_before,
                robot_state_after,
            )
            step += 1
            chunk_step += 1
            sleep_time = (1.0 / args.fps) - (time.time() - loop_start)
            time.sleep(max(0.0, sleep_time))
            if not action_queue and prefetcher.busy():
                print("[policy] waiting for prefetched chunk")
            if not action_queue:
                next_ready = pending_chunk or prefetcher.pop_result()
                if next_ready is not None:
                    pending_chunk = None
                    action_queue, _ = activate_chunk(next_ready, step)
                    active_chunk_index = chunk_index
                    active_obs_state = next_ready.obs_state
                    active_infer_ms = next_ready.infer_ms
                    chunk_step = 0
                    recorder.record_images(active_chunk_index, *next_ready.images)
                    print(
                        f"[policy] chunk={active_chunk_index} seq={next_ready.actions.shape} "
                        f"infer_ms={next_ready.infer_ms:.1f} prompt={next_ready.prompt}"
                    )
                    chunk_index += 1
    finally:
        prefetcher.close()


def validate_contract(args, metadata):
    expected = {
        "contract_version": "g1d-pi05-v1", "profile": args.profile,
        "action_layout": "left_arm_7,right_arm_7,left_gripper,right_gripper",
        "action_dim": 16, "model_action_dim": 32, "action_horizon": 32,
        "fps": 30, "prompt_mode": args.prompt_mode,
        "image_preprocess": "openpi_client.resize_with_pad.PIL_BILINEAR",
        "normalization": "legacy_quantile_clip" if args.profile == "legacy" else "official_quantile_unclipped",
    }
    profiles_path = args.profiles_file
    profiles = json.loads(profiles_path.read_text())
    expected["client_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    expected["checkpoint"] = profiles["profiles"][args.profile]["checkpoint"]
    expected["profiles_sha256"] = hashlib.sha256(profiles_path.read_bytes()).hexdigest()
    expected["norm_stats_sha256"] = profiles["profiles"][args.profile]["norm_stats_sha256"]
    expected["runtime_helper_sha256"] = hashlib.sha256(Path(__file__).with_name("robot_alignment.py").read_bytes()).hexdigest()
    errors = {k: (v, metadata.get(k)) for k, v in expected.items() if metadata.get(k) != v}
    if args.prompt_mode not in ("subtask", "whole") or args.fps != 30:
        errors["client_mode/fps"] = ("subtask|whole/30", (args.prompt_mode, args.fps))
    if not 1 <= args.execute_chunk_steps <= 32:
        errors["execute_chunk_steps"] = ("1..32", args.execute_chunk_steps)
    if not 1 <= args.interpolation_substeps <= 16:
        errors["interpolation_substeps"] = ("1..16", args.interpolation_substeps)
    if not 0 <= args.chunk_blend_steps <= 16:
        errors["chunk_blend_steps"] = ("0..16", args.chunk_blend_steps)
    own_image_hash = hashlib.sha256(Path(image_tools.__file__).read_bytes()).hexdigest()
    if metadata.get("image_tools_sha256") != own_image_hash:
        errors["image_tools_sha256"] = (own_image_hash, metadata.get("image_tools_sha256"))
    if not metadata.get("norm_stats_sha256"):
        errors["norm_stats_sha256"] = ("required", None)
    if errors:
        raise ValueError(f"G1D client/server contract mismatch: {errors}")


def main() -> None:
    args = make_argparser().parse_args()
    # Launch profiles resolve task-specific height; it is not uniform across datasets.
    if args.profile == "official":
        if args.fixed_height_m is None:
            args.fixed_height_m = READY_TORSO_HEIGHT_M
        if args.waist_yaw_target is None:
            args.waist_yaw_target = DEFAULT_WAIST_YAW_TARGET
    elif args.waist_yaw_target is None:
        # The waist is outside both legacy and official 16D action contracts.
        args.waist_yaw_target = DEFAULT_WAIST_YAW_TARGET
    if args.execute and args.disable_waist_yaw_lock:
        raise ValueError("real G1-D execution requires the waist-yaw lock")
    if args.execute and args.initial_state != "current":
        raise ValueError("Real execution requires current measured state")
    prompt = args.prompt or TASKS[args.task]
    prompt_schedule = PromptSchedule(
        mode=args.prompt_mode,
        task=args.task,
        total_steps=args.steps,
        fractions=parse_subtask_fractions(args.subtask_fractions),
        whole_prompt=prompt,
        schedule=args.subtask_schedule,
        fixed_index=args.subtask_index,
    )
    if args.prompt_mode == "subtask" and args.prompt:
        print("[warning] --prompt is ignored when --prompt-mode=subtask")
    print(f"[prompt] mode={args.prompt_mode} fractions={prompt_schedule.fractions}")
    print(
        f"[motion] async_prefetch={args.async_prefetch} execute_chunk_steps={args.execute_chunk_steps} "
        f"interpolation_substeps={args.interpolation_substeps} chunk_blend_steps={args.chunk_blend_steps} "
        f"action_smoothing={not args.disable_action_smoothing}",
        flush=True,
    )
    if args.initial_state == "ready" and not args.gripper_open_only:
        print(
            "[warning] --initial-state=ready feeds a fixed dataset-ready proprio state to the policy. "
            "Use --initial-state=current for real rollouts after --go-ready-before-start."
        )
    policy = PolicyClient(args.policy_host, args.policy_port)
    validate_contract(args, policy.metadata)
    print(f"[policy] metadata={policy.metadata}")
    if args.check_server_only:
        policy.close()
        print("[check] server contract matches; no cameras or robot controller initialized")
        return
    recorder = RolloutRecorder(args.record_dir, args.fps)
    next_key = NextSubtaskKey() if args.execute and args.prompt_mode == "subtask" and args.subtask_schedule == "fixed" else None
    if recorder.enabled:
        (recorder.dir / "resolved_config.json").write_text(json.dumps({
            "args": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, "server_metadata": policy.metadata,
            "control_revision": "late_prefetch_executed_step_alignment_20261007",
            "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "image_tools_sha256": hashlib.sha256(Path(image_tools.__file__).read_bytes()).hexdigest(),
        }, indent=2))

    image_client = ImageClient(host=args.img_server_ip, request_bgr=True)
    controller = None
    height_lock = None
    waist_monitor = None
    telemetry = None
    gravity = None
    left_gripper_value = Value("d", 5.0, lock=True)
    right_gripper_value = Value("d", 5.0, lock=True)
    motion_ready = Value("b", True, lock=True)

    try:
        if args.execute:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from teleop.robot_control.robot_arm import G1_29_Arm_Internal_Dex1_Controller

            validate_network_interface(args.network_interface)
            gravity = GravityCompensation(args.gravity_urdf,args.gravity_scale) if args.gravity_compensation else None
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)
            telemetry = RobotTelemetry()
            recorder.telemetry = telemetry
            recorder.gravity = gravity
            before = telemetry.wait_ready()
            recorder.record_event({"type":"physical_startup_before",**before})
            if gravity is not None:
                recorder.record_event({"type":"gravity_model",**gravity.metadata()})
            if not args.disable_height_lock and args.fixed_height_m is not None:
                height_lock = FixedHeightLock(
                    args.height_control_binary,
                    args.network_interface,
                    args.fixed_height_m,
                )
                try:
                    height_lock.start()
                    settled = telemetry.wait_height(args.fixed_height_m,args.height_tolerance_m)
                    recorder.record_event({"type":"height_settled","target_m":args.fixed_height_m,**settled})
                    print(f"[robot] measured height settled: {settled['height_m']:.4f}m",flush=True)
                except Exception:
                    policy.close()
                    image_client.close()
                    recorder.close()
                    raise
            try:
                controller = G1_29_Arm_Internal_Dex1_Controller(
                    left_gripper_value,
                    right_gripper_value,
                    Lock(),
                    Array("d", 2, lock=False),
                    Array("d", 2, lock=False),
                    simulation_mode=False,
                    use_waist=False,
                    xr_motion_data_ready_in=motion_ready,
                    waist_yaw_lock_target=args.waist_yaw_target,
                )
            except Exception:
                if height_lock is not None:
                    height_lock.close()
                policy.close()
                image_client.close()
                recorder.close()
                raise
            if gravity is not None:
                controller = GravityAlignedController(controller,gravity)
            controller.arm_velocity_limit = args.arm_velocity_limit
            controller.lock_waist_yaw(args.waist_yaw_target)
            time.sleep(1.0)
            if args.go_home_before_start:
                print("[robot] moving arms/waist to home before policy rollout")
                controller.ctrl_dual_arm_go_home()
                time.sleep(args.home_wait)
            wait_for_waist_yaw(controller, args.waist_yaw_target)
            if args.monitor_waist_yaw:
                monitor_candidates = [
                    candidate / "tools" / "monitor_waist_yaw.py"
                    for candidate in TELEOP_SEARCH_ROOTS
                ]
                monitor_script = next((path for path in monitor_candidates if path.is_file()), None)
                if monitor_script is None:
                    print("[robot] warning: monitor_waist_yaw.py not found; continuing without monitor")
                else:
                    waist_monitor = subprocess.Popen(
                        [
                            sys.executable,
                            str(monitor_script),
                            "--network-interface",
                            args.network_interface,
                            "--duration",
                            "3600",
                        ],
                        stdin=subprocess.DEVNULL,
                    )
                    print(f"[robot] waist-yaw read-only monitor started (pid={waist_monitor.pid})")
            if args.go_ready_before_start:
                move_to_ready_pose(controller, left_gripper_value, right_gripper_value, args.ready_wait)
            if args.gripper_open_only:
                open_q = float(np.clip(args.gripper_open_q, 0.0, 5.4))
                trigger_value = gripper_q_to_trigger_value(open_q)
                print(f"[robot] opening both grippers to q={open_q:.3f}; holding for {args.hold_seconds:.1f}s")
                end_time = time.time() + args.hold_seconds
                while time.time() < end_time:
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = trigger_value
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = trigger_value
                    time.sleep(0.05)
                return
            print("[robot] controller initialized; executing policy commands")
        else:
            print("[dry-run] not sending robot commands; add --execute to control G1-D")

        if args.async_prefetch:
            run_async_rollout(
                args,
                prompt_schedule,
                policy,
                image_client,
                controller,
                left_gripper_value,
                right_gripper_value,
                recorder,
                next_key,
            )
        else:
            run_sync_rollout(
                args,
                prompt_schedule,
                policy,
                image_client,
                controller,
                left_gripper_value,
                right_gripper_value,
                recorder,
                next_key,
            )
    except KeyboardInterrupt:
        print("[main] interrupted")
    finally:
        if waist_monitor is not None:
            try:
                waist_monitor.terminate()
                waist_monitor.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                waist_monitor.kill()
        if controller is not None:
            print("[robot] returning arms/grippers to home")
            try:
                controller.ctrl_dual_arm_go_home()
                with left_gripper_value.get_lock():
                    left_gripper_value.value = gripper_q_to_trigger_value(2.7)
                with right_gripper_value.get_lock():
                    right_gripper_value.value = gripper_q_to_trigger_value(2.7)
                time.sleep(3.0)
                controller.running = False
            except Exception as exc:
                print(f"[robot] shutdown warning: {exc}")
        if height_lock is not None:
            height_lock.close()
        if recorder.enabled:
            recorder.record_event({"type": "shutdown", "reason": "interrupt_or_complete", "safe_home_requested": controller is not None})
        if telemetry is not None:
            telemetry.close()
        image_client.close()
        policy.close()
        recorder.close()
        if next_key is not None:
            next_key.close()


if __name__ == "__main__":
    main()
