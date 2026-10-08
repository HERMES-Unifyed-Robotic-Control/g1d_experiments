#!/usr/bin/env python3
"""Record G1-D three-camera input stream from the teleimager server."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "teleop"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import logging_mp  # noqa: F401
except ModuleNotFoundError:
    logging_mp = types.ModuleType("logging_mp")
    for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        setattr(logging_mp, name, getattr(logging, name))
    logging_mp.getLogger = logging.getLogger
    logging_mp.basicConfig = logging.basicConfig
    sys.modules["logging_mp"] = logging_mp

from teleop.teleimager.src.teleimager.image_client import ImageClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.10.104")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = ImageClient(host=args.host, request_bgr=True)
    writers = {}
    names = ("base", "left_wrist", "right_wrist", "contact_sheet")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        start = time.time()
        frame_idx = 0
        while time.time() - start < args.seconds:
            cfg = client.get_cam_config()
            head = client.get_head_frame()
            left = client.get_left_wrist_frame()
            right = client.get_right_wrist_frame()
            if head is None or left is None or right is None or head.bgr is None or left.bgr is None or right.bgr is None:
                time.sleep(0.02)
                continue
            base = head.bgr
            if cfg["head_camera"].get("binocular", False):
                base = base[:, : base.shape[1] // 2]
            frames = {
                "base": cv2.resize(base, (224, 224), interpolation=cv2.INTER_AREA),
                "left_wrist": cv2.resize(left.bgr, (224, 224), interpolation=cv2.INTER_AREA),
                "right_wrist": cv2.resize(right.bgr, (224, 224), interpolation=cv2.INTER_AREA),
            }
            frames["contact_sheet"] = np.concatenate([frames["base"], frames["left_wrist"], frames["right_wrist"]], axis=1)
            for name, frame in frames.items():
                if name not in writers:
                    h, w = frame.shape[:2]
                    writers[name] = cv2.VideoWriter(str(out / f"{name}.mp4"), fourcc, args.fps, (w, h))
                cv2.putText(frame, f"{frame_idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                writers[name].write(frame)
                if frame_idx == 0:
                    cv2.imwrite(str(out / f"{name}_first.jpg"), frame)
            frame_idx += 1
            time.sleep(max(0.0, 1.0 / args.fps))
    finally:
        for writer in writers.values():
            writer.release()
        client.close()
    print(f"recorded {frame_idx} frames to {out}")


if __name__ == "__main__":
    main()
