#!/usr/bin/env python3
"""Mirror OpenPI stdout metrics into TensorBoard event files."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+grad_norm=(?P<grad_norm>[-+0-9.eE]+)"
)
RICH_METRIC_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[-+0-9.eE]+).*?"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+grad_norm=(?P<grad_norm>[-+0-9.eE]+)",
    re.DOTALL,
)
TQDM_RE = re.compile(
    r"loss=(?P<loss>[-+0-9.eE]+),\s+lr=(?P<lr>[-+0-9.eE]+),\s+step=(?P<step>\d+)"
)


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line.replace("\r", "\n"))


def emit_match(writer: SummaryWriter, match: re.Match[str], seen: set[int]) -> bool:
    step = int(match.group("step"))
    if step in seen:
        return False
    seen.add(step)
    writer.add_scalar("train/loss", float(match.group("loss")), step)
    writer.add_scalar("train/lr", float(match.group("lr")), step)
    grad_norm = match.groupdict().get("grad_norm")
    if grad_norm is not None:
        writer.add_scalar("train/grad_norm", float(grad_norm), step)
    return True


def emit_metrics(writer: SummaryWriter, line: str, seen: set[int]) -> int:
    count = 0
    for pattern in (METRIC_RE, TQDM_RE):
        for match in pattern.finditer(line):
            count += int(emit_match(writer, match, seen))
    if count:
        writer.flush()
    return count


def emit_rich_metrics(writer: SummaryWriter, text: str, seen: set[int]) -> int:
    count = 0
    for match in RICH_METRIC_RE.finditer(text):
        count += int(emit_match(writer, match, seen))
    if count:
        writer.flush()
    return count


def tail_log(log_path: Path, logdir: Path, poll: float) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    writer = SummaryWriter(log_dir=str(logdir))
    position = 0
    while True:
        if not log_path.exists():
            time.sleep(poll)
            continue
        size = log_path.stat().st_size
        if size < position:
            position = 0
            seen.clear()
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()
        cleaned = clean_line(chunk)
        emit_rich_metrics(writer, cleaned, seen)
        for line in cleaned.splitlines():
            emit_metrics(writer, line, seen)
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--logdir", required=True, type=Path)
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args()
    tail_log(args.log, args.logdir, args.poll)


if __name__ == "__main__":
    main()
