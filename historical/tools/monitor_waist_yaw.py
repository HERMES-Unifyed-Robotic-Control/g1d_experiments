#!/usr/bin/env python3
"""Read-only monitor for G1-D waist-yaw state and low-level command."""

from __future__ import annotations

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    command_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    state_sub.Init()
    command_sub.Init()

    print("READ ONLY: motor index 12 = WAIST_YAW; this tool publishes no commands.")
    deadline = time.monotonic() + max(0.1, args.duration)
    next_print = 0.0
    latest_state = None
    latest_command = None
    while time.monotonic() < deadline:
        state = state_sub.Read()
        command = command_sub.Read()
        if state is not None:
            latest_state = state.motor_state[12]
        if command is not None:
            latest_command = command.motor_cmd[12]
        now = time.monotonic()
        if latest_state is not None and latest_command is not None and now >= next_print:
            print(
                f"q={latest_state.q:+.6f} dq={latest_state.dq:+.6f} "
                f"cmd_q={latest_command.q:+.6f} cmd_dq={latest_command.dq:+.6f} "
                f"mode={latest_command.mode} kp={latest_command.kp:.1f} "
                f"kd={latest_command.kd:.1f}",
                flush=True,
            )
            next_print = now + 1.0 / max(0.1, args.rate)
        time.sleep(0.002)


if __name__ == "__main__":
    main()
