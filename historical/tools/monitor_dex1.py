#!/usr/bin/env python3
"""Read-only Dex1 state monitor for integrated G1-D or external serial service."""
import argparse
import signal
import statistics
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

RUNNING = True


def stop(*_):
    global RUNNING
    RUNNING = False


def value(state, name, default=float('nan')):
    return getattr(state, name, default)


def line(side, state):
    return (
        f"{side}: q={value(state, 'q'):+.6f} rad  "
        f"dq={value(state, 'dq'):+.6f} rad/s  "
        f"tau_est={value(state, 'tau_est'):+.4f}  "
        f"motorstate={value(state, 'motorstate', 'n/a')}  "
        f"temperature={value(state, 'temperature', 'n/a')}"
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=('internal', 'external'), default='internal')
parser.add_argument('--network-interface', default='eth0')
parser.add_argument('--rate', type=float, default=5.0, help='terminal print rate in Hz')
parser.add_argument('--duration', type=float, default=0.0, help='seconds; 0 runs until Ctrl-C')
args = parser.parse_args()

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
ChannelFactoryInitialize(0, networkInterface=args.network_interface)

if args.mode == 'internal':
    topic_description = 'rt/lowstate motor_state[31]=left, motor_state[33]=right'
    state_sub = ChannelSubscriber('rt/lowstate', LowState_)
    state_sub.Init()

    def read_pair():
        msg = state_sub.Read()
        return None if msg is None else (msg.motor_state[31], msg.motor_state[33])
else:
    topic_description = 'rt/dex1/left/state and rt/dex1/right/state, states[0]'
    left_sub = ChannelSubscriber('rt/dex1/left/state', MotorStates_)
    right_sub = ChannelSubscriber('rt/dex1/right/state', MotorStates_)
    left_sub.Init()
    right_sub.Init()

    def read_pair():
        left_msg = left_sub.Read()
        right_msg = right_sub.Read()
        if left_msg is None or right_msg is None or not left_msg.states or not right_msg.states:
            return None
        return left_msg.states[0], right_msg.states[0]

print(f'Mode: {args.mode}; source: {topic_description}')
print('READ ONLY: this program does not publish commands.')
start = time.monotonic()
next_print = start
period = 1.0 / max(args.rate, 0.1)
samples = {'left': [], 'right': []}
latest = None
while RUNNING and (args.duration <= 0 or time.monotonic() - start < args.duration):
    pair = read_pair()
    if pair is not None:
        latest = pair
        samples['left'].append(float(pair[0].q))
        samples['right'].append(float(pair[1].q))
    now = time.monotonic()
    if latest is not None and now >= next_print:
        print(f'{now - start:8.2f}s  {line("left ", latest[0])}  |  {line("right", latest[1])}', flush=True)
        next_print = now + period
    time.sleep(0.002)

print('\nSummary:')
for side in ('left', 'right'):
    values = samples[side]
    if values:
        print(
            f'{side}: mean={statistics.mean(values):.6f} rad  '
            f'min={min(values):.6f} rad  max={max(values):.6f} rad  samples={len(values)}'
        )
    else:
        print(f'{side}: NO DATA')
