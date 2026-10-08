#!/usr/bin/env python3
"""Resolve the versioned profile once; execution is an explicit CLI/environment option."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--check-server-only', action='store_true')
parser.add_argument('--execute', action='store_true')
parser.add_argument('--print-command', action='store_true')
args = parser.parse_args()
variant = os.environ.get('G1D_VARIANT', 'baseline')
if variant not in ('baseline', 'qc', 'joint197'):
    parser.error('G1D_VARIANT must be baseline, qc or joint197')
profiles_path = root / f'g1d_profiles_{variant}.json'
profiles = json.loads(profiles_path.read_text())
profile = os.environ.get('G1D_PROFILE', 'official')
if profile not in profiles['profiles']:
    parser.error('G1D_PROFILE must be legacy or official')
settings = dict(profiles['client'])
# Physical invariants for the official data/inference geometry.  Keep these
# visible in the resolved command without changing the versioned contract
# profile hash used by the policy server.
if profile == 'official':
    task = os.environ.get('TASK', settings.get('task', 'chrysanthemum'))
    settings.setdefault('fixed_height_m', profiles['height_by_task_m'][task])
    settings.setdefault('waist_yaw_target', 0.0)
if os.environ.get('MONITOR_WAIST_YAW', '0').lower() in ('1', 'true', 'yes'):
    settings['monitor_waist_yaw'] = True
# Only deliberate task/run/connection overrides; dynamics stay in the versioned JSON.
for env, key, cast in [('TASK','task',str), ('SUBTASK_INDEX','subtask_index',int),
                       ('STEPS','steps',int), ('POLICY_HOST','policy_host',str),
                       ('POLICY_PORT','policy_port',int), ('IMG_SERVER_IP','img_server_ip',str),
                       ('NETWORK_INTERFACE','network_interface',str),
                       ('FIXED_HEIGHT_M','fixed_height_m',float),
                       ('HEIGHT_TOLERANCE_M','height_tolerance_m',float),
                       ('GRAVITY_SCALE','gravity_scale',float),
                       ('EXECUTE_CHUNK_STEPS','execute_chunk_steps',int),
                       ('PREFETCH_REMAINING_STEPS','prefetch_remaining_steps',int),
                       ('INTERPOLATION_SUBSTEPS','interpolation_substeps',int),
                       ('CHUNK_BLEND_STEPS','chunk_blend_steps',int),
                       ('ACTION_SMOOTH_ALPHA','action_smooth_alpha',float),
                       ('ACTION_MAX_DQ','action_max_dq',float),
                       ('MAX_ARM_DELTA','max_arm_delta',float),
                       ('MAX_GRIPPER_DELTA','max_gripper_delta',float)]:
    if env in os.environ:
        settings[key] = cast(os.environ[env])
if 'ASYNC_PREFETCH' in os.environ:
    settings['async_prefetch'] = os.environ['ASYNC_PREFETCH'].lower() in ('1', 'true', 'yes')
if 'GRAVITY_COMPENSATION' in os.environ:
    settings['gravity_compensation'] = os.environ['GRAVITY_COMPENSATION'].lower() in ('1', 'true', 'yes')
if os.environ.get('ENABLE_ACTION_SMOOTHING', '0').lower() in ('1', 'true', 'yes'):
    settings['disable_action_smoothing'] = False
if os.environ.get('DISABLE_ACTION_SMOOTHING', '0').lower() in ('1', 'true', 'yes'):
    settings['disable_action_smoothing'] = True
cmd = [sys.executable, str(root / 'deploy_g1d_food_sorting_pi05.py'), '--profile', profile, '--profiles-file', str(profiles_path)]
for key, value in settings.items():
    if isinstance(value, bool):
        if value:
            cmd.append('--' + key.replace('_', '-'))
    else:
        cmd.extend(['--' + key.replace('_', '-'), str(value)])
if args.check_server_only:
    cmd.append('--check-server-only')
else:
    record = os.environ.get('RECORD_DIR', str(root / 'rollouts' / time.strftime('%Y%m%d_%H%M%S')))
    cmd.extend(['--record-dir', record])
if args.execute or os.environ.get('EXECUTE', '0') == '1':
    cmd.append('--execute')
if os.environ.get('GO_READY_BEFORE_START', '0') == '1':
    cmd.append('--go-ready-before-start')
print(json.dumps({'profile': profile, 'command': cmd}, indent=2), flush=True)
if not args.print_command:
    # Replace the launcher process so Ctrl-C is handled once by the robot
    # client. Waiting on a child here caused a second KeyboardInterrupt
    # traceback even after the client had already returned the robot home.
    os.execv(cmd[0], cmd)
