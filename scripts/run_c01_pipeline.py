#!/usr/bin/env python3
"""Sequential, restartable C01 native16 training and offline evaluation pipeline."""
from pathlib import Path
import argparse, subprocess, json, os, sys, datetime

p = argparse.ArgumentParser()
for name in ('train', 'validation', 'dino', 'dino-source', 'base-params', 'norm-stats', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--torch-python', default=sys.executable)
p.add_argument('--jax-python', default=sys.executable)
p.add_argument('--baseline-predictions',type=Path)
p.add_argument('--e-steps', type=int, default=30000)
p.add_argument('--pi-steps', type=int, default=40000)
a = p.parse_args()
root = Path(__file__).parents[1]
a.output.mkdir(parents=True, exist_ok=True)
(a.output/'logs').mkdir(exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_PREALLOCATE='false',
           PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')

def status(stage, state, **extra):
    record = dict(stage=stage, state=state, timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temp = a.output/'PIPELINE_STATUS.tmp.json'
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(a.output/'PIPELINE_STATUS.json')
    print(json.dumps(record), flush=True)

def run(stage, python, script, args):
    command = [python, str(root/'scripts'/script), *map(str, args)]
    status(stage, 'running', command=command)
    with (a.output/'logs'/f'{stage}.log').open('a') as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status(stage, 'failed', returncode=result.returncode)
        raise SystemExit(result.returncode)
    status(stage, 'complete')

common = ['--train', a.train, '--validation', a.validation]
normal = a.output/'normalization'
evidence = a.output/'evidence'
encoder = a.output/'E'
cache = a.output/'cache'
calibration = a.output/'calibration_lora'
policy = a.output/'pi_lora'
run('statistics', a.torch_python, 'prepare_c01_statistics.py', common+['--output', normal])
if not (evidence/'world_complete.json').exists():
    run('visual', a.torch_python, 'prepare_c01_visual.py', common+['--dino', a.dino,
        '--dino-source', a.dino_source, '--output', evidence])
if not (encoder/'TRAINING_COMPLETE.json').exists():
    args = common+['--evidence', evidence, '--statistics', normal/'statistics.json', '--output', encoder, '--steps', a.e_steps]
    if (encoder/'last.pt').exists(): args += ['--resume']
    run('E', a.torch_python, 'train_c01_e.py', args)
if not (cache/'manifest.json').exists():
    run('export', a.torch_python, 'export_c01_e.py', common+['--checkpoint', encoder/'best.pt', '--evidence', evidence, '--output', cache])
pi_args = common+['--cache', cache, '--base-params', a.base_params, '--norm-stats', a.norm_stats, '--variant', 'lora']
if not (calibration/'calibration.json').exists():
    run('calibration', a.jax_python, 'train_c01_pi.py', pi_args+['--output', calibration, '--calibrate-only'])
weight = json.loads((calibration/'calibration.json').read_text())['chosen_gtaux_weight']
if not (policy/'TRAINING_COMPLETE.json').exists():
    args = pi_args+['--output', policy, '--gtaux-weight', weight, '--steps', a.pi_steps]
    if list(policy.glob('cursor_*.json')): args += ['--resume']
    run('pi_lora', a.jax_python, 'train_c01_pi.py', args)
records = [json.loads(s) for s in (policy/'validation_metrics.jsonl').read_text().splitlines()]
best = min(records, key=lambda row: row['loss'])
(a.output/'PI_SELECTION.json').write_text(json.dumps(dict(best, selection='minimum heldout composite loss; not independent test'), indent=2)+'\n')
run('evaluation', a.jax_python, 'evaluate_c01_pi.py', ['--params', policy/'checkpoints'/str(best['step'])/'params',
    '--cache', cache, '--validation', a.validation, '--norm-stats', a.norm_stats, '--variant', 'lora', '--output', a.output/'evaluation'])
if a.baseline_predictions:
    run('comparison', a.torch_python, 'compare_c01_baseline.py', ['--baseline', a.baseline_predictions, '--c01', a.output/'evaluation/predictions.npz', '--output', a.output/'comparison/baseline7500_vs_c01.json'])
status('pipeline', 'complete', selected_pi_step=best['step'], baseline_comparison='Run compare_c01_baseline.py with baseline7500 heldout predictions')
