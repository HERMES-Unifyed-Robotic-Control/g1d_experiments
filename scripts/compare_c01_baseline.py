#!/usr/bin/env python3
"""Compare physical joint errors at the published anchors and first 30 steps."""
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--c01', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
anchors = json.loads((Path(__file__).parents[1] / 'data/validation_samples.json').read_text())
with np.load(a.baseline) as b, np.load(a.c01) as c:
    bp, bt = b['predictions'][:, :30, :16], b['targets'][:, :30, :16]
    cp, ct = c['predictions'], c['targets']
    tasks = c['task_indices']
    if not np.array_equal(c['indices'], anchors['indices']) or not np.array_equal(tasks, b['task_indices']):
        raise ValueError('Evaluation anchors/tasks differ')
    if bp.shape != cp.shape or not np.allclose(bt, ct, atol=1e-6, rtol=0):
        raise ValueError('Physical action targets differ')
    valid = c['action_valid']
    if not np.isfinite(bp).all() or not np.isfinite(cp).all():
        raise ValueError('Nonfinite predictions')
    errors = {'baseline7500': np.where(valid[..., None], bp-bt, np.nan),
              'c01': np.where(valid[..., None], cp-ct, np.nan)}

def metrics(e):
    return {'mae': float(np.nanmean(np.abs(e))),
            'rmse': float(np.sqrt(np.nanmean(e*e))),
            'per_dim_mae': np.nanmean(np.abs(e), axis=(0, 1)).tolist(),
            'left_arm_mae': float(np.nanmean(np.abs(e[..., :7]))),
            'right_arm_mae': float(np.nanmean(np.abs(e[..., 7:14]))),
            'left_gripper_mae': float(np.nanmean(np.abs(e[..., 14]))),
            'right_gripper_mae': float(np.nanmean(np.abs(e[..., 15])))}

report = {'observations': len(cp), 'horizon': 30, 'action_dim': 16,
          'overall': {k: metrics(v) for k, v in errors.items()},
          'per_task': {str(t): {k: metrics(v[tasks == t]) for k, v in errors.items()}
                       for t in sorted(set(tasks.tolist()))},
          'limitations': ['Holdout previously used for checkpoint selection, not independent test set.',
                         '7500-step baseline versus C01 training budget must be reported separately.',
                         'Action-space and effect-space noise have different dimensions.',
                         'Offline errors do not measure closed-loop robot success.']}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
