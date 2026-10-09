#!/usr/bin/env bash
set -euo pipefail
ulimit -Sn 65536
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
C01_RUN_ROOT=/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/g1d_c01_20261008
C01_BASE_ROOT=/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/g1d_subtask_compare40k_20261007/lora
C01_PYTHON=$C01_BASE_ROOT/venv/bin/python
C01_NORM=$C01_BASE_ROOT/assets/g1d_subtasks_pi05_compare_lora_40k_v1/g1d_subtasks_train_v1/norm_stats.json
C01_OUTPUT=$C01_RUN_ROOT/run/pi_lora_100k
"$C01_PYTHON" -u "$C01_RUN_ROOT/code/scripts/train_c01_pi.py" \
 --train "$C01_RUN_ROOT/data/g1d_subtasks_official_train_v1" \
 --validation "$C01_RUN_ROOT/data/g1d_subtasks_official_validation_v1" \
 --cache "$C01_RUN_ROOT/run/cache" --base-params "$C01_BASE_ROOT/models/pi05_base_jax/params" \
 --norm-stats "$C01_NORM" --variant lora --batch-size 1 \
 --output "$C01_OUTPUT" --gtaux-weight 1.3227095743492492 --steps 100000 --resume
C01_BEST=$(/usr/bin/python3 - "$C01_OUTPUT" <<'PY'
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);records=[json.loads(s) for s in (p/'validation_metrics.jsonl').read_text().splitlines()]
records=[r for r in records if (p/'checkpoints'/str(r['step'])/'params').is_dir()]
best=min(records,key=lambda r:r['loss']);(p/'PI_SELECTION.json').write_text(json.dumps(dict(best,selection='minimum heldout composite loss among continuation committed checkpoints; not independent test'),indent=2)+'\n');print(best['step'])
PY
)
"$C01_PYTHON" "$C01_RUN_ROOT/code/scripts/evaluate_c01_pi.py" \
 --params "$C01_OUTPUT/checkpoints/$C01_BEST/params" --cache "$C01_RUN_ROOT/run/cache" \
 --validation "$C01_RUN_ROOT/data/g1d_subtasks_official_validation_v1" --norm-stats "$C01_NORM" \
 --variant lora --output "$C01_RUN_ROOT/run/evaluation_100k"
"$C01_PYTHON" "$C01_RUN_ROOT/code/scripts/compare_c01_baseline.py" \
 --baseline "$C01_RUN_ROOT/code/baseline7500/heldout_predictions.npz" \
 --c01 "$C01_RUN_ROOT/run/evaluation_100k/predictions.npz" \
 --output "$C01_RUN_ROOT/run/comparison/baseline7500_vs_c01_100k.json"
