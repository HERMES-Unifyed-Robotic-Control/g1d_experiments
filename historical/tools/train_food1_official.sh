#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/yangky/test/Psi0
REPORT=/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_food_sorting1
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/mnt/data/yangky/test/Psi0/.cache/openpi
export PYTHONPATH="/mnt/data/yangky/test/Psi0/src:${PYTHONPATH:-}"
mkdir -p "$REPORT"
while [[ ! -f cache/checkpoints/openpi/pi05_base_jax/DOWNLOAD_COMPLETE ]]; do sleep 10; done
python3 - <<'PY'
from pathlib import Path
import os, signal, subprocess, time
connections=subprocess.check_output(['ss','-tnp'],text=True)
if any(':9016 ' in x and 'ESTAB' in x for x in connections.splitlines()):
    raise SystemExit('active 9016 client detected; refusing to stop policy server')
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try: cmd=(p/'cmdline').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if b'serve_official_holdout_v1.py\x00' in cmd and b'--port\x009016\x00' in cmd:
        os.kill(int(p.name), signal.SIGTERM)
        print('stopped idle 9016 server', p.name, flush=True)
        time.sleep(2)
PY
printf 'TRAINING %s\n' "$(date -Iseconds)" > "$REPORT/job_status.txt"
.venv-openpi/bin/python /mnt/data/yangky/test/xr_teleoperate_g1d/tools/train_food1_official.py 2>&1 | tee -a "$REPORT/train.log"
printf 'COMPLETE %s\n' "$(date -Iseconds)" > "$REPORT/job_status.txt"
