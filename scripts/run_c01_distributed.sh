#!/usr/bin/env bash
# Run once on each allocated one-GPU instance. Assets and code share a filesystem.
set -euo pipefail
C01_ROOT=$(realpath "${1:?experiment root required}")
C01_RANK=${2:?rank required}
C01_COORDINATOR=${3:?coordinator host:port required}
case "$C01_RANK" in 0|1) ;; *) echo 'rank must be 0 or 1' >&2; exit 2;; esac
ulimit -Sn 65536
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export OPENPI_DATA_HOME="$C01_ROOT/tokenizer"
mkdir -p "$C01_ROOT/run"
C01_PYTHON="$C01_ROOT/venv/bin/python"
# Do not start on incomplete transfers or an environment still being installed.
/usr/bin/python3 - "$C01_ROOT" "$C01_RANK" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);rank=int(sys.argv[2]);out=root/'run'/f'RANK{rank}_STATUS.json'
def status(state):out.write_text(json.dumps({'state':state,'rank':rank,'updated_unix':time.time()},indent=2)+'\n')
status('waiting_for_assets_and_environment')
while True:
    marker=root/'ENVIRONMENT_READY.json';sync=root/'ASSET_SYNC_STATUS.json'
    if sync.exists():
        data=json.loads(sync.read_text())
        if data['state']=='failed':
            retry=root/'AUTO_SYNC_RETRY_STATUS.json'
            auto=json.loads(retry.read_text()) if retry.exists() else {}
            if not (auto.get('active') and time.time()-auto.get('updated_unix',0)<60):
                status('asset_sync_failed');raise SystemExit(data['error'])
        if data['state']=='complete' and marker.exists():break
    time.sleep(15)
for rank in (0,1):
    assert json.loads((root/f'COLLECTIVE_RANK{rank}.json').read_text())['success']
status('starting_distributed_training')
PY
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY no_proxy NO_PROXY SL_PROXY_ADDR
C01_WEIGHT=$(/usr/bin/python3 -c "import json;print(json.load(open('$C01_ROOT/assets/calibration/calibration.json'))['chosen_gtaux_weight'])")
C01_OUTPUT="$C01_ROOT/run/pi_lora_dp2_40k"
C01_RESUME=()
if compgen -G "$C01_OUTPUT/cursor_*.json" >/dev/null; then C01_RESUME=(--resume); fi
set +e
"$C01_PYTHON" "$C01_ROOT/code/scripts/train_c01_pi.py" \
  --num-processes 2 --process-id "$C01_RANK" --coordinator-address "$C01_COORDINATOR" \
  --train "$C01_ROOT/assets/data/g1d_subtasks_official_train_v1" \
  --validation "$C01_ROOT/assets/data/g1d_subtasks_official_validation_v1" \
  --cache "$C01_ROOT/assets/cache" --base-params "$C01_ROOT/assets/base_params" \
  --norm-stats "$C01_ROOT/assets/norm_stats.json" --output "$C01_OUTPUT" \
  --variant lora --batch-size 2 --steps 40000 --gtaux-weight "$C01_WEIGHT" "${C01_RESUME[@]}"
C01_RESULT=$?
set -e
/usr/bin/python3 - "$C01_ROOT" "$C01_RANK" "$C01_RESULT" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);rank=int(sys.argv[2]);code=int(sys.argv[3])
(root/'run'/f'RANK{rank}_STATUS.json').write_text(json.dumps({'rank':rank,'state':'training_complete' if code==0 else 'training_failed','returncode':code,'updated_unix':time.time()},indent=2)+'\n')
PY
if (( C01_RESULT != 0 )); then exit "$C01_RESULT"; fi
if [[ "$C01_RANK" == 0 ]]; then
  C01_BEST=$(/usr/bin/python3 - "$C01_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);rows=[json.loads(s) for s in (root/'validation_metrics.jsonl').read_text().splitlines()]
rows=[r for r in rows if (root/'checkpoints'/str(r['step'])/'params').is_dir()]
best=min(rows,key=lambda r:r['loss']);(root/'PI_SELECTION.json').write_text(json.dumps(dict(best,selection='minimum heldout composite loss; not independent test'),indent=2)+'\n');print(best['step'])
PY
  )
  "$C01_PYTHON" "$C01_ROOT/code/scripts/evaluate_c01_pi.py" \
    --params "$C01_OUTPUT/checkpoints/$C01_BEST/params" --cache "$C01_ROOT/assets/cache" \
    --validation "$C01_ROOT/assets/data/g1d_subtasks_official_validation_v1" \
    --norm-stats "$C01_ROOT/assets/norm_stats.json" --variant lora --output "$C01_ROOT/run/evaluation_dp2"
  "$C01_PYTHON" "$C01_ROOT/code/scripts/compare_c01_baseline.py" \
    --baseline "$C01_ROOT/assets/baseline7500/heldout_predictions.npz" \
    --c01 "$C01_ROOT/run/evaluation_dp2/predictions.npz" \
    --output "$C01_ROOT/run/comparison/baseline7500_vs_c01_dp2.json"
fi
