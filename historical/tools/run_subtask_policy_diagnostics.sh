#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-/mnt/data/yangky/test/datasets/g1d_food_sorting_123_subtasks_lerobot}"
PORT="${PORT:-9015}"
COUNT="${COUNT:-20}"
OUT_ROOT="${OUT_ROOT:-/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_finetune/raw_policy_diagnostics_by_subtask}"
PYTHON="${PYTHON:-/mnt/data/yangky/test/Psi0/.venv-openpi/bin/python}"
SCRIPT="/mnt/data/yangky/test/xr_teleoperate_g1d/tools/diagnose_pi05_dataset_vs_policy.py"

mkdir -p "${OUT_ROOT}"
for task_index in 0 1 2 3 4 5 6; do
  echo "[subtask] task_index=${task_index} count=${COUNT}"
  PYTHONPATH=/mnt/data/yangky/test/Psi0/src \
    "${PYTHON}" "${SCRIPT}" \
      --dataset "${DATASET}" \
      --port "${PORT}" \
      --count "${COUNT}" \
      --task-index "${task_index}" \
      --seed "$((20260924 + task_index))" \
      --output "${OUT_ROOT}/task_${task_index}"
done

echo "[subtask] reports written to ${OUT_ROOT}"
