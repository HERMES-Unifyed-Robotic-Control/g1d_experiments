#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BASE_DEPLOY_DIR="${BASE_DEPLOY_DIR:-/home/unitree/pi05_g1d_deploy}"
XR_TELEOP_ROOT="${XR_TELEOP_ROOT:-/home/unitree/xr_teleoperate_g1d}"
export PYTHONPATH="${DEPLOY_DIR}:${BASE_DEPLOY_DIR}:${XR_TELEOP_ROOT}:${XR_TELEOP_ROOT}/teleop:${PYTHONPATH:-}"
cd "${DEPLOY_DIR}"
exec /home/unitree/miniconda3/envs/tv/bin/python "${DEPLOY_DIR}/launch_g1d_client.py" "$@"
