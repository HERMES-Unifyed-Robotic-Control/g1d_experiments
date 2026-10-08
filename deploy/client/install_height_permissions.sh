#!/usr/bin/env bash
# Run manually on G1D with sudo. Adds only the three configured task heights.
set -euo pipefail
if [[ ${EUID} -ne 0 ]]; then
  echo 'Run on G1D: sudo bash ./install_height_permissions.sh' >&2
  exit 1
fi
binary=/home/unitree/unitree_sdk2/build/bin/g1d_height_control
[[ -x "$binary" ]] || { echo "Missing executable: $binary" >&2; exit 1; }
command -v visudo >/dev/null
stage=$(mktemp -d /tmp/g1d-task-heights.XXXXXX)
trap 'rm -rf "$stage"' EXIT
cat > "$stage/sudoers" <<'RULES'
unitree ALL=(root) NOPASSWD: /home/unitree/unitree_sdk2/build/bin/g1d_height_control eth0 0.065
unitree ALL=(root) NOPASSWD: /home/unitree/unitree_sdk2/build/bin/g1d_height_control eth0 0.150
unitree ALL=(root) NOPASSWD: /home/unitree/unitree_sdk2/build/bin/g1d_height_control eth0 0.145
unitree ALL=(root) NOPASSWD: /usr/local/sbin/g1d_height_control_stop
RULES
cat > "$stage/stop" <<'STOP'
#!/bin/sh
set -eu
target='/home/unitree/unitree_sdk2/build/bin/g1d_height_control'
for pid in $(pgrep -f "^${target} eth0 0[.](065|145|150)$" || true); do
    kill -INT "$pid" 2>/dev/null || true
done
STOP
visudo -cf "$stage/sudoers"
if [[ -f /usr/local/sbin/g1d_height_control_stop && ! -f /usr/local/sbin/g1d_height_control_stop.before_task_heights ]]; then
  cp -p /usr/local/sbin/g1d_height_control_stop /usr/local/sbin/g1d_height_control_stop.before_task_heights
fi
install -o root -g root -m 0755 "$stage/stop" /usr/local/sbin/g1d_height_control_stop
install -o root -g root -m 0440 "$stage/sudoers" /etc/sudoers.d/g1d_policy_task_heights
printf 'Installed exact task-height start/stop permissions; no robot commands sent.\n'
