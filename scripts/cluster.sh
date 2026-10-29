#!/bin/bash
# Laptop-side access to the cluster that survives ssh drops: every call retries with exponential backoff.
#   bash scripts/cluster.sh run '<remote command>'      run once (retries only on ssh failure, exit 255)
#   bash scripts/cluster.sh watch <job_id> [log_path]   print state changes and the log tail until the job ends
# HOST (default bits-hpc), MAX_TRIES (default 8), POLL seconds (default 60).
set -uo pipefail
HOST=${HOST:-bits-hpc}

rssh() {  # rssh <command>: ssh with backoff 10, 20, 40 ... 300 s; returns the remote exit code
  local delay=10 try rc
  for try in $(seq 1 "${MAX_TRIES:-8}"); do
    ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$HOST" "$1"
    rc=$?
    [ $rc -ne 255 ] && return $rc
    echo "[cluster.sh] ssh failed (try $try), retrying in ${delay}s" >&2
    sleep "$delay"; delay=$(( delay * 2 > 300 ? 300 : delay * 2 ))
  done
  return 255
}

case "${1:-}" in
  run) rssh "$2" ;;
  watch)
    job=$2 log=${3:-} last=""
    while true; do
      state=$(rssh "squeue -h -j $job -o '%T %M %R' 2>/dev/null") || { echo "[cluster.sh] cluster unreachable"; exit 255; }
      if [ -z "$state" ]; then
        rssh "sacct -n -X -j $job -o JobID,State,ExitCode,Elapsed; ${log:+tail -40 $log}"
        exit 0
      fi
      s=${state%% *}
      [ "$s" != "$last" ] && { echo "[$(date +%H:%M:%S)] $job $state"; last=$s; }
      sleep "${POLL:-60}"
    done ;;
  *) echo "usage: $0 run '<cmd>' | watch <job_id> [log]"; exit 2 ;;
esac
