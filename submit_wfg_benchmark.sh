#!/bin/bash
# ─── submit_wfg_benchmark.sh ──────────────────────────────────────────────────
# Submits WFG_BENCHMARK.sbatch one problem at a time, waiting for each job to
# fully complete before submitting the next.  This keeps at most 1 array job
# (30 tasks) in the SLURM queue at any time, avoiding QOSMaxSubmitJobPerUser.
#
# IMPORTANT: run this inside a persistent session (screen / tmux / nohup) so
# it survives SSH disconnections while polling.
#
# Usage:
#   screen -S wfg_submit
#   bash submit_wfg_benchmark.sh [--dry-run]
#
# Optional overrides:
#   N_OBJ=2 POP_SIZE=20 N_GEN_INNER=20 INNER_POP_SIZE=200 bash submit_wfg_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

PROBLEMS=(wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9)
POLL_INTERVAL=60   # seconds between squeue checks

mkdir -p slurm_logs_wfg_benchmark

echo "=== WFG Benchmark submission ==="
echo "N_OBJ=${N_OBJ:-2}  POP_SIZE=${POP_SIZE:-20}  INNER_POP_SIZE=${INNER_POP_SIZE:-200}  N_GEN_INNER=${N_GEN_INNER:-20}"
echo "Submitting one job at a time (polling every ${POLL_INTERVAL}s)."
echo ""

_wait_for_job() {
    local jid="$1"
    echo "  Waiting for job ${jid} to finish ..."
    while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
        sleep "${POLL_INTERVAL}"
    done
    echo "  Job ${jid} done ($(date '+%H:%M:%S'))."
}

for PROB in "${PROBLEMS[@]}"; do
    CMD=(sbatch
        --parsable
        --export=ALL,PROBLEM=${PROB},N_OBJ=${N_OBJ:-2},POP_SIZE=${POP_SIZE:-20},N_GEN_INNER=${N_GEN_INNER:-20},INNER_POP_SIZE=${INNER_POP_SIZE:-200}
        WFG_BENCHMARK.sbatch
    )
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] ${CMD[*]}"
        echo "[DRY-RUN] would wait for job to finish before continuing"
    else
        JID=$( "${CMD[@]}" )
        echo "Submitted ${PROB}  -> job ${JID}"
        _wait_for_job "${JID}"
    fi
    echo ""
done

echo "=== All problems submitted and completed ==="
echo "Plots are in results/pymoo_benchmark/${N_OBJ:-2}_obj/"
