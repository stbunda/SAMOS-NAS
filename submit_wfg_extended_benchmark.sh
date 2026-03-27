#!/bin/bash
# ─── submit_wfg_extended_benchmark.sh ────────────────────────────────────────
# Submits WFG_EXTENDED_BENCHMARK.sbatch one problem at a time, waiting for
# each job to fully complete before submitting the next.
#
# Usage:
#   screen -S wfg_extended_submit
#   bash submit_wfg_extended_benchmark.sh [--dry-run]
#
# Optional overrides:
#   N_OBJ=2 POP_SIZE=20 N_GEN_INNER=20 INNER_POP_SIZE=200 bash submit_wfg_extended_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

POLL_INTERVAL=60   # seconds between squeue checks

mkdir -p slurm_logs_wfg_extended

echo "=== WFG Extended Benchmark submission ==="
echo "N_OBJ=${N_OBJ:-2}  POP_SIZE=${POP_SIZE:-20}  INNER_POP_SIZE=${INNER_POP_SIZE:-200}  N_GEN_INNER=${N_GEN_INNER:-20}"
echo "Problems: wfg1-9 (handled inside each task's for-loop)"
echo "Methods: random nsga2 samos-xgb samos-rfr mosmac parego cobra"
echo "         ssa-nsga2-{default,rfr,xgb}  gpsaf-{default,rfr,xgb}"
echo ""

CMD=(
    sbatch
    --parsable
    --export=ALL,N_OBJ=${N_OBJ:-2},POP_SIZE=${POP_SIZE:-20},N_GEN_INNER=${N_GEN_INNER:-20},INNER_POP_SIZE=${INNER_POP_SIZE:-200}
    WFG_EXTENDED_BENCHMARK.sbatch
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD[*]}"
else
    JID=$( "${CMD[@]}" )
    echo "Submitted -> job ${JID}"
    echo "Waiting for job ${JID} to finish (polling every ${POLL_INTERVAL}s) ..."
    while squeue -j "${JID}" -h 2>/dev/null | grep -q .; do
        sleep "${POLL_INTERVAL}"
    done
    echo "Job ${JID} done ($(date '+%H:%M:%S'))."
fi

echo ""
echo "=== Done ==="
echo "Plots are in results/pymoo_benchmark/${N_OBJ:-2}_obj/"
