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

PROBLEMS=(wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9)
POLL_INTERVAL=60   # seconds between squeue checks

mkdir -p slurm_logs_wfg_extended

echo "=== WFG Extended Benchmark submission ==="
echo "N_OBJ=${N_OBJ:-2}  POP_SIZE=${POP_SIZE:-20}  INNER_POP_SIZE=${INNER_POP_SIZE:-200}  N_GEN_INNER=${N_GEN_INNER:-20}"
echo "Methods: random nsga2 samos-xgb samos-rfr mosmac parego cobra"
echo "         ssa-nsga2-{default,rfr,xgb}  gpsaf-{default,rfr,xgb}"
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
        WFG_EXTENDED_BENCHMARK.sbatch
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
echo "Plots are in results/pymoo_benchmark_extended/${N_OBJ:-2}_obj/"
echo "  moo_hv_igd.png         — standard 2-panel overlay"
echo "  moo_hv_igd_extended.png — group-column layout (one column per family)"
