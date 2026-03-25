#!/bin/bash
# ─── submit_wfg_benchmark.sh ──────────────────────────────────────────────────
# Submits WFG_BENCHMARK.sbatch once per WFG problem (9 jobs × 30 seeds each).
# Each task runs all 5 methods for one seed, copies results back, then runs
# analyze_pymoo_benchmark.py inline so plots update as seeds finish.
#
# Usage:
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

mkdir -p slurm_logs_wfg_benchmark

echo "=== WFG Benchmark submission ==="
echo "N_OBJ=${N_OBJ:-2}  POP_SIZE=${POP_SIZE:-20}  INNER_POP_SIZE=${INNER_POP_SIZE:-200}  N_GEN_INNER=${N_GEN_INNER:-20}"
echo ""

JOB_IDS=()
PREV_JID=""

for PROB in "${PROBLEMS[@]}"; do
    # Chain: each problem waits for the previous to finish (afterany = success or fail)
    # so at most one problem's 30-task array is active/pending at a time, staying
    # under the QOS per-user submit limit.
    if [[ -n "$PREV_JID" ]]; then
        DEP="--dependency=afterany:${PREV_JID}"
    else
        DEP=""
    fi

    CMD=(sbatch
        --parsable
        ${DEP:+"$DEP"}
        --export=ALL,PROBLEM=${PROB},N_OBJ=${N_OBJ:-2},POP_SIZE=${POP_SIZE:-20},N_GEN_INNER=${N_GEN_INNER:-20},INNER_POP_SIZE=${INNER_POP_SIZE:-200}
        WFG_BENCHMARK.sbatch
    )
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] ${CMD[*]}"
        PREV_JID="DRYRUN_${PROB}"
    else
        JID=$( "${CMD[@]}" )
        echo "Submitted ${PROB}  -> job ${JID}${PREV_JID:+  (after ${PREV_JID})}"
        PREV_JID="$JID"
    fi
done

echo ""
echo "=== Done ==="
echo "Plots update after each seed finishes."
echo "Watch with: watch -n 30 ls -lh results/pymoo_benchmark/${N_OBJ:-2}_obj/"
