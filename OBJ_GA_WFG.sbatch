#!/bin/bash
# ─── OBJ_GA_WFG.sbatch ────────────────────────────────────────────────────────
# Experiment 1 (GA comparison under varying objectives) — WFG continuous suite.
#
# One array task == one seed. Each task loops over every METHOD x every WFG
# problem (wfg1..wfg9) for the given N_OBJ sub-experiment. The Python entry point
# auto-skips already-completed pickles, so re-submitting an array is safe.
#
# Submit one sub-experiment at a time via --export:
#   sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=2 OBJ_GA_WFG.sbatch
#   sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=3 OBJ_GA_WFG.sbatch
#   sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=4 OBJ_GA_WFG.sbatch
#
# Selective single-seed rerun (force overwrite):
#   sbatch --array=5 --export=ALL,BENCHMARK=wfg,N_OBJ=2,OVERWRITE=1 OBJ_GA_WFG.sbatch
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J obj_ga_wfg
#SBATCH -c 2
#SBATCH --array=0-19%5            # 20 tasks = 20 seeds, 5 concurrent
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs_obj_ga_wfg/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_obj_ga_wfg

# ─── environment ──────────────────────────────────────────────────────────────
module load nvidia/nvhpc/23.3
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

echo "Activated: SAMOS311"

# ─── parameters from submission env ───────────────────────────────────────────
BENCHMARK=${BENCHMARK:-wfg}
N_OBJ=${N_OBJ:-2}
SEED=$SLURM_ARRAY_TASK_ID
CONFIG=config/experiment_obj_ga.yaml

# ─── method / problem sets ────────────────────────────────────────────────────
METHODS=(random nsga2 nsga3 moead sms-emoa rvea age-moea age-moea2)
# WFG problem names are identical across n_obj sub-experiments (only n_var differs).
PROBLEMS=(wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9)

# ─── logging ──────────────────────────────────────────────────────────────────
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "BENCHMARK         = $BENCHMARK"
echo "N_OBJ             = $N_OBJ"
echo "SEED              = $SEED"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "METHODS           = ${METHODS[*]}"
echo "PROBLEMS          = ${PROBLEMS[*]}"

# ─── run: METHOD x PROBLEM ────────────────────────────────────────────────────
for METHOD in "${METHODS[@]}"; do
    for PROBLEM in "${PROBLEMS[@]}"; do
        echo ""
        echo "─── benchmark: $BENCHMARK  n_obj: $N_OBJ  problem: $PROBLEM  method: $METHOD  seed: $SEED"

        python experiment_obj_ga.py \
            --benchmark "$BENCHMARK" \
            --problem   "$PROBLEM" \
            --n_obj     "$N_OBJ" \
            --method    "$METHOD" \
            --seed      "$SEED" \
            --config    "$CONFIG" \
            ${OVERWRITE:+--overwrite}
    done
done

echo ""
echo "Date (done)       = $(date)"
