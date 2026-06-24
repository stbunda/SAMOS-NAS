#!/bin/bash
# --------- OBJ_GA_PRED_1K_WFG.sbatch ------------------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2 variant: GA performance under fixed predictors (1000-sample DOE) --- WFG continuous suite.
#
# One array task == one seed. Each task loops over every PREDICTOR x METHOD x WFG
# problem (wfg1..wfg9) for the given N_OBJ sub-experiment. The Python entry point
# auto-skips already-completed pickles, so re-submitting an array is safe.
#
# Submit one sub-experiment at a time via --export:
#   sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=2 experiments/obj_ga_pred/OBJ_GA_PRED_1K_WFG.sbatch
#   sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=3 experiments/obj_ga_pred/OBJ_GA_PRED_1K_WFG.sbatch
#   sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=4 experiments/obj_ga_pred/OBJ_GA_PRED_1K_WFG.sbatch
#
# Restrict to one predictor (optional) via PREDICTORS:
#   sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=2,PREDICTORS=xgboost experiments/obj_ga_pred/OBJ_GA_PRED_1K_WFG.sbatch
#
# Selective single-seed rerun (force overwrite):
#   sbatch --array=5 --export=ALL,BENCHMARK=wfg,N_OBJ=2,OVERWRITE=1 experiments/obj_ga_pred/OBJ_GA_PRED_1K_WFG.sbatch
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J obj_ga_pred_1k_wfg
#SBATCH -c 2
#SBATCH --array=0-19%2            # 20 tasks = 20 seeds, 2 concurrent
#SBATCH --mem=8gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=12:00:00
#SBATCH --output=slurm/slurm_logs_obj_ga_pred_1k_wfg/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_obj_ga_pred_1k_wfg

# --------- scratch setup ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "$SCRATCH_DIRECTORY" ]; then
    echo "'$SCRATCH_DIRECTORY' already found!"
else
    echo "'$SCRATCH_DIRECTORY' not found, creating!"
    mkdir -p "$SCRATCH_DIRECTORY"
fi

RSYNC_SOURCES=(
    analysis
    problem
    strategy
    experiments/obj_ga_pred
)

# Pre-copy any already-computed results for WFG problems
for _PROBLEM in wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9; do
    for _N_OBJ in 2 3 4; do
        _SRC="results/obj_ga_pred/wfg/${_N_OBJ}_obj/${_PROBLEM}"
        [ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")
    done
done

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/"

cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 3 -type d

# --------- environment ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
module load nvidia/nvhpc/26.3
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

echo "Activated: SAMOS311"

# --------- parameters from submission env ---------------------------------------------------------------------------------------------------------------------------------
BENCHMARK=${BENCHMARK:-wfg}
N_OBJ=${N_OBJ:-2}
SEED=$SLURM_ARRAY_TASK_ID
CONFIG=experiments/obj_ga_pred/config/experiment_obj_ga_pred_1k.yaml

# --------- predictor / method / problem sets ------------------------------------------------------------------------------------------------------------------------
PREDICTORS=(${PREDICTORS:-xgboost rf rbf})
METHODS=(random nsga2 nsga3 moead sms-emoa rvea age-moea age-moea2)
# WFG problem names are identical across n_obj sub-experiments (only n_var differs).
PROBLEMS=(wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9)

# --------- logging ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"
echo "BENCHMARK         = $BENCHMARK"
echo "N_OBJ             = $N_OBJ"
echo "SEED              = $SEED"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "PREDICTORS        = ${PREDICTORS[*]}"
echo "METHODS           = ${METHODS[*]}"
echo "PROBLEMS          = ${PROBLEMS[*]}"

# --------- run: PREDICTOR x METHOD x PROBLEM ------------------------------------------------------------------------------------------------------------------------
for PREDICTOR in "${PREDICTORS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for PROBLEM in "${PROBLEMS[@]}"; do
            echo ""
            echo "--------- benchmark: $BENCHMARK  n_obj: $N_OBJ  problem: $PROBLEM  predictor: $PREDICTOR  method: $METHOD  seed: $SEED"

            python experiments/obj_ga_pred/experiment_obj_ga_pred.py \
                --benchmark "$BENCHMARK" \
                --problem   "$PROBLEM" \
                --n_obj     "$N_OBJ" \
                --predictor "$PREDICTOR" \
                --method    "$METHOD" \
                --seed      "$SEED" \
                --config    "$CONFIG" \
                ${OVERWRITE:+--overwrite}
        done
    done

    # ------ copy results back (per predictor) ------------------------------------------------------------------------------------------------------------------
    for PROBLEM in "${PROBLEMS[@]}"; do
        SRC="${SCRATCH_DIRECTORY}/results/obj_ga_pred/wfg/${N_OBJ}_obj/${PROBLEM}/${PREDICTOR}"
        [ -d "$SRC" ] || continue
        DEST="${SLURM_SUBMIT_DIR}/results/obj_ga_pred/wfg/${N_OBJ}_obj/${PROBLEM}/${PREDICTOR}"
        mkdir -p "$DEST"
        rsync -a "$SRC/" "$DEST/"
        echo "Results copied to $DEST"
    done
done

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
