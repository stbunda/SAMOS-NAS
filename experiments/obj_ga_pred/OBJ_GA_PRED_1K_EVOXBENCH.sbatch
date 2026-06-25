#!/bin/bash
# --------- OBJ_GA_PRED_1K_EVOXBENCH.sbatch -------------------------------------------------------------------------------------------------------------------------------
# Experiment 2 variant: GA performance under fixed predictors (1000-sample DOE) --- EvoXBench NAS suites.
#
# One array task == one seed. Each task loops over every PREDICTOR x METHOD x PID
# in the (BENCHMARK, N_OBJ) sub-experiment. The Python entry point auto-skips
# already-completed pickles, so re-submitting an array is safe.
#
# Walltime is 72h: in1kmop uses surrogate evaluators and is slow.
#
# Submit one sub-experiment at a time via --export:
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=2 experiments/obj_ga_pred/OBJ_GA_PRED_1K_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=4 experiments/obj_ga_pred/OBJ_GA_PRED_1K_EVOXBENCH.sbatch
#   ... and the remaining (BENCHMARK, N_OBJ) combinations.
#
# Restrict to one predictor (optional) via PREDICTORS:
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=2,PREDICTORS=rf experiments/obj_ga_pred/OBJ_GA_PRED_1K_EVOXBENCH.sbatch
#
# Selective single-seed rerun (force overwrite):
#   sbatch --array=5 --export=ALL,BENCHMARK=in1kmop,N_OBJ=4,OVERWRITE=1 experiments/obj_ga_pred/OBJ_GA_PRED_1K_EVOXBENCH.sbatch
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J obj_ga_pred_1k_evoxbench
#SBATCH -c 2
#SBATCH --array=0-19%4            # 20 tasks = 20 seeds, 4 concurrent
#SBATCH --mem=8gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=12:00:00
#SBATCH --output=slurm/slurm_logs_obj_ga_pred_evoxbench/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_obj_ga_pred_evoxbench

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

# Pre-copy any already-computed results for EvoXBench (all suites and pids)
for _SUITE in c10mop in1kmop; do
    for _N_OBJ in 2 3 4; do
        _SRC="results/obj_ga_pred/evoxbench/${_SUITE}/pid"*
        [ -d "$_SRC" ] 2>/dev/null && RSYNC_SOURCES+=("$_SRC")
    done
done

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/" 2>/dev/null || true

cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 4 -type d 2>/dev/null | head -20

# --------- environment ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

# CPU-only job: hide all GPUs so CUDA-enabled builds (xgboost, torch via
# evoxbench) cannot create a CUDA context / reserve VRAM on this shared node.
export CUDA_VISIBLE_DEVICES=""

echo "Activated: SAMOS311"

# --------- parameters from submission env ---------------------------------------------------------------------------------------------------------------------------------
BENCHMARK=${BENCHMARK:-c10mop}
N_OBJ=${N_OBJ:-2}
SEED=$SLURM_ARRAY_TASK_ID
CONFIG=experiments/obj_ga_pred/config/experiment_obj_ga_pred_1k.yaml

# --------- predictor / method sets ------------------------------------------------------------------------------------------------------------------------------------------------------
PREDICTORS=(${PREDICTORS:-xgboost rf rbf})
METHODS=(random nsga2 nsga3 moead sms-emoa rvea age-moea age-moea2)

# --------- PID set: depends on (BENCHMARK, N_OBJ) ---------------------------------------------------------------------------------------------------------
case "${BENCHMARK}_${N_OBJ}" in
    c10mop_2)  PIDS=(1 8) ;;
    c10mop_3)  PIDS=(2 3 9) ;;
    c10mop_4)  PIDS=(4 5 6 7) ;;
    in1kmop_2) PIDS=(1 2 4 5 7) ;;
    in1kmop_3) PIDS=(3 6 8) ;;
    in1kmop_4) PIDS=(9) ;;
    *)
        echo "ERROR: unsupported BENCHMARK/N_OBJ combination: ${BENCHMARK}_${N_OBJ}" >&2
        echo "       Expected BENCHMARK in {c10mop,in1kmop} and N_OBJ in {2,3,4}." >&2
        exit 1
        ;;
esac

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
echo "PIDS              = ${PIDS[*]}"

# --------- run: PREDICTOR x METHOD x PID ------------------------------------------------------------------------------------------------------------------------------------
for PREDICTOR in "${PREDICTORS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for PID in "${PIDS[@]}"; do
            echo ""
            echo "--------- benchmark: $BENCHMARK  n_obj: $N_OBJ  pid: $PID  predictor: $PREDICTOR  method: $METHOD  seed: $SEED"

            python experiments/obj_ga_pred/experiment_obj_ga_pred.py \
                --benchmark "$BENCHMARK" \
                --pid       "$PID" \
                --n_obj     "$N_OBJ" \
                --predictor "$PREDICTOR" \
                --method    "$METHOD" \
                --seed      "$SEED" \
                --config    "$CONFIG" \
                ${OVERWRITE:+--overwrite}

            # ------ copy results back ------------------------------------------------------------------------------------------------------------------------------------------
            SRC="${SCRATCH_DIRECTORY}/results/obj_ga_pred/evoxbench/${BENCHMARK}/pid${PID}/${PREDICTOR}"
            [ -d "$SRC" ] || continue
            DEST="${SLURM_SUBMIT_DIR}/results/obj_ga_pred/evoxbench/${BENCHMARK}/pid${PID}/${PREDICTOR}"
            mkdir -p "$DEST"
            rsync -a "$SRC/" "$DEST/" 2>/dev/null || true
            echo "Results copied to $DEST"
        done
    done
done

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
