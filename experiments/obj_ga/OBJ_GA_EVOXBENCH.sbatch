#!/bin/bash
# --------- OBJ_GA_EVOXBENCH.sbatch ------------------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1 (GA comparison under varying objectives) --- EvoXBench NAS suites.
#
# One array task == one seed. Each task loops over every METHOD x every PID in
# the (BENCHMARK, N_OBJ) sub-experiment. The Python entry point auto-skips
# already-completed pickles, so re-submitting an array is safe.
#
# Walltime is 72h: in1kmop uses surrogate evaluators and is slow.
#
# Submit one sub-experiment at a time via --export:
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=2 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=3 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=4 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=2 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=3 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=4 OBJ_GA_EVOXBENCH.sbatch
#
# Selective single-seed rerun (force overwrite):
#   sbatch --array=5 --export=ALL,BENCHMARK=in1kmop,N_OBJ=4,OVERWRITE=1 OBJ_GA_EVOXBENCH.sbatch
#
# PROBLEM RUNS (YAML parsing error - hidden control chars in config file):
# Issue: yaml.reader.ReaderError (character #x0094 at position 134)
# Root cause: File corruption during job launch or editor encoding issue
# Fix applied: Config file rewritten cleanly with proper UTF-8 encoding (no control chars)
# Rerun failed seeds with OVERWRITE=1:
#   sbatch --array=5-7,10-14 --export=ALL,BENCHMARK=c10mop,N_OBJ=3,OVERWRITE=1 experiments/obj_ga/OBJ_GA_EVOXBENCH.sbatch
#   sbatch --array=5-9,10-12,14-15,18-19 --export=ALL,BENCHMARK=c10mop,N_OBJ=4,OVERWRITE=1 experiments/obj_ga/OBJ_GA_EVOXBENCH.sbatch
#   sbatch --array=0,10-19 --export=ALL,BENCHMARK=in1kmop,N_OBJ=2,OVERWRITE=1 experiments/obj_ga/OBJ_GA_EVOXBENCH.sbatch
#   sbatch --array=5-9,10-15,17-19 --export=ALL,BENCHMARK=in1kmop,N_OBJ=3,OVERWRITE=1 experiments/obj_ga/OBJ_GA_EVOXBENCH.sbatch
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J obj_ga_evoxbench
#SBATCH -c 2
#SBATCH --array=0-19%2            # 20 tasks = 20 seeds, 2 concurrent
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=12:00:00
#SBATCH --output=slurm/slurm_logs_obj_ga_evoxbench/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_obj_ga_evoxbench

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
    experiments/obj_ga
)

# Pre-copy any already-computed results for EvoXBench (all suites and pids)
for _SUITE in c10mop in1kmop; do
    for _N_OBJ in 2 3 4; do
        _SRC="results/obj_ga/evoxbench/${_SUITE}/pid"*
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
CONFIG=experiments/obj_ga/config/experiment_obj_ga.yaml

# --------- method set ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
echo "METHODS           = ${METHODS[*]}"
echo "PIDS              = ${PIDS[*]}"

# --------- run: METHOD x PID ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for METHOD in "${METHODS[@]}"; do
    for PID in "${PIDS[@]}"; do
        echo ""
        echo "--------- benchmark: $BENCHMARK  n_obj: $N_OBJ  pid: $PID  method: $METHOD  seed: $SEED"

        python experiments/obj_ga/experiment_obj_ga.py \
            --benchmark "$BENCHMARK" \
            --pid       "$PID" \
            --n_obj     "$N_OBJ" \
            --method    "$METHOD" \
            --seed      "$SEED" \
            --config    "$CONFIG" \
            ${OVERWRITE:+--overwrite}

        # ------ copy results back ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        DEST="${SLURM_SUBMIT_DIR}/results/obj_ga/evoxbench/${BENCHMARK}/pid${PID}"
        mkdir -p "$DEST"
        rsync -a \
            "${SCRATCH_DIRECTORY}/results/obj_ga/evoxbench/${BENCHMARK}/pid${PID}/" \
            "$DEST/" 2>/dev/null || true
        echo "Results copied to $DEST"
    done
done

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
