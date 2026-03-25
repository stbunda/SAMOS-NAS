#!/bin/bash
# ─── WFG_BENCHMARK.sbatch ─────────────────────────────────────────────────────
# Run Random / NSGA-II / SAMOS-XGB / SAMOS-RFR / MOSMAC on a single WFG problem
# for 30 independent seeds.  After each seed's results are copied back to the
# submit directory, analyze_pymoo_benchmark.py is run there so plots/tables
# accumulate incrementally and can be inspected while the job is running.
#
# Submit for one problem at a time:
#   sbatch --export=ALL,PROBLEM=wfg1 WFG_BENCHMARK.sbatch
#
# Or use the wrapper script to submit all 9 problems at once:
#   bash submit_wfg_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J wfg_benchmark
#SBATCH -c 2
#SBATCH --array=0-29           # 30 seeds; SLURM cap ≤ 100 tasks
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs_wfg_benchmark/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_wfg_benchmark

# ─── tunable parameters ───────────────────────────────────────────────────────
N_OBJ=${N_OBJ:-2}
POP_SIZE=${POP_SIZE:-20}
N_GEN_INNER=${N_GEN_INNER:-20}
INNER_POP_SIZE=${INNER_POP_SIZE:-200}
# PROBLEM is injected via --export=ALL,PROBLEM=wfg1 at submission time
PROBLEM=${PROBLEM:-wfg1}

# ─── derived parameters ───────────────────────────────────────────────────────
# WFG: n_var = 2*(n_obj-1) + 10; budget = 100*n_var; n_gen = budget / pop_size
N_VAR=$(( 2 * (N_OBJ - 1) + 10 ))
TOTAL_BUDGET=$(( 100 * N_VAR ))
N_GEN=$(( TOTAL_BUDGET / POP_SIZE ))

SEED=${SLURM_ARRAY_TASK_ID}

SAMOS_XGB="samos-xgb-i${INNER_POP_SIZE}-g${N_GEN_INNER}"
SAMOS_RFR="samos-rfr-i${INNER_POP_SIZE}-g${N_GEN_INNER}"
EXPERIMENT_NAME="pymoo_benchmark/${N_OBJ}_obj"

# ─── scratch setup ─────────────────────────────────────────────────────────────
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "$SCRATCH_DIRECTORY" ]; then
    echo "'$SCRATCH_DIRECTORY' already found!"
else
    echo "'$SCRATCH_DIRECTORY' not found, creating!"
    mkdir -p "$SCRATCH_DIRECTORY"
fi

rsync -aR \
    analysis \
    problem/pymoo \
    strategy \
    utils \
    main_pymoo_benchmark.py \
    analyze_pymoo_benchmark.py \
    "$SCRATCH_DIRECTORY/"

# Pre-populate scratch with any already-computed results for this problem so
# that seeds that are already done are skipped (when --overwrite is not set).
RESULTS_SRC="results/${EXPERIMENT_NAME}/${PROBLEM}"
if [ -d "$RESULTS_SRC" ]; then
    rsync -a "$RESULTS_SRC/" "${SCRATCH_DIRECTORY}/${RESULTS_SRC}/"
fi

cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 3 -type d

mkdir -p slurm_logs_wfg_benchmark

# ─── environment ──────────────────────────────────────────────────────────────
module load nvidia/nvhpc/23.3
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

echo "Activated: SAMOS311"

# ─── logging ──────────────────────────────────────────────────────────────────
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"
echo "PROBLEM           = $PROBLEM"
echo "SEED              = $SEED"
echo "N_OBJ             = $N_OBJ"
echo "N_VAR             = $N_VAR"
echo "N_GEN             = $N_GEN"
echo "POP_SIZE          = $POP_SIZE"
echo "BUDGET            = $TOTAL_BUDGET"
echo "SAMOS_XGB         = $SAMOS_XGB"
echo "SAMOS_RFR         = $SAMOS_RFR"

# ─── run ──────────────────────────────────────────────────────────────────────
python main_pymoo_benchmark.py \
    --problem       "$PROBLEM" \
    --n_obj         "$N_OBJ" \
    --n_var         "$N_VAR" \
    --methods       random nsga2 "$SAMOS_XGB" "$SAMOS_RFR" mosmac \
    --seeds         "$SEED" \
    --pop_size      "$POP_SIZE" \
    --n_gen         "$N_GEN" \
    --experiment_name "$EXPERIMENT_NAME" \
    --no_plot \
#    --overwrite

# ─── copy results back ────────────────────────────────────────────────────────
DEST="${SLURM_SUBMIT_DIR}/results/${EXPERIMENT_NAME}/${PROBLEM}"
mkdir -p "$DEST"
rsync -a \
    "${SCRATCH_DIRECTORY}/results/${EXPERIMENT_NAME}/${PROBLEM}/" \
    "$DEST/"

echo "Results copied to $DEST"

# ─── inline analysis (runs in submit dir so plots update incrementally) ───────
# Multiple seeds may finish concurrently; flock serialises analyze per-problem
# so plot files are never half-written.
cd "${SLURM_SUBMIT_DIR}"

flock --wait 60 \
    "${SLURM_SUBMIT_DIR}/results/${EXPERIMENT_NAME}/${PROBLEM}/.analyze.lock" \
    python analyze_pymoo_benchmark.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --problems        "$PROBLEM" \
        --n_obj           "$N_OBJ" \
        --n_var           "$N_VAR" \
        --methods         random nsga2 "$SAMOS_XGB" "$SAMOS_RFR" mosmac \
        --pop_size        "$POP_SIZE" \
        --n_gen           "$N_GEN" \
        --checkpoints_d   10 25 50 100 \
    && echo "Analysis updated for seed ${SEED}" \
    || echo "[WARN] Analysis failed or timed out for seed ${SEED} (non-fatal)"

echo "Date (done)       = $(date)"
