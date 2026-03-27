#!/bin/bash
# ─── WFG_EXTENDED_BENCHMARK.sbatch ───────────────────────────────────────────
# Run ALL comparison methods on a single WFG problem for 30 independent seeds:
#   Baselines  : random, nsga2
#   SAMOS      : samos-xgb-i{IPS}-g{NGI}, samos-rfr-i{IPS}-g{NGI}
#   MO-Bayesian: mosmac, parego, cobra
#   SSA-NSGA-II: ssa-nsga2-default, ssa-nsga2-rfr, ssa-nsga2-xgb
#   GPSAF-NSGA-II: gpsaf-default, gpsaf-rfr, gpsaf-xgb
#
# All methods share the same evaluation budget: pop_size × n_gen evaluations.
# After each seed, results are rsync'd back and analyze_pymoo_benchmark.py is
# run with --extended_plot to produce both moo_hv_igd.png and
# moo_hv_igd_extended.png (group-column layout for readability).
#
# Submit for one problem at a time:
#   sbatch --export=ALL,PROBLEM=wfg1 WFG_EXTENDED_BENCHMARK.sbatch
#
# Or use the wrapper script to submit all 9 problems sequentially:
#   bash submit_wfg_extended_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J wfg_extended
#SBATCH -c 2
# Uncomment ONE of the following:
#SBATCH --array=0-79              # Seeds 0-9   (8 methods × 10 seeds)
## --array=80-159            # Seeds 10-19 (8 methods × 10 seeds)
## --array=160-239           # Seeds 20-29 (8 methods × 10 seeds)
#SBATCH --dependency=afterok:$SLURM_ARRAY_JOB_ID%8  # Wait for prev seed (all 8 methods)
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs_wfg_extended/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_wfg_extended

# ─── tunable parameters ───────────────────────────────────────────────────────
N_OBJ=${N_OBJ:-2}
POP_SIZE=${POP_SIZE:-20}
N_GEN_INNER=${N_GEN_INNER:-20}
INNER_POP_SIZE=${INNER_POP_SIZE:-200}
PROBLEM=${PROBLEM:-wfg1}

# ─── derived parameters ───────────────────────────────────────────────────────
N_VAR=$(( 2 * (N_OBJ - 1) + 10 ))
TOTAL_BUDGET=$(( 100 * N_VAR ))
N_GEN=$(( TOTAL_BUDGET / POP_SIZE ))

SAMOS_XGB="samos-xgb-i${INNER_POP_SIZE}-g${N_GEN_INNER}"
SAMOS_RFR="samos-rfr-i${INNER_POP_SIZE}-g${N_GEN_INNER}"
EXPERIMENT_NAME="pymoo_benchmark/${N_OBJ}_obj"

# ─── method/seed from array task ID ──────────────────────────────────────────
METHODS=(parego cobra ssa-nsga2-default ssa-nsga2-rfr ssa-nsga2-xgb gpsaf-default gpsaf-rfr gpsaf-xgb)
N_METHODS=${#METHODS[@]}   # 8
SEED=$(( SLURM_ARRAY_TASK_ID / N_METHODS ))
METHOD_IDX=$(( SLURM_ARRAY_TASK_ID % N_METHODS ))
METHOD=${METHODS[$METHOD_IDX]}

# ─── scratch setup ─────────────────────────────────────────────────────────────
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "$SCRATCH_DIRECTORY" ]; then
    echo "'$SCRATCH_DIRECTORY' already found!"
else
    echo "'$SCRATCH_DIRECTORY' not found, creating!"
    mkdir -p "$SCRATCH_DIRECTORY"
fi

RESULTS_SRC="results/${EXPERIMENT_NAME}/${PROBLEM}"
RSYNC_SOURCES=(
    analysis
    problem/pymoo
    strategy
    utils
    main_pymoo_benchmark.py
    analyze_pymoo_benchmark.py
)
[ -d "$RESULTS_SRC" ] && RSYNC_SOURCES+=("$RESULTS_SRC")
rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/"

cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 3 -type d

mkdir -p slurm_logs_wfg_extended

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
echo "METHOD            = $METHOD"
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
    --methods       "$METHOD" \
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
cd "${SLURM_SUBMIT_DIR}"

flock --wait 60 \
    "${SLURM_SUBMIT_DIR}/results/${EXPERIMENT_NAME}/${PROBLEM}/.analyze.lock" \
    python analyze_pymoo_benchmark.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --problems        "$PROBLEM" \
        --n_obj           "$N_OBJ" \
        --n_var           "$N_VAR" \
        --methods         \
            random nsga2 \
            "$SAMOS_XGB" "$SAMOS_RFR" \
            mosmac parego cobra \
            ssa-nsga2-default ssa-nsga2-rfr ssa-nsga2-xgb \
            gpsaf-default gpsaf-rfr gpsaf-xgb \
        --pop_size        "$POP_SIZE" \
        --n_gen           "$N_GEN" \
        --checkpoints_d   10 25 50 100 \
        --extended_plot \
    && echo "Analysis updated for seed ${SEED}" \
    || echo "[WARN] Analysis failed or timed out for seed ${SEED} (non-fatal)"
