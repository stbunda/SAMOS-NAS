#!/bin/bash
# --------- COMPARE_METHODS_EVOXBENCH.sbatch -------------------------------------------------------------------------------------------------------------
# Runs experiments/samos/compare_methods.py (random / samos-xgb / samos2-baseline /
# samos2-encoding / samos2-isomorphism / samos2-uncertainty) across every
# c10mop + in1kmop pid, 20 seeds each, at the depreciated/main_evoxbench.py
# budget (pop_size=20, n_gen=60, n_gen_inner=20, ga_pop_size=pop_size*10).
#
# One array task == one (suite, pid, method) triplet; each task loops over
# all 20 seeds internally (compare_methods.py already skips seeds whose
# output pkl exists, so re-submitting is safe without --overwrite).
#
# Task count: 2 suites x 9 pids x 5 methods, plus samos2-encoding restricted
# to the 7 c10mop pids with a categorical-op search space (NB101/NB201/NATS;
# DARTS pids 8-9 and all in1kmop pids have none) = 18*5 + 7 = 97 tasks.
# The COMBOS array below is built to match compare_methods.py's own
# samos2-encoding skip logic exactly -- keep the two in sync if either changes.
#
# Submit:
#   sbatch experiments/samos/COMPARE_METHODS_EVOXBENCH.sbatch
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/samos/COMPARE_METHODS_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=42 experiments/samos/COMPARE_METHODS_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J samos2_compare
#SBATCH -c 2
#SBATCH --array=0-96%10            # 97 tasks (suite,pid,method), 10 concurrent
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=24:00:00
#SBATCH --output=slurm/slurm_logs_samos2_compare/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_samos2_compare

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}
SEEDS=$(seq 0 19)

# --------- (suite, pid, method) task table -----------------------------------------------------------------------------------------------------------
# Mirrors compare_methods.py's METHODS list and its samos2-encoding
# get_op_var_group() skip (c10mop pids 1-7 = NB101/NATS/NB201; 8-9 = DARTS).
COMBOS=()
for SUITE in c10mop in1kmop; do
    for PID in 1 2 3 4 5 6 7 8 9; do
        for METHOD in random samos-xgb samos2-baseline samos2-isomorphism samos2-uncertainty; do
            COMBOS+=("${SUITE}:${PID}:${METHOD}")
        done
        if [ "$SUITE" = "c10mop" ] && [ "$PID" -le 7 ]; then
            COMBOS+=("${SUITE}:${PID}:samos2-encoding")
        fi
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r SUITE PID METHOD <<< "$TASK_ENTRY"

# --------- scratch setup -----------------------------------------------------------------------------------------------------------------------------------
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "$SCRATCH_DIRECTORY" ]; then
    echo "'$SCRATCH_DIRECTORY' already found!"
else
    echo "'$SCRATCH_DIRECTORY' not found, creating!"
    mkdir -p "$SCRATCH_DIRECTORY"
fi

RSYNC_SOURCES=(
    problem/evoxbench
    problem/pymoo
    strategy
    experiments/samos
)

# Pre-copy any already-computed results for this (suite, pid, method) so
# compare_methods.py's own skip-if-exists logic can resume mid-seed-range.
BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
_SRC="results/samos/compare/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}"
[ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/" 2>/dev/null || true

cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 4 -type d 2>/dev/null | head -20

# --------- environment ---------------------------------------------------------------------------------------------------------------------------------------
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

# CPU-only job: hide all GPUs so CUDA-enabled builds (xgboost, torch via
# evoxbench) cannot create a CUDA context / reserve VRAM on this shared node.
export CUDA_VISIBLE_DEVICES=""

echo "Activated: SAMOS311"

# --------- logging -------------------------------------------------------------------------------------------------------------------------------------------
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"
echo "Array Task ID     = $SLURM_ARRAY_TASK_ID"
echo "SUITE             = $SUITE"
echo "PID               = $PID"
echo "METHOD            = $METHOD"
echo "SEEDS             = 0-19"
echo "POP_SIZE          = $POP_SIZE"
echo "N_GEN             = $N_GEN"
echo "N_GEN_INNER       = $N_GEN_INNER"
echo "BUDGET            = $(( N_GEN * POP_SIZE ))"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"

# --------- run -------------------------------------------------------------------------------------------------------------------------------------------------
python experiments/samos/compare_methods.py \
    --suites        "$SUITE" \
    --pids          "$PID" \
    --methods       "$METHOD" \
    --seeds         $SEEDS \
    --pop_size      "$POP_SIZE" \
    --n_gen         "$N_GEN" \
    --n_gen_inner   "$N_GEN_INNER" \
    ${OVERWRITE:+--overwrite}

# --------- copy results back --------------------------------------------------------------------------------------------------------------------------------
DEST="${SLURM_SUBMIT_DIR}/results/samos/compare/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}"
mkdir -p "$DEST"
rsync -a "${SCRATCH_DIRECTORY}/results/samos/compare/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/" "$DEST/" 2>/dev/null || true
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
