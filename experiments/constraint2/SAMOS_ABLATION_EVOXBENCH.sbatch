#!/bin/bash
# --------- SAMOS_ABLATION_EVOXBENCH.sbatch ----------------------------------------------------------------------------------------------------------
# The two MISSING corners of the samos vs samos-cheap 2x2 exact/predicted
# ablation (experiments/constraint2/run_samos_ablation.py, Experiment 2 of
# analyse_samos.py). samos = (FLOPs predicted, constraint predicted) and
# samos-cheap = (FLOPs exact, constraint exact) are already on disk from the
# main campaign; this job produces the off-diagonal:
#
#     samos-cheapflops  : FLOPs EXACT,     constraint PREDICTED
#     samos-cheapconstr : FLOPs PREDICTED, constraint EXACT
#
# Only the cheap single-constraint scenarios s1 (hard, #Params) and s3 (soft,
# #Params) qualify -- there the constraint is structural and so free to make
# exact. Same B1200_P20 budget, same seeds 0-19 and the SAME scenario-default
# handler (h4-cdp hard / h2-penalty soft) as the campaign, so all four 2x2
# corners seed-pair blockwise in analyse_samos.py. The handler is held FIXED
# so the measured axis effect is not confounded by it.
#
# Results -> results/constraint2/samos_ablation/{scenario}/{suite}/pid{pid}/{budget}/
# {method}/{handler}/seed_N.pkl -- a DEDICATED tree apart from the campaign's
# results/constraint2, so all four 2x2 corners sit together for analysis. The
# two diagonal reference corners (samos, samos-cheap) are NOT re-run; copy
# them in from the campaign once (scenario-default handlers) so analysis sees
# all four in one root:
#   for S in s1 s3; do H=$([ "$S" = s1 ] && echo h4-cdp || echo h2-penalty)
#     for I in c10mop/pid4 c10mop/pid5 in1kmop/pid9 citysegmop/pid5; do
#       for M in samos samos-cheap; do
#         mkdir -p results/constraint2/samos_ablation/$S/$I/B1200_P20/$M/$H
#         cp -r results/constraint2/$S/$I/B1200_P20/$M/$H/. \
#               results/constraint2/samos_ablation/$S/$I/B1200_P20/$M/$H/; done; done; done
#
# This is a SEPARATE file with its own 0-15 index map (the campaign files'
# maps are frozen). One array task == one (suite, pid, scenario, method)
# tuple, looping over all 20 seeds with one run_samos_ablation.py call + one
# results rsync-back PER SEED -- a killed task only ever loses its in-flight
# seed; re-submitting resumes from the next missing seed (run_samos_ablation's
# skip-if-exists + the results pre-copy below).
#
# Task table (16 tasks). Cheap instance set (run_constraint.py
# SCENARIO_INSTANCES['cheap']): c10mop:4, c10mop:5, in1kmop:9, citysegmop:5.
# Per (scenario, instance) cell -- 2 tasks (the two ablation corners); random
# and the diagonal corners are NOT re-run (already on disk).
# Index map (each block contiguous, instances in the order above, corners in
# the order samos-cheapflops, samos-cheapconstr):
#     0- 7  s1  (hard, #Params, h4-cdp)
#     8-15  s3  (soft, #Params, h2-penalty)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint2/SAMOS_ABLATION_EVOXBENCH.sbatch
#
# Submit one scenario block:
#   sbatch --array=0-7%8 experiments/constraint2/SAMOS_ABLATION_EVOXBENCH.sbatch    # s1 (hard)
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint2/SAMOS_ABLATION_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=3 experiments/constraint2/SAMOS_ABLATION_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J samosablate
#SBATCH -c 2
#SBATCH --array=0-15%8                # full ablation block (16 tasks, existing results are skipped)
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_constraint2/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_constraint2

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# --------- (suite, pid, scenario, method) task table --------------------------------------------------------------------------------------
# Mirrors run_samos_ablation.py: cheap scenarios s1/s3, the cheap instance
# set, the two ablation corners, scenario-default handler -- keep in sync.
default_handler() {
    case "$1" in
        s1) echo h4-cdp ;;
        s3) echo h2-penalty ;;
    esac
}
ABLATION_METHODS="samos-cheapflops samos-cheapconstr"
CHEAP_INSTANCES="c10mop:4 c10mop:5 in1kmop:9 citysegmop:5"

COMBOS=()
for SCENARIO in s1 s3; do
    for INSTANCE in $CHEAP_INSTANCES; do
        for METHOD in $ABLATION_METHODS; do
            COMBOS+=("${INSTANCE}:${SCENARIO}:${METHOD}")
        done
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r SUITE PID SCENARIO METHOD <<< "$TASK_ENTRY"
HANDLER="$(default_handler "$SCENARIO")"

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
    experiments/constraint2
)

# Pre-copy any already-computed results for this (scenario, suite, pid,
# method, handler) so run_samos_ablation.py's own skip-if-exists logic can
# resume mid-seed-range.
BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
_SRC="results/constraint2/samos_ablation/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
[ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/" 2>/dev/null || true

# EvoXBench SQLite database (~2.8G): copied to node-local scratch because the
# tabular suites issue one ORM query PER ARCHITECTURE (thousands of tiny reads
# per outer generation) -- brutal over the shared filesystem, cheap on /local.
# Separate rsync with -L (--copy-links): problem/data/evoxbench may be a
# symlink/junction to another checkout, and the scratch copy must be real
# files, not a link back to the shared FS.
rsync -aRL problem/data/evoxbench/database "$SCRATCH_DIRECTORY/" 2>/dev/null || true

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

# --------- point evoxbench's SQLite at the scratch copy -------------------------------------------------------------------------------
# Exporting these BEFORE python starts wins over the shared config.json with
# no race between array tasks; PYTHONPATH puts the scratch database dir ahead
# of the json's sys.path.append, so ORM + db.sqlite3 resolve from /local
# scratch while EVOXBENCH_MODEL keeps the one-time-loaded model files on the
# shared FS (see problem/evoxbench/utils.py).
export PYTHONPATH="${SCRATCH_DIRECTORY}/problem/data/evoxbench/database${PYTHONPATH:+:$PYTHONPATH}"
export EVOXBENCH_DATA="${SCRATCH_DIRECTORY}/problem/data/evoxbench/database"
export EVOXBENCH_MODEL="${SLURM_SUBMIT_DIR}/problem/data/evoxbench/data"

echo "Activated: SAMOS311"

# --------- logging -------------------------------------------------------------------------------------------------------------------------------------------
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"
echo "Array Task ID     = $SLURM_ARRAY_TASK_ID"
echo "SUITE             = $SUITE"
echo "PID               = $PID"
echo "SCENARIO          = $SCENARIO"
echo "METHOD            = $METHOD"
echo "HANDLER           = $HANDLER"
echo "SEEDS             = 0-19"
echo "POP_SIZE          = $POP_SIZE"
echo "N_GEN             = $N_GEN"
echo "N_GEN_INNER       = $N_GEN_INNER"
echo "BUDGET            = $(( N_GEN * POP_SIZE ))"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "EVOXBENCH_DATA    = $EVOXBENCH_DATA"
echo "EVOXBENCH_MODEL   = $EVOXBENCH_MODEL"

# --------- run: one seed at a time, rsync-back after each ------------------------------------------------------------------------------------------------
DEST="${SLURM_SUBMIT_DIR}/results/constraint2/samos_ablation/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- SUITE: $SUITE  PID: $PID  SCENARIO: $SCENARIO  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED"

    python experiments/constraint2/run_samos_ablation.py \
        --scenarios     "$SCENARIO" \
        --instances     "${SUITE}/${PID}" \
        --methods       "$METHOD" \
        --handler       "$HANDLER" \
        --seeds         "$SEED" \
        --pop_size      "$POP_SIZE" \
        --n_gen         "$N_GEN" \
        --n_gen_inner   "$N_GEN_INNER" \
        --results_root  results/constraint2/samos_ablation \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results/constraint2/samos_ablation/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
