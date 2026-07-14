#!/bin/bash
# --------- CONSTRAINT2_EVOXBENCH.sbatch -----------------------------------------------------------------------------------------------------------
# The constraint2 campaign: the full hard/soft x cheap/expensive 2x2
# (experiments/constraint2/run_constraint.py) at ~Q25 thresholds
# (experiments/constraint2/THRESHOLDS.md), objectives {Err., FLOPs}
# everywhere. The mode axis is physical: hard scenarios (s1, s2) gate
# evaluability (infeasible high-fidelity evaluations are counted as waste
# and discarded -- objective surrogates train on feasible points only, the
# constraint surrogate on archive + rejection log); soft scenarios (s3, s4)
# archive infeasible points with raw F/G. Results ->
# results/constraint2/{scenario}/{suite}/pid{pid}/{budget}/{method}/
# {handler}/seed_N.pkl (no objective-tag path level: one fixed objective
# set, fresh tree).
#
# One array task == one (suite, pid, scenario, method, handler) tuple,
# looping over all 20 seeds with one run_constraint.py call + one results
# rsync-back PER SEED (not one call for all 20 seeds). The scratch directory
# is destroyed when a task ends for any reason -- normal exit, walltime,
# preemption, node failure -- so per-seed rsync-back means a killed task
# only ever loses its current in-flight seed, not every seed it had already
# finished. Combined with run_constraint.py's own skip-if-exists check and
# the results pre-copy below, re-submitting a killed/interrupted array index
# resumes from the next missing seed rather than seed 0.
#
# Task table (378 tasks). Instance sets per constraint type (see
# run_constraint.py, 'Instances'):
#   cheap (s1/s3):     c10mop:4, c10mop:5, in1kmop:9, citysegmop:5
#   expensive (s2/s4): the same four + citysegmop:10 (same MoSegNAS search
#                      space under a second device, H2 latency; excluded
#                      from the cheap scenarios where, without its latency
#                      column, it duplicates citysegmop:5 exactly)
# Per (scenario, instance) cell -- 21 tasks:
#   random x scenario-default handler (1; random runs nothing else: no
#     selection pressure, no sampling strategy, no replacement)
#   samos / samos-cheap x the shared 8-handler row {h1-rejection,
#     h2-penalty, h3-adaptive-penalty, h4-cdp, h5-eps, h6-sr,
#     h1-cdp-reject, b0-as-obj} (16)
#   samos / samos-cheap x the two inner-GA controls {h4-cdp-sms, b0-nsga2}
#     (4; complete the {formulation} x {inner GA} factorial)
# Totals: s1 4x21=84, s2 5x21=105, s3 4x21=84, s4 5x21=105 -> 378.
#
# NB101 does not appear in this campaign; the slow tasks are the citysegmop
# instances (RankNet/lookup predictor per evaluation) -- expect those to run
# longer than the tabular instances at the same budget.
#
# Index map (each block contiguous, instances in the order above):
#     0- 83  s1 (hard, #Params)  on c10mop:4, c10mop:5, in1kmop:9,
#            citysegmop:5 (21 per instance: random:h4-cdp, samos x 8-row,
#            samos-cheap x 8-row, samos x h4-cdp-sms, samos-cheap x
#            h4-cdp-sms, samos x b0-nsga2, samos-cheap x b0-nsga2)
#    84-188  s2 (hard, latency)  on c10mop:4, c10mop:5, in1kmop:9,
#            citysegmop:5, citysegmop:10 (21 per instance, same shape,
#            random:h4-cdp)
#   189-272  s3 (soft, #Params)  on the s1 instances (21 per instance,
#            random:h2-penalty)
#   273-377  s4 (soft, latency)  on the s2 instances (21 per instance,
#            random:h2-penalty)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint2/CONSTRAINT2_EVOXBENCH.sbatch
#
# Submit one scenario block:
#   sbatch --array=0-83%10    experiments/constraint2/CONSTRAINT2_EVOXBENCH.sbatch
#   sbatch --array=84-188%10  experiments/constraint2/CONSTRAINT2_EVOXBENCH.sbatch
#   sbatch --array=189-272%10 experiments/constraint2/CONSTRAINT2_EVOXBENCH.sbatch
#   sbatch --array=273-377%10 experiments/constraint2/CONSTRAINT2_EVOXBENCH.sbatch
#
# Submit a single (scenario, instance) cell (21-blocks per the index map):
#   e.g. s1 on c10mop:4 --array=0-20 ; s2 on citysegmop:10 --array=168-188
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint2/CONSTRAINT2_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments/constraint2/CONSTRAINT2_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J constraint2
#SBATCH -c 2
#SBATCH --array=0-377%20              # full campaign (378 tasks total, existing results are skipped)
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

# --------- (suite, pid, scenario, method, handler) task table --------------------------------------------------------------------------------------
# Mirrors run_constraint.py's SCENARIO_INSTANCES, HANDLER_ROW,
# CONTROL_HANDLERS and DEFAULT_HANDLER -- keep in sync.
default_handler() {
    case "$1" in
        s1|s2) echo h4-cdp ;;
        s3|s4) echo h2-penalty ;;
    esac
}
HANDLER_ROW="h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h5-eps h6-sr h1-cdp-reject b0-as-obj"
CONTROL_HANDLERS="h4-cdp-sms b0-nsga2"
CHEAP_INSTANCES="c10mop:4 c10mop:5 in1kmop:9 citysegmop:5"
EXPENSIVE_INSTANCES="c10mop:4 c10mop:5 in1kmop:9 citysegmop:5 citysegmop:10"

instances_for() {
    case "$1" in
        s1|s3) echo "$CHEAP_INSTANCES" ;;
        s2|s4) echo "$EXPENSIVE_INSTANCES" ;;
    esac
}

COMBOS=()
for SCENARIO in s1 s2 s3 s4; do
    for INSTANCE in $(instances_for "$SCENARIO"); do
        COMBOS+=("${INSTANCE}:${SCENARIO}:random:$(default_handler "$SCENARIO")")
        for METHOD in samos samos-cheap; do
            for HANDLER in $HANDLER_ROW; do
                COMBOS+=("${INSTANCE}:${SCENARIO}:${METHOD}:${HANDLER}")
            done
        done
        for HANDLER in $CONTROL_HANDLERS; do
            for METHOD in samos samos-cheap; do
                COMBOS+=("${INSTANCE}:${SCENARIO}:${METHOD}:${HANDLER}")
            done
        done
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r SUITE PID SCENARIO METHOD HANDLER <<< "$TASK_ENTRY"

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
# method, handler) so run_constraint.py's own skip-if-exists logic can
# resume mid-seed-range.
BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
_SRC="results/constraint2/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
[ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/" 2>/dev/null || true

# EvoXBench SQLite database (~2.8G): copied to node-local scratch because
# the tabular suites issue one ORM query PER ARCHITECTURE (thousands of tiny
# reads per outer generation) -- brutal over the shared filesystem, cheap on
# /local. Separate rsync with -L (--copy-links): problem/data/evoxbench may
# be a symlink/junction to another checkout, and the scratch copy must be
# real files, not a link back to the shared FS. The model/data dir (1.3G) is
# NOT copied: those files are loaded once per benchmark construction, so
# shared-FS latency is irrelevant there (EVOXBENCH_MODEL below keeps them on
# submit dir).
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
# evoxbench resolves its Django `ORM` package (and thus db.sqlite3, via that
# package's BASE_DIR) from sys.path, and only `setdefault`s EVOXBENCH_DATA /
# EVOXBENCH_MODEL from the shared ~/.config/evoxbench/config.json. Exporting
# these BEFORE python starts therefore wins over the shared json with no race
# between array tasks (the json itself keeps its submit-dir value, same as
# before -- see problem/evoxbench/utils.py). PYTHONPATH puts the scratch
# database dir ahead of the json's sys.path.append, so ORM + db.sqlite3
# resolve from /local scratch; EVOXBENCH_MODEL keeps the (one-time-loaded)
# model files on the shared FS, matching what utils.py configures.
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
DEST="${SLURM_SUBMIT_DIR}/results/constraint2/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- SUITE: $SUITE  PID: $PID  SCENARIO: $SCENARIO  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED"

    python experiments/constraint2/run_constraint.py \
        --scenario      "$SCENARIO" \
        --suite         "$SUITE" \
        --pid           "$PID" \
        --method        "$METHOD" \
        --handler       "$HANDLER" \
        --seeds         "$SEED" \
        --pop_size      "$POP_SIZE" \
        --n_gen         "$N_GEN" \
        --n_gen_inner   "$N_GEN_INNER" \
        --results_root  results/constraint2 \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results/constraint2/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
