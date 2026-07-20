#!/bin/bash
# --------- MULTI_CONSTRAINT_EVOXBENCH.sbatch ---------------------------------------------------------------------------------------------------------
# The multi-constraint scenarios s5-s12 of the constraint2 campaign
# (experiments/constraint2/run_constraint.py): every constraint-TYPE
# combination -- CC (#Params+FLOPs), CE (#Params+latency), EE
# (latency+energy), ALL (every constrained metric at once) -- each in both
# hard and soft mode, at the same per-metric ~Q25 thresholds as s1-s4
# (experiments/constraint2/THRESHOLDS.md, multi-constraint section), the
# same B1200_P20 budget and seeds 0-19, so every cell seed-pairs against
# the single-constraint campaign. Results -> results/constraint2/
# {scenario}/{suite}/pid{pid}/{budget}/{method}/{handler}/seed_N.pkl -- the
# SAME tree as CONSTRAINT2_EVOXBENCH.sbatch (s5-s12 are new scenario
# directories; nothing collides), one analysis pass covers both files.
#
# This is a SEPARATE file from CONSTRAINT2_EVOXBENCH.sbatch (whose 0-377
# index map is frozen); it owns its own 0-475 map.
#
# One array task == one (suite, pid, scenario, method, handler) tuple,
# looping over all 20 seeds with one run_constraint.py call + one results
# rsync-back PER SEED (not one call for all 20 seeds) -- a killed task only
# ever loses its in-flight seed; re-submitting resumes from the next
# missing seed (skip-if-exists + results pre-copy below).
#
# Task table (476 tasks). Instance sets per combo (run_constraint.py,
# SCENARIO_INSTANCES; EE/ALL need >= 2 expensive metric columns):
#   cc/ce  (s5-s8):   c10mop:4, c10mop:5, in1kmop:9, citysegmop:5
#   ee/all (s9-s12):  c10mop:5, citysegmop:5, citysegmop:10
# Per (scenario, instance) cell -- 17 tasks:
#   random x scenario-default handler (1)
#   samos / samos-cheap x the full multi row {h1-rejection, h2-penalty,
#     h3-adaptive-penalty, h4-cdp, h5-eps, h6-sr, b0-as-obj, b0-nsga2} (16)
#     -- the full HANDLER_ROW minus the h1-cdp-reject hybrid, plus BOTH b0
#     inner-GA variants: at 4-6 search objectives the inner GA may start to
#     matter, so b0-nsga2 is a first-class row here, not a control.
# Totals: s5-s8 4x17=68 each, s9-s12 3x17=51 each -> 4*68 + 4*51 = 476.
#
# Index map (each block contiguous, instances in the order above):
#     0- 67  s5  (hard, #Params+FLOPs)          on the cc/ce set
#    68-135  s6  (soft, #Params+FLOPs)
#   136-203  s7  (hard, #Params+latency)
#   204-271  s8  (soft, #Params+latency)
#   272-322  s9  (hard, latency+energy)         on the ee/all set
#   323-373  s10 (soft, latency+energy)
#   374-424  s11 (hard, all four metrics)
#   425-475  s12 (soft, all four metrics)
# Within a cell: random, samos x {h1-rejection, h2-penalty,
# h3-adaptive-penalty, h4-cdp, h5-eps, h6-sr, b0-as-obj, b0-nsga2},
# samos-cheap x the same eight.
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint2/MULTI_CONSTRAINT_EVOXBENCH.sbatch
#
# Submit one scenario block:
#   sbatch --array=136-203%10 experiments/constraint2/MULTI_CONSTRAINT_EVOXBENCH.sbatch   # s7 (CE hard)
#
# Priority order if capacity is tight (direct 2x test of the headline b0
# claim first): CE (136-271) > ALL (374-475) > EE (272-373) > CC (0-135).
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint2/MULTI_CONSTRAINT_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments/constraint2/MULTI_CONSTRAINT_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J multiconstr2
#SBATCH -c 2
#SBATCH --array=0-475%20              # full multi-constraint block (476 tasks, existing results are skipped)
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
# Mirrors run_constraint.py's SCENARIO_INSTANCES, MULTI_HANDLER_ROW and
# DEFAULT_HANDLER -- keep in sync.
default_handler() {
    case "$1" in
        s5|s7|s9|s11)  echo h4-cdp ;;
        s6|s8|s10|s12) echo h2-penalty ;;
    esac
}
MULTI_HANDLER_ROW="h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h5-eps h6-sr b0-as-obj b0-nsga2"
CCCE_INSTANCES="c10mop:4 c10mop:5 in1kmop:9 citysegmop:5"
EEALL_INSTANCES="c10mop:5 citysegmop:5 citysegmop:10"

instances_for() {
    case "$1" in
        s5|s6|s7|s8)     echo "$CCCE_INSTANCES" ;;
        s9|s10|s11|s12)  echo "$EEALL_INSTANCES" ;;
    esac
}

COMBOS=()
for SCENARIO in s5 s6 s7 s8 s9 s10 s11 s12; do
    for INSTANCE in $(instances_for "$SCENARIO"); do
        COMBOS+=("${INSTANCE}:${SCENARIO}:random:$(default_handler "$SCENARIO")")
        for METHOD in samos samos-cheap; do
            for HANDLER in $MULTI_HANDLER_ROW; do
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
