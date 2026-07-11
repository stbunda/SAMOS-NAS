#!/bin/bash
# --------- CONSTRAINT_EVOXBENCH.sbatch ------------------------------------------------------------------------------------------------------------
# Runs experiments/constraint/run_constraint.py (S1-S4 constraint-scenario
# campaign: random / samos / samos-cheap on c10mop/pid4 AND in1kmop/pid9,
# objectives {Err, FLOPs}, one inequality constraint per scenario -- see
# text/plans/EXPERIMENT_PLAN_constraint_scenarios.md and
# experiments/constraint/THRESHOLDS.md), 20 seeds each, at the
# compare_methods.py budget (pop_size=20, n_gen=60, n_gen_inner=20,
# ga_pop_size=pop_size*10).
#
# Cloned from experiments/samos/COMPARE_METHODS_EVOXBENCH.sbatch: one array
# task == one (suite, pid, scenario, method, handler) tuple, looping over all
# 20 seeds with one run_constraint.py call + one results rsync-back PER SEED
# (not one call for all 20 seeds). The scratch directory is destroyed when a
# task ends for any reason -- normal exit, walltime, preemption, node
# failure -- so per-seed rsync-back means a killed task only ever loses its
# current in-flight seed, not every seed it had already finished. Combined
# with run_constraint.py's own skip-if-exists check and the results pre-copy
# below, re-submitting a killed/interrupted array index resumes from the
# next missing seed rather than seed 0.
#
# Task count: 100 = 24 round-1 default-handler tasks + 76 round-2 handler
# additions (plan Round 2 / D22), APPENDED after so indices 0-23 keep their
# meaning (they now explicitly name the scenario-default handler: s1/s2 ->
# h4-cdp, s3/s4 -> h2-penalty -- same runs as before, new {handler} output
# level). Round-2 additions per scenario: random x h1-rejection, then
# samos / samos-cheap x the scenario's NON-default D22 handlers
# (s1: h1,h2,h5,h6 | s2: h1,h2,h3,h5,h6 | s3/s4: h1,h3,h4,h6); `random`
# skips all other handlers (no selection pressure, D22).
#
# Index map (each scenario block contiguous):
#   ROUND-1 DEFAULTS (24):
#     0- 2  c10mop/pid4  s1 (random,samos,samos-cheap x h4-cdp)
#     3- 5  c10mop/pid4  s2 (x h4-cdp)      6- 8  c10mop/pid4  s3 (x h2-penalty)
#     9-11  c10mop/pid4  s4 (x h2-penalty) 12-14  in1kmop/pid9 s1 (x h4-cdp)
#    15-17  in1kmop/pid9 s2 (x h4-cdp)     18-20  in1kmop/pid9 s3 (x h2-penalty)
#    21-23  in1kmop/pid9 s4 (x h2-penalty)
#   ROUND-2 ADDITIONS (76): per scenario: 1 random:h1 + 2 methods x non-default handlers
#    24-32  c10mop/pid4  s1 (9: random:h1; samos x h1,h2,h5,h6; samos-cheap x h1,h2,h5,h6)
#    33-43  c10mop/pid4  s2 (11: random:h1; samos x h1,h2,h3,h5,h6; samos-cheap x same)
#    44-52  c10mop/pid4  s3 (9: random:h1; samos x h1,h3,h4,h6; samos-cheap x same)
#    53-61  c10mop/pid4  s4 (9: same shape as s3)
#    62-70  in1kmop/pid9 s1 (9)   71-81  in1kmop/pid9 s2 (11)
#    82-90  in1kmop/pid9 s3 (9)   91-99  in1kmop/pid9 s4 (9)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#
# Submit only the round-1 defaults / only the round-2 additions:
#   sbatch --array=0-23%6   experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#   sbatch --array=24-99%10 experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#
# Submit a single scenario (defaults + additions, per the index map):
#   e.g. c10mop/pid4 s2 defaults: --array=3-5 ; its handler additions: --array=33-43
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J constraint_s1s4
#SBATCH -c 2
#SBATCH --array=0-99%10             # 100 tasks (suite,pid,scenario,method,handler), 10 concurrent
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_constraint/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_constraint

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# --------- (suite, pid, scenario, method, handler) task table --------------------------------------------------------------------------------------
# Mirrors run_constraint.py's THRESHOLDS instances, SCENARIOS dict, METHODS
# list, DEFAULT_HANDLER, and SCENARIO_HANDLERS (D22 matrix) -- keep in sync.
# Round-1 block first (indices 0-23 preserve their meaning; the handler
# field just names each scenario's default explicitly), round-2 handler
# additions APPENDED after (COMPARE_METHODS 2026-07-06-additions pattern).
default_handler() {
    case "$1" in
        s1|s2) echo h4-cdp ;;
        s3|s4) echo h2-penalty ;;
    esac
}
nondefault_handlers() {    # = SCENARIO_HANDLERS row minus the default
    case "$1" in
        s1)    echo "h1-rejection h2-penalty h5-eps h6-sr" ;;
        s2)    echo "h1-rejection h2-penalty h3-adaptive-penalty h5-eps h6-sr" ;;
        s3|s4) echo "h1-rejection h3-adaptive-penalty h4-cdp h6-sr" ;;
    esac
}

COMBOS=()
# indices 0-23: round-1 tasks, scenario-default handler (order unchanged)
for INSTANCE in c10mop:4 in1kmop:9; do
    for SCENARIO in s1 s2 s3 s4; do
        for METHOD in random samos samos-cheap; do
            COMBOS+=("${INSTANCE}:${SCENARIO}:${METHOD}:$(default_handler "$SCENARIO")")
        done
    done
done
# indices 24-99: round-2 handler additions (random runs only h1 beyond its
# default -- D22; samos/samos-cheap run every non-default handler of the row)
for INSTANCE in c10mop:4 in1kmop:9; do
    for SCENARIO in s1 s2 s3 s4; do
        COMBOS+=("${INSTANCE}:${SCENARIO}:random:h1-rejection")
        for METHOD in samos samos-cheap; do
            for HANDLER in $(nondefault_handlers "$SCENARIO"); do
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
    experiments/samos
    experiments/constraint
)

# Pre-copy any already-computed results for this (suite, pid, scenario,
# method, handler) so run_constraint.py's own skip-if-exists logic can
# resume mid-seed-range.
BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
_SRC="results/constraint/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
[ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/" 2>/dev/null || true

# EvoXBench SQLite database (~2.8G): copied to node-local scratch because
# NB101/NATS issue one ORM query PER ARCHITECTURE (thousands of tiny reads
# per outer generation) -- brutal over the shared filesystem, cheap on
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
DEST="${SLURM_SUBMIT_DIR}/results/constraint/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- SUITE: $SUITE  PID: $PID  SCENARIO: $SCENARIO  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED"

    python experiments/constraint/run_constraint.py \
        --scenario      "$SCENARIO" \
        --suite         "$SUITE" \
        --pid           "$PID" \
        --method        "$METHOD" \
        --handler       "$HANDLER" \
        --seeds         "$SEED" \
        --pop_size      "$POP_SIZE" \
        --n_gen         "$N_GEN" \
        --n_gen_inner   "$N_GEN_INNER" \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results/constraint/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
