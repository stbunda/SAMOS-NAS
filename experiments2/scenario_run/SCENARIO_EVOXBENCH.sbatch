#!/bin/bash
# --------- SCENARIO_EVOXBENCH.sbatch ----------------------------------------------------------------------------------------------------------------
# The scenario_run campaign: S1-S8 (experiments2/scenario_run/scenarios.py)
# at a single 10% feasibility operating point, hard/soft mode x
# {nsga2, samos} x 7 handlers + random x 1 (scenario/mode default) handler.
# Termination is EVALUATION-based (('n_evals', N), never ('n_gen', N)) --
# see run_scenario.py's module docstring for why this matters (h1-rejection
# and SAMOS2's hard-gate DOE retries spend real evaluations beyond the
# nominal pop_size-per-generation rate; under n_gen termination those would
# be bought off-budget). Results ->
# results2/scenario_run/{sid}/{mode}/{method}/{handler}/seed_N.pkl.
#
# One array task == one (scenario, mode, method, handler) tuple, looping
# over all 20 seeds with one run_scenario.py call + one results rsync-back
# PER SEED (not one call for all 20 seeds). The scratch directory is
# destroyed when a task ends for any reason -- normal exit, walltime,
# preemption, node failure -- so per-seed rsync-back means a killed task
# only ever loses its current in-flight seed, not every seed it had already
# finished. Combined with run_scenario.py's own skip-if-exists check and the
# results pre-copy below, re-submitting a killed/interrupted array index
# resumes from the next missing seed rather than seed 0.
#
# Task table (240 tasks). Per (scenario, mode) cell -- 15 tasks:
#   random x scenario/mode-default handler (1; random has no selection
#     pressure or sampling strategy for a handler to act on)
#   nsga2 x the 7-handler row {h1-rejection, h2-static_penalty, h3-adaptive,
#     h4-cdp, h5-epsilon, h6-DSR, b0-as-obj} (7)
#   samos x the same 7-handler row (7)
# Totals: 8 scenarios x 2 modes x 15 = 240.
#
# Index map: each (scenario, mode) block is 15 CONTIGUOUS tasks, blocks in
# scenario-major, mode-minor order (S1/hard, S1/soft, S2/hard, S2/soft, ...,
# S8/hard, S8/soft); within a block, item 0 is random:default, items 1-7 are
# nsga2 x the 7-handler row (in HANDLER_ROW order), items 8-14 are
# samos x the same 7-handler row:
#
#   block(sid, mode) = (scenario_idx * 2 + mode_idx) * 15
#     scenario_idx: S1=0, S2=1, ..., S8=7 ; mode_idx: hard=0, soft=1
#   task = block(sid, mode) + item   (item: 0=random, 1-7=nsga2 row, 8-14=samos row)
#
# e.g. index   0 = S1/hard/random/h4-cdp        (block 0,  item 0)
#      index   1 = S1/hard/nsga2/h1-rejection   (block 0,  item 1)
#      index   7 = S1/hard/nsga2/b0-as-obj      (block 0,  item 7)
#      index   8 = S1/hard/samos/h1-rejection   (block 0,  item 8)
#      index  15 = S1/soft/random/h2-static_penalty (block 15, item 0)
#      index  30 = S2/hard/random/h4-cdp        (block 30, item 0)
#      index 210 = S8/hard/random/h4-cdp        (block 210, item 0)
#      index 225 = S8/soft/random/h2-static_penalty (block 225, item 0)
#      index 239 = S8/soft/samos/b0-as-obj      (block 225, item 14; last task)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments2/scenario_run/SCENARIO_EVOXBENCH.sbatch
#
# Submit one (scenario, mode) cell (15-blocks per the index map above):
#   sbatch --array=0-14    experiments2/scenario_run/SCENARIO_EVOXBENCH.sbatch   # S1 hard
#   sbatch --array=15-29   experiments2/scenario_run/SCENARIO_EVOXBENCH.sbatch   # S1 soft
#   sbatch --array=210-224 experiments2/scenario_run/SCENARIO_EVOXBENCH.sbatch   # S8 hard
#   sbatch --array=225-239 experiments2/scenario_run/SCENARIO_EVOXBENCH.sbatch   # S8 soft
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments2/scenario_run/SCENARIO_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments2/scenario_run/SCENARIO_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J scenario_run
#SBATCH -c 2
#SBATCH --array=0-239%20              # full campaign (240 tasks total, existing results are skipped)
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_scenario_run/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_scenario_run

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_EVALS=${N_EVALS:-1200}

# --------- (scenario, mode, method, handler) task table -----------------------------------------------------------------------------------------
# Mirrors run_scenario.py's scenarios.SCENARIOS / HANDLERS / DEFAULT_HANDLER
# -- keep in sync.
SCENARIOS_LIST="S1 S2 S3 S4 S5 S6 S7 S8"
MODES_LIST="hard soft"
HANDLER_ROW="h1-rejection h2-static_penalty h3-adaptive h4-cdp h5-epsilon h6-DSR b0-as-obj"
default_handler() {
    case "$1" in
        hard) echo h4-cdp ;;
        soft) echo h2-static_penalty ;;
    esac
}

COMBOS=()
for SCENARIO in $SCENARIOS_LIST; do
    for MODE in $MODES_LIST; do
        COMBOS+=("${SCENARIO}:${MODE}:random:$(default_handler "$MODE")")
        for METHOD in nsga2 samos; do
            for HANDLER in $HANDLER_ROW; do
                COMBOS+=("${SCENARIO}:${MODE}:${METHOD}:${HANDLER}")
            done
        done
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r SCENARIO MODE METHOD HANDLER <<< "$TASK_ENTRY"

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
    experiments2/scenario_run
)

# Pre-copy any already-computed results for this (scenario, mode, method,
# handler) so run_scenario.py's own skip-if-exists logic can resume
# mid-seed-range.
_SRC="results2/scenario_run/${SCENARIO}/${MODE}/${METHOD}/${HANDLER}"
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
echo "SCENARIO          = $SCENARIO"
echo "MODE              = $MODE"
echo "METHOD            = $METHOD"
echo "HANDLER           = $HANDLER"
echo "SEEDS             = 0-19"
echo "POP_SIZE          = $POP_SIZE"
echo "N_EVALS           = $N_EVALS"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "EVOXBENCH_DATA    = $EVOXBENCH_DATA"
echo "EVOXBENCH_MODEL   = $EVOXBENCH_MODEL"

# --------- run: one seed at a time, rsync-back after each ------------------------------------------------------------------------------------------------
DEST="${SLURM_SUBMIT_DIR}/results2/scenario_run/${SCENARIO}/${MODE}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- SCENARIO: $SCENARIO  MODE: $MODE  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED"

    python experiments2/scenario_run/run_scenario.py \
        --scenario     "$SCENARIO" \
        --mode         "$MODE" \
        --method       "$METHOD" \
        --handler      "$HANDLER" \
        --seeds        "$SEED" \
        --pop_size     "$POP_SIZE" \
        --n_evals      "$N_EVALS" \
        --results_root results2/scenario_run \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results2/scenario_run/${SCENARIO}/${MODE}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
