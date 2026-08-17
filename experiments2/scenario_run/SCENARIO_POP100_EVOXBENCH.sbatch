#!/bin/bash
# --------- SCENARIO_POP100_EVOXBENCH.sbatch --------------------------------------------------------------------------------------------------
# Population-size arm of the scenario_run campaign: the SAME S1-S6 scenarios
# (experiments2/scenario_run/scenarios.py), the SAME 10% feasibility operating
# point and the SAME 1200-evaluation budget as the pop-20 campaign, run at
# POP_SIZE=100. 1200 / 100 = 12 outer generations, against 60 at pop 20.
#
# The question this arm answers: does the ranking of the seven handlers -- and
# of the methods that carry them -- survive a 5x wider population searched for
# a 5x shorter time? A handler that only pays off through many generations of
# accumulated selection pressure (h3-adaptive's weight schedule, h5-epsilon's
# decay clock, h6-DSR) has 12 generations here instead of 60, while a handler
# that mostly buys diversity in the initial draw (h1-rejection, b0-as-obj)
# gets 5x more of it per generation. Budget-matched by construction, so any
# difference is attributable to the pop/gen split rather than to spend.
#
# SEPARATE RESULTS TREE: results2/scenario_run_pop100/{sid}/{mode}/{method}/
# {handler}/seed_N.pkl -- NOT results2/scenario_run. The two arms differ only
# in pop_size, so writing them into one tree would silently overwrite the
# pop-20 pkls path-for-path (the path carries no pop_size). Analyse this arm
# with --results_root results2/scenario_run_pop100; every pkl also records its
# pop_size / n_evals / inner_pop_size in data['meta'], so the two trees stay
# distinguishable after the fact.
#
# INNER GA IS PINNED, NOT SCALED. samos derives its surrogate-side inner GA
# population from the outer pop by default (ga_pop_size = pop_size * 10, i.e.
# 1000 here against 200 at pop 20), so leaving it at the default would move the
# outer population AND the surrogate search width in the same arm and confound
# the comparison. INNER_POP_SIZE=200 below pins it to the pop-20 value via
# run_scenario.py's --inner_pop_size, leaving the outer population as the only
# thing this arm changes. n_doe and n_infill are NOT pinned: they are the outer
# batch itself (both default to pop_size), so they scale to 100 by definition --
# that IS the population size being varied. Methods with no surrogate-side inner
# GA (random, nsga2, ctaea) ignore the flag.
#
# Termination is EVALUATION-based (('n_evals', N), never ('n_gen', N)) -- see
# run_scenario.py's module docstring for why. It matters identically here: 12
# generations is the budget divided by the population, not a termination
# criterion, so handlers that spend evaluations on infeasible probes
# (h1-rejection, SAMOS2's hard-gate DOE retries) simply complete fewer than 12
# rather than buying the overspend off-budget. The realised generation count is
# recorded in meta.
#
# SSA-NSGA-II (ssansga2, ssansga2-stock) and IOC-SAMO-COBRA are deliberately
# NOT in this arm.
#
# One array task == one (scenario, mode, method, handler) tuple, looping over
# all 20 seeds with one run_scenario.py call + one results rsync-back PER SEED
# (not one call for all 20 seeds). The scratch directory is destroyed when a
# task ends for any reason -- normal exit, walltime, preemption, node failure --
# so per-seed rsync-back means a killed task only ever loses its current
# in-flight seed, not every seed it had already finished. Combined with
# run_scenario.py's own skip-if-exists check and the results pre-copy below,
# re-submitting a killed/interrupted array index resumes from the next missing
# seed rather than seed 0.
#
# Task table (216 tasks). Per (scenario, mode) cell -- 18 tasks:
#   random x scenario/mode-default handler (1; random has no selection
#     pressure or sampling strategy for a handler to act on)
#   nsga2 x the 7-handler row {h1-rejection, h2-static_penalty, h3-adaptive,
#     h4-cdp, h5-epsilon, h6-DSR, b0-as-obj} (7)
#   samos x the same 7-handler row (7)
#   nsga2 x b1-unconstrained, samos x b1-unconstrained (2; the unconstrained
#     floor -- not part of the 7-handler row, never produced by
#     --all_handlers, so it is asked for by explicit --handler)
#   ctaea x h4-cdp (1; its constraint handling is intrinsic, so it runs the
#     single slot it implements -- see scenarios.FIXED_HANDLER_METHODS)
# Totals: 6 scenarios x 2 modes x 18 = 216.
#
# Items 0-14 of each block carry the SAME meaning as items 0-14 of
# SCENARIO_EVOXBENCH.sbatch's 15-task block, so the two arms cross-reference
# directly; the b1 and ctaea tasks (which are their own sbatch files at pop 20)
# are appended as items 15-17 rather than interleaved.
#
# Index map: each (scenario, mode) block is 18 CONTIGUOUS tasks, blocks in
# scenario-major, mode-minor order (S1/hard, S1/soft, S2/hard, S2/soft, ...,
# S6/hard, S6/soft):
#
#   block(sid, mode) = (scenario_idx * 2 + mode_idx) * 18
#     scenario_idx: S1=0, S2=1, ..., S6=5 ; mode_idx: hard=0, soft=1
#   task = block(sid, mode) + item
#     item 0     = random x scenario/mode-default handler
#     items 1-7  = nsga2 x the 7-handler row (in HANDLER_ROW order)
#     items 8-14 = samos x the same 7-handler row
#     item 15    = nsga2 x b1-unconstrained
#     item 16    = samos x b1-unconstrained
#     item 17    = ctaea x h4-cdp
#
# e.g. index   0 = S1/hard/random/h4-cdp            (block 0,   item 0)
#      index   1 = S1/hard/nsga2/h1-rejection       (block 0,   item 1)
#      index   8 = S1/hard/samos/h1-rejection       (block 0,   item 8)
#      index  15 = S1/hard/nsga2/b1-unconstrained   (block 0,   item 15)
#      index  17 = S1/hard/ctaea/h4-cdp             (block 0,   item 17)
#      index  18 = S1/soft/random/h2-static_penalty (block 18,  item 0)
#      index  36 = S2/hard/random/h4-cdp            (block 36,  item 0)
#      index 180 = S6/hard/random/h4-cdp            (block 180, item 0)
#      index 215 = S6/soft/ctaea/h4-cdp             (block 198, item 17; last task)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch
#
# Submit one (scenario, mode) cell (18-blocks per the index map above):
#   sbatch --array=0-17    experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch   # S1 hard
#   sbatch --array=18-35   experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch   # S1 soft
#   sbatch --array=180-197 experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch   # S6 hard
#   sbatch --array=198-215 experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch   # S6 soft
#
# Sanity-check one scenario before committing the whole arm:
#   sbatch --array=0-17 experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments2/scenario_run/SCENARIO_POP100_EVOXBENCH.sbatch
#
# Analyse (note the results root -- the default targets the pop-20 arm):
#   python experiments2/scenario_run/analyse_scenarios.py \
#       --results_root results2/scenario_run_pop100 \
#       --methods random nsga2 samos ctaea
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J scenario_pop100
#SBATCH -c 2
#SBATCH --array=0-215%20              # full pop-100 arm (216 tasks total, existing results are skipped)
#SBATCH --mem=4gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_scenario_run_pop100/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_scenario_run_pop100

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-100}
N_EVALS=${N_EVALS:-1200}
INNER_POP_SIZE=${INNER_POP_SIZE:-200}   # pinned to the pop-20 value -- see header
RESULTS_ROOT=${RESULTS_ROOT:-results2/scenario_run_pop100}

# --------- (scenario, mode, method, handler) task table -----------------------------------------------------------------------------------------
# Mirrors run_scenario.py's scenarios.SCENARIOS / HANDLERS / DEFAULT_HANDLER
# -- keep in sync.
SCENARIOS_LIST="S1 S2 S3 S4 S5 S6"
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
        COMBOS+=("${SCENARIO}:${MODE}:nsga2:b1-unconstrained")
        COMBOS+=("${SCENARIO}:${MODE}:samos:b1-unconstrained")
        COMBOS+=("${SCENARIO}:${MODE}:ctaea:h4-cdp")
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
    experiments2/scenario_run
)

# Pre-copy any already-computed results for this (scenario, mode, method,
# handler) so run_scenario.py's own skip-if-exists logic can resume
# mid-seed-range.
_SRC="${RESULTS_ROOT}/${SCENARIO}/${MODE}/${METHOD}/${HANDLER}"
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
echo "INNER_POP_SIZE    = $INNER_POP_SIZE"
echo "RESULTS_ROOT      = $RESULTS_ROOT"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "EVOXBENCH_DATA    = $EVOXBENCH_DATA"
echo "EVOXBENCH_MODEL   = $EVOXBENCH_MODEL"

# --------- run: one seed at a time, rsync-back after each ------------------------------------------------------------------------------------------------
DEST="${SLURM_SUBMIT_DIR}/${RESULTS_ROOT}/${SCENARIO}/${MODE}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- SCENARIO: $SCENARIO  MODE: $MODE  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED"

    python experiments2/scenario_run/run_scenario.py \
        --scenario       "$SCENARIO" \
        --mode           "$MODE" \
        --method         "$METHOD" \
        --handler        "$HANDLER" \
        --seeds          "$SEED" \
        --pop_size       "$POP_SIZE" \
        --n_evals        "$N_EVALS" \
        --inner_pop_size "$INNER_POP_SIZE" \
        --results_root   "$RESULTS_ROOT" \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/${RESULTS_ROOT}/${SCENARIO}/${MODE}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
