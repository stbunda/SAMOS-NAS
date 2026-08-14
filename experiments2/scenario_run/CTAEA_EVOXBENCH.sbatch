#!/bin/bash
# --------- CTAEA_EVOXBENCH.sbatch ------------------------------------------------------------------------------------------------------------------
# CTAEA add-on to the scenario_run campaign: S1-S6
# (experiments2/scenario_run/scenarios.py) at the same 10% feasibility
# operating point, hard/soft mode, pymoo CTAEA only.
#
# Its own sbatch rather than extra rows in SCENARIO_EVOXBENCH.sbatch: that
# file's COMBOS array is 15 tasks per (scenario, mode) block, so adding a
# 16th item would re-number every index in its documented map (and therefore
# every already-submitted / re-submittable array index). CTAEA writes into
# the SAME results tree and is read by the same analyse_scenarios.py, so the
# split is purely a submission-side one.
#
# CTAEA runs a SINGLE handler slot, 'h4-cdp' in BOTH modes
# (scenarios.FIXED_HANDLER_METHODS / fixed_handler): its constraint handling
# is intrinsic -- the convergence/diversity archive pair plus the CV-first
# restricted-mating tournament -- so there is no seam for one of the 7
# handlers to act on. h4-cdp is the mechanism it implements (feasibility-first
# constraint domination), which also puts it next to nsga2/samos x h4-cdp in
# the analysis grid. The mode axis still applies: it governs gating vs
# archiving in the outer problem, not the handler.
#
# Termination is EVALUATION-based (('n_evals', N), never ('n_gen', N)) --
# see run_scenario.py's module docstring. Results ->
# results2/scenario_run/{sid}/{mode}/ctaea/h4-cdp/seed_N.pkl.
#
# One array task == one (scenario, mode) tuple, looping over all 20 seeds
# with one run_scenario.py call + one results rsync-back PER SEED (not one
# call for all 20 seeds). The scratch directory is destroyed when a task ends
# for any reason -- normal exit, walltime, preemption, node failure -- so
# per-seed rsync-back means a killed task only ever loses its current
# in-flight seed, not every seed it had already finished. Combined with
# run_scenario.py's own skip-if-exists check and the results pre-copy below,
# re-submitting a killed/interrupted array index resumes from the next
# missing seed rather than seed 0.
#
# Task table (12 tasks), scenario-major, mode-minor:
#   task = scenario_idx * 2 + mode_idx
#     scenario_idx: S1=0, S2=1, ..., S6=5 ; mode_idx: hard=0, soft=1
#
# e.g. index  0 = S1/hard    index  1 = S1/soft
#      index  2 = S2/hard    index 10 = S6/hard
#      index 11 = S6/soft    (last task)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments2/scenario_run/CTAEA_EVOXBENCH.sbatch
#
# Submit one scenario (both modes):
#   sbatch --array=0-1   experiments2/scenario_run/CTAEA_EVOXBENCH.sbatch   # S1
#   sbatch --array=10-11 experiments2/scenario_run/CTAEA_EVOXBENCH.sbatch   # S6
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments2/scenario_run/CTAEA_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments2/scenario_run/CTAEA_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J scenario_ctaea
#SBATCH -c 2
#SBATCH --array=0-11%12               # full CTAEA add-on (12 tasks total, existing results are skipped)
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_scenario_run/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_scenario_run

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
# Keep these identical to SCENARIO_EVOXBENCH.sbatch -- CTAEA is only
# comparable to the nsga2/samos cells under the same budget.
POP_SIZE=${POP_SIZE:-20}
N_EVALS=${N_EVALS:-1200}

# --------- (scenario, mode) task table ----------------------------------------------------------------------------------------------------------
METHOD=ctaea
HANDLER=h4-cdp                        # scenarios.fixed_handler('ctaea', mode), both modes

SCENARIOS_LIST="S1 S2 S3 S4 S5 S6"
MODES_LIST="hard soft"

COMBOS=()
for SCENARIO in $SCENARIOS_LIST; do
    for MODE in $MODES_LIST; do
        COMBOS+=("${SCENARIO}:${MODE}")
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r SCENARIO MODE <<< "$TASK_ENTRY"

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

# Pre-copy any already-computed results for this (scenario, mode) so
# run_scenario.py's own skip-if-exists logic can resume mid-seed-range.
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
