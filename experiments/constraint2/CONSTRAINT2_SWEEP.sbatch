#!/bin/bash
# --------- CONSTRAINT2_SWEEP.sbatch -----------------------------------------------------------------------------------------------------------
# Quantile-tightness sweep: constraint2 at Q10 and Q50 variants
# (experiments/constraint2/run_constraint.py --threshold_variant q10 or q50)
# at ~10% and ~50% feasible regions. Q25 results reuse from results/constraint2.
# Scenario: s2 (hard mode, expensive/latency constraint) only.
# Instances: c10mop/pid4, citysegmop/pid5 only (the two where q10/q50 are defined).
# Rows per (quantile, instance): 7 rows mirroring s2 structure:
#   random×h4-cdp (1) + {samos, samos-cheap} × {h4-cdp, h2-penalty, b0-nsga2} (6)
#
# Task table (28 tasks):
#   task id  quantile  instance       method        handler      results_root
#   -------  --------  --------       ------        -------      ---------------
#   0-6      q10       c10mop:4       (varied)      (varied)     results/constraint2_q10
#   7-13     q10       citysegmop:5   (varied)      (varied)     results/constraint2_q10
#   14-20    q50       c10mop:4       (varied)      (varied)     results/constraint2_q50
#   21-27    q50       citysegmop:5   (varied)      (varied)     results/constraint2_q50
#
# One array task == one (quantile, instance, method, handler) tuple,
# looping over all 20 seeds with one run_constraint.py call + one results
# rsync-back PER SEED (same per-seed-iteration strategy as CONSTRAINT2_EVOXBENCH.sbatch).
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint2/CONSTRAINT2_SWEEP.sbatch
#
# Submit one quantile block:
#   sbatch --array=0-13%10  experiments/constraint2/CONSTRAINT2_SWEEP.sbatch
#   sbatch --array=14-27%10 experiments/constraint2/CONSTRAINT2_SWEEP.sbatch
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint2/CONSTRAINT2_SWEEP.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=5 experiments/constraint2/CONSTRAINT2_SWEEP.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J constraint2-sweep
#SBATCH -c 2
#SBATCH --array=0-27%20              # 28 tasks total (2 quantiles x 2 instances x 7 rows)
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_constraint2_sweep/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_constraint2_sweep

# --------- tunable parameters (B1200_P20: identical to main campaign) -------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# --------- task table: (quantile, instance, method, handler) tuples ---------------------------------------------------------
# Builds 28 tasks: 2 quantiles x 2 instances x 7 rows
COMBOS=()

for QUANTILE in q10 q50; do
    for INSTANCE in c10mop:4 citysegmop:5; do
        # Row 0: random × h4-cdp
        COMBOS+=("${QUANTILE}:${INSTANCE}:random:h4-cdp")
        # Rows 1-3: samos × {h4-cdp, h2-penalty, b0-nsga2}
        for HANDLER in h4-cdp h2-penalty b0-nsga2; do
            COMBOS+=("${QUANTILE}:${INSTANCE}:samos:${HANDLER}")
        done
        # Rows 4-6: samos-cheap × {h4-cdp, h2-penalty, b0-nsga2}
        for HANDLER in h4-cdp h2-penalty b0-nsga2; do
            COMBOS+=("${QUANTILE}:${INSTANCE}:samos-cheap:${HANDLER}")
        done
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r QUANTILE SUITE PID METHOD HANDLER <<< "$TASK_ENTRY"

# Fixed scenario for this sweep (s2: hard, expensive/latency constraint)
SCENARIO="s2"

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

# Pre-copy any already-computed results for this (quantile, scenario, suite, pid,
# method, handler) so run_constraint.py's own skip-if-exists logic can
# resume mid-seed-range.
BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
_SRC="results/constraint2_${QUANTILE}/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
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
echo "QUANTILE          = $QUANTILE"
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
DEST="${SLURM_SUBMIT_DIR}/results/constraint2_${QUANTILE}/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- QUANTILE: $QUANTILE  SUITE: $SUITE  PID: $PID  SCENARIO: $SCENARIO  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED"

    python experiments/constraint2/run_constraint.py \
        --scenario           "$SCENARIO" \
        --suite              "$SUITE" \
        --pid                "$PID" \
        --method             "$METHOD" \
        --handler            "$HANDLER" \
        --seeds              "$SEED" \
        --pop_size           "$POP_SIZE" \
        --n_gen              "$N_GEN" \
        --n_gen_inner        "$N_GEN_INNER" \
        --threshold_variant  "$QUANTILE" \
        --results_root       results/constraint2_${QUANTILE} \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results/constraint2_${QUANTILE}/${SCENARIO}/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
