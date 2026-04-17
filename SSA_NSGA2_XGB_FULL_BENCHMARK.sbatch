#!/bin/bash
# ─── SSA_NSGA2_XGB_FULL_BENCHMARK.sbatch ─────────────────────────────────────
# Run SSA-NSGA-II with XGBoost surrogates on all three benchmark suites
# (WFG1-9, C10MOP1-9, IN1KMOP1-9) for 30 independent seeds.
#
# 1 method × 30 seeds = 30 array tasks (0-29).
# SLURM_ARRAY_TASK_ID maps directly to the seed index.
#
# Each task runs all benchmarks sequentially:
#   1. WFG1-9   (main_pymoo_benchmark.py)
#   2. C10MOP1-9  (main_evoxbench.py --suite c10mop)
#   3. IN1KMOP1-9 (main_evoxbench.py --suite in1kmop)
#
# Submit:
#   sbatch SSA_NSGA2_XGB_FULL_BENCHMARK.sbatch
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J ssa_nsga2_xgb_full
#SBATCH -c 2
#SBATCH --array=0-29
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs_ssa_nsga2_xgb_full/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_ssa_nsga2_xgb_full

# ─── parameters ───────────────────────────────────────────────────────────────
METHOD=ssa-nsga2-xgb
SEED=${SLURM_ARRAY_TASK_ID}

# WFG
N_OBJ=${N_OBJ:-2}
POP_SIZE=${POP_SIZE:-20}
N_VAR=$(( 2 * (N_OBJ - 1) + 10 ))
TOTAL_BUDGET=$(( 100 * N_VAR ))
N_GEN_WFG=$(( TOTAL_BUDGET / POP_SIZE ))
WFG_EXPERIMENT="pymoo_benchmark/${N_OBJ}_obj"

# EvoXBench (C10 & IN1K)
N_GEN_EVX=${N_GEN_EVX:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# ─── scratch setup ────────────────────────────────────────────────────────────
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIRECTORY"

RSYNC_SOURCES=(
    analysis
    problem/pymoo
    problem/evoxbench
    strategy
    utils
    main_pymoo_benchmark.py
    main_evoxbench.py
)

# Pre-copy existing WFG results
for _P in wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9; do
    _SRC="results/${WFG_EXPERIMENT}/${_P}"
    [ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")
done

# Pre-copy existing evoxbench results
BUDGET_FOLDER_EVX="B$(( N_GEN_EVX * POP_SIZE ))_P${POP_SIZE}"
for _SUITE in c10mop in1kmop; do
    for _PID in 1 2 3 4 5 6 7 8 9; do
        _SRC="results/evoxbench/${_SUITE}/pid${_PID}/${BUDGET_FOLDER_EVX}/${METHOD}"
        [ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")
    done
done

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/"
cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 4 -type d

mkdir -p slurm_logs_ssa_nsga2_xgb_full

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
echo "METHOD            = $METHOD"
echo "SEED              = $SEED"
echo "N_OBJ (WFG)       = $N_OBJ"
echo "N_VAR (WFG)       = $N_VAR"
echo "N_GEN (WFG)       = $N_GEN_WFG"
echo "POP_SIZE          = $POP_SIZE"
echo "WFG budget        = $TOTAL_BUDGET"
echo "N_GEN (EvoX)      = $N_GEN_EVX"
echo "N_GEN_INNER       = $N_GEN_INNER"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. WFG1-9
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════ WFG benchmark ══════════════════"

for PROBLEM in wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9; do
    echo ""
    echo "─── Problem: $PROBLEM  method: $METHOD  seed: $SEED"

    python main_pymoo_benchmark.py \
        --problem         "$PROBLEM" \
        --n_obj           "$N_OBJ" \
        --n_var           "$N_VAR" \
        --methods         "$METHOD" \
        --seeds           "$SEED" \
        --pop_size        "$POP_SIZE" \
        --n_gen           "$N_GEN_WFG" \
        --experiment_name "$WFG_EXPERIMENT" \
        --no_plot

    # ── copy results back ──────────────────────────────────────────────────────
    DEST="${SLURM_SUBMIT_DIR}/results/${WFG_EXPERIMENT}/${PROBLEM}"
    mkdir -p "$DEST"
    rsync -a \
        "${SCRATCH_DIRECTORY}/results/${WFG_EXPERIMENT}/${PROBLEM}/" \
        "$DEST/"
    echo "Results copied to $DEST"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 2. C10MOP1-9
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════ C10MOP benchmark ══════════════════"

for PID in 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo "─── Suite: c10mop  PID: $PID  method: $METHOD  seed: $SEED"

    python main_evoxbench.py \
        --suite         c10mop \
        --pids          "$PID" \
        --methods       "$METHOD" \
        --seeds         "$SEED" \
        --pop_size      "$POP_SIZE" \
        --n_gen         "$N_GEN_EVX" \
        --n_gen_inner   "$N_GEN_INNER"

    # ── copy results back ──────────────────────────────────────────────────────
    DEST="${SLURM_SUBMIT_DIR}/results/evoxbench/c10mop/pid${PID}/${BUDGET_FOLDER_EVX}/${METHOD}"
    mkdir -p "$DEST"
    rsync -a \
        "${SCRATCH_DIRECTORY}/results/evoxbench/c10mop/pid${PID}/${BUDGET_FOLDER_EVX}/${METHOD}/" \
        "$DEST/"
    echo "Results copied to $DEST"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 3. IN1KMOP1-9
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════ IN1KMOP benchmark ══════════════════"

for PID in 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo "─── Suite: in1kmop  PID: $PID  method: $METHOD  seed: $SEED"

    python main_evoxbench.py \
        --suite         in1kmop \
        --pids          "$PID" \
        --methods       "$METHOD" \
        --seeds         "$SEED" \
        --pop_size      "$POP_SIZE" \
        --n_gen         "$N_GEN_EVX" \
        --n_gen_inner   "$N_GEN_INNER"

    # ── copy results back ──────────────────────────────────────────────────────
    DEST="${SLURM_SUBMIT_DIR}/results/evoxbench/in1kmop/pid${PID}/${BUDGET_FOLDER_EVX}/${METHOD}"
    mkdir -p "$DEST"
    rsync -a \
        "${SCRATCH_DIRECTORY}/results/evoxbench/in1kmop/pid${PID}/${BUDGET_FOLDER_EVX}/${METHOD}/" \
        "$DEST/"
    echo "Results copied to $DEST"
done

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Date (done)       = $(date)"

# ─── scratch cleanup ──────────────────────────────────────────────────────────
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
