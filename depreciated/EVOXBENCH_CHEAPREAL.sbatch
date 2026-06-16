#!/bin/bash
# ─── EVOXBENCH_CHEAPREAL.sbatch ───────────────────────────────────────────────
# Run SAMOS-CheapReal (val_err via surrogate; params/flops via real eval) on
# c10mop and in1kmop PIDs 1-9, 30 independent seeds.
# Each array task = one seed; both suites × 9 PIDs run sequentially inside.
#
# 30 seeds = 30 tasks total.  Submit in three batches of 10:
#
#   Batch 1 (seeds  0-29):  sbatch EVOXBENCH_CHEAPREAL.sbatch    # array 0-9
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J evoxbench_cheapreal
#SBATCH -c 2
# Activate exactly ONE of the three lines below for each submission:
#SBATCH --array=0-29                # Batch 1: seeds  0-29
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs_evoxbench_cheapreal/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_evoxbench_cheapreal

# ─── tunable parameters ───────────────────────────────────────────────────────
METHOD=samos-cheapreal
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}
SEED=$SLURM_ARRAY_TASK_ID

# ─── scratch setup ────────────────────────────────────────────────────────────
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "$SCRATCH_DIRECTORY" ]; then
    echo "'$SCRATCH_DIRECTORY' already found!"
else
    echo "'$SCRATCH_DIRECTORY' not found, creating!"
    mkdir -p "$SCRATCH_DIRECTORY"
fi

rsync -aR \
    analysis \
    problem/evoxbench \
    problem/pymoo \
    strategy \
    utils \
    main_evoxbench.py \
    "$SCRATCH_DIRECTORY/"

cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 3 -type d

mkdir -p slurm_logs_evoxbench_cheapreal

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
echo "SEED              = $SEED"
echo "METHOD            = $METHOD"
echo "N_GEN             = $N_GEN"
echo "POP_SIZE          = $POP_SIZE"
echo "BUDGET            = $(( N_GEN * POP_SIZE ))"

# ─── run: both suites × all 9 PIDs ───────────────────────────────────────────
for SUITE in c10mop in1kmop; do
    for PID in 1 2 3 4 5 6 7 8 9; do
        echo ""
        echo "─── Suite: $SUITE  PID: $PID  method: $METHOD  seed: $SEED"

        python main_evoxbench.py \
            --suite         "$SUITE" \
            --pids          "$PID" \
            --methods       "$METHOD" \
            --seeds         "$SEED" \
            --pop_size      "$POP_SIZE" \
            --n_gen         "$N_GEN" \
            --n_gen_inner   "$N_GEN_INNER" \
            --overwrite

        # ── copy results back ──────────────────────────────────────────────────
        BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
        DEST="${SLURM_SUBMIT_DIR}/results/evoxbench/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}"
        mkdir -p "$DEST"
        rsync -a \
            "${SCRATCH_DIRECTORY}/results/evoxbench/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/" \
            "$DEST/"
        echo "Results copied to $DEST"
    done
done

echo ""
echo "Date (done)       = $(date)"

# ─── scratch cleanup ──────────────────────────────────────────────────────────
rm -rf "$SCRATCH_DIRECTORY"
echo "Scratch cleaned up."
