#!/bin/bash
# ─── EVOXBENCH_IN1K.sbatch ────────────────────────────────────────────────────
# Run Random / NSGA-II / SAMOS-XGB / SAMOS2 / ParEGO / GPSAF / MOSMAC on
# in1kmop PIDs 1-9 for 30 independent seeds.
# Each array task = one method × one seed; all 9 PIDs run sequentially inside.
#
# 7 methods × 30 seeds = 210 tasks total — exceeds the 100-task array limit.
# Submit in three seed batches by activating the matching --array line below:
#
#   Batch 1 (seeds  0-9):  sbatch EVOXBENCH_IN1K.sbatch          # array 0-69
#   Batch 2 (seeds 10-19): edit array to 70-139, then sbatch     # array 70-139
#   Batch 3 (seeds 20-29): edit array to 140-209, then sbatch    # array 140-209
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J evoxbench_in1k
#SBATCH -c 2
# Activate exactly ONE of the three lines below for each submission:
##SBATCH --array=0-69               # Batch 1: seeds  0-9  (7 methods × 10 seeds)
##SBATCH --array=70-139                  # Batch 2: seeds 10-19 (7 methods × 10 seeds)
#SBATCH --array=140-209                 # Batch 3: seeds 20-29 (7 methods × 10 seeds)
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs_evoxbench_in1k/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_evoxbench_in1k

# ─── tunable parameters ───────────────────────────────────────────────────────
SUITE=in1kmop
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# ─── methods ──────────────────────────────────────────────────────────────────
METHODS=(random nsga2 samos-xgb samos2 parego gpsaf-default mosmac)
N_METHODS=${#METHODS[@]}   # 7

# ─── task decomposition ───────────────────────────────────────────────────────
SEED=$(( SLURM_ARRAY_TASK_ID / N_METHODS ))
METHOD_IDX=$(( SLURM_ARRAY_TASK_ID % N_METHODS ))
METHOD=${METHODS[$METHOD_IDX]}

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

mkdir -p slurm_logs_evoxbench_in1k

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
echo "SUITE             = $SUITE"
echo "SEED              = $SEED"
echo "METHOD            = $METHOD"
echo "N_GEN             = $N_GEN"
echo "POP_SIZE          = $POP_SIZE"
echo "BUDGET            = $(( N_GEN * POP_SIZE ))"

# ─── run: loop over all 9 PIDs ────────────────────────────────────────────────
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
        --n_gen_inner   "$N_GEN_INNER"
    #    --overwrite

    # ── copy results back ──────────────────────────────────────────────────────
    BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
    DEST="${SLURM_SUBMIT_DIR}/results/evoxbench/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}"
    mkdir -p "$DEST"
    rsync -a \
        "${SCRATCH_DIRECTORY}/results/evoxbench/${SUITE}/pid${PID}/${BUDGET_FOLDER}/${METHOD}/" \
        "$DEST/"
    echo "Results copied to $DEST"
done

echo ""
echo "Date (done)       = $(date)"

# ─── scratch cleanup ──────────────────────────────────────────────────────────
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
