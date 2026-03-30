#!/bin/bash
# ─── WFG_SAMOS_ABLATION.sbatch ──────────────────────────────────────────────
# Run the SAMOS ablation study on WFG problems with SSA-NSGA-II and GPSAF
# as reference baselines.
#
# Each array task runs ONE ablation type on ONE problem for ALL seeds.
# 5 ablation types × 9 WFG problems = 45 tasks.
#
# Submit:
#   sbatch WFG_SAMOS_ABLATION.sbatch
#
# Or run a single ablation:
#   sbatch --export=ALL,ABLATION=warm_start WFG_SAMOS_ABLATION.sbatch
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J wfg_ablation
#SBATCH -c 2
#SBATCH --array=0-44              # 5 ablations × 9 problems
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs_wfg_ablation/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_wfg_ablation

# ─── tunable parameters ───────────────────────────────────────────────────────
N_OBJ=${N_OBJ:-2}
POP_SIZE=${POP_SIZE:-20}
N_GEN_INNER=${N_GEN_INNER:-20}
N_SEEDS=${N_SEEDS:-10}

# ─── derived parameters ───────────────────────────────────────────────────────
N_VAR=$(( 2 * (N_OBJ - 1) + 10 ))
TOTAL_BUDGET=$(( 100 * N_VAR ))
N_GEN=$(( TOTAL_BUDGET / POP_SIZE ))

EXPERIMENT_NAME="samos_ablation/${N_OBJ}_obj"

# ─── ablation/problem from array task ID ─────────────────────────────────────
ABLATIONS=(surrogate warm_start selection inner_pop subset_sel)
PROBLEMS=(wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9)

N_ABLATIONS=${#ABLATIONS[@]}   # 5
N_PROBLEMS=${#PROBLEMS[@]}     # 9

# If ABLATION is set via --export, run only that ablation for all problems
if [ -n "${ABLATION:-}" ]; then
    PROBLEM_IDX=$(( SLURM_ARRAY_TASK_ID % N_PROBLEMS ))
    PROBLEM=${PROBLEMS[$PROBLEM_IDX]}
    ABL="$ABLATION"
else
    ABL_IDX=$(( SLURM_ARRAY_TASK_ID / N_PROBLEMS ))
    PROBLEM_IDX=$(( SLURM_ARRAY_TASK_ID % N_PROBLEMS ))
    ABL=${ABLATIONS[$ABL_IDX]}
    PROBLEM=${PROBLEMS[$PROBLEM_IDX]}
fi

# ─── Build seed list ─────────────────────────────────────────────────────────
SEEDS=""
for (( s=0; s<N_SEEDS; s++ )); do
    SEEDS="$SEEDS $s"
done

# ─── scratch setup ────────────────────────────────────────────────────────────
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIRECTORY"

RSYNC_SOURCES=(
    analysis
    problem/pymoo
    strategy
    utils
    main_samos_ablation.py
)
# Pre-copy existing results
_SRC="results/${EXPERIMENT_NAME}/${PROBLEM}"
[ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/"
cd "$SCRATCH_DIRECTORY"

# ─── environment ──────────────────────────────────────────────────────────────
module load nvidia/nvhpc/23.3
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

# ─── logging ──────────────────────────────────────────────────────────────────
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "PROBLEM           = $PROBLEM"
echo "ABLATION          = $ABL"
echo "N_OBJ             = $N_OBJ"
echo "N_VAR             = $N_VAR"
echo "N_GEN             = $N_GEN"
echo "POP_SIZE          = $POP_SIZE"
echo "SEEDS             = $SEEDS"

# ─── run ──────────────────────────────────────────────────────────────────────
python main_samos_ablation.py \
    --problem       "$PROBLEM" \
    --n_obj         "$N_OBJ" \
    --n_var         "$N_VAR" \
    --ablation      "$ABL" \
    --seeds         $SEEDS \
    --pop_size      "$POP_SIZE" \
    --n_gen         "$N_GEN" \
    --n_gen_inner   "$N_GEN_INNER" \
    --experiment_name "$EXPERIMENT_NAME"

# ── copy results back ────────────────────────────────────────────────────────
DEST="${SLURM_SUBMIT_DIR}/results/${EXPERIMENT_NAME}/${PROBLEM}"
mkdir -p "$DEST"
rsync -a \
    "${SCRATCH_DIRECTORY}/results/${EXPERIMENT_NAME}/${PROBLEM}/" \
    "$DEST/"
echo "Results copied to $DEST"

# ─── scratch cleanup ─────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"
rm -rf "$SCRATCH_DIRECTORY"

echo ""
echo "Date (done)       = $(date)"
