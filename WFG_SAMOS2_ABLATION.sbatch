#!/bin/bash
# ─── WFG_SAMOS2_ABLATION.sbatch ─────────────────────────────────────────────
# Run the SAMOS2 ablation study on WFG problems.
# Uses SAMOS2 tuned defaults (warm_start=0.25, ga_pop=100, n_gen_inner=5,
# alpha=3, beta=10, pop_schedule=exp_dec 500→50).
#
# Reference baselines are hard-copied from the V1 experiment folder so they
# never need to be recomputed.
#
# Each array task runs ONE ablation type on ONE problem for ALL seeds.
# 16 ablation types × 9 WFG problems = 144 tasks.
#
# Submit:
#   sbatch WFG_SAMOS2_ABLATION.sbatch
#
# Or run a single ablation:
#   sbatch --export=ALL,ABLATION=warm_start WFG_SAMOS2_ABLATION.sbatch
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J wfg2_ablation
#SBATCH -c 2
#SBATCH --array=0-143
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs_wfg2_ablation/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_wfg2_ablation

# ─── tunable parameters ───────────────────────────────────────────────────────
N_OBJ=${N_OBJ:-2}
POP_SIZE=${POP_SIZE:-20}
N_GEN_INNER=${N_GEN_INNER:-5}      # SAMOS2 default
N_SEEDS=${N_SEEDS:-10}
POP_START=${POP_START:-500}        # SAMOS2 default (pop_schedule ablation range)
POP_END=${POP_END:-50}             # SAMOS2 default

# ─── derived parameters ───────────────────────────────────────────────────────
N_VAR=$(( 2 * (N_OBJ - 1) + 10 ))
TOTAL_BUDGET=$(( 100 * N_VAR ))
N_GEN=$(( TOTAL_BUDGET / POP_SIZE ))

EXPERIMENT_NAME="samos2_ablation/${N_OBJ}_obj"
V1_EXPERIMENT_NAME="samos_ablation/${N_OBJ}_obj"

# ─── ablation/problem from array task ID ─────────────────────────────────────
ABLATIONS=(surrogate warm_start selection inner_pop subset_sel
           surrogate_ext inner_gens pop_schedule alpha_beta
           noise_tournament trace uncertainty
           ensemble doe_strategy rbf_surrogate infill_criterion)
PROBLEMS=(wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9)

N_PROBLEMS=${#PROBLEMS[@]}   # 9

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

# ─── seed list ────────────────────────────────────────────────────────────────
SEEDS=""
for (( s=0; s<N_SEEDS; s++ )); do
    SEEDS="$SEEDS $s"
done

# ─── hard-copy V1 references (run in submit dir before scratch) ───────────────
BUDGET_FOLDER="B${TOTAL_BUDGET}_P${POP_SIZE}"
V1_REF="${SLURM_SUBMIT_DIR}/results/${V1_EXPERIMENT_NAME}/${PROBLEM}/${BUDGET_FOLDER}/reference"
V2_REF="${SLURM_SUBMIT_DIR}/results/${EXPERIMENT_NAME}/${PROBLEM}/${BUDGET_FOLDER}/reference"
if [ -d "$V1_REF" ] && [ ! -d "$V2_REF" ]; then
    mkdir -p "$(dirname "$V2_REF")"
    cp -r "$V1_REF" "$V2_REF"
    echo "Copied V1 references -> $V2_REF"
fi

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
# Pre-copy existing V2 results (skip + overwrite logic)
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
echo "BASE_ALGORITHM    = samos2"
echo "N_OBJ             = $N_OBJ"
echo "N_VAR             = $N_VAR"
echo "N_GEN             = $N_GEN"
echo "POP_SIZE          = $POP_SIZE"
echo "N_GEN_INNER       = $N_GEN_INNER"
echo "SEEDS             = $SEEDS"
echo "POP_START         = $POP_START"
echo "POP_END           = $POP_END"

# ─── run ──────────────────────────────────────────────────────────────────────
python main_samos_ablation.py \
    --problem           "$PROBLEM" \
    --n_obj             "$N_OBJ" \
    --n_var             "$N_VAR" \
    --ablation          "$ABL" \
    --seeds             $SEEDS \
    --pop_size          "$POP_SIZE" \
    --n_gen             "$N_GEN" \
    --n_gen_inner       "$N_GEN_INNER" \
    --pop_start         "$POP_START" \
    --pop_end           "$POP_END" \
    --base_algorithm    samos2 \
    --experiment_name   "$EXPERIMENT_NAME"

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
