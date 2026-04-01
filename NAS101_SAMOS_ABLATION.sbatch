#!/bin/bash
# ─── NAS101_SAMOS_ABLATION.sbatch ────────────────────────────────────────────
# Run the SAMOS ablation study on NASBench-101 (2-objective: error + params).
#
# Each array task runs ONE ablation type for ALL seeds.
# 15 ablations (all 16 minus doe_strategy, which requires continuous QMC).
#
# Submit:
#   sbatch NAS101_SAMOS_ABLATION.sbatch
#
# Or run a single ablation:
#   sbatch --export=ALL,ABLATION=warm_start NAS101_SAMOS_ABLATION.sbatch
#
# Ablations (15 total — doe_strategy excluded):
#   surrogate, warm_start, selection, inner_pop, subset_sel,
#   surrogate_ext, inner_gens, pop_schedule, alpha_beta,
#   noise_tournament, trace, uncertainty,
#   ensemble, rbf_surrogate, infill_criterion
#
# Budget: N_VAR=26 → TOTAL_BUDGET=2600 → N_GEN=130 (with POP_SIZE=20)
# Results: results/samos_ablation/nasbench101/B2600_P20/…
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J nas101_ablation
#SBATCH -c 2
#SBATCH --array=0-14
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs_nas_ablation/101_%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_nas_ablation

# ─── tunable parameters ───────────────────────────────────────────────────────
N_VAR=26
POP_SIZE=${POP_SIZE:-20}
N_GEN_INNER=${N_GEN_INNER:-20}
N_SEEDS=${N_SEEDS:-10}
POP_START=${POP_START:-50}
POP_END=${POP_END:-1000}

# ─── derived parameters ───────────────────────────────────────────────────────
TOTAL_BUDGET=$(( 100 * N_VAR ))          # 2600
N_GEN=$(( TOTAL_BUDGET / POP_SIZE ))     # 130

EXPERIMENT_NAME="samos_ablation"

# ─── ablation from array task ID ─────────────────────────────────────────────
ABLATIONS=(surrogate warm_start selection inner_pop subset_sel
           surrogate_ext inner_gens pop_schedule alpha_beta
           noise_tournament trace uncertainty
           ensemble rbf_surrogate infill_criterion)

if [ -n "${ABLATION:-}" ]; then
    ABL="$ABLATION"
else
    ABL=${ABLATIONS[$SLURM_ARRAY_TASK_ID]}
fi

# ─── seed list ────────────────────────────────────────────────────────────────
SEEDS=""
for (( s=0; s<N_SEEDS; s++ )); do
    SEEDS="$SEEDS $s"
done

# ─── scratch setup ────────────────────────────────────────────────────────────
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIRECTORY"

RSYNC_SOURCES=(
    analysis
    problem/nasbench101
    problem/data/data_nasbench101.pkl
    strategy
    utils
    main_samos_ablation.py
)
# Pre-copy any existing NAS-101 results so --overwrite is respected
_SRC="results/${EXPERIMENT_NAME}/nasbench101"
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
echo "Benchmark         = nasbench101"
echo "ABLATION          = $ABL"
echo "N_VAR             = $N_VAR"
echo "TOTAL_BUDGET      = $TOTAL_BUDGET"
echo "N_GEN             = $N_GEN"
echo "POP_SIZE          = $POP_SIZE"
echo "N_GEN_INNER       = $N_GEN_INNER"
echo "SEEDS             = $SEEDS"

# ─── run ──────────────────────────────────────────────────────────────────────
python main_samos_ablation.py \
    --benchmark     nasbench101 \
    --ablation      "$ABL" \
    --seeds         $SEEDS \
    --pop_size      "$POP_SIZE" \
    --n_gen         "$N_GEN" \
    --n_gen_inner   "$N_GEN_INNER" \
    --pop_start     "$POP_START" \
    --pop_end       "$POP_END" \
    --experiment_name "$EXPERIMENT_NAME"

# ── copy results back ────────────────────────────────────────────────────────
DEST="${SLURM_SUBMIT_DIR}/results/${EXPERIMENT_NAME}/nasbench101"
mkdir -p "$DEST"
rsync -a \
    "${SCRATCH_DIRECTORY}/results/${EXPERIMENT_NAME}/nasbench101/" \
    "$DEST/"
echo "Results copied to $DEST"

# ─── scratch cleanup ─────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"
rm -rf "$SCRATCH_DIRECTORY"

echo ""
echo "Date (done)       = $(date)"
