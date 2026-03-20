#!/bin/bash
# parameters for slurm
#SBATCH -J nasbench101_baseline        # job name
#SBATCH -c 2                           # number of cores

#SBATCH --array=0-25

#SBATCH --mem=16gb                     # job memory request
#SBATCH --gres=gpu:0                   # no GPUs needed
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=15-00:00
#SBATCH --output=slurm_logs_nasbench101_baseline/trial_%A_%a.log # make sure this folder exists before running
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

# Variables
EXPERIMENT_NAME=nasbench101_baseline
POP_SIZE=50
N_GEN=20
N_GEN_INNER=20
WARM_START_RATIO=1.0

# Create a scratch directory for this job
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}"
if [ -d "$SCRATCH_DIRECTORY" ]; then
    echo "'$SCRATCH_DIRECTORY' already found!"
else
    echo "'$SCRATCH_DIRECTORY' not found, creating!"
    mkdir $SCRATCH_DIRECTORY
fi

# Copy inputs and executable to the node
rsync -aR \
    analysis \
    problem/*.py \
    problem/data/data_nasbench101.pkl \
    strategy \
    utils \
    *.py \
    "$SCRATCH_DIRECTORY/"

cd $SCRATCH_DIRECTORY

echo "Copied structure:"
find ${SCRATCH_DIRECTORY} -maxdepth 2 -type d

# Load software modules
module load nvidia/nvhpc/23.3
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

echo "Activated: SAMOS311"

# Logging info
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Hostname -i       = $(hostname -I)"
echo "Working Directory = $(pwd)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"
echo "Slurm Submit Dir  = $SLURM_SUBMIT_DIR"
echo "Num nodes         = $SLURM_NNODES"
echo "Num tasks         = $SLURM_NTASKS"
echo "Threads/task      = $SLURM_CPUS_PER_TASK"

SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)

# -----------------------------------------------------------------
# Method index (13 total methods):
#   0  → random_ga                      (generational random search)
#   1  → nsga2-uniform                  (NSGA-II + xo + uniform mutation)
#   2  → nsga2-xo-single                (NSGA-II + xo + single mutation)
#   3  → nsga2-no-xo-single             (NSGA-II + no xo + single mutation)
#   4  → samos-xgb-xo-uniform-200       (SAMOS-XGB + xo + uniform + inner_pop=200)
#   5  → samos-xgb-xo-single-200        (SAMOS-XGB + xo + single + inner_pop=200)
#   6  → samos-xgb-no-xo-single-200     (SAMOS-XGB + no xo + single + inner_pop=200)
#   7  → samos-xgb-xo-uniform-50        (SAMOS-XGB + xo + uniform + inner_pop=50)
#   8  → samos-xgb-xo-single-50         (SAMOS-XGB + xo + single + inner_pop=50)
#   9  → samos-xgb-no-xo-single-50      (SAMOS-XGB + no xo + single + inner_pop=50)
#  10  → samos-xgb-xo-uniform-1000      (SAMOS-XGB + xo + uniform + inner_pop=1000)
#  11  → samos-xgb-xo-single-1000       (SAMOS-XGB + xo + single + inner_pop=1000)
#  12  → samos-xgb-no-xo-single-1000    (SAMOS-XGB + no xo + single + inner_pop=1000)
# -----------------------------------------------------------------
METHODS=(random_ga nsga2-uniform nsga2-xo-single nsga2-no-xo-single \
          samos-xgb-xo-uniform-200 samos-xgb-xo-single-200 samos-xgb-no-xo-single-200 \
          samos-xgb-xo-uniform-50 samos-xgb-xo-single-50 samos-xgb-no-xo-single-50 \
          samos-xgb-xo-uniform-1000 samos-xgb-xo-single-1000 samos-xgb-no-xo-single-1000)

N_METHODS=${#METHODS[@]}

TOTAL=$(( ${#SEEDS[@]} * N_METHODS ))

# Helper: decode index into (remainder, quotient)
pop() {
    local size=$1
    local idx=$2
    echo $(( idx % size )) $(( idx / size ))
}

STREAMS=26
IDX=$SLURM_ARRAY_TASK_ID
STREAM_ID=$SLURM_ARRAY_TASK_ID
IDX_START=$SLURM_ARRAY_TASK_ID
IDX_LAST=$(( ((TOTAL - 1 - STREAM_ID) / STREAMS) * STREAMS + STREAM_ID ))

while (( IDX < TOTAL )); do

    CUR=$IDX

    DONE=$(( (IDX - IDX_START) / STREAMS ))
    TOTAL_STREAM=$(( (IDX_LAST - IDX_START) / STREAMS + 1 ))
    PCT=$(( 100 * DONE / TOTAL_STREAM ))
    echo "[STREAM ${STREAM_ID}] ${DONE}/${TOTAL_STREAM} (${PCT}%) — IDX=${IDX}"

    # --- Seed ---
    read s CUR <<< $(pop ${#SEEDS[@]} $CUR)
    SEED=${SEEDS[$s]}

    # --- Method ---
    read m CUR <<< $(pop $N_METHODS $CUR)
    METHOD=${METHODS[$m]}

    case $METHOD in
        random_ga)
            METHOD_FLAG="--random_ga"
            LABEL="RandomGA"
            ;;
        nsga2-uniform)
            METHOD_FLAG="--nsga2_uniform"
            LABEL="NSGA2_uniform"
            ;;
        nsga2-xo-single)
            METHOD_FLAG="--nsga2_xo_single"
            LABEL="NSGA2_xo_single"
            ;;
        nsga2-no-xo-single)
            METHOD_FLAG="--nsga2_no_xo_single"
            LABEL="NSGA2_no_xo_single"
            ;;
        samos-xgb-xo-uniform-200)
            METHOD_FLAG="--samos_xgb_uniform_200"
            LABEL="SAMOS_XGB_xo_uniform_200"
            ;;
        samos-xgb-xo-single-200)
            METHOD_FLAG="--samos_xgb_single_200"
            LABEL="SAMOS_XGB_xo_single_200"
            ;;
        samos-xgb-no-xo-single-200)
            METHOD_FLAG="--samos_xgb_no_xo_single_200"
            LABEL="SAMOS_XGB_no_xo_single_200"
            ;;
        samos-xgb-xo-uniform-50)
            METHOD_FLAG="--samos_xgb_uniform_50"
            LABEL="SAMOS_XGB_xo_uniform_50"
            ;;
        samos-xgb-xo-single-50)
            METHOD_FLAG="--samos_xgb_single_50"
            LABEL="SAMOS_XGB_xo_single_50"
            ;;
        samos-xgb-no-xo-single-50)
            METHOD_FLAG="--samos_xgb_no_xo_single_50"
            LABEL="SAMOS_XGB_no_xo_single_50"
            ;;
        samos-xgb-xo-uniform-1000)
            METHOD_FLAG="--samos_xgb_uniform_1000"
            LABEL="SAMOS_XGB_xo_uniform_1000"
            ;;
        samos-xgb-xo-single-1000)
            METHOD_FLAG="--samos_xgb_single_1000"
            LABEL="SAMOS_XGB_xo_single_1000"
            ;;
        samos-xgb-no-xo-single-1000)
            METHOD_FLAG="--samos_xgb_no_xo_single_1000"
            LABEL="SAMOS_XGB_no_xo_single_1000"
            ;;
    esac

    echo "Running:"
    echo "  POP_SIZE=$POP_SIZE"
    echo "  N_GEN=$N_GEN"
    echo "  N_GEN_INNER=$N_GEN_INNER"
    echo "  seed=$SEED"
    echo "  method=$LABEL"

    python main_nasbench101_baseline.py \
        --experiment_name ${EXPERIMENT_NAME} \
        --pop_size ${POP_SIZE} \
        --n_gen ${N_GEN} \
        --n_gen_inner ${N_GEN_INNER} \
        --warm_start_ratio ${WARM_START_RATIO} \
        --seeds ${SEED} \
        ${METHOD_FLAG}

    rsync -a --ignore-existing \
        "$SCRATCH_DIRECTORY/results/" \
        "$SLURM_SUBMIT_DIR/results/"

    # Call analyzer to generate plots after each result is copied back
    cd $SLURM_SUBMIT_DIR
    python analyze_nasbench101_baseline.py \
        --experiment_name ${EXPERIMENT_NAME} \
        --pop_size ${POP_SIZE} \
        --n_gen ${N_GEN} \
        --n_gen_inner ${N_GEN_INNER}
    cd $SCRATCH_DIRECTORY

    IDX=$(( IDX + STREAMS ))
done

# Clean up scratch
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
