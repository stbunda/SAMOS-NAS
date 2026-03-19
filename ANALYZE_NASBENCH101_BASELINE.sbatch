#!/bin/bash
# parameters for slurm
#SBATCH -J nasbench101_analyze        # job name
#SBATCH -c 1                          # single core — analysis is sequential
#SBATCH --mem=16gb                    # job memory request (bench_db + LUT + results)
#SBATCH --gres=gpu:0                  # no GPUs needed
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs_nasbench101_baseline/analyze_%j.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

# Variables
POP_SIZE=20
N_GEN=50

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
    results/nasbench101_baseline \
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
echo "Working Directory = $(pwd)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"

# Run analysis
python analyze_nasbench101_baseline.py \
    --pop_size ${POP_SIZE} \
    --n_gen ${N_GEN}

echo "Date (done)       = $(date)"

# Copy plots and table back to submit directory
rsync -a \
    "$SCRATCH_DIRECTORY/results/nasbench101_baseline/" \
    "$SLURM_SUBMIT_DIR/results/nasbench101_baseline/"

echo "Results copied to $SLURM_SUBMIT_DIR/results/nasbench101_baseline/"

# Clean up scratch
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
