#!/bin/bash
# parameters for slurm
#SBATCH -J nasbench101_lut              # job name
#SBATCH -c 1                            # single core — enumeration is sequential
#SBATCH --mem=8gb                       # job memory request
#SBATCH --gres=gpu:0                    # no GPUs needed
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=01:30:00                 # 30 min is well above the expected 3-5 min
#SBATCH --output=slurm_logs_nasbench101_baseline/precompute_lut_%j.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

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
    problem/*.py \
    problem/data/data_nasbench101.pkl \
    strategy \
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

# Build the LUT
python -m problem.precompute_nasbench101_lut

echo "Date (done)       = $(date)"

# Copy the generated LUT back to the submit directory
rsync -a \
    "$SCRATCH_DIRECTORY/problem/data/nasbench101_lut.pkl" \
    "$SLURM_SUBMIT_DIR/problem/data/"

echo "LUT copied to $SLURM_SUBMIT_DIR/problem/data/nasbench101_lut.pkl"

# Clean up scratch
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
