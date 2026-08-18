#!/bin/bash
# --------- BIAS_EVOXBENCH.sbatch ----------------------------------------------------------------------------------------------------------------
# The sample-selection-bias campaign (experiments2/sample_selection_bias/):
# NB201 c10mop/pid7, 4 #Params tightness levels x 3 roles for #Params x 4
# hardware metrics x 2 surrogate-training arms, samos + CDP + hard gate
# throughout. Results ->
# results2/sample_selection_bias/{tightness}/{role}/{hw}/{arm}/seed_N.pkl.
#
# Termination is EVALUATION-based (('n_evals', N)): the hard gate's DOE
# redraws whole batches when too few draws are evaluable, and every draw is a
# real benchmark query, so an n_gen budget would silently hand the tighter
# levels more evaluations for the "same" run.
#
# SPACE selects the benchmark and therefore the task table:
#
#   NB201 (default, 80 tasks) -- tabular, exact reference. 10 runnable
#     (role, hw) pairs, not 12: the two roles that put arithmetic intensity in
#     G alongside the size gate have an EMPTY feasible set (config.SPACES's
#     'excluded'), so they are skipped here and by run_bias.py itself.
#       roles x hw : obj x 3 + constr x 4 + multi x 3 = 10
#       x 4 tightness levels x 2 arms                 = 80
#
#   MobileNetV3 (24 tasks) -- surrogate benchmark, 21 variables, sampled
#     reference. Exists because NB201 SATURATES: 1 200 evaluations enumerate
#     7.7% of it and cells reach 95% of the exact optimum in 40-120
#     evaluations, so final HV cannot separate the arms. One hardware metric
#     (Latency -- FLOPs is a static count, not a hardware budget), no
#     exclusions: 3 roles x 1 hw x 4 tightness x 2 arms = 24.
#     SUBMIT WITH AN EXPLICIT ARRAY RANGE (the #SBATCH default below is
#     NB201's 0-79; indices 24-79 would just exit 1 on this space):
#       sbatch --export=ALL,SPACE=MobileNetV3 --array=0-23%20 \
#           experiments2/sample_selection_bias/BIAS_EVOXBENCH.sbatch
#
# Index map: tightness-major, then (role, hw) in config order, then arm.
#   task = tightness_idx * 20 + pair_idx * 2 + arm_idx
#     tightness_idx: T1=0..T4=3 ; arm_idx: feasible=0, oracle=1
#
#   index  0 = T1 / obj    / edgegpu_latency / feasible
#   index  1 = T1 / obj    / edgegpu_latency / oracle
#   index 19 = T1 / multi  / edgegpu_energy  / oracle   (last T1 task)
#   index 20 = T2 / obj    / edgegpu_latency / feasible
#   index 79 = T4 / multi  / edgegpu_energy  / oracle   (last task)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments2/sample_selection_bias/BIAS_EVOXBENCH.sbatch
#
# Submit one tightness level (20-blocks per the index map above):
#   sbatch --array=0-19  experiments2/sample_selection_bias/BIAS_EVOXBENCH.sbatch  # T1
#   sbatch --array=60-79 experiments2/sample_selection_bias/BIAS_EVOXBENCH.sbatch  # T4
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments2/sample_selection_bias/BIAS_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J bias_run
#SBATCH -c 2
#SBATCH --array=0-79%20
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_sample_selection_bias/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_sample_selection_bias

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
SPACE=${SPACE:-NB201}
POP_SIZE=${POP_SIZE:-20}
N_EVALS=${N_EVALS:-1200}
PROBE_N=${PROBE_N:-2000}
SEEDS_LAST=${SEEDS_LAST:-19}

# --------- (tightness, role, hw, arm) task table -----------------------------------------------------------------------------------------
# Mirrors config.SPACES[$SPACE] -- keep in sync. Verified index-for-index
# against config.cells() for both spaces.
TIGHTNESS_LIST="T1 T2 T3 T4"
ARMS_LIST="feasible oracle"
case "$SPACE" in
    NB201)
        HW_LIST="edgegpu_latency eyeriss_latency edgegpu_energy eyeriss_arithmetic_intensity"
        RESULTS_ROOT="results2/sample_selection_bias"
        ;;
    MobileNetV3)
        HW_LIST="latency"
        RESULTS_ROOT="results2/sample_selection_bias_mobilenetv3"
        ;;
    *)
        echo "ERROR: unknown SPACE='$SPACE' (expected NB201 or MobileNetV3)" >&2
        exit 1
        ;;
esac

COMBOS=()
for TIGHTNESS in $TIGHTNESS_LIST; do
    for ROLE in obj constr multi; do
        for HW in $HW_LIST; do
            # config 'excluded': AI as a CONSTRAINT next to the size gate is empty.
            if [ "$HW" = "eyeriss_arithmetic_intensity" ] && [ "$ROLE" != "constr" ]; then
                continue
            fi
            for ARM in $ARMS_LIST; do
                COMBOS+=("${TIGHTNESS}:${ROLE}:${HW}:${ARM}")
            done
        done
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r TIGHTNESS ROLE HW ARM <<< "$TASK_ENTRY"

# --------- scratch setup -----------------------------------------------------------------------------------------------------------------------------------
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIRECTORY"

RSYNC_SOURCES=(
    problem/evoxbench
    problem/pymoo
    strategy
    experiments2/sample_selection_bias
)

# Pre-copy already-computed results for this cell so run_bias.py's own
# skip-if-exists check can resume mid-seed-range.
_SRC="${RESULTS_ROOT}/${TIGHTNESS}/${ROLE}/${HW}/${ARM}"
[ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/" 2>/dev/null || true

# EvoXBench SQLite database (~2.8G) onto node-local scratch: the tabular
# suites issue one ORM query PER ARCHITECTURE, which is brutal over the shared
# filesystem. -L because problem/data/evoxbench may be a symlink to another
# checkout and the scratch copy must be real files.
rsync -aRL problem/data/evoxbench/database "$SCRATCH_DIRECTORY/" 2>/dev/null || true

cd "$SCRATCH_DIRECTORY"

# --------- environment ---------------------------------------------------------------------------------------------------------------------------------------
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

# CPU-only job: hide all GPUs so CUDA-enabled builds (xgboost, torch via
# evoxbench) cannot create a CUDA context on this shared node.
export CUDA_VISIBLE_DEVICES=""

# Point evoxbench's SQLite at the scratch copy (see SCENARIO_EVOXBENCH.sbatch
# for the full explanation of why these three exports must precede python).
export PYTHONPATH="${SCRATCH_DIRECTORY}/problem/data/evoxbench/database${PYTHONPATH:+:$PYTHONPATH}"
export EVOXBENCH_DATA="${SCRATCH_DIRECTORY}/problem/data/evoxbench/database"
export EVOXBENCH_MODEL="${SLURM_SUBMIT_DIR}/problem/data/evoxbench/data"

# --------- logging -------------------------------------------------------------------------------------------------------------------------------------------
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"
echo "Array Task ID     = $SLURM_ARRAY_TASK_ID"
echo "SPACE             = $SPACE"
echo "TIGHTNESS         = $TIGHTNESS"
echo "ROLE              = $ROLE"
echo "HW                = $HW"
echo "ARM               = $ARM"
echo "SEEDS             = 0-$SEEDS_LAST"
echo "POP_SIZE          = $POP_SIZE"
echo "N_EVALS           = $N_EVALS"
echo "PROBE_N           = $PROBE_N"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"

# --------- run: one seed at a time, rsync-back after each ------------------------------------------------------------------------------------------------
DEST="${SLURM_SUBMIT_DIR}/${RESULTS_ROOT}/${TIGHTNESS}/${ROLE}/${HW}/${ARM}"
mkdir -p "$DEST"

for SEED in $(seq 0 "$SEEDS_LAST"); do
    echo ""
    echo "--------- $SPACE / $TIGHTNESS / $ROLE / $HW / $ARM  seed: $SEED"

    python experiments2/sample_selection_bias/run_bias.py \
        --space        "$SPACE" \
        --tightness    "$TIGHTNESS" \
        --role         "$ROLE" \
        --hw           "$HW" \
        --arm          "$ARM" \
        --seeds        "$SEED" \
        --pop_size     "$POP_SIZE" \
        --n_evals      "$N_EVALS" \
        --probe_n      "$PROBE_N" \
        --results_root "$RESULTS_ROOT" \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/${RESULTS_ROOT}/${TIGHTNESS}/${ROLE}/${HW}/${ARM}/" \
        "$DEST/" 2>/dev/null || true
done

echo ""
echo "Results copied to $DEST"
echo "Date (done)       = $(date)"

echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
