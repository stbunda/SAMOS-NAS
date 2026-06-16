#!/bin/bash
# ─── OBJ_GA_EVOXBENCH.sbatch ──────────────────────────────────────────────────
# Experiment 1 (GA comparison under varying objectives) — EvoXBench NAS suites.
#
# One array task == one seed. Each task loops over every METHOD x every PID in
# the (BENCHMARK, N_OBJ) sub-experiment. The Python entry point auto-skips
# already-completed pickles, so re-submitting an array is safe.
#
# Walltime is 72h: in1kmop uses surrogate evaluators and is slow.
#
# Submit one sub-experiment at a time via --export:
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=2 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=3 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=4 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=2 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=3 OBJ_GA_EVOXBENCH.sbatch
#   sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=4 OBJ_GA_EVOXBENCH.sbatch
#
# Selective single-seed rerun (force overwrite):
#   sbatch --array=5 --export=ALL,BENCHMARK=in1kmop,N_OBJ=4,OVERWRITE=1 OBJ_GA_EVOXBENCH.sbatch
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -J obj_ga_evoxbench
#SBATCH -c 2
#SBATCH --array=0-19%5            # 20 tasks = 20 seeds, 5 concurrent
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs_obj_ga_evoxbench/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm_logs_obj_ga_evoxbench

# ─── environment ──────────────────────────────────────────────────────────────
module load nvidia/nvhpc/26.3
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

echo "Activated: SAMOS311"

# ─── parameters from submission env ───────────────────────────────────────────
BENCHMARK=${BENCHMARK:-c10mop}
N_OBJ=${N_OBJ:-2}
SEED=$SLURM_ARRAY_TASK_ID
CONFIG=config/experiment_obj_ga.yaml

# ─── method set ───────────────────────────────────────────────────────────────
METHODS=(random nsga2 nsga3 moead sms-emoa rvea age-moea age-moea2)

# ─── PID set: depends on (BENCHMARK, N_OBJ) ───────────────────────────────────
case "${BENCHMARK}_${N_OBJ}" in
    c10mop_2)  PIDS=(1 8) ;;
    c10mop_3)  PIDS=(2 3 9) ;;
    c10mop_4)  PIDS=(4 5 6 7) ;;
    in1kmop_2) PIDS=(1 4 7) ;;
    in1kmop_3) PIDS=(3 6) ;;
    in1kmop_4) PIDS=(9) ;;
    *)
        echo "ERROR: unsupported BENCHMARK/N_OBJ combination: ${BENCHMARK}_${N_OBJ}" >&2
        echo "       Expected BENCHMARK in {c10mop,in1kmop} and N_OBJ in {2,3,4}." >&2
        exit 1
        ;;
esac

# ─── logging ──────────────────────────────────────────────────────────────────
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "BENCHMARK         = $BENCHMARK"
echo "N_OBJ             = $N_OBJ"
echo "SEED              = $SEED"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "METHODS           = ${METHODS[*]}"
echo "PIDS              = ${PIDS[*]}"

# ─── run: METHOD x PID ────────────────────────────────────────────────────────
for METHOD in "${METHODS[@]}"; do
    for PID in "${PIDS[@]}"; do
        echo ""
        echo "─── benchmark: $BENCHMARK  n_obj: $N_OBJ  pid: $PID  method: $METHOD  seed: $SEED"

        python experiment_obj_ga.py \
            --benchmark "$BENCHMARK" \
            --pid       "$PID" \
            --n_obj     "$N_OBJ" \
            --method    "$METHOD" \
            --seed      "$SEED" \
            --config    "$CONFIG" \
            ${OVERWRITE:+--overwrite}
    done
done

echo ""
echo "Date (done)       = $(date)"
