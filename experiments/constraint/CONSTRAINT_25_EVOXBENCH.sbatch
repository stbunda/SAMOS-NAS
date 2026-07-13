#!/bin/bash
# --------- CONSTRAINT_25_EVOXBENCH.sbatch ---------------------------------------------------------------------------------------------------------
# GATED Q25 campaign: the S1-S4 constraint-scenario campaign
# (run_constraint.py) at the Q25 (~25%-feasible) tight thresholds
# (THRESHOLDS_Q25) WITH the hard-evaluability gate -- every task
# call passes --gate --threshold_set q25 --results_root
# results/constraint_25_gated. Under the gate, hard-mode scenarios (s1, s2)
# discard every infeasible high-fidelity evaluation (counted as waste,
# F-masked to inf, never archived; objective surrogates train on feasible
# points only, the constraint surrogate on archive + rejection log); soft
# scenarios (s3, s4) run with ungated internals as the controls -- the
# hard/soft axis is physical in this campaign, so s4 is a REAL run set here
# (gated s2 != soft s4; the s2/s4 dedup applies only ungated) rather than an
# analysis view. See run_constraint.py's module docstring
# ('Hard-evaluability gate').
#
# This is a clean fresh-campaign table: results/constraint_25_gated starts
# empty, so there is no historical index map to preserve. The results root
# is results/constraint_25_gated (run_constraint.py's --gate guard refuses
# the ungated trees; the name carries both axes: threshold set + gate).
#
# Cloned from experiments/samos/COMPARE_METHODS_EVOXBENCH.sbatch: one array
# task == one (suite, pid, scenario, method, handler) tuple, looping over all
# 20 seeds with one run_constraint.py call + one results rsync-back PER SEED
# (not one call for all 20 seeds). The scratch directory is destroyed when a
# task ends for any reason -- normal exit, walltime, preemption, node
# failure -- so per-seed rsync-back means a killed task only ever loses its
# current in-flight seed, not every seed it had already finished. Combined
# with run_constraint.py's own skip-if-exists check and the results pre-copy
# below, re-submitting a killed/interrupted array index resumes from the
# next missing seed rather than seed 0.
#
# Task table (426 tasks, gated semantics):
#   * NO random x h1-rejection rows anywhere: under --gate run_constraint.py
#     [SKIP]s them (random has no strategy -- an infeasible draw
#     consumes budget with no replacement; rejection sampling would be both
#     a strategy and, for expensive constraints, an unbudgeted oracle), so
#     submitting them would burn array tasks on no-ops. Random runs ONLY its
#     scenario default.
#   * s4 is a physical block (soft counterpart of s2, same instances --
#     incl. citysegmop via the s4 SCENARIO_OVERRIDES entries).
#   * h1-cdp-reject is folded into every scenario's handler row (this
#     is a fresh table; the frozen-map append constraint of the Q50 file
#     does not exist here) -- rows mirror run_constraint.py's
#     SCENARIO_HANDLERS exactly, b0-as-obj included.
#   * h4-cdp-sms / b0-nsga2 inner-GA control blocks cover all four
#     scenarios' instance sets, s4 and citysegmop included, so the
#     {formulation} x {inner GA} factorial exists under the gate too.
#
#   s1 (hard, GATED) on the eight #Params-threshold instances (c10mop:4,
#     in1kmop:9, c10mop:2,3,9, in1kmop:3,6,8): 15 tasks per instance --
#     random x h4-cdp, then samos/samos-cheap x s1's full 7-handler row
#     {h1-rejection, h2-penalty, h4-cdp, h5-eps, h6-sr, b0-as-obj,
#     h1-cdp-reject} = 8 x 15 = 120
#   s3 (soft, control) on the six 3-obj instances (c10mop:2,3,9,
#     in1kmop:3,6,8): 15 per instance -- random x h2-penalty, then
#     samos/samos-cheap x s3's row {h1-rejection, h2-penalty,
#     h3-adaptive-penalty, h4-cdp, h6-sr, b0-as-obj, h1-cdp-reject}
#     = 6 x 15 = 90
#   s2 (hard, GATED) on the four latency instances (c10mop:4, in1kmop:9,
#     citysegmop:2, citysegmop:3): 17 per instance -- random x h4-cdp, then
#     samos/samos-cheap x s2's full 8-handler row {h1-rejection, h2-penalty,
#     h3-adaptive-penalty, h4-cdp, h5-eps, h6-sr, b0-as-obj, h1-cdp-reject}
#     = 4 x 17 = 68
#   s4 (soft, control -- s2's soft counterpart, PHYSICAL under --gate) on
#     the same four latency instances: 15 per instance -- random x
#     h2-penalty, then samos/samos-cheap x s4's row (same as s3's)
#     = 4 x 15 = 60
#   h4-cdp-sms controls: (s1's 8 + s3's 6 + s2's 4 + s4's 4) instances x
#     {samos, samos-cheap} = 22 x 2 = 44
#   b0-nsga2 controls: same shape = 44
#   total: 120 + 90 + 68 + 60 + 44 + 44 = 426
#
# run_constraint.py always writes an objective-set-tagged output path
# ({scenario}/{suite}/pid{pid}/{objtag}/{budget}/{method}/{handler}/
# seed_N.pkl -- see objtag_for() below and run_constraint.py's module
# docstring, 'Output layout'). Gated runs additionally carry meta.gate
# (True on s1/s2 rows, False on the s3/s4 controls), meta.penalty, and
# data['handler_state'] (h3 w(t) / h5 eps(t) trajectories).
#
# NB101 instances (c10mop/pid2, in the s1/s3/control blocks) are the slow
# tasks here: NB101 issues one ORM query PER ARCHITECTURE (thousands of tiny
# reads per outer generation), which the scratch-DB copy below mitigates but
# does not eliminate -- expect these tasks to run noticeably longer than the
# other instances at the same budget.
#
# Index map (each block contiguous):
#     0-119  s1 GATED on c10mop:4, in1kmop:9, c10mop:2,3,9, in1kmop:3,6,8
#            (120: 15 tasks per instance, contiguous 15-blocks in that
#            instance order -- random:h4-cdp, then samos x s1's 7-handler
#            row, samos-cheap x same)
#   120-209  s3 control on c10mop:2,3,9, in1kmop:3,6,8 (90: 15 per instance,
#            random:h2-penalty + samos/samos-cheap x s3's 7-handler row)
#   210-277  s2 GATED on c10mop:4, in1kmop:9, citysegmop:2, citysegmop:3
#            (68: 17 per instance, random:h4-cdp + samos/samos-cheap x s2's
#            8-handler row)
#   278-337  s4 control on the same four latency instances (60: 15 per
#            instance, random:h2-penalty + samos/samos-cheap x s4's
#            7-handler row)
#   338-381  h4-cdp-sms controls (44: 2 tasks -- samos,samos-cheap -- per
#            instance; s1's 8 instances, then s3's 6, then s2's 4, then
#            s4's 4)
#   382-425  b0-nsga2 controls, same shape (44)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#
# Submit one scenario block or the control blocks:
#   sbatch --array=0-119%10   experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=120-209%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=210-277%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=278-337%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=338-425%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#
# Submit a single scenario/instance block (per the index map):
#   e.g. c10mop/pid4 s1 (first 15-block of 0-119): --array=0-14
#   e.g. citysegmop/pid2 s2 (third 17-block of 210-277): --array=244-260
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J constraint_25g
#SBATCH -c 2
#SBATCH --array=0-425%20              # full gated campaign (426 tasks total, existing results are skipped)
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_constraint_25_gated/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_constraint_25_gated

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# --------- (suite, pid, scenario, method, handler) task table --------------------------------------------------------------------------------------
# Mirrors run_constraint.py's THRESHOLDS_Q25 instances, SCENARIOS dict +
# citysegmop SCENARIO_OVERRIDES (s2 AND s4), METHODS list, DEFAULT_HANDLER,
# and SCENARIO_HANDLERS (which includes h1-cdp-reject) -- keep in sync.
default_handler() {
    case "$1" in
        s1|s2) echo h4-cdp ;;
        s3|s4) echo h2-penalty ;;
    esac
}
# Each scenario's FULL handler row incl. b0-as-obj and h1-cdp-reject. Must
# mirror run_constraint.py's SCENARIO_HANDLERS exactly (fresh table, no
# frozen-index constraint here -- unlike the Q50 file).
scenario_handlers() {
    case "$1" in
        s1)    echo "h1-rejection h2-penalty h4-cdp h5-eps h6-sr b0-as-obj h1-cdp-reject" ;;
        s2)    echo "h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h5-eps h6-sr b0-as-obj h1-cdp-reject" ;;
        s3|s4) echo "h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h6-sr b0-as-obj h1-cdp-reject" ;;
    esac
}
# Output-path objective-set tag per scenario -- mirrors run_constraint.py's
# _objtag(scenario_cfg(scenario, suite, pid)['obj_metrics']) for every
# handler except b0-as-obj/b0-nsga2, which get their own 3-column tag
# (mirrors _objtag_b0()). citysegmop needs suite+pid (not just scenario) to
# resolve its tag -- s2/s4 are per-instance overridden there (see
# run_constraint.py's SCENARIO_OVERRIDES).
objtag_for() {
    local scenario="$1" handler="$2" suite="$3" pid="$4"
    if [ "$handler" = "b0-as-obj" ] || [ "$handler" = "b0-nsga2" ]; then
        # both b0 rows share the 3-column search-objective tag; they are
        # separated by their handler directory.
        if [ "$suite" = "citysegmop" ]; then
            case "$pid" in
                2) echo "obj-err-flops-h1lat-b0" ;;
                3) echo "obj-err-params-h1lat-b0" ;;
            esac
            return
        fi
        case "$scenario" in
            s1)       echo "obj-err-flops-params-b0" ;;
            s2|s4)    echo "obj-err-params-latency-b0" ;;
            s3)       echo "obj-err-params-flops-b0" ;;
        esac
        return
    fi
    if [ "$suite" = "citysegmop" ]; then
        case "$pid" in
            2) echo "obj-err-flops" ;;
            3) echo "obj-err-params" ;;
        esac
        return
    fi
    case "$scenario" in
        s1)          echo "obj-err-flops" ;;
        s2|s3|s4)    echo "obj-err-params" ;;
    esac
}

COMBOS=()
# indices 0-119: s1 GATED on the eight instances (15 per instance: random x
# default handler only -- NO random x h1-rejection under the gate, see
# header -- then samos/samos-cheap x s1's full handler row).
for INSTANCE in c10mop:4 in1kmop:9 c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    COMBOS+=("${INSTANCE}:s1:random:$(default_handler s1)")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s1); do
            COMBOS+=("${INSTANCE}:s1:${METHOD}:${HANDLER}")
        done
    done
done
# indices 120-209: s3 soft control on the six 3-obj instances (15 per
# instance, same shape with s3's handler row).
for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    COMBOS+=("${INSTANCE}:s3:random:$(default_handler s3)")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s3); do
            COMBOS+=("${INSTANCE}:s3:${METHOD}:${HANDLER}")
        done
    done
done
# indices 210-277: s2 GATED on the four latency instances (17 per instance,
# s2's full 8-handler row).
for INSTANCE in c10mop:4 in1kmop:9 citysegmop:2 citysegmop:3; do
    COMBOS+=("${INSTANCE}:s2:random:$(default_handler s2)")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s2); do
            COMBOS+=("${INSTANCE}:s2:${METHOD}:${HANDLER}")
        done
    done
done
# indices 278-337: s4 soft control -- s2's soft counterpart, PHYSICAL under
# --gate (citysegmop via its s4 SCENARIO_OVERRIDES entries) -- on
# the same four latency instances (15 per instance, s4's handler row).
for INSTANCE in c10mop:4 in1kmop:9 citysegmop:2 citysegmop:3; do
    COMBOS+=("${INSTANCE}:s4:random:$(default_handler s4)")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s4); do
            COMBOS+=("${INSTANCE}:s4:${METHOD}:${HANDLER}")
        done
    done
done
# indices 338-381: h4-cdp-sms inner-GA controls (constrained CDP with
# SMS-EMOA inner instead of NSGA-II), samos/samos-cheap only, all four
# scenarios' instance sets (inner-GA factorial under the gate).
for HANDLER in h4-cdp-sms; do
    for INSTANCE in c10mop:4 in1kmop:9 c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s1:${METHOD}:${HANDLER}")
        done
    done
    for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s3:${METHOD}:${HANDLER}")
        done
    done
    for INSTANCE in c10mop:4 in1kmop:9 citysegmop:2 citysegmop:3; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s2:${METHOD}:${HANDLER}")
        done
    done
    for INSTANCE in c10mop:4 in1kmop:9 citysegmop:2 citysegmop:3; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s4:${METHOD}:${HANDLER}")
        done
    done
done
# indices 382-425: b0-nsga2 controls (constraint-as-objective with NSGA-II
# inner instead of SMS-EMOA), same shape (44).
for HANDLER in b0-nsga2; do
    for INSTANCE in c10mop:4 in1kmop:9 c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s1:${METHOD}:${HANDLER}")
        done
    done
    for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s3:${METHOD}:${HANDLER}")
        done
    done
    for INSTANCE in c10mop:4 in1kmop:9 citysegmop:2 citysegmop:3; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s2:${METHOD}:${HANDLER}")
        done
    done
    for INSTANCE in c10mop:4 in1kmop:9 citysegmop:2 citysegmop:3; do
        for METHOD in samos samos-cheap; do
            COMBOS+=("${INSTANCE}:s4:${METHOD}:${HANDLER}")
        done
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r SUITE PID SCENARIO METHOD HANDLER <<< "$TASK_ENTRY"
OBJTAG=$(objtag_for "$SCENARIO" "$HANDLER" "$SUITE" "$PID")   # mirrors run_constraint.py's output path -- see header

# --------- scratch setup -----------------------------------------------------------------------------------------------------------------------------------
SCRATCH_DIRECTORY="/local/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "$SCRATCH_DIRECTORY" ]; then
    echo "'$SCRATCH_DIRECTORY' already found!"
else
    echo "'$SCRATCH_DIRECTORY' not found, creating!"
    mkdir -p "$SCRATCH_DIRECTORY"
fi

RSYNC_SOURCES=(
    problem/evoxbench
    problem/pymoo
    strategy
    experiments/samos
    experiments/constraint
)

# Pre-copy any already-computed results for this (suite, pid, scenario,
# objtag, method, handler) so run_constraint.py's own skip-if-exists logic
# can resume mid-seed-range.
BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
_SRC="results/constraint_25_gated/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
[ -d "$_SRC" ] && RSYNC_SOURCES+=("$_SRC")

rsync -aR "${RSYNC_SOURCES[@]}" "$SCRATCH_DIRECTORY/" 2>/dev/null || true

# EvoXBench SQLite database (~2.8G): copied to node-local scratch because
# NB101/NATS issue one ORM query PER ARCHITECTURE (thousands of tiny reads
# per outer generation) -- brutal over the shared filesystem, cheap on
# /local. Separate rsync with -L (--copy-links): problem/data/evoxbench may
# be a symlink/junction to another checkout, and the scratch copy must be
# real files, not a link back to the shared FS. The model/data dir (1.3G) is
# NOT copied: those files are loaded once per benchmark construction, so
# shared-FS latency is irrelevant there (EVOXBENCH_MODEL below keeps them on
# submit dir).
rsync -aRL problem/data/evoxbench/database "$SCRATCH_DIRECTORY/" 2>/dev/null || true

cd "$SCRATCH_DIRECTORY"

echo "Copied structure:"
find "${SCRATCH_DIRECTORY}" -maxdepth 4 -type d 2>/dev/null | head -20

# --------- environment ---------------------------------------------------------------------------------------------------------------------------------------
module load anaconda3/2023.09
source activate
conda deactivate
conda activate SAMOS311

# CPU-only job: hide all GPUs so CUDA-enabled builds (xgboost, torch via
# evoxbench) cannot create a CUDA context / reserve VRAM on this shared node.
export CUDA_VISIBLE_DEVICES=""

# --------- point evoxbench's SQLite at the scratch copy -------------------------------------------------------------------------------
# evoxbench resolves its Django `ORM` package (and thus db.sqlite3, via that
# package's BASE_DIR) from sys.path, and only `setdefault`s EVOXBENCH_DATA /
# EVOXBENCH_MODEL from the shared ~/.config/evoxbench/config.json. Exporting
# these BEFORE python starts therefore wins over the shared json with no race
# between array tasks (the json itself keeps its submit-dir value, same as
# before -- see problem/evoxbench/utils.py). PYTHONPATH puts the scratch
# database dir ahead of the json's sys.path.append, so ORM + db.sqlite3
# resolve from /local scratch; EVOXBENCH_MODEL keeps the (one-time-loaded)
# model files on the shared FS, matching what utils.py configures.
export PYTHONPATH="${SCRATCH_DIRECTORY}/problem/data/evoxbench/database${PYTHONPATH:+:$PYTHONPATH}"
export EVOXBENCH_DATA="${SCRATCH_DIRECTORY}/problem/data/evoxbench/database"
export EVOXBENCH_MODEL="${SLURM_SUBMIT_DIR}/problem/data/evoxbench/data"

echo "Activated: SAMOS311"

# --------- logging -------------------------------------------------------------------------------------------------------------------------------------------
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Scratch Directory = $SCRATCH_DIRECTORY"
echo "Array Task ID     = $SLURM_ARRAY_TASK_ID"
echo "SUITE             = $SUITE"
echo "PID               = $PID"
echo "SCENARIO          = $SCENARIO"
echo "METHOD            = $METHOD"
echo "HANDLER           = $HANDLER"
echo "OBJTAG            = $OBJTAG"
echo "GATE              = --gate (hard-mode rows gated, s3/s4 controls ungated internals)"
echo "SEEDS             = 0-19"
echo "POP_SIZE          = $POP_SIZE"
echo "N_GEN             = $N_GEN"
echo "N_GEN_INNER       = $N_GEN_INNER"
echo "BUDGET            = $(( N_GEN * POP_SIZE ))"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "EVOXBENCH_DATA    = $EVOXBENCH_DATA"
echo "EVOXBENCH_MODEL   = $EVOXBENCH_MODEL"

# --------- run: one seed at a time, rsync-back after each ------------------------------------------------------------------------------------------------
DEST="${SLURM_SUBMIT_DIR}/results/constraint_25_gated/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- SUITE: $SUITE  PID: $PID  SCENARIO: $SCENARIO  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED  (GATED q25)"

    python experiments/constraint/run_constraint.py \
        --scenario      "$SCENARIO" \
        --suite         "$SUITE" \
        --pid           "$PID" \
        --method        "$METHOD" \
        --handler       "$HANDLER" \
        --seeds         "$SEED" \
        --pop_size      "$POP_SIZE" \
        --n_gen         "$N_GEN" \
        --n_gen_inner   "$N_GEN_INNER" \
        --threshold_set q25 \
        --gate \
        --results_root  results/constraint_25_gated \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results/constraint_25_gated/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
