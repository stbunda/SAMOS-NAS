#!/bin/bash
# --------- CONSTRAINT_25_EVOXBENCH.sbatch ---------------------------------------------------------------------------------------------------------
# Q25 (~25%-feasible) tight-threshold variant of CONSTRAINT_EVOXBENCH.sbatch:
# same S1-S3 constraint-scenario campaign (run_constraint.py), but its OWN
# clean (suite, pid, scenario, method, handler) task table and index map --
# NOT the same as that file's (see Task count / Index map below) -- plus
# every task call adds --threshold_set q25 --results_root
# results/constraint_25, so runs use the tighter THRESHOLDS_Q25 table
# (run_constraint.py) instead of the default THRESHOLDS median table, and
# land under results/constraint_25, never under results/constraint
# (run_constraint.py's --threshold_set guard refuses to mix the two in one
# results tree) -- see experiments/constraint/THRESHOLDS.md for the Q25
# provenance (same fixed-seed 10k samples the medians came from).
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
# Task count: 292, a clean fresh-campaign table (results/constraint_25 starts
# empty, so there is no historical index map to preserve -- see below):
#   s1 on all eight instances (c10mop:4, in1kmop:9, c10mop:2,3,9, in1kmop:3,
#     6,8): 14 tasks per instance -- random x default handler, random x
#     h1-rejection, then samos/samos-cheap x s1's full handler row
#     {h1-rejection, h2-penalty, h4-cdp, h5-eps, h6-sr, b0-as-obj} = 8 x 14
#     = 112
#   s3 on the six 3-obj instances only (c10mop:2,3,9, in1kmop:3,6,8 -- c10mop:4
#     / in1kmop:9 carry no FLOPs threshold, so s3 skips them): same
#     per-instance shape with s3's handler row {h1-rejection, h2-penalty,
#     h3-adaptive-penalty, h4-cdp, h6-sr, b0-as-obj} = 6 x 14 = 84
#   s2 on c10mop:4 / in1kmop:9 only: 16 tasks per instance -- random x
#     default, random x h1-rejection, then samos/samos-cheap x s2's full
#     7-handler row {h1-rejection, h2-penalty, h3-adaptive-penalty, h4-cdp,
#     h5-eps, h6-sr, b0-as-obj} = 2 x 16 = 32 -- s2 and s4 are the physically
#     identical run set, s4 is an analysis view (run_constraint.py refuses
#     --scenario s4), so no s4 row is ever built below, and s2 is built
#     exactly once (not once per round, as in the frozen table)
#   h4-cdp-sms controls (constrained CDP with SMS-EMOA inner instead of
#     NSGA-II): same three scenario/instance sets as above, samos/samos-cheap
#     only = (8 + 6 + 2) x 2 = 32
#   b0-nsga2 controls (constraint-as-objective with NSGA-II inner instead of
#     SMS-EMOA): same shape = 32
#   citysegmop suite (s5 on pid2, s6 on pid3 -- a hard constraint on H1 Lat.,
#     the genuinely predictor-backed expensive metric on this suite, see
#     run_constraint.py's module docstring): 16 tasks per instance -- random
#     x default handler, random x h1-rejection, then samos/samos-cheap x the
#     scenario's full 7-handler row {h1-rejection, h2-penalty,
#     h3-adaptive-penalty, h4-cdp, h5-eps, h6-sr, b0-as-obj} = 2 x 16 = 32
#   total: 112 + 84 + 32 + 32 + 32 + 32 = 324
#
# b0-as-obj (unconstrained 3-objective search -- scenario's 2 objectives
# plus its constrained metric as an ordinary 3rd objective, no G, SMS-EMOA
# inner) is folded straight into each scenario's full handler row instead of
# being a separate appended round -- see scenario_handlers() below, which
# returns the FULL row including b0-as-obj for every scenario, mirroring
# run_constraint.py's SCENARIO_HANDLERS exactly.
#
# This file's indices are NOT compatible with CONSTRAINT_EVOXBENCH.sbatch's:
# that file keeps a FROZEN historical map (368 tasks) so its old array
# indices stay meaningful against already-existing Q50 results. This
# campaign starts fresh against an empty results/constraint_25, so the dead
# weight that map carries is simply absent here: no s4 rows (they would exit
# 1 -- s4 is an analysis view of s2, see above), no s3 rows on c10mop:4 /
# in1kmop:9 (that file's round-1 s3 entries there error out: those instances
# have no FLOPs threshold, and their objectives predate the per-scenario
# objective change), and no duplicated s2 block (that file runs s2 on
# pid4/pid9 twice -- once in its round-1/2 block, once again in its round-3
# union block; this table runs it once).
#
# run_constraint.py always writes an objective-set-tagged output path
# ({scenario}/{suite}/pid{pid}/{objtag}/{budget}/{method}/{handler}/
# seed_N.pkl, objtag = 'obj-err-flops' for s1 / 'obj-err-params' for s2/s3,
# its own 3-column tag for b0-as-obj/b0-nsga2 -- see objtag_for() below and
# run_constraint.py's module docstring, 'Output layout').
#
# NB101 instances (c10mop/pid2, in the s1-on-eight and h4-cdp-sms/b0-nsga2
# blocks) are the slow tasks here: NB101 issues one ORM query PER
# ARCHITECTURE (thousands of tiny reads per outer generation), which the
# scratch-DB copy below mitigates but does not eliminate -- expect these
# tasks to run noticeably longer than the other instances at the same
# budget.
#
# Index map (each block contiguous):
#     0-111  s1 on c10mop:4, in1kmop:9, c10mop:2,3,9, in1kmop:3,6,8 (112: 14
#            tasks per instance, contiguous 14-blocks in that instance
#            order -- random:default, random:h1-rejection, then samos x
#            {h1,h2,h4,h5,h6,b0-as-obj}, samos-cheap x same)
#   112-195  s3 on c10mop:2,3,9, in1kmop:3,6,8 (84: 14 tasks per instance,
#            same shape with s3's handler row {h1,h2,h3-adaptive-penalty,h4,
#            h6,b0-as-obj})
#   196-227  s2 on c10mop:4, in1kmop:9 (32: 16 tasks per instance --
#            random:default, random:h1-rejection, then samos/samos-cheap x
#            the full 7-handler s2 row {h1,h2,h3-adaptive-penalty,h4,h5,h6,
#            b0-as-obj}); also s4's physical data (see above)
#   228-259  h4-cdp-sms controls, same scenario/instance sets as above,
#            samos/samos-cheap only (32: s1 on the eight instances (16),
#            then s3 on the six 3-obj instances (12), then s2 on pid4/pid9
#            (4), each instance a contiguous 2-block -- samos, samos-cheap)
#   260-291  b0-nsga2 controls, same shape (32)
#   292-307  s5 on citysegmop:2 (16: random x {h4-cdp,h1-rejection}, then
#            samos/samos-cheap x s5's full 7-handler row)
#   308-323  s6 on citysegmop:3, same shape (16)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#
# Submit only s1, only s3, only s2, only the SMS-EMOA/NSGA-II control
# blocks, or only the citysegmop suite (new submissions should usually use
# one of these ranges):
#   sbatch --array=0-111%10   experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=112-195%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=196-227%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=228-291%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#   sbatch --array=292-323%10 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#
# Submit a single scenario/instance block (per the index map):
#   e.g. c10mop/pid4 s1 (first 14-block of 0-111): --array=0-13
#   e.g. c10mop/pid2 s3 (first 14-block of 112-195): --array=112-125
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments/constraint/CONSTRAINT_25_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J constraint_25
#SBATCH -c 2
#SBATCH --array=0-323%20              # full campaign (324 tasks total, existing results are skipped)
##SBATCH --array=228-291%20           # h4-cdp-sms / b0-nsga2 control blocks only
##SBATCH --array=196-227%20           # s2 block only
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_constraint_25/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_constraint_25

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# --------- (suite, pid, scenario, method, handler) task table --------------------------------------------------------------------------------------
# Mirrors run_constraint.py's THRESHOLDS instances, SCENARIOS dict, METHODS
# list, DEFAULT_HANDLER, and SCENARIO_HANDLERS (D22 matrix) -- keep in sync.
# Each scenario's block below is random x default handler, random x
# h1-rejection, then samos/samos-cheap x the scenario's FULL handler row
# (scenario_handlers(), which now includes b0-as-obj) -- one pass per
# scenario, nothing appended in a later round (see Task count above).
default_handler() {
    case "$1" in
        s1|s2|s5|s6) echo h4-cdp ;;
        s3|s4)       echo h2-penalty ;;
    esac
}
# Each scenario's FULL handler row, including b0-as-obj. Must mirror
# run_constraint.py's SCENARIO_HANDLERS exactly (plus b0-as-obj, folded in
# here rather than appended as a separate round -- see Task count above).
scenario_handlers() {
    case "$1" in
        s1)    echo "h1-rejection h2-penalty h4-cdp h5-eps h6-sr b0-as-obj" ;;
        s3)    echo "h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h6-sr b0-as-obj" ;;
        s2)    echo "h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h5-eps h6-sr b0-as-obj" ;;
        s5|s6) echo "h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h5-eps h6-sr b0-as-obj" ;;
    esac
}
# Output-path objective-set tag per scenario -- mirrors run_constraint.py's
# _objtag(SCENARIOS[scenario]['obj_metrics']) for every handler except
# b0-as-obj/b0-nsga2, which get their own 3-column tag (mirrors
# _objtag_b0(), see run_constraint.py's 'b0-as-obj' docstring / 'Output
# layout'). Applies to every index in COMBOS (0-323).
objtag_for() {
    local scenario="$1" handler="$2"
    if [ "$handler" = "b0-as-obj" ] || [ "$handler" = "b0-nsga2" ]; then
        # both b0 rows share the 3-column search-objective tag; they are
        # separated by their handler directory.
        case "$scenario" in
            s1)       echo "obj-err-flops-params-b0" ;;
            s2|s4)    echo "obj-err-params-latency-b0" ;;
            s3)       echo "obj-err-params-flops-b0" ;;
            s5)       echo "obj-err-flops-h1lat-b0" ;;
            s6)       echo "obj-err-params-h1lat-b0" ;;
        esac
        return
    fi
    case "$scenario" in
        s1)          echo "obj-err-flops" ;;
        s2|s3|s4)    echo "obj-err-params" ;;
        s5)          echo "obj-err-flops" ;;
        s6)          echo "obj-err-params" ;;
    esac
}

COMBOS=()
# indices 0-111: s1 on all eight instances (14 per instance: random x
# default handler, random x h1-rejection, then samos/samos-cheap x s1's
# full handler row, which includes b0-as-obj -- see scenario_handlers()
# above)
for INSTANCE in c10mop:4 in1kmop:9 c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    COMBOS+=("${INSTANCE}:s1:random:$(default_handler s1)")
    COMBOS+=("${INSTANCE}:s1:random:h1-rejection")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s1); do
            COMBOS+=("${INSTANCE}:s1:${METHOD}:${HANDLER}")
        done
    done
done
# indices 112-195: s3 on the six 3-obj instances (84: 14 tasks per instance,
# same shape; c10mop:4 / in1kmop:9 have no FLOPs threshold, so s3 skips them)
for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    COMBOS+=("${INSTANCE}:s3:random:$(default_handler s3)")
    COMBOS+=("${INSTANCE}:s3:random:h1-rejection")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s3); do
            COMBOS+=("${INSTANCE}:s3:${METHOD}:${HANDLER}")
        done
    done
done
# indices 196-227: s2 on c10mop:4 / in1kmop:9 (32: 16 tasks per instance --
# random x default, random x h1-rejection, then samos/samos-cheap x s2's
# full 7-handler row). This is ALSO s4's physical data (analysis view, no
# separate submission -- run_constraint.py refuses --scenario s4, see
# header).
for INSTANCE in c10mop:4 in1kmop:9; do
    COMBOS+=("${INSTANCE}:s2:random:$(default_handler s2)")
    COMBOS+=("${INSTANCE}:s2:random:h1-rejection")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s2); do
            COMBOS+=("${INSTANCE}:s2:${METHOD}:${HANDLER}")
        done
    done
done
# indices 228-259: h4-cdp-sms controls (constrained CDP with SMS-EMOA inner
# instead of NSGA-II), same scenario/instance sets as above, samos/samos-cheap
# only (32 = s1 on the eight instances (16) + s3 on the six 3-obj instances
# (12) + s2 on pid4/pid9 (4)).
for INSTANCE in c10mop:4 in1kmop:9 c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s1:${METHOD}:h4-cdp-sms")
    done
done
for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s3:${METHOD}:h4-cdp-sms")
    done
done
for INSTANCE in c10mop:4 in1kmop:9; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s2:${METHOD}:h4-cdp-sms")
    done
done
# indices 260-291: b0-nsga2 controls (constraint-as-objective with NSGA-II
# inner instead of SMS-EMOA), same shape as the h4-cdp-sms block (32).
for INSTANCE in c10mop:4 in1kmop:9 c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s1:${METHOD}:b0-nsga2")
    done
done
for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s3:${METHOD}:b0-nsga2")
    done
done
for INSTANCE in c10mop:4 in1kmop:9; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s2:${METHOD}:b0-nsga2")
    done
done
# indices 292-323: citysegmop suite (s5 on pid2, s6 on pid3 -- a hard
# constraint on H1 Lat., see header). 16 tasks per instance: random x
# default handler, random x h1-rejection, then samos/samos-cheap x the
# scenario's full 7-handler row (scenario_handlers(), which already
# includes b0-as-obj for s5/s6 -- see above).
for SCENARIO in s5 s6; do
    INSTANCE=$([ "$SCENARIO" = "s5" ] && echo citysegmop:2 || echo citysegmop:3)
    COMBOS+=("${INSTANCE}:${SCENARIO}:random:$(default_handler "$SCENARIO")")
    COMBOS+=("${INSTANCE}:${SCENARIO}:random:h1-rejection")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers "$SCENARIO"); do
            COMBOS+=("${INSTANCE}:${SCENARIO}:${METHOD}:${HANDLER}")
        done
    done
done

TASK_ENTRY="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$TASK_ENTRY" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in COMBOS (size ${#COMBOS[@]})" >&2
    exit 1
fi
IFS=':' read -r SUITE PID SCENARIO METHOD HANDLER <<< "$TASK_ENTRY"
OBJTAG=$(objtag_for "$SCENARIO" "$HANDLER")   # mirrors run_constraint.py's output path -- see header

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
# can resume mid-seed-range. The objtag level is always present -- see header
# ('run_constraint.py now ALWAYS writes an objective-set-tagged output path').
BUDGET_FOLDER="B$(( N_GEN * POP_SIZE ))_P${POP_SIZE}"
_SRC="results/constraint_25/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
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
echo "SEEDS             = 0-19"
echo "POP_SIZE          = $POP_SIZE"
echo "N_GEN             = $N_GEN"
echo "N_GEN_INNER       = $N_GEN_INNER"
echo "BUDGET            = $(( N_GEN * POP_SIZE ))"
echo "OVERWRITE         = ${OVERWRITE:-<unset>}"
echo "EVOXBENCH_DATA    = $EVOXBENCH_DATA"
echo "EVOXBENCH_MODEL   = $EVOXBENCH_MODEL"

# --------- run: one seed at a time, rsync-back after each ------------------------------------------------------------------------------------------------
DEST="${SLURM_SUBMIT_DIR}/results/constraint_25/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
mkdir -p "$DEST"

for SEED in $(seq 0 19); do
    echo ""
    echo "--------- SUITE: $SUITE  PID: $PID  SCENARIO: $SCENARIO  METHOD: $METHOD  HANDLER: $HANDLER  seed: $SEED"

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
        --results_root  results/constraint_25 \
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results/constraint_25/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
