#!/bin/bash
# --------- CONSTRAINT_EVOXBENCH.sbatch ------------------------------------------------------------------------------------------------------------
# Runs experiments/constraint/run_constraint.py (S1-S4 constraint-scenario
# campaign: random / samos / samos-cheap on c10mop/pid4 AND in1kmop/pid9,
# objectives {Err, FLOPs}, one inequality constraint per scenario -- see
# text/plans/EXPERIMENT_PLAN_constraint_scenarios.md and
# experiments/constraint/THRESHOLDS.md), 20 seeds each, at the
# compare_methods.py budget (pop_size=20, n_gen=60, n_gen_inner=20,
# ga_pop_size=pop_size*10).
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
# Task count: 304 = 100 rounds-1/2 tasks (indices 0-99, FROZEN -- same
# (suite,pid,scenario,method,handler) table as before) + 172 round-3
# additions (indices 100-271, FROZEN -- same as before) + 32 round-4 B0
# additions (indices 272-303, APPENDED after so 0-271 keep their meaning).
#
# Round-3 (instance axis: the constrained metric on every 3-objective problem
# is its own "removed third objective" -- s1 constrains #Params with
# objectives {Err, FLOPs}, s3 constrains FLOPs with objectives {Err,
# #Params}; s2 drops FLOPs and switches to objectives {Err, #Params} too, and
# is now run ONCE per latency instance under a union handler row -- s2 and s4
# are the physically identical run set, s4 is an analysis view, see
# run_constraint.py's --scenario s4 refusal):
#   s1 on 6 new 3-obj instances x (samos,samos-cheap x {h1,h2,h4,h5,h6} +
#       random x {default=h4-cdp,h1}) = 6 x 12 = 72
#   s3 on the SAME 6 instances x (samos,samos-cheap x {h1,h2,h3,h4,h6} +
#       random x {default=h2-penalty,h1}) = 6 x 12 = 72
#   s2-union on pid4/pid9 x (samos,samos-cheap x {h1,h2,h3,h4,h5,h6} +
#       random x {default=h4-cdp,h1}) = 2 x 14 = 28
#
# Round-4 (B0 baseline, constraint-as-objective, D31: unconstrained
# 3-objective search -- scenario's 2 objectives + its constrained metric as
# an ordinary 3rd objective, no G -- SMS-EMOA as the search/inner algorithm,
# see run_constraint.py's 'b0-as-obj' handler docstring). samos/samos-cheap
# only -- random has no selection pressure to lose at 3 objectives and is
# skipped by run_constraint.py itself (prints [SKIP], byte-identical to its
# default row), so no random entries are appended below:
#   s1 on 8 instances (the six 3-obj pids + c10mop/4 + in1kmop/9, all of
#       which already carry a #Params threshold) x {samos,samos-cheap} =
#       8 x 2 = 16
#   s3 on the six 3-obj pids only (pid4/pid9 have no FLOPs threshold) x
#       {samos,samos-cheap} = 6 x 2 = 12
#   s2-union on pid4/pid9 x {samos,samos-cheap} = 2 x 2 = 4
#   total: 16 + 12 + 4 = 32, indices 272-303, own b0-as-obj objtag per
#   scenario (see objtag_for() below) -- never the scenario's normal
#   2-objective objtag, so these paths can never collide with any other
#   handler's.
#
# run_constraint.py now ALWAYS writes an objective-set-tagged output path
# ({scenario}/{suite}/pid{pid}/{objtag}/{budget}/{method}/{handler}/
# seed_N.pkl, objtag = 'obj-err-flops' for s1 / 'obj-err-params' for
# s2/s3/s4) -- this applies uniformly to EVERY index below, including the
# frozen 0-99 block: resubmitting an old index writes into the (initially
# empty) tagged path, never the pre-round-3 untagged path its original data
# lives under, so it is NOT a resume of that legacy data -- see
# run_constraint.py's module docstring ('Output layout') and
# analyse_constraint.py (which reads both layouts, distinguished by a
# 'config' column). The scratch pre-copy / rsync-back below is updated to
# use the tagged path for all indices accordingly.
#
# IMPORTANT: scenario s4 entries in the frozen 0-99 block (indices 9-11,
# 53-61, 91-99) now exit 1 immediately without writing anything --
# run_constraint.py refuses --scenario s4 (it has no runs of its own since
# round 3, see above). Their existing legacy pkls remain valid and readable
# by analyse_constraint.py; do not resubmit those specific indices expecting
# new output.
#
# NB101 instances (c10mop/pid2, all its indices in the round-3 s1 block)
# are the slow tasks here: NB101 issues one ORM query PER ARCHITECTURE
# (thousands of tiny reads per outer generation), which the scratch-DB copy
# below mitigates but does not eliminate -- expect these tasks to run
# noticeably longer than the other instances at the same budget.
#
# Index map (each scenario block contiguous):
#   ROUND-1/2 (0-99, FROZEN -- unchanged from before round 3):
#     0- 2  c10mop/pid4  s1 (random,samos,samos-cheap x h4-cdp)
#     3- 5  c10mop/pid4  s2 (x h4-cdp)      6- 8  c10mop/pid4  s3 (x h2-penalty)
#     9-11  c10mop/pid4  s4 (x h2-penalty) 12-14  in1kmop/pid9 s1 (x h4-cdp)
#    15-17  in1kmop/pid9 s2 (x h4-cdp)     18-20  in1kmop/pid9 s3 (x h2-penalty)
#    21-23  in1kmop/pid9 s4 (x h2-penalty)
#    24-32  c10mop/pid4  s1 (9: random:h1; samos x h1,h2,h5,h6; samos-cheap x h1,h2,h5,h6)
#    33-43  c10mop/pid4  s2 (11: random:h1; samos x h1,h2,h3,h5,h6; samos-cheap x same)
#    44-52  c10mop/pid4  s3 (9: random:h1; samos x h1,h3,h4,h6; samos-cheap x same)
#    53-61  c10mop/pid4  s4 (9: same shape as s3)
#    62-70  in1kmop/pid9 s1 (9)   71-81  in1kmop/pid9 s2 (11)
#    82-90  in1kmop/pid9 s3 (9)   91-99  in1kmop/pid9 s4 (9)
#   ROUND-3 (100-271, new instance axis -- see above):
#   100-171  s1 on c10mop:2,3,9 / in1kmop:3,6,8 (72: 12 tasks per instance,
#            contiguous 12-blocks in that instance order)
#   172-243  s3 on the same 6 instances, same shape (72)
#   244-271  s2-union on c10mop:4, in1kmop:9 (28: 14 tasks per instance)
#   ROUND-4 (272-303, B0 baseline -- see above):
#   272-287  s1 x b0-as-obj on c10mop:4, in1kmop:9, c10mop:2,3,9,
#            in1kmop:3,6,8 (16: 2 tasks -- samos,samos-cheap -- per instance,
#            contiguous 2-blocks in that instance order)
#   288-299  s3 x b0-as-obj on c10mop:2,3,9 / in1kmop:3,6,8, same shape (12)
#   300-303  s2-union x b0-as-obj on c10mop:4, in1kmop:9 (4: 2 tasks per
#            instance)
#
# Submit everything (existing results are skipped, so this is safe):
#   sbatch experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#
# Submit only rounds 1-2, only round 3, or only round 4 / B0 (new
# submissions should usually use one of these ranges):
#   sbatch --array=0-99%10   experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#   sbatch --array=100-271%10 experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#   sbatch --array=272-303%10 experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#
# Submit a single scenario (defaults + additions, per the index map):
#   e.g. c10mop/pid4 s2 defaults: --array=3-5 ; its handler additions: --array=33-43
#   e.g. round-3 s1 on c10mop/pid2 (first 12-block of 100-171): --array=100-111
#
# Force re-run (ignore existing pkls):
#   sbatch --export=ALL,OVERWRITE=1 experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
#
# Re-run a single failed task (see array index in its log filename):
#   sbatch --array=7 experiments/constraint/CONSTRAINT_EVOXBENCH.sbatch
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#SBATCH -J constraint_s1s4
#SBATCH -c 2
#SBATCH --array=272-303%20            # 304 tasks (suite,pid,scenario,method,handler), 10 concurrent
##SBATCH --array=0-303%10            # 304 tasks (suite,pid,scenario,method,handler), 10 concurrent
#SBATCH --mem=2gb
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=48:00:00
#SBATCH --output=slurm/slurm_logs_constraint/%x_%A_%a.log
#SBATCH -p main,dmb
#SBATCH --qos research-deadline

mkdir -p slurm/slurm_logs_constraint

# --------- tunable parameters -----------------------------------------------------------------------------------------------------------------------------
POP_SIZE=${POP_SIZE:-20}
N_GEN=${N_GEN:-60}
N_GEN_INNER=${N_GEN_INNER:-20}

# --------- (suite, pid, scenario, method, handler) task table --------------------------------------------------------------------------------------
# Mirrors run_constraint.py's THRESHOLDS instances, SCENARIOS dict, METHODS
# list, DEFAULT_HANDLER, and SCENARIO_HANDLERS (D22 matrix) -- keep in sync.
# Round-1 block first (indices 0-23 preserve their meaning; the handler
# field just names each scenario's default explicitly), round-2 handler
# additions APPENDED after (COMPARE_METHODS 2026-07-06-additions pattern).
default_handler() {
    case "$1" in
        s1|s2) echo h4-cdp ;;
        s3|s4) echo h2-penalty ;;
    esac
}
nondefault_handlers() {    # = SCENARIO_HANDLERS row minus the default
    case "$1" in
        s1)    echo "h1-rejection h2-penalty h5-eps h6-sr" ;;
        s2)    echo "h1-rejection h2-penalty h3-adaptive-penalty h5-eps h6-sr" ;;
        s3|s4) echo "h1-rejection h3-adaptive-penalty h4-cdp h6-sr" ;;
    esac
}
# Round-3's FULL handler row per scenario (= SCENARIO_HANDLERS itself, not
# "minus the default" -- these are fresh instances, nothing to dedupe
# against). Must mirror run_constraint.py's SCENARIO_HANDLERS exactly.
scenario_handlers() {
    case "$1" in
        s1) echo "h1-rejection h2-penalty h4-cdp h5-eps h6-sr" ;;
        s3) echo "h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h6-sr" ;;
        s2) echo "h1-rejection h2-penalty h3-adaptive-penalty h4-cdp h5-eps h6-sr" ;;
    esac
}
# Output-path objective-set tag per scenario -- mirrors run_constraint.py's
# _objtag(SCENARIOS[scenario]['obj_metrics']) for every handler except
# b0-as-obj, which gets its own 3-column tag (mirrors _objtag_b0(), see
# run_constraint.py's 'b0-as-obj' docstring / 'Output layout'). Applies to
# every index in COMBOS (0-303): run_constraint.py always writes the tagged
# path now, so the scratch pre-copy / rsync-back below must use it too, even
# for the frozen 0-271 block.
objtag_for() {
    local scenario="$1" handler="$2"
    if [ "$handler" = "b0-as-obj" ]; then
        case "$scenario" in
            s1)       echo "obj-err-flops-params-b0" ;;
            s2|s4)    echo "obj-err-params-latency-b0" ;;
            s3)       echo "obj-err-params-flops-b0" ;;
        esac
        return
    fi
    case "$scenario" in
        s1)          echo "obj-err-flops" ;;
        s2|s3|s4)    echo "obj-err-params" ;;
    esac
}

COMBOS=()
# indices 0-23: round-1 tasks, scenario-default handler (order unchanged)
for INSTANCE in c10mop:4 in1kmop:9; do
    for SCENARIO in s1 s2 s3 s4; do
        for METHOD in random samos samos-cheap; do
            COMBOS+=("${INSTANCE}:${SCENARIO}:${METHOD}:$(default_handler "$SCENARIO")")
        done
    done
done
# indices 24-99: round-2 handler additions (random runs only h1 beyond its
# default -- D22; samos/samos-cheap run every non-default handler of the row)
for INSTANCE in c10mop:4 in1kmop:9; do
    for SCENARIO in s1 s2 s3 s4; do
        COMBOS+=("${INSTANCE}:${SCENARIO}:random:h1-rejection")
        for METHOD in samos samos-cheap; do
            for HANDLER in $(nondefault_handlers "$SCENARIO"); do
                COMBOS+=("${INSTANCE}:${SCENARIO}:${METHOD}:${HANDLER}")
            done
        done
    done
done
# indices 100-171: round-3, s1 on the six new 3-obj instances (72 = 6 x
# (random x {default,h1} + samos/samos-cheap x the full s1 handler row))
for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    COMBOS+=("${INSTANCE}:s1:random:$(default_handler s1)")
    COMBOS+=("${INSTANCE}:s1:random:h1-rejection")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s1); do
            COMBOS+=("${INSTANCE}:s1:${METHOD}:${HANDLER}")
        done
    done
done
# indices 172-243: round-3, s3 on the SAME six instances (72, same shape)
for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    COMBOS+=("${INSTANCE}:s3:random:$(default_handler s3)")
    COMBOS+=("${INSTANCE}:s3:random:h1-rejection")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s3); do
            COMBOS+=("${INSTANCE}:s3:${METHOD}:${HANDLER}")
        done
    done
done
# indices 244-271: round-3, s2-union on pid4/pid9 (28 = 2 x (random x
# {default,h1} + samos/samos-cheap x the full s2 handler row)). This is
# ALSO s4's physical data (analysis view, no separate submission --
# run_constraint.py refuses --scenario s4, see header).
for INSTANCE in c10mop:4 in1kmop:9; do
    COMBOS+=("${INSTANCE}:s2:random:$(default_handler s2)")
    COMBOS+=("${INSTANCE}:s2:random:h1-rejection")
    for METHOD in samos samos-cheap; do
        for HANDLER in $(scenario_handlers s2); do
            COMBOS+=("${INSTANCE}:s2:${METHOD}:${HANDLER}")
        done
    done
done
# indices 272-303: round-4, B0 baseline (constraint-as-objective, D31).
# samos/samos-cheap only per scenario's instance set (see header) -- no
# random entries, run_constraint.py itself [SKIP]s that combination.
for INSTANCE in c10mop:4 in1kmop:9 c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s1:${METHOD}:b0-as-obj")
    done
done
for INSTANCE in c10mop:2 c10mop:3 c10mop:9 in1kmop:3 in1kmop:6 in1kmop:8; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s3:${METHOD}:b0-as-obj")
    done
done
for INSTANCE in c10mop:4 in1kmop:9; do
    for METHOD in samos samos-cheap; do
        COMBOS+=("${INSTANCE}:s2:${METHOD}:b0-as-obj")
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
_SRC="results/constraint/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
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
DEST="${SLURM_SUBMIT_DIR}/results/constraint/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}"
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
        ${OVERWRITE:+--overwrite}

    rsync -a "${SCRATCH_DIRECTORY}/results/constraint/${SCENARIO}/${SUITE}/pid${PID}/${OBJTAG}/${BUDGET_FOLDER}/${METHOD}/${HANDLER}/" "$DEST/" 2>/dev/null || true
done
echo ""
echo "Results copied to $DEST"

echo ""
echo "Date (done)       = $(date)"

# --------- scratch cleanup ------------------------------------------------------------------------------------------------------------------------------------
echo "Removing scratch directory: $SCRATCH_DIRECTORY"
rm -rf "$SCRATCH_DIRECTORY"
