#!/usr/bin/env bash
# One GPU draining a shared job queue.
#
#   ./worker.sh <gpu> <jobs_file> <log_dir> <deadline_epoch>
#
# Jobs are popped under flock, so four workers share one list and a slow job on
# one GPU does not stall the others -- round-robin partitioning would, since the
# runs here differ by 2x in length (iter4000 against convnext_atto).
#
# A worker stops *starting* jobs past the deadline; it never kills one that is
# already running, because a half-trained model writes a checkpoint and a score
# that look real and are not.
set -u

GPU=$1; JOBS=$2; LOGDIR=$3; DEADLINE=$4
CURSOR="$LOGDIR/.cursor"
PY=/data/kaggle-cell-tracking/.venv/bin/python

mkdir -p "$LOGDIR"
[ -f "$CURSOR" ] || echo 0 > "$CURSOR"

# Open the lock file once, here. Putting the `9>>file` redirect on a `$( ... )`
# does not reach the subshell that runs inside it -- flock then fails with "bad
# file descriptor" and, because the failure is non-fatal, every worker reads the
# same cursor and runs the same job.
exec 9>>"$CURSOR.lock"

while :; do
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then
        echo "[gpu $GPU] deadline reached, stopping" >> "$LOGDIR/worker_$GPU.log"
        break
    fi

    # Pop the next index atomically. The lock is held only for the
    # read-increment-write, never across the run itself.
    flock 9 || { echo "[gpu $GPU] cannot lock, refusing to run" >&2; exit 1; }
    idx=$(cat "$CURSOR")
    echo $((idx + 1)) > "$CURSOR"
    flock -u 9

    line=$(sed -n "$((idx + 1))p" "$JOBS")
    [ -z "$line" ] && { echo "[gpu $GPU] queue empty" >> "$LOGDIR/worker_$GPU.log"; break; }

    name=${line%%$'\t'*}
    args=${line#*$'\t'}

    # `sweep.py` writes the args shell-quoted, so they have to go back through
    # the shell's quote removal before they are words. Plain `$args` only
    # word-splits: the quotes survive as literal characters and any JSON value
    # containing a space is torn in half --
    #   --augmentations '{"flip_p": 0}'  ->  [--augmentations] ['{"flip_p":] [0}']
    # which argparse rejects with exit 2 before `prepare_log_folder` runs, so the
    # job leaves no run directory at all. That silently dropped 25 of 71 jobs on
    # the night of 2026-09-09 -- every job carrying JSON, i.e. the entire
    # augmentation arm of the sweep. See README, "The bug that cost the first
    # night".
    eval "argv=($args)"

    # `prepare_log_folder` races when two runs create the same numbered dir at
    # the same instant; a short jittered pause is enough to separate them.
    sleep $((RANDOM % 12 + GPU * 3))

    echo "[gpu $GPU] $(date +%H:%M:%S) start #$idx $name" >> "$LOGDIR/worker_$GPU.log"
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU $PY main_division.py "${argv[@]}" > "$LOGDIR/$name.log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))
    echo "[gpu $GPU] $(date +%H:%M:%S) done  #$idx $name rc=$rc ($(( elapsed / 60 ))m)" \
        >> "$LOGDIR/worker_$GPU.log"

    # A job that fails in under a minute failed to *start* -- bad argument, bad
    # backbone name, no GPU -- and the queue would otherwise march past it in
    # silence at one job per second. Say so on stderr and keep a list, because
    # the cost of not noticing is a whole sweep arm that never ran.
    if [ "$rc" -ne 0 ] && [ "$elapsed" -lt 60 ]; then
        echo "[gpu $GPU] FAST FAIL #$idx $name rc=$rc after ${elapsed}s -- did not train" \
            | tee -a "$LOGDIR/failed.log" >&2
        tail -3 "$LOGDIR/$name.log" >> "$LOGDIR/failed.log"
    fi
done
