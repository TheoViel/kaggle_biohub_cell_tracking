#!/usr/bin/env bash
# Retrain every model the kernel (notebooks/biohub-ens-edu.ipynb) loads, end to end.
#
#   ./train.sh                      # every step, in order
#   ./train.sh netr edge            # some steps; earlier runs come from $STATE
#   SMOKE=1 ./train.sh              # the whole chain on 3 stacks per embryo and
#                                   # a few iterations per stage, in input_smoke/
#                                   # and logs_smoke/ -- checks that it runs
#
# Steps: detector_r0 pseudo_labels detector merge netr edge forks division package.
# Each training step appends the run folder it created to $STATE, which later
# steps read, so a step can be rerun on its own. See README.md for what each
# step does and what it needs.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
    INPUT=${INPUT:-input_smoke}
    LOGS=${LOGS:-logs_smoke}
else
    INPUT=${INPUT:-input}
    LOGS=${LOGS:-logs}
fi
# The teammate's out-of-fold detections (README, "Data").
EDU_PREDS=${EDU_PREDS:-$LOGS/eduardo_current_mean3/preds_r6_0.05}
# The merged node set. The kernel checks that the edge model was trained on a
# folder whose name contains "v16_".
NODES=${NODES:-$LOGS/ensemble/v16_retrain}
STATE=$REPO/$LOGS/pipeline_runs.env

# Paths the training code resolves relative to src/ (src/params.py).
export BIOHUB_INPUT=../$INPUT/
export BIOHUB_LOGS=../$LOGS/
export OMP_NUM_THREADS=1
# torch reads MKL_NUM_THREADS before OMP_NUM_THREADS; set both.
export MKL_NUM_THREADS=1

N_GPUS=${N_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
# --local-addr: this host's FQDN does not resolve, so torchrun's default lookup
# hangs when it hands the address to the workers.
PY=${PY:-$REPO/.venv/bin/python}
TORCHRUN="$PY -m torch.distributed.run --standalone --local-addr=127.0.0.1 --nproc_per_node=$N_GPUS"

mkdir -p "$LOGS"
touch "$STATE"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
log() { printf '\n\033[1m== %s  %s\033[0m\n' "$(date '+%H:%M:%S')" "$*"; }

# Run folders under a log root: <root>/<date>/<n>/.
runs() { ls -d "$1"/????-??-??/*/ 2>/dev/null | sort || true; }

# `trained VAR ROOT cmd...` runs a training command and records the run folder
# it created under ROOT as VAR in $STATE. Recorded relative to src/, where every
# training command runs, which is also how the runs refer to each other in
# their args.json.
trained() {
    local var=$1 root=$2; shift 2
    local before; before=$(runs "$REPO/$root")
    "$@"
    local new; new=$(comm -13 <(echo "$before") <(runs "$REPO/$root") | tail -1)
    [ -n "$new" ] || { echo "no new run folder under $root" >&2; exit 1; }
    new=../$root/${new#"$REPO/$root/"}
    new=${new%/}
    echo "$var=$new" >> "$STATE"
    eval "$var=$new"
    echo "-> $var=$new"
}

# A src/-relative path (as recorded in $STATE) from the repository root.
from_root() { echo "${1#../}"; }

# Merge JSON objects: the training scripts take a config block as one JSON
# value that overrides the defaults key by key.
json() { "$PY" -c 'import json,sys; d={}; [d.update(json.loads(a)) for a in sys.argv[1:]]; print(json.dumps(d))' "$@"; }

need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "$v is not set: run its step first ($STATE)" >&2; exit 1; }; done; }

# shellcheck disable=SC1090
source "$STATE"

# Smoke mode: a few iterations per stage, everything else unchanged.
if [ "$SMOKE" = 1 ]; then
    S_TRACK='{"iterations": 4, "n_val_videos": 1, "eval_every": 2, "batch_size": 2}'
    S_TRACK_ARGS=(--num_workers 2 --log_every 1 --save_every 1000)
    S_NETR=(--iterations 4 --max_frames_per_video 4 --num_workers 2)
    S_EDGE=(--iterations 4 --eval_every 2 --n_val_videos 1 --num_workers 2)
    S_DIV=(--iterations 4 --eval_every 2 --num_workers 2)
else
    S_TRACK='{}'
    S_TRACK_ARGS=()
    S_NETR=()
    S_EDGE=()
    S_DIV=()
fi

# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

# Bootstrap detector: synthetic pretraining (-> fold_0/ckpt_synthetic.pt) and
# the fold models that pseudo-label the train split. No pseudo-labels of its
# own: weak positives come from image peaks alone.
detector_r0() {
    log "detector_r0"
    cd "$REPO/src"
    trained DET0 "$LOGS" $TORCHRUN main_track.py \
        --no-fullfit \
        --synthetic_pretraining "$(json '{"enabled": true}' "$S_TRACK")" \
        --pretraining "$(json '{"weights": null}' "$S_TRACK")" \
        --edge "$(json '{}' "$S_TRACK")" \
        --weak_positives '{"pl_folder": null}' \
        "${S_TRACK_ARGS[@]}"
    cd "$REPO"
}

# Each fold model predicts its own training embryo; the best threshold is
# post-processed into DET0/preds_pl_pp.
pseudo_labels() {
    need DET0
    log "pseudo_labels"
    cd "$REPO/src"
    $TORCHRUN main_predict.py --log_folder "$DET0/" --pl --det_threshold 0.3,0.4,0.5,0.6
    "$PY" main_postprocess.py --log_folder "$DET0/" --pl
    cd "$REPO"
}

# The shipped detector recipe (main_track.py defaults), on the bootstrap's
# pretraining checkpoint and pseudo-labels; then out-of-fold detections.
detector() {
    need DET0
    log "detector"
    cd "$REPO/src"
    trained DET "$LOGS" $TORCHRUN main_track.py \
        --synthetic_pretraining "$(json '{}' "$S_TRACK")" \
        --pretraining "$(json "{\"weights\": \"$DET0/fold_0/ckpt_synthetic.pt\"}" "$S_TRACK")" \
        --edge "$(json '{}' "$S_TRACK")" \
        --weak_positives "{\"pl_folder\": \"$DET0/preds_pl_pp/\"}" \
        "${S_TRACK_ARGS[@]}"
    # 0.30 feeds the merge (thresholded at 0.45 there), 0.60 trains NETr.
    $TORCHRUN main_predict.py --log_folder "$DET/" --det_threshold 0.30,0.60 --nms_radius_um 6
    cd "$REPO"
}

# Our detections >= 0.45, plus the teammate's >= 0.35 further than 7 um away.
merge() {
    need DET
    log "merge"
    "$PY" scripts/merge_detectors.py \
        --base "$(from_root "$DET")/preds_0.30" --add "$EDU_PREDS" \
        --cfg 0.45 0.35 7 --out "$NODES"
}

# NETr on the detector's own detections, then one embedding per merged node,
# out of fold, written into the merged npz files.
netr() {
    need DET
    log "netr"
    cd "$REPO/src"
    # Smoke mode: a 4-iteration detector finds nothing at 0.60, so NETr crops
    # around the merged nodes instead.
    local coords="$DET/preds_0.60/"
    [ "$SMOKE" = 1 ] && coords="../$NODES/"
    trained NETR "$LOGS/contrastive" $TORCHRUN main_contrastive.py \
        --coords_log_folder "$coords" "${S_NETR[@]}"
    $TORCHRUN inference_contrastive.py --exp_folder "$NETR/" --coords_log_folder "../$NODES/"
    cd "$REPO"
}

emb_key() { (cd "$REPO/src" && "$PY" -c "import sys; from inference_contrastive import embeddings_key; print(embeddings_key(sys.argv[1]))" "$NETR"); }

# The edge transformer, then its out-of-fold candidates, which the division
# forks are harvested from.
edge() {
    need NETR
    log "edge"
    local key; key=$(emb_key)
    cd "$REPO/src"
    trained EDGE "$LOGS/edge_run" $TORCHRUN main_edge.py \
        --nodes_folder "../$NODES/" --embeddings_key "$key" "${S_EDGE[@]}"
    $TORCHRUN main_edge_predict.py --log_folder "$EDGE/" --threshold 0.45
    # Scores the kernel's post-processing out of fold; not needed downstream.
    "$PY" main_edge_postprocess.py --ilp --preds_dir "$EDGE/preds_0.45/"
    cd "$REPO"
}

# The three fork pools the division classifiers train on.
forks() {
    need EDGE
    log "forks"
    local pools=$REPO/$LOGS/fork_cls
    local preds="$EDGE/preds_0.45/"
    mkdir -p "$pools"
    if [ "$SMOKE" = 1 ]; then
        # A 4-iteration edge model proposes no forks; harvest the smoke stacks
        # from the shipped edge run's candidates instead.
        preds=../$LOGS/edge_preds_shipped/
        mkdir -p "$LOGS/edge_preds_shipped"
        for f in "$(from_root "$EDGE")"/preds_0.45/*.npz; do
            ln -sf "$REPO/logs/edge_run/2026-09-28/0/preds_0.45/$(basename "$f")" "$LOGS/edge_preds_shipped/"
        done
    fi
    cd "$REPO/src"
    # The ILP's own forks.
    "$PY" division_cls/gen_fork_candidates.py --preds_dir "$preds" \
        --out "$pools/pool_ours.npz" --pp_overrides '{"recover_max_gap": 2}'
    # The geometric proposer with the ILP off.
    "$PY" division_cls/gen_fork_candidates.py --preds_dir "$preds" \
        --out "$pools/pool_edu.npz" --pp_overrides '{"ilp": 0, "ilp_div_forks": 0,
            "div_min_prob": 0.15, "div_require_mutual_nn": 0, "div_require_divergence": 0,
            "div_symmetry_tau": 0.0, "div_frame_frac_cap": 1.0, "div_global_frac_cap": 1.0,
            "div_max_parent_um": 11.0, "div_max_existing_child_um": 10.0,
            "div_max_sister_um": 14.0, "recover_max_gap": 3, "recover_max_residual": 5.0,
            "recover_min_dup": 3.0, "threshold": 0.4, "min_track": 5, "linefit": 0.6,
            "linefit_window": 3}'
    "$PY" ../scripts/union_pools.py --out "$pools/pool_union.npz" \
        "$pools/pool_ours.npz" "$pools/pool_edu.npz"
    cd "$REPO"
}

# One fold_fullfit classifier per pool; that is what the kernel loads.
division() {
    log "division"
    local pools=$REPO/$LOGS/fork_cls
    cd "$REPO/src"
    for pool in ours edu union; do
        local var="DIV_${pool^^}"
        trained "$var" "$LOGS/division" "$PY" main_division.py \
            --candidates "$pools/pool_$pool.npz" --selected_folds '[]' "${S_DIV[@]}"
    done
    cd "$REPO"
}

# Everything the kernel loads, into dataset/ (or $LOGS/dataset in smoke mode).
package() {
    need EDGE DIV_OURS DIV_EDU DIV_UNION
    log "package"
    local dest=dataset
    [ "$SMOKE" = 1 ] && dest=$LOGS/dataset
    "$PY" scripts/copy_to_dataset.py "$(from_root "$EDGE")" --dest "$dest" \
        --division "pool28_ours=$(from_root "$DIV_OURS")" \
        --division "pool28_edu=$(from_root "$DIV_EDU")" \
        --division "pool28_union=$(from_root "$DIV_UNION")"
}

# --------------------------------------------------------------------------
# Smoke data: 3 stacks per embryo, a few synthetic sequences, the teammate's
# detections for those stacks.
# --------------------------------------------------------------------------
smoke_data() {
    [ -d "$INPUT/train" ] && return
    log "smoke data in $INPUT/"
    mkdir -p "$INPUT/train"
    cp input/metadata.csv "$INPUT/"
    for emb in 44b6 6bba; do
        for z in $(ls -d input/train/${emb}_*.zarr | head -3); do
            name=$(basename "$z" .zarr)
            ln -s "$REPO/input/train/$name.zarr" "$INPUT/train/"
            [ -e "input/train/$name.geff" ] && ln -s "$REPO/input/train/$name.geff" "$INPUT/train/"
        done
    done
    "$PY" scripts/make_synthetic.py --input "$INPUT" --version 5 \
        --n_sequences 8 --n_calib_videos 2 --jobs 8
    ln -s v5 "$INPUT/synthetic/v4"
    mkdir -p "$EDU_PREDS"
    for z in "$INPUT"/train/*.zarr; do
        name=$(basename "$z" .zarr)
        ln -s "$REPO/logs/eduardo_current_mean3/preds_r6_0.05/$name.npz" "$EDU_PREDS/"
    done
}

STEPS=(detector_r0 pseudo_labels detector merge netr edge forks division package)
[ "$SMOKE" = 1 ] && smoke_data
[ $# -gt 0 ] || set -- "${STEPS[@]}"
for step in "$@"; do
    "$step"
done
log "done -- run folders in $STATE"
