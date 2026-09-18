#!/usr/bin/env bash
# Re-run edge post-processing, with the current DEFAULTS, on every strong run.
#
#   ./rerun_edge_pp.sh                 # every run at cv >= 0.863
#   ./rerun_edge_pp.sh 0.87            # a different bar
#   MIN_CV=0.85 FORCE=1 ./rerun_edge_pp.sh
#   DRY=1 ./rerun_edge_pp.sh           # print the commands, run nothing
#
# Why this exists: the DEFAULTS block in `main_edge_postprocess.py` moves --
# 2026-09-18 turned the ILP on and changed `ilp_disappearance`, `ilp_threshold`,
# `div_min_prob` and `div_diverge_um` -- and every `scores_*.json` already on
# disk was written under whatever settings were current when it ran. Those are
# not comparable to each other, which is the thing `Experiments.md` keeps
# warning about. This puts one common configuration across all of them, written
# as `<preds>_pp_default`, so the ranking reflects the models rather than the
# post-processing each happened to be tuned with.
#
# Each run is chosen by its own best two-embryo `scores_*.json`, and the
# candidate folder comes from that file's `preds_dir` -- the same folder that
# score was measured on, not a guess from the folder name.
set -uo pipefail
cd "$(dirname "$0")/src"

MIN_CV="${1:-${MIN_CV:-0.863}}"
NAME="${NAME:-default}"
LOGDIR="${LOGDIR:-/tmp/rerun_edge_pp}"
mkdir -p "$LOGDIR"

mapfile -t JOBS < <(../.venv/bin/python - "$MIN_CV" <<'PY'
import glob, json, os, sys
floor = float(sys.argv[1])
best = {}
for f in glob.glob('../logs/edge_run/*/*/scores_*.json'):
    b = os.path.basename(f)
    # `_pl` is in-sample and `f0_tuned` is a different protocol; neither says
    # how good the model is.
    if '_pl' in b or 'f0_tuned' in b:
        continue
    try:
        j = json.load(open(f))
    except Exception:
        continue
    pe = j.get('per_embryo') or {}
    if len(pe) != 2 or any(v.get('n', 0) < 60 for v in pe.values()):
        continue          # needs both embryos to be a comparable score
    run = '/'.join(f.split('/')[2:5])
    cv = j['macro']['score']
    preds = (j.get('preds_dir') or '').rstrip('/')
    if preds and (run not in best or cv > best[run][0]):
        best[run] = (cv, preds)
for run, (cv, preds) in sorted(best.items(), key=lambda kv: -kv[1][0]):
    if cv >= floor:
        print(f"{cv:.4f}\t{run}\t{preds}")
PY
)

if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "no runs at cv >= $MIN_CV"; exit 1
fi

echo "== $(date '+%H:%M:%S')  ${#JOBS[@]} runs at cv >= $MIN_CV, writing _pp_${NAME}"
printf '   cv %s  %-22s %s\n' $(printf '%s\n' "${JOBS[@]}" | tr '\t' ' ')
echo

ok=0; skip=0; fail=0
for job in "${JOBS[@]}"; do
  cv=$(cut -f1 <<<"$job"); run=$(cut -f2 <<<"$job"); preds=$(cut -f3 <<<"$job")
  out="${preds}_pp_${NAME}"
  tag=$(tr '/' '_' <<<"$run")
  log="$LOGDIR/${tag}.log"

  if [ -d "$out" ] && [ -z "${FORCE:-}" ]; then
    echo "-- skip $run ($(basename "$out") exists; FORCE=1 to redo)"
    skip=$((skip + 1)); continue
  fi

  cmd=(../.venv/bin/python main_edge_postprocess.py --ilp
       --preds_dir "$preds/" --name "$NAME")
  if [ -n "${DRY:-}" ]; then printf '   %s\n' "${cmd[*]}"; continue; fi

  echo "-- $(date '+%H:%M:%S')  $run  (was cv $cv)  -> $log"
  # Sequential on purpose: each invocation already fans out over a worker pool,
  # and the ILP solve is the expensive part at ~9-16 s a stack.
  if "${cmd[@]}" > "$log" 2>&1; then
    new=$(grep -oE 'cv_score[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
    echo "   done  cv ${new:-?}  (was $cv)"
    ok=$((ok + 1))
  else
    echo "   FAILED -- see $log"; tail -3 "$log" | sed 's/^/     /'
    fail=$((fail + 1))
  fi
done

echo
echo "== $(date '+%H:%M:%S')  $ok ok, $skip skipped, $fail failed"
[ "$fail" -eq 0 ]
