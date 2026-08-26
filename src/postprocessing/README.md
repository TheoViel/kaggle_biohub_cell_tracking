# Post-processing strategies for `adj_edge_jaccard`

Everything here runs from a saved `preds/*.npz` (coords + edges). Nothing
modifies `notebooks/TrackingInf.ipynb`.

A whole run — every stack of `<log_folder>/preds`, out to `preds_pp` and
`geff_pp`, with per-stack metrics:

```bash
cd src
python main_postprocess.py --log_folder ../logs/2026-08-20/20/ --jobs 24
```

One saved npz, from anywhere, with nothing but the npz:

```bash
python src/postprocessing/postprocess_npz.py PRED.npz [-o OUT.npz] [--geff DIR]
       [--no-pipeline] [--no-pp-public] [--ilp]
```

Pipeline and `pp_public` are on, the ILP is off. Output is an npz in the same
format as the input, so it drops into anything that reads the tracker's output.

From Python — the notebook does `cd ../src`, so this resolves there:

```python
from postprocessing.pipeline import postprocess
coords, edges = postprocess(coords, edges)
```

```bash
cd src/postprocessing
python baseline.py     # reproduce the notebook's three numbers, and decompose them
python diagnose.py     # where the score is lost
python apply.py        # run the pipeline on one npz and score it
python tune.py         # grid over all 71 validation stacks
python run_all.py      # before/after on all 71, with the notebook's own scorer
python exp_final.py --jobs 24        # every chain below, on all 71

# tuning the individual pieces (--spread takes a subset spanning the difficulty range)
python exp_zero_div.py
python exp_ilp_all.py    --limit 10 --spread
python exp_pp_public.py  --limit 10 --spread          # ablation
python exp_pp_public.py  --limit 10 --spread --tune   # knob grid
```

## Results

**All 71 validation stacks** of `logs/2026-08-20/20/`, adjusted edge Jaccard
(`exp_final.py`). "better/worse" counts stacks moved by more than 0.005:

| chain | adjJ | Δ | better | worse | s/stack |
|---|---|---|---|---|---|
| raw | 0.7592 | — | — | — | — |
| raw → ILP (notebook weights) | 0.7889 | +0.030 | 62 | 2 | 209 |
| raw → ILP (tuned) | 0.7895 | +0.030 | 62 | 2 | 169 |
| **raw → cap-one-child** | **0.7898** | **+0.031** | 63 | 3 | **0.1** |
| raw → pp_public (default) | 0.8064 | +0.047 | 53 | 16 | 57 |
| raw → pp_public (tuned) | 0.8101 | +0.051 | 54 | 15 | 56 |
| **raw → pipeline** | **0.8065** | **+0.047** | **68** | **1** | 6.5 |
| pipeline → ILP (tuned) | 0.8065 | +0.047 | 68 | 1 | 42 |
| pipeline → pp_public (default) | 0.8162 | +0.057 | 57 | 11 | 54 |
| **pipeline → pp_public (tuned)** | **0.8197** | **+0.061** | 59 | 10 | 53 |
| pipeline → pp_public (tuned) → ILP | 0.8203 | +0.061 | 59 | 10 | 95 |

Three things to take from this.

**The ILP is the zero-division prior, at 2000x the price.** Both score +0.030.
`probe_ilp_vs_cap.py` compares the two edge sets directly: they contain the
*same number* of edges and agree on **99.6–99.9%** of them, the solver adds no
edge the raw graph did not already have, and it leaves zero dividing nodes. Its
weights barely matter (the whole `appearance ∈ [0.02, 0.4]` sweep spans 0.009,
and `division_weight` 1 vs 100 is bit-identical — at these appearance costs
divisions are already unattractive). Layered on the pipeline it changes
literally nothing: 17800/2057/2026 before and after, because the re-linking
already enforces the constraint the solver would impose.

**`pp_public` is much better than one stack suggested,** and `motion_relink` is
its engine, not its problem — turning it off costs 0.05. The earlier reading
came from `44b6_2f31fc2f`, which is the one stack `pp_public` makes *worse*.

**The pipeline is the robust choice; `pp_public` on top is the strong one.**
The pipeline improves 68 of 71 and regresses 1. Adding tuned `pp_public` buys
another 0.013 but regresses 10 stacks. Which to prefer depends on whether the
run-level score or the worst case is what you are protecting.

**On `44b6_2f31fc2f`** — the stack you pointed at, which is *the worst of the 71*
(0.496 against a run-level 0.759):

| | adjJ | J | tp | fp | fn | nodes |
|---|---|---|---|---|---|---|
| raw npz | 0.5172 | 0.5484 | 68 | 39 | 17 | 38243 |
| notebook ILP | 0.5307 | 0.5593 | 66 | 33 | 19 | 36818 |
| notebook ILP + `pp_public` | 0.5048 | 0.5333 | 64 | 35 | 21 | 37423 |
| pipeline, run-tuned settings | 0.5504 | 0.5789 | 66 | 29 | 19 | 36365 |
| pipeline, settings tuned on this stack | **0.6447** | 0.6667 | 68 | 17 | 17 | 32387 |

The two rows disagree on purpose, and the gap is the main finding — see
"Which settings" below.

## What the metric actually rewards

Worth deriving before optimising anything, because it is not what it looks like.

The ground truth is ~89 annotated nodes out of an estimated 24 368. A predicted
edge is counted **only** when its source matched a GT node that has a child, or
its target matched one that has a parent (`pred_valid` in
`tracking_cellmot/metrics.py`). On this stack the metric looks at 107 of 38 054
predicted edges. Everything else is free.

With every GT node detected (`node_recall = 1.0`) that collapses to

```
J = TP / (n_gt_edges + FP)
```

and a GT link we get wrong costs **two** false positives — one for the parent we
linked elsewhere, one for the child we linked from elsewhere — so `FP ≈ 2·FN`
and

```
J ≈ TP / (3·n_gt − 2·TP)
```

Two consequences drive every strategy here:

1. **Link accuracy is the whole score.** 68/85 gives 0.55; 75/85 gives 0.71.
   Nothing else moves it comparably.
2. **A coin-toss link scores worse than no link.** Differentiating the above,
   a link is worth making only when it is better than **54%** likely to be
   right. Declining is a real option, and the tracker never takes it.

Separately, `adj = J · (1 − 0.1 · (N_pred/N_total − 1))`. At 38 243 nodes
against 24 368 that is a flat ×0.943.

## Diagnosis

Run `diagnose.py`, `explore.py`, `analysis_flow.py`, `analysis_candidates.py`.

- **The ground truth contains no divisions at all** (out-degree ≤ 1 for all 89
  nodes), while the tracker emits 4128 second children. Every one that touches
  a matched node is a guaranteed FP; 10 of the 39 FPs are exactly this.
- **The 17 lost links are lost to the runner-up.** Every one of the 85 GT links
  is inside the model's top-12 candidates — the ceiling on TP is 85, not 68 —
  and where the tracker errs the true parent is the model's rank 1 in 10 cases,
  rank 2–3 in 7. Once the frame's flow is removed, it is the second-nearest
  candidate in 13 of 17 and within the top 5 in all 17.
- **Detections are poorly localised**: median 3.7 µm from the GT centroid, with
  a systematic −1.9 µm bias in z. 47 of 89 GT cells have more than one
  detection within the 7 µm matching radius. Links whose endpoints matched
  badly are the ones that go wrong (median endpoint error 5.3 µm on lost links
  against 4.0 µm on kept ones).
- **The 57% node excess is not spurious peaks.** Raising the detection
  threshold removes it and the recall together — at 0.4 the node count halves
  but only 71% of GT cells are still found, and J drops to 0.40.

## The pipeline

`pipeline.py`, three steps, in this order:

1. **Re-link each frame pair as a global assignment** on flow-corrected
   distance (`relink.py`). The tracker gives each child its own best parent
   independently, so children compete for parents and two can take the same
   one. Solving the pair jointly removes the spurious divisions and reassigns
   contested parents. Flow — the median displacement of a cell's 30 nearest
   already-linked neighbours — is re-estimated for three rounds, because the
   embryo slides bodily between frames (up to 7 grid voxels on this stack) and
   that bias otherwise swamps each cell's own motion.
2. **Drop short tracks.** The links now form chains, so track length means
   something and a 1–2 node chain is a run of spurious detections. This is the
   only node reduction that survived.
3. **Decline the uncertain links** — the assignment margin over the runner-up,
   and the link's residual against the local flow. This is the 54% rule above.

Pruning comes before gating: the gate cuts tracks in two, and pruning after it
would discard both halves of a track uncertain at a single frame.

## Which settings — and the honest caveat

Tuned over all 71 stacks (`tune.py`), the winner is
`max_distance=7.0, min_track=3, min_margin=0, max_residual=inf`: **the two
confidence gates end up all but off**. Essentially the whole run-level gain is
step 1, the re-linking. That is the robust, non-gaming result, and it is what
`DEFAULTS` is set to.

Tuned on `44b6_2f31fc2f` alone the winner is
`max_distance=7.5, min_track=8, min_margin=2, max_residual=4` — worth +0.13 on
that stack and −0.02 across the run (0.7897 vs 0.8065), improving 54 stacks and
regressing 16, one of them by 0.12.

The gates are genuinely trading TP for FP at about 1:2, which is only a good
deal on a stack whose linking is already poor. If per-stack settings are
acceptable, gate hard when a stack's own margin distribution is wide; otherwise
leave them off.

## What did not work, and why

Recorded so they are not re-tried.

- **Raising the detection threshold** — recall collapses first (see above).
- **Duplicate-track suppression** (`dedup.py`) — barely fires. The excess
  detections are not shadowing duplicates; they track independently, a full NMS
  radius apart.
- **Gap-closed span pruning** (`assemble.py`) — worse than plain track-length
  pruning. Tracks fragment badly (only 54% of nodes are in a track spanning 20+
  frames), so span is not the cleaner signal it looks like.
- **Global assignment on the model's own scores** (`model_assign.py`,
  `exp_model.py`) — no better than the greedy argmax it replaces (TP 62–67 vs
  68). The model's *ranking* is the error; re-ordering under a one-to-one
  constraint does not fix a ranking that is simply wrong, and the constraint
  forces links where over-detection means cells should stay unlinked.
- **A whole-video ILP over the model's *candidate* graph** (`exp_ilp.py`) — the
  notebook's ILP can only delete edges the tracker already committed to, never
  reach the runner-up, so feeding it the top-4 candidates instead should have
  been the principled fix. It does not solve: 15 minutes on the ~150k-edge
  candidate graph without a first incumbent. Worth retrying with the candidate
  set trimmed to top-2 or gated on flow, since this is the one form of the ILP
  that could do more than restate the degree constraint.
- **Per-cell velocity** instead of neighbourhood flow (`track.py`) — worse at
  every weight. Tracks fragment too often for a cell's own history to be
  available when it is needed.
- **Flow-residual filtering alone** — TP and FP overlap too much (median 1.6 µm
  against 2.3 µm); the best single cut is worth +0.02.

## Where the remaining headroom is

The ceiling is TP = 85 (every GT link is in the model's top-12), against 68
today. Two things stand between:

- **Detection localisation.** A median 3.7 µm centroid error with a −1.9 µm z
  bias means the node the GT matches is often not the node that tracks the
  cell. This is a detector fix, not a post-processing one; the systematic
  component is measurable on the training split and correctable.
- **A calibrated link confidence.** The 54% rule is exactly a decision
  threshold, and neither the model's softmax (TP median 0.41 vs FP 0.29) nor
  the geometric margin separates well enough to exploit it. A small classifier
  over (model probability, assignment margin, flow residual, track context),
  fitted on the training embryos' GT, is the principled version of step 3 and
  the thing most likely to close the gap.

## Files

| file | what it is |
|---|---|
| `postprocess_npz.py` | **npz in, npz out** — the best chain, with each stage switchable |
| `common.py` | loading and the notebook's scoring, shared |
| `pipeline.py` | **the pipeline**, and its tuned defaults |
| `relink.py` | flow-corrected assignment per frame pair |
| `apply.py` / `run_all.py` | run it on one stack / all 71 |
| `tune.py`, `fast_score.py` | grid search; an exact, fast reimplementation of the metric |
| `baseline.py`, `diagnose.py`, `explore.py`, `analysis_*.py` | the diagnosis above |
| `dump_edge_scores.py` | re-score all candidate parents with the model (needs a GPU) |
| `stages.py`, `sweep.py` | chainable post-processing stages, and a parallel harness to score named chains |
| `exp_zero_div.py`, `exp_ilp_all.py`, `exp_pp_public.py`, `exp_final.py` | the zero-division prior, the ILP, `pp_public`, and all of them together |
| `probe.py`, `probe_ilp_vs_cap.py` | GT division counts and ILP timing; ILP-vs-cap edge-set comparison |
| `model_assign.py`, `track.py`, `dedup.py`, `assemble.py`, `exp_*.py` | the strategies that did not pay off |

`fast_score.py` reproduces `tracking_cellmot.metrics.evaluate` exactly
(`validate_fast.py`: largest disagreement 0.0000) while being fast enough to
sweep 256 settings × 71 stacks.
