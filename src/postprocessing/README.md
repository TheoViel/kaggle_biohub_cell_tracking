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
| pipeline → pp_public (tuned) | 0.8197 | +0.061 | 59 | 10 | 53 |
| pipeline → pp_public (tuned) → ILP | 0.8203 | +0.061 | 59 | 10 | 95 |
| **+ grid-centring and rounding fixes** | **0.8251** | **+0.066** | — | — | 53 |

The last row is two coordinate bugs found afterwards, both worth more than any
of the gating work — see **Two half-voxel bugs** below.

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

## Two half-voxel bugs

Found by chasing the systematic component of the detection offset
(`analysis_localisation.py`, `check_centring.py`). Both are the same mistake:
throwing away half a voxel by taking a corner or a floor where a centre was
meant.

**1. The detection grid was reported at its corner.** The grid is 4x coarser
than the image in y and x. `tracking/targets.py` assigns a ground-truth cell to
a grid cell with `floor(gt / s)`, and `tracking/predict.py` mapped it back with
a plain `coords *= unet_scaling` — so a cell spanning image voxels `[4g, 4g+3]`
was reported at `4g`, its corner, when the cell it stands for averages `4g+1.5`.
That predicts a fixed −1.5 voxel (−0.61 µm) error in y and x and none in z.
Measured against the annotation over 71 stacks: y −0.591 µm, x −0.536 µm,
z −0.225 µm. Applying the predicted shift drives them to −0.039 and +0.030.

Fixed at source in `tracking/predict.py`. For npz files already written, the
pipeline applies it — but only to those that still need it, which it reads off
the coordinates: a corner-referenced axis has every value at a multiple of the
grid step, a centred one at half a step off. `--no_recentre` disables it.

**2. `astype(int)` truncates.** `pp_public`'s linefit smoothing leaves sub-voxel
positions, and the conversion back dragged every one of them toward zero — the
same half-voxel bias, reintroduced after the first fix removed it. Now rounds,
in `stages.run_pp_public` and in both npz writers.

| | adjJ |
|---|---|
| pipeline → pp_public, as before | 0.8197 |
| + round instead of truncate | 0.8236 |
| + grid centring | **0.8251** |

For scale: the calibrated link classifier in `src/edge_classifier/`, which was
the thing most expected to help, is worth +0.0000.

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
2. **An uncertain link scores worse than no link.** Differentiating the above,
   keeping a link is worth it only when `p > TP / (TP + n_gt + FP)`, which at
   the pipeline's operating point is **45%**. Declining is a real option, and
   the tracker never takes it. (Note the asymmetry above is about *ground-truth
   links* — one we get wrong does produce two false-positive edges — but each of
   those edges is a separate keep-or-drop decision worth one FP, so the
   threshold does not carry the factor of two. An earlier version of this note
   said 54%, then 62%; both were wrong. See `src/edge_classifier/README.md`.)

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
- **Detections are poorly localised _on this stack_**: median 3.7 µm from the
  GT centroid, and 47 of 89 GT cells have more than one detection within the
  7 µm matching radius. This is not typical — across all 71 stacks the median
  is 1.86 µm and only 21% of cells have a rival candidate
  (`analysis_localisation.py`). Localisation quality tracks stack difficulty,
  and this is the hardest stack.
- **The 57% node excess is not spurious peaks.** Raising the detection
  threshold removes it and the recall together — at 0.4 the node count halves
  but only 71% of GT cells are still found, and J drops to 0.40.

## The pipeline

`pipeline.py`, four steps, in this order:

1. **Re-link each frame pair as a global assignment** on flow-corrected
   distance (`relink.py`). The tracker gives each child its own best parent
   independently, so children compete for parents and two can take the same
   one. Solving the pair jointly removes the spurious divisions and reassigns
   contested parents. Flow — the median displacement of a cell's 30 nearest
   already-linked neighbours — is re-estimated for three rounds, because the
   embryo slides bodily between frames (up to 7 grid voxels on this stack) and
   that bias otherwise swamps each cell's own motion.
2. **Recover the detections dropped for a single frame** (`recover.py`, below).
3. **Drop short tracks.** The links now form chains, so track length means
   something and a 1–2 node chain is a run of spurious detections. This is the
   only node reduction that survived.
4. **Decline the uncertain links** — the assignment margin over the runner-up,
   and the link's residual against the local flow. This is the 45% rule above.

Recovery comes before pruning and pruning before gating: the gate cuts tracks
in two, and pruning after it would discard both halves of a track uncertain at
a single frame — while pruning *before* recovery would discard the fragments
recovery exists to join.

## Recovering the detections the detector dropped

The largest single gain since the re-linking, and it attacks the thing the
headroom section below names: *links we never proposed*.

At threshold 0.25 node recall is 0.984, and a missed node costs **two**
ground-truth edges — the link into it and the link out of it — which nothing
downstream can win back, because the node those links need does not exist. So:
find a track that ends at frame `u` and one that begins at `u + L + 1`, decide
they are the same cell, and interpolate `L` nodes into the hole. No image
evidence, no per-cell velocity model — just the two endpoints.

**The whole thing turns on removing the drift first.** `estimate_drift`
(`model_zoo/simple_node_transformer.py`) gives the whole-field translation
between two frames from the node sets alone, by matching each child to its
nearest parent and taking the median offset. On `44b6_267148e4` that
translation is a **median 2.0 µm per frame, p90 5.2, max 11.0** — across a gap
it is the same size as the gate, so without removing it the gate is measuring
the microscope, not the cell. One refinement on top of the function as written:
its median is quantised to the nearest-neighbour offsets (1.625 µm in z, a
quarter of the matching radius), so `recover.py` re-averages the offsets that
agree with the median to recover the sub-voxel part.

The gate is then the *residual* displacement, `‖(b − a) − ΣD‖`, and the pairing
is a Hungarian assignment per gap frame. The gate is a **speed, not a
distance**: a cell crossing a three-frame hole has had three times as long to
move, so the limit scales with the span. Gap lengths are tried shortest first
and an endpoint consumed by a short gap is not offered to a longer one — where
a one-frame explanation exists it is the better one. Two refusals matter: a
proposal within 4 µm of a detection that already exists at that frame sinks the
whole bridge — that cell was not missed, its *link* was wrong, and a node on
top of another one only steals the ground-truth match — and both ends must
already belong to a track, since bridging two isolated nodes invents a whole
one. The recovered nodes carry no links; `relink` runs a second time and wires
them in with the same assignment every other frame pair got.

**All 71 stacks**, `logs/2026-08-26/4/preds_0.25` (`exp_recover_all.py`):

| chain | adjJ | Δ vs pipeline | tp | fp | fn | recall | better/worse | s/stack |
|---|---|---|---|---|---|---|---|---|
| raw | 0.7603 | — | 18066 | 3641 | 1760 | 0.984 | — | 0.1 |
| pipeline | 0.8076 | — | 17772 | 1985 | 2054 | 0.980 | — | 5.5 |
| + recover, gap ≤ 1 | 0.8192 | +0.0115 | 17965 | 1878 | 1861 | 0.981 | 38 / 4 | 11.1 |
| + recover, gap ≤ 2 | 0.8227 | +0.0151 | 18054 | 1861 | 1772 | 0.983 | 41 / 7 | 11.0 |
| **+ recover, gap ≤ 3** | **0.8238** | **+0.0161** | 18096 | 1861 | 1730 | 0.985 | 42 / 6 | 11.1 |
| pipeline → pp_public | 0.8217 | +0.0141 | 18097 | 1991 | 1729 | 0.979 | 47 / 14 | 48.8 |
| **+ recover, gap ≤ 3** | **0.8280** | **+0.0203** | 18246 | 1929 | 1580 | 0.983 | 48 / 15 | 56.5 |

True positives go up and false positives go *down* — the recovered node is not
the prize, the repaired chain is. Run-level node recall moves only 0.980 →
0.985 while tp gains 324, because a repaired gap lets the assignment get *both*
sides of it right. `pp_public` absorbs part of the gain (it re-links too), but
+0.0063 survives under it.

**Most of the value is in the multi-frame gaps.** Going from `L = 1` to `L ≤ 2`
is worth another +0.0036 and `L ≤ 3` a further +0.0010, on top of the +0.0115
the one-frame filler gets — the detector does not drop a cell for exactly one
frame and then recover, it drops it for a run.

The gate is flat from 4 to 6 µm (0.8235 / 0.8238 / 0.8233 at `gap ≤ 3`) and
clearly worse at 8 (0.8218, and 13 regressed stacks), which says the linear
span-scaling is about right. 5.0 is the default.

**Iterating the stage is not worth it.** A second round of recover + relink
gains +0.0002 (0.8238 → 0.8240) for 50% more time, and only at `gap ≤ 3` —
`recover_passes` exists, defaults to 1. The idea was that filling a gap shifts
the assignment and exposes new ends and starts; it does, barely. The bridges
that matter are found in one pass.

The one caution: the hardest stack, `44b6_2f31fc2f`, goes 0.494 → 0.487
standalone and 0.547 → 0.544 under `pp_public`. Small, but this stage is the
only one here that makes it worse.

**On `44b6_267148e4`**, which has 8 GT nodes with no detection within the 7 µm
matching radius:

| | adjJ | tp | fp | fn | recall |
|---|---|---|---|---|---|
| raw | 0.6377 | 235 | 77 | 42 | 0.971 |
| pipeline | 0.7080 | 230 | 37 | 47 | 0.964 |
| + recover, gap ≤ 1 (r=6) | 0.7366 | 239 | 36 | 38 | 0.982 |
| **+ recover, defaults** | **0.7774** | 250 | 32 | 27 | 0.993 |
| + recover, gap ≤ 3, r=6, ×2 | 0.7828 | 252 | 32 | 25 | 0.996 |

`gap ≤ 1` recovers 3 of the 8 missed nodes, the shipped defaults 6, and `r=6`
7 — at 1.0–4.3 µm from the annotation, so interpolation localises them better
than the detector's own median of 1.86 µm. The one it never reaches sits at
`x = 0`, on the edge of the field. Note the last row: the settings that win on
this stack lose across the run, which is the same trade the gates below make.
`exp_recover.py --diagnose` prints the missed-node table for any stack; it is
the right thing to run before tuning anything, since it gives the ceiling.

One caveat on the "no model, no images" property of this directory:
`recover.py` imports `estimate_drift` from `model_zoo`. It is a pure function
of two node sets — no weights are loaded — but it is an import out of the
tracker.

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
- **Cutting the rival density** (`suppress.py`, `exp_suppress.py`) — the
  intervention the −0.66 correlation above seemed to call for, and it does not
  work. Over all 71: suppression at 5 µm after `pp_public` is **−0.0002** (3
  better, 6 worse), at 7 µm **−0.068**, and raising `recover_min_dup` from 4 to
  7 µm is **−0.0061** (6 better, 27 worse). Three things came out of it, all
  worth keeping:

  **Post-processing does break the detector's spacing floor.** Greedy NMS leaves
  a hard 7.08 µm gap between detections in a frame — *not one pair below it* in
  the raw npz. After the pipeline and `pp_public`, 14–21% of nodes have a
  neighbour inside 7 µm and ~1% inside 5 µm, some sitting on top of each other.
  `recover` contributes ~3–6% of that and `pp_public`'s linefit smoothing the
  rest. Real, and still not worth undoing.

  **The rivals that cost the score are out of suppression's reach.** What hurts
  is two detections within 7 µm *of the annotation*, and those two are typically
  7–14 µm apart from **each other** — precisely what a 7 µm NMS permits.
  Suppressing at 7 µm moves the rival rate 58%→54%, 30%→19%, 38%→31% on the
  three worst stacks. Reaching zero needs 9–11 µm, which takes node recall from
  100%/97%/99% to 78%/55%/41%. The median spacing between cells is 8.0 µm and
  the matching radius is 7 µm, so **any** detection set that covers the cells
  has rivals inside it. Rival density is a marker of how hard a stack is, not a
  lever on it — the correlation is real and the intervention is not.

  **Greedy NMS at 7 µm is slightly too aggressive.** The clearest signal in the
  sweep: letting `recover` insert nodes 4–7 µm from an existing detection beats
  forbidding it by 0.006 over 71 stacks, 27 stacks to 6. Those insertions are
  real cells NMS suppressed, and `recover` is partly working as a recall fix for
  over-suppression. Lowering `nms_radius_um` is the direct version of that, and
  needs a re-run of inference rather than post-processing.
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

## A centroid vote on the detection grid

The answer to the section below, and the one thing here that is not
post-processing: `--offset_head` in `main_track.py`. A second head on the
detection grid predicts, per voxel, the vector to its cell's centre; a detection
is then reported at `(g + o) * s` instead of at the middle of its grid cell.
This is `segmentation/`'s centroid voting, moved onto the tracker.

**A sub-voxel offset would not have worked, and measuring first is what showed
it.** The grid is 1.625 µm isotropic, so a CenterNet-style offset confined to
the peak's own cell can move a detection by at most half of that. Against the
detections actually produced:

| | total error | irreducible (peak in the wrong cell) | reachable |
|---|---|---|---|
| worst 10 | 2.73 µm | 2.33 | **0.40** |
| best 10 | 2.06 µm | 1.63 | **0.43** |

The peak is in the *wrong* grid cell 89% of the time on the worst stacks and
76% on the best, and two cells or more away 39% / 27% of the time. A head
confined to its own cell recovers 0.4 µm of a 2.7 µm error, which the oracle
curve says is worth almost nothing. Reaching over a couple of cells is the whole
point, and it is why the target is a *vote* — every voxel within `offset_radius`
of a cell is asked for the vector to it, nearest cell winning a contested voxel
— rather than a residual.

Supervised on ground truth only. Weak positives are image intensity maxima and
carry no opinion about where the cell centre is beyond the peak they already sit
on, so training the vote toward them would teach the head to reproduce the error
it exists to remove. The trade is sparse supervision: about three annotated cells
a frame, each claiming `(2r + 1)³` voxels.

Off by default — it changes the architecture, so checkpoints trained before it
must have it off, which `main_predict._reconcile_arch` reads off the weights.
`PredictConfig.use_offset` turns the correction off at inference on a model that
has one, which is the honest A/B.

**Not yet scored.** The head trains (offset loss 0.797 → 0.415 over 360 steps of
the pretraining stage) and the three inference paths behave — no head, head
used, head ignored — but what it is worth needs a full run.

## Why the worst stacks are the worst

Run `analysis_worst.py`, then `analysis_steal.py`, then `analysis_oracle.py`.
Measured on `logs/2026-08-20/20` post-processed (`scores_pp.json`, run-level
0.8348), worst 10 against best 10 by score.

**Every ground-truth link, by what happened to it:**

| | tp | stolen (wrong parent) | declined (no parent) | lost (node missing) |
|---|---|---|---|---|
| worst 10 (2048 links) | 85.0% | **11.5%** | 1.5% | 2.1% |
| best 10 (1980 links) | 98.4% | 0.8% | 0.5% | 0.4% |

So detection recall is not the problem any more (2.1% lost), the gates are not
the problem (1.5% declined), and the entire spread between a 0.95 stack and a
0.48 stack is **links given to the wrong parent**.

**And they are not close calls.** In raw microns, per link, p25/median/p75:

| worst 10 | how far the *annotation* moved | distance between the two matched detections | the link we made instead | error on the parent detection | on the child |
|---|---|---|---|---|---|
| stolen (235) | 1.68 / **2.33** / 5.03 | 6.91 / **8.53** / 10.10 | 1.18 / 1.99 / 2.63 | 3.07 / **3.98** / 5.16 | 3.37 / **4.14** / 5.27 |
| kept (1770) | 1.15 / 1.82 / 2.47 | 0.91 / **1.72** / 2.19 | — | 1.68 / 2.07 / 3.45 | 1.68 / 2.07 / 3.37 |

The cell moved **2.3 µm**; the two detections matched to its endpoints are
**8.5 µm apart** — four times the true motion and more than the 8.0 µm median
spacing between cells. Each endpoint is ~4 µm off the annotation, against
2.1 µm on the links we get right, and **there was nothing closer in the
detection set**: the nearest detection to the annotation is, to two decimals,
the one the matcher used. The true parent then ranks 2–3 (151 of 235) or worse
(81), and the cell we link instead is ~6 µm nearer after flow correction. No
re-ranking recovers this. The coordinates are wrong.

**The oracle settles it.** Move only the ~1% of detections matched to an
annotated cell onto the annotation, leave every other node alone, re-run the
same pipeline:

| stack | pipeline | +oracle localisation | +1 µm | +2 µm | +3 µm |
|---|---|---|---|---|---|
| 44b6_2f31fc2f | 0.568 | **0.856** | 0.822 | 0.529 | 0.370 |
| 44b6_e57ff5c6 | 0.668 | **0.775** | 0.741 | 0.504 | 0.293 |
| 44b6_87bba6c4 | 0.623 | **0.809** | 0.810 | 0.554 | 0.319 |
| 44b6_a2bb48bb | 0.621 | **0.869** | 0.830 | 0.515 | 0.313 |

False positives collapse with it (91 → 8 on `a2bb48bb`). Note the shape: **1 µm
of residual error is nearly free and 2 µm costs the entire gain.** The detector
sits at a 1.7–2.2 µm median — exactly on that cliff, which is why the same
pipeline scores 0.95 on one stack and 0.48 on another.

**What sets the cliff is crowding, not the error alone.** Across all 71 stacks
the strongest correlate of the score is the fraction of ground-truth nodes with
**more than one** detection inside the 7 µm matching radius:

| | score | median loc err | mean loc err | > 3 µm | rival rate |
|---|---|---|---|---|---|
| worst 10 | 0.689 | 2.24 µm | 2.70 | 39.3% | **42.5%** |
| middle 10 | 0.848 | 1.68 µm | 1.85 | 17.0% | 22.5% |
| best 10 | 0.951 | 2.05 µm | 2.28 | 22.6% | **9.9%** |

`corr(score, rival) = −0.66`, `corr(score, mean loc err) = −0.61`, but
`corr(score, median loc err)` is only −0.41 — the best group's *median* error is
no better than the worst group's. The tolerance for a link is roughly **half the
distance to the nearest rival detection**: 2 µm of error is harmless when a cell
stands alone and fatal when a second detection sits inside its radius. Node
excess feeds it directly — 1.42× the estimated node count on the worst 10
against 1.19× on the best.

Removing the rivals was tried and does not work — see *Cutting the rival
density* below. The radius that reaches them also deletes the cells. **Rival
density is a marker of a hard stack, not a lever on one**, which leaves
localisation as the only side of this that can actually be moved.

One consequence worth stating plainly: **node recall of 0.99 is an artefact of
the 7 µm matching radius.** At the ~2 µm the linking actually needs, 39% of
annotated cells on the worst stacks have no detection at all.

Two things ruled out here, so they are not re-tried:

- **Smoothing node positions along their track** (`analysis_jitter.py`, line fit
  or moving average, windows 1–5). The error is not zero-mean per-frame jitter:
  smoothing *raises* the median (2.19 → 2.43 at ±1 frame, 2.94 at ±5).
- **Per-stack detection threshold from the node count.** `metadata.csv` gives an
  estimated node count per sample, so targeting it needs no ground truth — but
  every stack is over-detected at every threshold in the sweep, so the rule
  picks the top of the range for 63 of 71 and does not discriminate.

## Where the remaining headroom is

The ceiling is TP = 85 (every GT link is in the model's top-12), against 68
today. Two things stand between:

- **Detection localisation.** Across all 71 stacks the median error is 1.86 µm
  — near the 0.81 µm the detection grid costs by rounding alone — so typical
  localisation is *not* the problem. But it is strongly associated with lost
  links: the worst endpoint of a link we reproduce is 2.07 µm off, against
  **5.28 µm** for a link we miss. There is also a small fixed shift of 0.83 µm
  (y −0.59, x −0.54, z −0.23), about a third of a grid cell and far larger than
  its standard error — that part looks like a centring convention rather than
  detector noise, and is correctable by measuring it once and subtracting.
  ~~Whether poor localisation *causes* the lost links or both follow from
  crowding is not established here.~~ **Established — it causes them, and it is
  now the binding constraint.** See *Why the worst stacks are the worst*: the
  oracle that fixes localisation on the annotated cells alone takes the worst
  stacks from 0.57–0.67 to 0.78–0.87, and 1 µm of residual error is free while
  2 µm is fatal. **Acted on** — see *A centroid vote on the detection grid*.
- ~~**A calibrated link confidence.**~~ **Tried — it does not pay.** See
  `src/edge_classifier/`. The classifier is well calibrated and ranks wrong
  links 3x better than chance, but it largely learns the assignment margin, and
  at the threshold where gating pays it declines the links the margin cut
  already declines: 0.8065 either way, and identical once `pp_public` runs
  after it. The break-even threshold quoted here was also wrong — it is 0.45,
  not 0.54; a wrong ground-truth link produces two false-positive edges, but
  each is a separate keep-or-drop decision worth one. The real conclusion is
  that **precision was never the binding constraint**: 2,026 false negatives
  against 2,057 false positives. What is missing is links we never proposed —
  and the reason we never proposed them is that the coordinates put the true
  parent a full cell-spacing away. That is a detector problem, not a linker one.
- **Links we never proposed, part 1: the missing node.** Addressed — see
  *Recovering the detections the detector dropped*, **+0.0161**, of which two
  thirds is the one-frame gap and a third the multi-frame ones. What is left is
  the run of missing frames longer than three, and the cells lost at the edge of
  the field, where there is no chain on the far side to bracket them.

## Files

| file | what it is |
|---|---|
| `postprocess_npz.py` | **npz in, npz out** — the best chain, with each stage switchable |
| `common.py` | loading and the notebook's scoring, shared |
| `pipeline.py` | **the pipeline**, and its tuned defaults |
| `relink.py` | flow-corrected assignment per frame pair |
| `recover.py` | put back the detections dropped for a single frame, drift removed |
| `exp_recover.py` / `exp_recover_all.py` | its ceiling and score on one stack / the gate over all 71 |
| `apply.py` / `run_all.py` | run it on one stack / all 71 |
| `tune.py`, `fast_score.py` | grid search; an exact, fast reimplementation of the metric |
| `baseline.py`, `diagnose.py`, `explore.py`, `analysis_*.py` | the diagnosis above |
| `dump_edge_scores.py` | re-score all candidate parents with the model (needs a GPU) |
| `analysis_worst.py` | every GT link classified: tp / stolen / declined / lost, plus stack features |
| `analysis_steal.py` | the raw geometry of a stolen link — the measurement that names the cause |
| `analysis_oracle.py` | perfect localisation on the annotated cells only; the ceiling it buys |
| `analysis_jitter.py`, `analysis_duplicates.py` | two readings of the error that turned out to be wrong |
| `suppress.py`, `exp_suppress.py` | restoring the detector's spacing floor — measured, does not pay |
| `stages.py`, `sweep.py` | chainable post-processing stages, and a parallel harness to score named chains |
| `exp_zero_div.py`, `exp_ilp_all.py`, `exp_pp_public.py`, `exp_final.py` | the zero-division prior, the ILP, `pp_public`, and all of them together |
| `probe.py`, `probe_ilp_vs_cap.py` | GT division counts and ILP timing; ILP-vs-cap edge-set comparison |
| `model_assign.py`, `track.py`, `dedup.py`, `assemble.py`, `exp_*.py` | the strategies that did not pay off |

`fast_score.py` reproduces `tracking_cellmot.metrics.evaluate` exactly
(`validate_fast.py`: largest disagreement 0.0000) while being fast enough to
sweep 256 settings × 71 stacks.
