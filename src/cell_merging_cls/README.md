# Deduplication: is this two cells, or one cell found twice?

One question about a **pair of detections in one frame**. The input is a short
clip of crops cut around the pair and rotated onto its axis; the output is one
logit, "these two are the same physical cell".

```bash
cd src
python main_merging.py --iterations 3000

# score every close pair once, then sweep the threshold against the real metric
python main_merging_predict.py --preds_dir <preds> --run_dir <run> --scores_dir <scores>
python main_merging_predict.py --preds_dir <preds> --scores_dir <scores> \
    --apply --threshold 0.5 --out_dir <preds>_merged
python main_edge_postprocess.py --preds_dir <preds>_merged
```

## Why this exists

The detector doubles cells. Measured against the annotation on 40 stacks of
`preds_0.50`:

| detections within 5 µm of an annotated cell | share |
|---|---|
| 0 (a miss) | 6.32% |
| 1 | 90.83% |
| **≥2 (a duplicate)** | **2.85%** |

2.85% is the base rate. Among the 151 annotated divisions the rate is **23%** —
`edge_pp.merge_duplicate_parents` found that in 34 of them the second daughter
is claimed by a duplicate of the parent. That 8× enrichment is not a
coincidence: a dividing cell is bilobed, which is exactly the shape that makes a
peak finder fire twice.

`postprocessing/README_edge.md` records that geometry cannot fix this. A 7 µm
merge on distance alone fires 10 508 times for 34 opportunities, because rival
pairs sit at median 8.1 µm and so do thousands of genuine neighbours. The
temporal version, `merge_shadow_forks`, is a no-op at any safe setting. Both are
off by default. This reads the pixels instead.

## What it is worth: the oracle

Before building anything, the ceiling was measured directly, using the
annotation as a perfect duplicate detector (`postprocessing/exp_oracle_dedup.py`)
and re-running the shipped chain on the result. All 199 out-of-fold stacks of
`logs/edge_run/2026-09-08/13/preds_0.50`:

| variant | cv | Δcv | adj_edge_J | div_J | node_recall |
|---|---|---|---|---|---|
| baseline | 0.8513 | — | 0.8368 | 0.1452 | 0.9468 |
| oracle, **delete** the duplicate | 0.8563 | +0.0050 | 0.8448 | 0.1162 | 0.9423 |
| oracle, **merge** the duplicate | **0.8642** | **+0.0129** | 0.8534 | 0.1084 | 0.9463 |

Three things this settles.

**It is worth building.** +0.0129 is more than the entire shipped division-gate
chain (+0.0114), and it touches only 0.110% of detections.

**Merge, never delete.** Deleting is worth 2.6× less, because in the fork bucket
the duplicate *is* the copy carrying the second daughter — delete it and the
division goes with it. Deleting also costs node recall (−0.0045 against
−0.0005): a shortened track falls under `prune_short` and takes annotated nodes
with it. So the model's output is a *pairing*, and `merge.apply_merges`
re-points the loser's links onto the survivor.

**The gain is trajectories, not divisions.** `adj_edge_jaccard` rises +0.0165,
`division_jaccard` *falls* −0.0368. The score weights them 1.0 and 0.1, so the
trade is strongly positive, but it means `propose_divisions` is tuned to the
pre-merge geometry and needs re-tuning afterwards. If retuning merely held
`division_jaccard` flat the ceiling would be +0.0165.

The oracle only touches cells the annotation covers, so it is the ceiling for
the *direct* effect. Duplicates of the other ~99.5% of cells cost the score
indirectly, by competing for links, and are not in that number.

## Training it: the annotation labels one lineage in 223

This is the hard part, and it is a labelling problem rather than a modelling one.

The annotation covers **one lineage per stack** — a median of 10 cells per frame
against ~5 000 detections, and 133 318 annotated nodes against 5 436 435
detections overall, a ratio of **223:1**. So a detection matching no annotated
node is almost always a real, unannotated cell. The rule "unmatched, therefore
spurious" would call nearly every cell in the embryo a duplicate. It is not used
anywhere here, and avoiding it is most of what `pairs.py` is for.

What the annotation *can* say exactly, by charging each detection to its nearest
annotated cell within 5 µm:

| both members charged to | label | `kind` |
|---|---|---|
| the same annotated cell | duplicate | `dup` |
| two annotated cells sharing a parent | distinct — **the hard case** | `sister` |
| two other annotated cells | distinct | `distinct` |
| at least one uncharged | see below | `unlabelled` |

Two centres of distinct cells cannot be much closer than a cell radius, because
cells do not interpenetrate, so a second detection inside 5 µm of an annotated
cell is a duplicate of it rather than a neighbour.

### Unlabelled pairs are used as negatives, and the direction is the point

The trap above runs *unmatched → duplicate*. Going the other way is a different
claim with a measurable error rate. Of the ~675k pairs within 9 µm, roughly 3.5%
are duplicates, so a pair drawn from the unlabelled pool is genuinely two cells
about **96.5%** of the time.

That 3.5% label noise buys two things worth more than it costs. Without it the
labelled set runs about **2:1 positive** — two annotated cells within 14 µm of
each other are rarer than duplicates — against an inference population that is
~28:1 negative, and a model fitted to that prior over-fires. And the labelled
negatives all come from one lineage, so they show the model a single
neighbourhood, where the unlabelled pool is every cell density in the embryo.
With them the training mix lands at 1:20.5. `--unlabelled_per_pos 0` trains on
the exact labels alone.

### Sisters are never subsampled

Sister pairs are ~0.1% of the negatives and 100% of the damage a false merge can
do. They are exempt from the negative cap and oversampled 4×. Thinning them to
keep a ratio tidy would be optimising the loss against the one thing that
matters.

## The crop

Three things differ from `division_cls`, each forced by a measurement.

**It spans z.** 61% of duplicate pairs are separated more in z than in the
plane, at a median of 3 z-slices, because the voxel is 4× coarser in z (1.625 µm
against 0.40625) and the detector resolves one cell into two peaks on different
planes. The division classifier's `patch_z=3` reaches ±1 slice and would not
contain the second detection. `patch_z=9` reaches ±4, covering the 90th
percentile.

**It is oriented along the pair.** The pair's axis is rotated onto the crop's
+x, so the two detections always sit at the same two places. Along that axis one
cell is a single lobe and two cells are two lobes with a valley between — the
same instrument as the division classifier's "the parent's own two kernels",
read with the opposite sign.

**Rotation augmentation is therefore off.** `rot_p` and `rot90_p` would undo the
canonicalisation. Flips stay on: across the axis is a mirror, along it swaps the
two members, and a pair is unordered so the label is symmetric in them.

What the crop cannot show — how far apart the detections are, and how much of
that is in z — is fed back as three scalars, *after* the transformer rather than
as tokens, so a linear term on three numbers can shift the boundary but cannot
substitute for reading the image. That substitution is exactly what the
distance-only merge did at 10 508 firings for 34 opportunities.

The backbone and transformer are `division_cls.model`'s, imported unchanged.

## Evaluation

**During training**, the annotated divisions give a probe with no ambiguity in
it (`pairs.division_probe`): a dividing cell is *one* cell at frame `t` and
*two* at `t+1`, at a separation where the cases are otherwise
indistinguishable. So every pair of detections on the parent is a duplicate, and
the pair on the two daughters is not. The number to watch is not AP but
`sister_fp@0.9` — how often a genuine daughter pair is called a duplicate.

**What actually decides it** is the competition metric. `main_merging_predict.py`
separates scoring from thresholding for exactly this reason: scoring 6.5M pairs
is the expensive half, the threshold is the half that has to be swept, and the
only honest read of a threshold is `adj_edge_jaccard` out of the real chain.
Ranking metrics cannot see the asymmetry that matters — merging two real cells
destroys a trajectory, missing a duplicate leaves one spurious track.

## Files

| file | what it is |
|---|---|
| `pairs.py` | candidate pairs, and what the annotation can and cannot label |
| `dataset.py` | `MergePairDataset`: oriented, z-spanning pair clips |
| `crops.py` | batched extraction with a per-sample rotation (2× the shared-angle path, bit-exact) |
| `model.py` | `MergeNet`: `division_cls`'s encoder plus three geometry scalars |
| `train.py` | the fit loop; ranking metrics plus the sister false-merge rate |
| `merge.py` | choosing merges and re-pointing links; shared with the oracle |
| `infer.py` | scoring every close pair in a stack |
| `../main_merging.py` | training entrypoint and defaults |
| `../main_merging_predict.py` | score, then threshold and merge |
| `../postprocessing/exp_oracle_dedup.py` | the oracle upper bound above |

---

## Measured

### The classifier learns the task

Two variants, both `convnext_tiny`, 3 000 steps, leave-one-embryo-out. They
differ only in whether the unlabelled pool is used as negatives.

| | val AP | chance | AUC | rec@0.9 | sisterFP@0.9 |
|---|---|---|---|---|---|
| with unlabelled, fold 0 | 0.440 | 0.042 | 0.964 | 0.010 | **0.0000** |
| clean labels, fold 0 | 0.886 | 0.292 | 0.961 | 0.983 | 0.190 |
| clean labels, fold 1 | 0.993 | 0.539 | 0.992 | 0.909 | 0.084 |

The two are not comparable on AP -- they are scored against different negative
populations -- but the last two columns say something the AP does not. The
unlabelled-trained model never merges a sister and almost never fires; the
clean-label model fires on nearly every duplicate and merges 8-19% of sister
pairs. Neither is calibrated: at p >= 0.9 the first fires on 0.4% of all pairs
and the second on 51%.

### And it does not survive contact with the metric

All 199 out-of-fold stacks, real chain, real metric. Thresholds are chosen by
*merge budget* rather than by probability, because neither model is calibrated:

| | merges | % of nodes | cv | Δ |
|---|---|---|---|---|
| baseline | — | — | 0.8513 | — |
| **oracle merge** | 5 985 | 0.110% | **0.8642** | **+0.0129** |
| clean labels, p ≥ 0.99976 | 12 134 | 0.223% | 0.8505 | −0.0008 |
| clean labels, p ≥ 0.99953 | 58 147 | 1.070% | 0.8506 | −0.0007 |

and on 44b6 alone, where the with-unlabelled model was also tested:

| | merges | adj_edge_J | 44b6 score | Δ |
|---|---|---|---|---|
| baseline | — | 0.8619 | 0.8724 | — |
| oracle merge | 0.11% of nodes | 0.8676 | 0.8755 | +0.0031 |
| with unlabelled, p ≥ 0.9 | 8 046 | 0.8615 | 0.8721 | −0.0003 |
| clean labels, p ≥ 0.9998 | 6 164 | 0.8609 | 0.8709 | −0.0015 |

**Every model setting is neutral to slightly negative, against an oracle worth
+0.0129.** The pipeline is correct end to end -- the oracle is applied through
the same `merge.apply_merges` and reproduces byte-for-byte after that code was
shared -- so this is a model result, not a plumbing one.

One detail worth keeping: at the looser threshold the two embryos move in
opposite directions, 6bba's `adj_edge_jaccard` +0.0029 against 44b6's −0.0045.
The fold that trained on 5 509 duplicates does *worse* on its held-out embryo
than the fold that trained on 519. With two embryos that is not a finding, but
it does mean nothing here should be read as "more data would fix it" without a
replicate.

### Why: precision, by a factor of three

The measurable cause is precision at the operating point. At the oracle's merge
budget on 44b6 (6 624 merges, 0.22% of nodes) the clean model recovers 13 of the
519 confirmed duplicates. Scaling by the 0.65% of detections the annotation
covers puts its precision near **30%**, against the oracle's 100%. Seven of ten
merges join two real cells, and `adj_edge_jaccard` charges every one of them.

The score distributions show where it goes wrong. Ranking in the *middle* is
good -- confirmed duplicates sit at median 0.589 against 0.095 for distinct
pairs and 0.113 for sisters -- but the confident tail is not clean: at p >= 0.9
the unlabelled-trained model is only 2.4x enriched in duplicates over the
unlabelled pool. A model can have AUC 0.96 and still be useless here, because
the population is 2 million pairs at 3.4% prevalence and the merge budget is
0.1% of nodes. **This task is decided in the top 0.1% of the ranking, and AUC
does not look there.**

The two variants fail in opposite directions, which is diagnostic rather than
disappointing:

* **with unlabelled negatives**, ~3.4% of the majority class *are* duplicates,
  so the model is explicitly taught to score real duplicates as 0. That flattens
  exactly the tail the merge budget reads from.
* **with clean labels only**, the negatives come from a single lineage and the
  prior is 2:1 positive, so the model saturates -- half of all pairs score above
  0.9 -- and the ranking inside the tail carries little information.

### What to try next, in order

**Harvest positives from the temporal shadow signature.** This is the piece of
the original design that is *not* implemented here. A duplicate shadows its
original within a cell's width frame after frame until one copy dies, while
sisters separate permanently -- `edge_pp.merge_shadow_forks` already encodes
that distinction. Used as a *label generator* rather than as an inference rule,
it needs no annotation, so it runs on the 99.5% of cells the annotation misses:
calibrate it against the confirmed duplicates on the annotated lineage, then
harvest across the whole volume. That fixes both failure modes at once -- it
multiplies the ~6k confirmed positives by two orders of magnitude, and it
removes the 3.4% contamination by labelling the unlabelled pool instead of
assuming it.

**Then re-tune `propose_divisions`.** The oracle's `division_jaccard` fell
−0.0368 because the division gates are tuned to pre-merge geometry. That is
worth up to +0.0037 of the +0.0129 and cannot be collected until a merge model
works.

Until the first of those, the honest statement is that the *opportunity* is
measured at +0.0129 and the *model* does not yet reach it.
