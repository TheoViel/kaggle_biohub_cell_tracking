# A division classifier from image crops

One question about one annotated cell: **does it divide at this frame?** The
input is what `notebooks/Divisions.ipynb` plots — a clip of crops centred on the
cell, `n_frames` frames of time, `patch_z` z-slices each — and the output is one
logit.

```bash
cd src
python main_division.py --iterations 2000
torchrun --standalone --nproc_per_node=8 main_division.py --iterations 2000
```

## Why this exists

The fold score is `adj_edge_jaccard + 0.1 * division_jaccard`. The tracking
chain converts about one annotated division in seven, and
`postprocessing/README_edge.md` records what happened to every attempt at fixing
that downstream: a hand-tuned geometric gate (+0.0114, shipped), a duplicate
merge (rejected), a fork inside the assignment (rejected — it finds 85 of 151
divisions and drowns them in 5 289 false ones), and a classifier over candidate
pairs (rejected — it wins on one embryo and loses on the other).

Those all rank the same features: distances, and the edge model's probability.
The conclusion there is that **precision is the constraint** and needs roughly a
7x improvement at fixed recall, which no re-weighting of existing features
supplies. Divisions are 0.13% of the annotated links, so the edge transformer
has little reason to have learned them.

This is the other way at it: look at the image. Its output is a per-node prior
that `edge_pp.propose_divisions` can gate on, alongside the geometry it already
uses.

An earlier probe found the *contrastive* embeddings do not separate dividing
cells (AUC 0.558 / 0.435 across folds, precision@k = 0). That is weaker evidence
than it looks: NETr is trained to recognise the same cell across frames, so its
objective explicitly rewards invariance to the temporal appearance change a
division is. Reading the pixels directly is a different question, and this asks
it.

## The negatives are the design

At 151 divisions among 133k annotated nodes the natural imbalance is 1:880, and
almost all of those negatives are trivially easy — a cell in a different stack,
in a different part of the embryo, at a different brightness. They teach nothing
and cost the whole epoch.

`negatives="track"` (the default) takes them from **the dividing cells' own
lineages**, at their other frames. Same cell, same neighbourhood, same
brightness, different answer. The model cannot win by learning which cells look
distinctive; it has to learn *when*. The imbalance falls to about **1:12**.

| setting | negatives drawn from | imbalance |
|---|---|---|
| `"track"` | lineages containing a division | ~1:12 |
| `"stack"` | any annotated cell in a stack that has a division | ~1:250 |
| `"all"` | every stack, division or not | ~1:880 |

`exclude_window=1` drops the frames flanking a division from the negatives. A
cell one frame from dividing looks much like a cell dividing, and the annotation
does not claim otherwise — it says the division is at `t` and nothing more.
Those frames are dropped rather than relabelled, because inventing a label the
data does not carry is how a model learns to be confidently wrong.

`positive_multiplier` repeats the positives inside the epoch. Oversampling
rather than `pos_weight` so the rare class is seen under *different*
augmentation draws, which is the only reason to augment at all; both knobs exist
and only a sweep says how they trade off. It applies to training only —
oversampling the validation split would inflate the positive rate and flatter
every ranking metric.

## The model

A 2D CNN per frame, then **one transformer over every frame's feature map at
once**.

`embeddings/model.py` pools each frame to a vector and aggregates those. That
cannot work here: a division is a *spatial* event — one blob becomes two, and
which way they part is the evidence — so pooling the frame away before the
frames are compared discards the thing being classified.

So the CNN keeps its map. Each frame's `(H, W, d)` becomes `H*W` tokens, the
`n_frames` maps are concatenated into one sequence with a CLS token, and the
transformer attends over all of it. A token in the last frame can look directly
at the position it occupied in the first.

Two embeddings carry what concatenation destroys:

* a **frame embedding**, learned per time index — without it the clip is a bag
  and "one blob became two" is indistinguishable from the reverse;
* a **spatial embedding**, learned per grid cell and **shared across frames** —
  so the same position in different frames gets the same code, and "moved" is
  distinguishable from "appeared elsewhere". Shared is the whole point; a
  per-frame table would make position `p` at `t-1` unrelated to position `p` at
  `t+1`.

The backbone is a config string. At a 32-voxel crop most pyramids have collapsed
to 1x1 by their last stage, so `FrameEncoder` measures the stages and cuts at the
deepest one still at least `min_tokens` wide — 4x4 for every backbone tried
(`convnext_tiny`, `resnet18`, `maxvit_tiny_tf_224`, `efficientnet_b0`).

## What this cannot fix

**151 divisions, two embryos.** The split is by embryo and has to be — a cell
appears at many frames and its neighbours at all of them, so a random split puts
near-copies of a training crop in validation. That leaves a fold training on 26
divisions or on 125.

Read validation AP against `chance_ap`, never in absolute terms, and treat a
fold that disagrees with the other as noise until shown otherwise. That has
already happened once in this repo: `postprocessing/div_classifier.py` beat its
baseline trained on 61 positives and collapsed trained on 12, and the pooled
number averaged one reasonable fit with one hopeless one.

If the signal is there but the labels are too few, the obvious lead is
synthetic data — 601 divisions in 300 sequences against 151 in the whole
annotation. It was tried and it **hurts**; see "Synthetic divisions hurt" below.
The generator's division *geometry* has since been fixed (sisters now 11.8 µm
median against the real 10.6, from 6.8 before), but geometry was never the part
that mattered for a classifier reading appearance.

## Files

| file | what it is |
|---|---|
| `dataset.py` | `DivisionDataset`: lineages, negative mining, crops, augmentation |
| `model.py` | `FrameEncoder` (backbone, stage chosen by measurement) and `DivisionNet` |
| `train.py` | the fit loop; AP / AUC / precision@k, since accuracy is meaningless at 1:12 |
| `main.py` | loaders, model build, per-embryo cross-validation |
| `../main_division.py` | entrypoint and defaults |

---

## Measured

All numbers are leave-one-embryo-out, against a positive rate (`chance_ap`) of
0.077 / 0.082.

### It works

| | fold 0 (train 6bba, 125 div) | fold 1 (train 44b6, 26 div) |
|---|---|---|
| AP | **0.705** | **0.449** |
| AUC | 0.872 | 0.824 |
| precision@k | 0.65 | 0.46 |

Nine times chance on one fold, five and a half on the other, and **both folds
learn** -- including the one with 26 training divisions, which is where the
candidate classifier in `postprocessing/div_classifier.py` collapsed. At fold
1's operating point, 57 of the top 125 ranked cells are real divisions out of
1 522 candidates.

This also settles a question the contrastive embeddings answered wrongly. A
probe on those embeddings scored AUC 0.558 / 0.435 with precision@k of *zero*,
and was read as evidence that mitotic appearance is not visible here. It is not:
NETr is trained to recognise the same cell across frames, so its objective
rewards invariance to exactly the temporal appearance change a division is.
Reading the pixels finds what the embedding discards.

### The noise floor is wider than almost every effect measured here

**Read this before any table below.** Four runs of the *identical* default
configuration, differing only in `--seed`:

| seed | fold 0 | fold 1 |
|---|---|---|
| 0 | 0.767 | 0.314 |
| 1 | 0.699 | 0.446 |
| 2 | 0.663 | 0.462 |
| 3 | 0.735 | 0.457 |
| **spread** | **0.104** | **0.148** |

Same data, same hyper-parameters, same code. A configuration has to beat the
default by more than about 0.10 on fold 0 or 0.15 on fold 1 before the
difference is the configuration rather than the draw, and **nothing measured so
far does.** The tuning table below ranks ten configs across a 0.23 range of mean
AP; the top half of it is inside this band. So is every backbone and every
`positive_multiplier` in the 2026-09-09 sweep.

This is what 26 and 125 training positives buys, and it is not fixable by
sweeping harder. The honest uses of a single run are (a) detecting a collapse —
`bb_scratch` at 0.255 / 0.104 is far outside the band, so *pretraining matters*
is a real finding — and (b) shortlisting candidates for replication. Ranking two
runs 0.05 apart is not one of them.

### The 71-job sweep: the defaults survive all of it

Run in two halves. The first (2026-09-09, 22:31–01:17) reached job 51 of 71
before the machine died; the second (2026-09-10, 05:48–10:07) drained the
remaining 43. Every number below is one run per config unless it says
otherwise, so **read all of it against the noise floor above** — the band is
0.10 on fold 0 and 0.15 on fold 1, and almost nothing here clears it.

Two blocks are worth stating outright.

**The clip-level augmentation is not doing anything measurable.** `aug_none`
turns off all nine knobs `division_cls/augment.py` added — the direct
before/after against the pipeline that preceded them:

| config | fold 0 | fold 1 | mean |
|---|---|---|---|
| default (4 seeds) | 0.663–0.767 | 0.314–0.462 | — |
| `aug_none` (all nine off) | 0.705 | 0.404 | 0.554 |
| `no_flip_p` | 0.714 | 0.369 | 0.542 |
| `no_rot90_p` | 0.658 | 0.317 | 0.487 |
| `no_z_jitter` | 0.720 | 0.373 | 0.546 |
| `no_frame_shift_um` | 0.722 | 0.269 | 0.496 |
| `no_frame_brightness` | 0.753 | 0.402 | 0.578 |
| `no_frame_contrast` | 0.716 | 0.393 | 0.555 |
| `no_shot_noise` | 0.728 | 0.422 | 0.575 |
| `no_blur_p` | 0.741 | 0.376 | 0.558 |
| `no_cutout_p` | 0.760 | 0.348 | 0.554 |

`aug_none` lands mid-band on both folds, and every leave-one-out is inside it
too. Nine knobs, no measurable effect from any of them or from all of them at
once.

**More augmentation hurts, monotonically in how much it adds:**

| config | fold 0 | fold 1 | mean |
|---|---|---|---|
| `aug_light` | 0.681 | 0.404 | 0.542 |
| `aug_strong` | 0.612 | 0.336 | 0.474 |
| `aug_extreme` | 0.532 | 0.244 | 0.388 |

`aug_strong` is below the band on fold 0, `aug_extreme` below it on both. This
is the same shape as the crop-size result in the table below — every enlargement
lost, ordered by how much input it added. Two independent axes now say this
model wants *less* perturbation, not more, which is not what 151 positives would
lead you to expect.

Everything else stayed inside the band: transformer depth (`layers2` 0.765/0.389
through `layers8` 0.679/0.304), width (`d128` 0.629/0.279 to `d512`
0.700/0.309), heads, dropout, `min_tokens`, batch size, `exclude_window`,
learning rate (`lr0.0006` 0.741/0.436 is top-of-band, not above it), and
`iter4000` at double the compute (0.725/0.387). The backbone does not matter
either — `convnext_femto` 0.737/0.469 down to `efficientnet_b0` 0.492/0.276,
nearly all of it inside the band — though *having* a pretrained backbone does:
`bb_scratch` collapses to 0.255/0.104, the one unambiguous result in 71 jobs.

The `time_reverse` jobs post the two highest fold-0 numbers in the sweep
(`reverse0.5` 0.785, `reverse0.15` 0.768) while going *low* on fold 1 (0.307,
0.354). That is the fold-disagreement pattern this README already says to treat
as noise, and it is treated as noise here.

#### `neg_per_pos` cannot be compared on AP

`neg6` finished top of the raw ranking at 0.781 / 0.476 — the only config above
the band on both folds. It is an artifact. `neg_per_pos` sets how many negatives
the *validation* split carries, so it moves the positive rate `chance_ap` with
it, from 0.143 at `neg6` to 0.020 at `neg48` — a 7x swing:

| config | chance | AP f0 / f1 | AP / chance | AUC f0 / f1 |
|---|---|---|---|---|
| `neg6` | 0.143 | 0.781 / 0.476 | 5.5x | 0.909 / 0.799 |
| default | 0.077 | 0.663–0.767 / 0.314–0.462 | 9–10x | 0.872–0.900 / 0.804–0.867 |
| `neg24` | 0.040 | 0.649 / 0.273 | 16x | 0.884 / 0.805 |
| `neg48` | 0.020 | 0.538 / 0.261 | 26x | 0.875 / 0.835 |

The AP ordering and the lift ordering are exactly reversed, and on AUC — which
does not move with prevalence — every setting is inside the seed band. The axis
measures its own validation split, not the model. Compare `neg_per_pos` settings
on AUC or not at all.

#### The bug that cost the first night

25 of the 71 jobs never started and left no trace. `worker.sh` expanded the
job's arguments unquoted, so every job carrying a JSON value was torn apart by
word-splitting and rejected by argparse *before* `prepare_log_folder` ran — exit
2, no run directory, queue moves on at one job per second. That is precisely the
set passing `--augmentations` or `--optimizer_config`: the entire augmentation
arm above, plus the learning-rate block.

The failure is invisible by construction — a job that fails to *start* looks
exactly like a queue draining fast. Fixed with `eval "argv=($args)"` and a
`failed.log` that records any job exiting non-zero inside a minute. The relaunch
ran 43/43 with `rc=0` and an empty `failed.log`.

### Tuning found nothing better than the defaults

Ten configurations, ranked by the cross-fold mean AP — but see the noise floor
above, which the top half of this table sits inside:

| config | fold 0 | fold 1 | mean |
|---|---|---|---|
| **convnext_tiny 32/3/3 (default)** | 0.705 | 0.449 | **0.577** |
| n_frames 5 | 0.669 | 0.360 | 0.515 |
| maxvit_tiny | 0.617 | 0.393 | 0.505 |
| positive_multiplier 8 | 0.583 | 0.330 | 0.457 |
| min_tokens 8 | 0.622 | 0.281 | 0.451 |
| patch_yx 64 | 0.627 | 0.270 | 0.448 |
| patch_z 5 | 0.607 | 0.277 | 0.442 |
| patch_yx 48 + patch_z 5 | 0.571 | 0.308 | 0.440 |
| patch_yx 48 | 0.575 | 0.225 | 0.400 |
| 48 + z5 + t5 | 0.426 | 0.261 | 0.343 |

**Every enlargement lost, and the ordering is monotone in how much input it
adds.** The loss concentrates on fold 1 -- more to overfit against the same 26
positives. The prediction going in was the opposite: sisters sit 6.15 um from
the parent and a 32-voxel crop reaches only 6.5 um, so a wider crop *should*
capture daughters the current one truncates. It does not, and the reason is
worth keeping: the cue is the parent's own two kernels, which are inside 32 px.
The model never needed to see where the daughters landed.

### In post-processing: helps where the model is strong

`main_division_predict.py` writes a per-node probability and
`edge_pp.propose_divisions` gates on it. Against the shipped chain's +0.0114:

| | 44b6 (fold-0 model, AP 0.705) | 6bba (fold-1 model, AP 0.449) |
|---|---|---|
| no gate | 4 tp / 6 fp, divJ 0.125, +0.0123 | 17 tp / 36 fp, divJ 0.106, +0.0104 |
| gate >= 0.6 | **4 tp / 1 fp, divJ 0.148, +0.0149** | 3 tp / 3 fp, divJ 0.023, +0.0024 |

On the embryo scored by the *strong* model the gate removes five of six false
positives and keeps every true one. On the embryo scored by the weak one it
removes fourteen of seventeen true divisions. The two models do not share a
probability scale, and no single threshold serves both -- 0.02 through 0.95 were
tried and all cost more on 6bba than they win on 44b6.

So it is **off by default** (`div_prob_min=0.0`). Were fold 1 as good as fold 0,
the gate would be worth roughly +0.015 per embryo against the current +0.0114,
so about **+0.004 cv** -- the whole return is conditional on that one model.

Ranking by the classifier instead of by `parent_dist + 0.15 * sister_dist`
changes nothing (0.8436 vs 0.8437), because the caps are not binding: 3 113 of
3 136 proposals are accepted, so the order barely decides anything.

### Synthetic divisions hurt

The obvious fix for fold 1 is generated data -- `input/synthetic/v5` yields 601
divisions against 26 real. Mixed into training at four proportions, it degraded
**every** configuration, monotonically, including the fold it was meant to help:

| synthetic share | fold 0 | fold 1 | mean |
|---|---|---|---|
| 0% (real only) | **0.705** | **0.449** | **0.577** |
| 25% | 0.391 | 0.330 | 0.361 |
| 50% | 0.363 | 0.322 | 0.342 |
| 75% | 0.359 | 0.187 | 0.273 |
| 90% | 0.308 | 0.187 | 0.248 |

The cause is a distinction worth stating plainly, because the generator work
looked complete without it. Synthetic division *geometry* is calibrated -- the
sister separation is fitted to 151 real divisions and lands at 11.8 um against
10.6. Synthetic mitotic *appearance* is not: `sequences._lobe` draws two spheres
at 0.78x the parent radius and half the daughters' separation, and those numbers
were chosen, not measured. This classifier's entire premise is that it reads
mitotic appearance, so an invented appearance is precisely the input that
poisons it -- and a quarter of the batches is enough to outvote 26 real
positives.

**What would make it work.** Measure real dividing cells the way `SISTER_UM` was
measured: elongation, intensity profile along the division axis, how separated
the two kernels are at the frame the annotation names.
`notebooks/Divisions.ipynb` already locates and renders all 151 of them, so the
population is in hand. Until `_lobe` is driven by that, generated divisions are
a liability rather than an asset, whatever their count.
