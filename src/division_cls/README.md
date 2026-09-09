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

If the signal is there but the labels are too few, the lead is synthetic data —
871 divisions in 200 sequences against 151 in the whole annotation. See the
`Divisions` section of `notebooks/SyntheticData.ipynb`: the generator's division
*geometry* is currently wrong (sisters at a median 6.8 µm against the real 10.6,
with 54% inside the 7 µm NMS radius against 15%), so that has to be fixed before
the volume is worth anything.

## Files

| file | what it is |
|---|---|
| `dataset.py` | `DivisionDataset`: lineages, negative mining, crops, augmentation |
| `model.py` | `FrameEncoder` (backbone, stage chosen by measurement) and `DivisionNet` |
| `train.py` | the fit loop; AP / AUC / precision@k, since accuracy is meaningless at 1:12 |
| `main.py` | loaders, model build, per-embryo cross-validation |
| `../main_division.py` | entrypoint and defaults |
