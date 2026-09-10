# Ceilings: what the pipeline could score if a stage were perfect

Everything here is measured on the 199 out-of-fold stacks of
`logs/edge_run/2026-09-08/13/preds_0.50`, through the shipped post-processing
chain and the real metric. The question throughout is **where the score is
lost**, answered by replacing one stage's output with the ground truth and
re-running everything downstream unchanged.

Reproduce with:

```bash
cd src
# the annotated cells, embedded by the out-of-fold contrastive model
python main_gt_nodes.py --exp_folder ../logs/contrastive/2026-09-08/0 \
    --out_dir ../logs/gt_nodes/contrastive_2026-09-08_0
# ... plus every detection that is not one of them
python main_mix_nodes.py --gt_dir ../logs/gt_nodes/contrastive_2026-09-08_0 \
    --det_dir ../logs/2026-09-07/4/preds_r6_0.50 \
    --out_dir ../logs/gt_nodes/mix_7um --distractor_um 7

python main_edge_predict.py --log_folder ../logs/edge_run/2026-09-08/13 \
    --nodes_folder <one of the above> --threshold 0.5 --out_dir <preds>
python main_edge_postprocess.py --preds_dir <preds>

# and, with no model at all, how much of the GT the detections can even reach
python postprocessing/exp_reachable.py \
    --preds_dir ../logs/edge_run/2026-09-08/13/preds_0.50 --data_dir ../input/train/
```

---

## Read `adj_edge_jaccard` carefully

```
adj_edge_jaccard = edge_jaccard · (1 − 0.1 · total_node_ratio)
total_node_ratio = (N_pred − N_total) / N_total
score            = adj_edge_jaccard + 0.1 · division_jaccard
```

`N_total` is the *true* number of cells in the volume, not the number annotated.
So **emitting fewer nodes than there are cells raises the score**, and a node set
built only from the annotation gets a large bonus for being nearly empty. It can
exceed 1.

This is not a curiosity, it is the first thing that has to be controlled for.
The GT-only node set below carries `total_node_ratio = −0.939`, which multiplies
its Jaccard by 1.094. Comparing its `adj_edge_jaccard` against the real
pipeline's measures mostly the node count, not the tracking. **Compare node sets
on `edge_jaccard`, or on `adj_edge_jaccard` only between sets with the same node
ratio.**

## The three node sets

| node set | nodes kept | node ratio | node_recall | edge_J | adj_edge_J | div_J | cv |
|---|---|---|---|---|---|---|---|
| detections + pp (shipped) | 5 086 752 | +0.038 | 0.9468 | 0.8405 | 0.8368 | 0.1452 | 0.8513 |
| GT + distractors + pp | 5 069 833 | +0.038 | 0.9958 | 0.9864 | 0.9821 | 0.1191 | 0.9940 |
| GT nodes only + pp | 132 102 | −0.939 | 0.9864 | 0.9848 | 1.0779 | 0.1447 | 1.0924 |

* **detections** — what a submission runs.
* **GT + distractors** — every annotated cell at its exact position, plus every
  detection further than 7 µm from an annotated cell. A detection closer than
  that *is* that cell, so it is dropped and the annotated node stands in for it.
  Perfect detection on the cells that get scored, full distractor load, and the
  same node ratio as the real run — this is the honest comparison.
* **GT nodes only** — the annotation alone, a median of 6 cells a frame against
  the detections' 209. Inflated by the node-count bonus; kept here because the
  contrast with the row above is what proves the distractors are not the problem.

`node ratio` is `total_node_ratio`, recovered as `10 · (1 − adj_edge_J / edge_J)`
from the per-sample values, since the score files do not carry the column
directly. `nodes kept` is the summed `n_nodes_pred` after post-processing.

Per embryo:

| | node_recall | edge_J | adj_edge_J | div_J | score | div tp/fp/fn |
|---|---|---|---|---|---|---|
| detections 44b6 | 0.9687 | 0.8688 | 0.8619 | 0.1053 | 0.8724 | 4 / 12 / 22 |
| detections 6bba | 0.9249 | 0.8121 | 0.8118 | 0.1850 | 0.8302 | 32 / 48 / 93 |
| GT+distr. 44b6 | 0.9957 | 0.9896 | 0.9818 | 0.0690 | 0.9887 | 2 / 3 / 24 |
| GT+distr. 6bba | 0.9958 | 0.9831 | 0.9825 | 0.1691 | 0.9994 | 23 / 11 / 102 |
| GT only 44b6 | 0.9869 | 0.9851 | 1.0822 | 0.0769 | 1.0899 | 2 / 0 / 24 |
| GT only 6bba | 0.9858 | 0.9845 | 1.0737 | 0.2126 | 1.0949 | 27 / 2 / 98 |

## 1. The edge model is not where the score is lost

Given the right nodes it is very nearly exact. Before any post-processing:

| node set | edge tp | edge fp | edge fn | edge_J |
|---|---|---|---|---|
| GT nodes only | 128 380 | 219 | 503 | 0.9940 |
| GT + distractors | 128 039 | 874 | 844 | 0.9902 |

Hand it the annotated node set and it recovers 99.4% of the annotated edges.

## 2. Distractors cost almost nothing

`edge_jaccard` with 5 282 107 distractor nodes present is **0.9864**, against
**0.9848** with none — the distractor set is, if anything, marginally *better*.
The linker declines to link to five million spurious nodes almost perfectly.

Whatever intuition says about a crowded frame, crowding is not the problem. Note
that the whole apparent GT-only advantage in `adj_edge_jaccard` (1.0779 vs
0.9821) is the node-count term and nothing else.

## 3. The loss is the node set, on the cells that get scored

`edge_jaccard` 0.8405 → 0.9864 when the annotated cells are placed at their
exact positions, with every distractor left in. That is the entire gap.

Splitting it needs one more measurement, which needs no model:
`exp_reachable.py` counts the GT edges whose two endpoints both have a detection
within 7 µm. An edge failing that test cannot be produced by *any* assignment.

| | GT nodes | node match | GT edges | reachable | unreachable |
|---|---|---|---|---|---|
| 44b6 | 20 197 | 0.9796 | 19 826 | 0.9692 | 0.0308 |
| 6bba | 113 121 | 0.9424 | 109 057 | 0.9286 | 0.0714 |
| **all** | **133 318** | **0.9480** | **128 883** | **0.9348** | **0.0652** |

The 0.9480 node match rate corroborates the metric's own `node_recall` of
0.9468, computed independently.

So the shortfall decomposes as:

```
total shortfall (1 − 0.8405)                             0.1595
  unreachable: no detection within 7 µm (false negative)  0.0652   41%
  lost among edges it could have made                     0.0943   59%
```

and the second bucket splits further, because with exact positions, duplicates
dropped and no misses the linker loses only 0.0136:

```
detector misses (false negatives)      ~0.065     measured
position error on matched cells        ~0.064     inferred
duplicate detections                   ~0.017     from the dedup oracle
the linker itself                      ~0.014     measured
```

**False negatives are ~40% of the loss, not the whole of it.** Localization
error on cells the detector already fires on is about the same again. Those want
different fixes — recall wants a lower threshold or a better head, localization
wants a better centre estimate — and a threshold sweep trades one against the
other rather than fixing both.

The middle two rows are inferred by combining experiments rather than measured
in one, so treat them as indicative. The duplicate row comes from the oracle in
`cell_merging_cls/README.md` (+0.0165 `adj_edge_jaccard` for merging duplicates
on annotated cells). The clean way to nail the false-negative row on its own is
to keep every detection exactly as it is and append only the GT nodes with no
detection within 7 µm — nothing else changes, so the delta is purely recall.
Not yet run.

**The two embryos are not alike.** 6bba reaches 0.9286 against 44b6's 0.9692,
and it is the weaker embryo on every other metric too. Most of the recall
problem is one embryo.

## 4. Divisions do not improve with perfect nodes

| node set | division_jaccard | div tp (of 151) | div fp |
|---|---|---|---|
| detections | 0.1452 | 36 | 60 |
| GT + distractors | 0.1191 | 25 | 14 |
| GT nodes only | 0.1447 | 29 | 2 |

Hand the chain exact node positions, with both daughters present and the second
one unclaimed, and it finds **fewer** divisions than it does from real
detections. The division term sits between 0.119 and 0.145 across a node set
that varies from 5.1M noisy detections to 133k exact annotations.

**The division failure is not downstream of detection.** No amount of detector
work will move that term, and every division experiment that assumed otherwise
was chasing the wrong stage. At a weight of 0.1 it is worth up to +0.085 cv.

The false-positive counts point at where to look: 60 → 14 → 2 as the node set
cleans up. On a clean graph the gates spend almost none of their error budget —
they are tuned to survive a graph full of distractors and are far too
conservative when the nodes are good.

## 5. Post-processing helps only because of the distractors

Before and after the chain, same node set:

| node set | | node_recall | edge_J | adj_edge_J |
|---|---|---|---|---|
| GT + distractors | edge model | 1.0000 | 0.9902 | 0.9779 |
| | + pp | 0.9958 | 0.9864 | **0.9821** |
| GT nodes only | edge model | 1.0000 | 0.9940 | 1.0880 |
| | + pp | 0.9864 | 0.9848 | **1.0779** |

The chain deletes real nodes in both cases — node recall never survives it — and
always costs `edge_jaccard`. On the distractor-laden set it still wins
(+0.0042 `adj_edge_jaccard`) because the nodes it prunes are mostly spurious and
the node-count term rewards that. On the clean set it loses (−0.0101): there is
nothing spurious to prune, so `prune_short` and friends only remove real cells.

The chain is tuned for an over-detected graph. If detection improves
substantially, `min_track` and the recover/prune block have to be re-tuned or
they will start costing what they currently earn.

## What this does and does not bound

The GT node set is **one lineage per stack**, and the metric only scores that
lineage. Over all 199 stacks and 19 900 frames:

| | total | per frame (median / p90 / max) |
|---|---|---|
| detections | 5 436 435 | 209 / 656 / 1028 |
| annotated cells | 133 318 | 6 / 14 / 33 |

**40.8 detections per annotated node** overall, and the two embryos are far
apart: 149:1 on 44b6 (20 197 annotated nodes) against 21:1 on 6bba (113 121).

So "perfect detection" here means *perfect on the cells that are scored*. A real
detector must also place the other ~99.5% of cells, and errors there cost the
score only indirectly, by competing for links. The `GT + distractors` row keeps
those competitors, which is why it is the row to quote; the `GT only` row does
not, and is a ceiling on a node set no submission can have.

## Summary

| | ceiling | worth |
|---|---|---|
| edge model / linker | `edge_J` 0.9864 given the right nodes | ~0.014 left on the table |
| distractor rejection | already solved | ~0 |
| node set on scored cells | `adj_edge_J` 0.8368 → 0.9821 | **+0.145** |
| — of which false negatives | | ~0.065 |
| — of which localization | | ~0.064 |
| — of which duplicates | | ~0.017 |
| divisions | insensitive to node quality | up to +0.085 cv, separate problem |
