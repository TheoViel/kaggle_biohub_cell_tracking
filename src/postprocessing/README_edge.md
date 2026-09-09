# Post-processing the edge transformer's graph

`README.md` next to this one is about the detector's npz: a graph whose links
are already committed and whose probabilities are not worth much, so the
pipeline there throws the links away and re-decides them on flow-corrected
geometry. This file is about the other input — the edge transformer
(`src/edge_transformer/`, run `logs/edge_run/2026-09-04/3`), which scores every
candidate link and reaches AP 0.948 on held-out frames. Its probabilities are
the best ranking available, so nothing here re-ranks them; the stages fill in
what the model could not see.

Everything runs from the npz alone — no model, no images, no GPU — at 2.7 s a
stack.

```bash
cd src
# once, on the GPU: the npz now also carries every scored candidate
python main_edge_predict.py --weights ../logs/edge_run/2026-09-04/3/fold_0/ckpt_final.pt \
       --embryos '["44b6"]' --out_dir ../logs/edge_run/2026-09-04/3/preds_edge \
       --dump_candidates

# the chain, scored with the competition metric
python main_edge_postprocess.py --preds_dir ../logs/edge_run/2026-09-04/3/preds_edge

cd postprocessing
python exp_edge_pp.py --preds_dir <abs path>/preds_edge --spread 16 \
       --chains 'raw,[recover=1,min_track=4],[recover=1,min_track=4,linefit=0.6,linefit_window=3]'
python diag_edge.py  --preds_dir <abs path>/preds_edge --chain assign
```

`exp_edge_pp.py` takes an absolute `--preds_dir`: it imports `common`, which
chdirs to `src/` so that `params`' relative paths resolve.

## The result

Fold 0 — all 71 `44b6` stacks, adjusted edge Jaccard, which for a single embryo
*is* the fold score:

| chain | score | Δ | J | tp | fp | fn | node recall | N/N_est |
|---|---|---|---|---|---|---|---|---|
| raw (`link_video` at 0.4) | 0.8054 | — | 0.8107 | 17528 | 1795 | 2298 | 0.962 | 1.085 |
| + recover (gap ≤ 5) + prune (min 4) | 0.8399 | +0.0345 | 0.8444 | 18158 | 1679 | 1668 | 0.969 | 1.068 |
| + linefit (0.6, ±3) | 0.8547 | +0.0493 | 0.8593 | 18300 | 1471 | 1526 | 0.970 | 1.068 |
| **+ momentum (w=0.02, m=0.7)** | **0.8565** | **+0.0511** | 0.8608 | 18299 | 1431 | 1527 | 0.970 | 1.065 |

For scale, the detector + the tuned pipeline of `README.md`
(`logs/2026-09-02/9/scores_pp.json`) scores **0.8333** on the same 71 stacks.

Recovery with the pruning it pays for is worth +0.035, the smoothing another
+0.015, the momentum term +0.002. Every row is the competition metric, run by
`main_edge_postprocess.py`; `fast_score` agrees with it to 0.0000 here.

**Scoring caveat, and it is worth 0.005.** `submission.py` writes
`int(round(z))`, so anything sub-voxel is thrown away before the score is
computed — and the z voxel is 1.625 µm, up to 0.81 µm of exactly the error the
smoothing exists to remove. `exp_edge_pp.py` therefore rounds before scoring.
An earlier pass here did not, and read 0.8570 for a chain that submits at
0.8524. Anything that moves a coordinate has to be measured after the round.

**Where the settings were tuned.** The stages and their coarse settings came
off a 16-stack subset spanning the node-excess range (`--spread 16`); the
smoothing weight and window, the final `recover_max_gap` / `min_track` choice,
and the momentum weight were picked on all 71. That last part is the fold being
reported, so read those knobs as fitted: the smoothing plateau is flat
(0.8523–0.8547 over weight 0.5–0.8) and the choice inside it is worth ~0.002,
and the momentum weight is a narrow optimum worth ~0.002 at its peak.

## Retuned for `preds_pp` input — and the chain is now nearly worthless

The table above is what these stages were worth against the graph they were
tuned on. They are worth **+0.0019** against the current one, and the cause is
not the chain: it is that `main_edge.py`'s `nodes_folder` moved to
`../logs/2026-09-02/9/preds_pp/`, whose detections the detector's own pipeline
has *already* recovered. The stages were built for a node-starved graph and are
now handed an over-detected one.

Fold 0, all 71 `44b6` stacks, off `logs/edge_run/2026-09-05/24/preds_0.40`:

| chain | score | Δ | node recall | nodes | ratio |
|---|---|---|---|---|---|
| raw (`link_video` at 0.4) | 0.8485 | — | 0.966 | 39626 | 1.081 |
| + recover (gap ≤ 5) | 0.8486 | +0.0001 | 0.966 | 39697 | 1.083 |
| + recover + prune (min 4) | 0.8488 | +0.0003 | 0.967 | 40213 | 1.095 |
| + linefit (0.6, ±3) | 0.8496 | +0.0011 | 0.967 | 40213 | 1.095 |
| + momentum (w=0.02, m=0.7) | 0.8504 | +0.0019 | 0.968 | 40210 | 1.095 |

Read the `ratio` column, which is nodes over `estimated_number_of_nodes`. The
input already sits at **1.081** — 8% more nodes than the truth expects — and
`recover` pushes it to 1.095 by inserting 587 more, for +0.0001. In the table
above, pruning pulled 1.085 *down* to 1.068. The stage's premise is inverted.

**What was retuned.** `min_track` 4 → 2: with little left to recover, pruning at
4 was deleting real short tracks rather than fragments. Worth +0.0007, and the
only setting whose gain held on both halves of the fold (+0.0008 / +0.0006).

**What was rejected, and why it looks tempting.** Re-deciding the assignment at
`threshold=0.45` scores +0.0010 overall and wins on 45 stacks against 23 — but
the two halves of the fold disagree in sign (+0.0034 / −0.0008), so its whole
contribution is a handful of large stacks in one half. Combined with
`min_track=2` it reads +0.0022, which is what a greedy sweep would ship. Around
45 chains were tried against these same 71 stacks; at that search width a +0.002
effect that splits like this is selection noise, not a setting. It needs a
second embryo before it goes in.

**The honest conclusion.** There is very little left here for geometry to fix.
The oracle on these detections is 0.963 and the model reaches 0.849 on its own;
what separates them is the model mis-ranking links in the ~7% of cases that are
not the mutual nearest neighbour, and no stage in this file can see that.

## Retuned again for `preds_0.45`

The block above was tuned on `logs/edge_run/2026-09-05/24/preds_0.40`. A later
model/checkpoint run, `logs/edge_run/2026-09-07/2/preds_0.45`, over-detects
harder still: raw node ratio **1.157**, against 1.081 for the graph the
`min_track=2` retune above was made against. Same axis, further along it, so
the two retunes point the same direction and it is worth reading them
together rather than as separate incidents.

Fold 0, all 71 `44b6` stacks, `main_edge_postprocess.py`'s own scoring (not
`fast_score` — see the caveat below):

| chain | score | Δ | node recall | nodes | ratio |
|---|---|---|---|---|---|
| DEFAULTS as tuned above (`min_track=2`, `recover_min_dup=4`) | 0.8590 | — | 0.984 | 43960 | 1.196 |
| **+ `min_track=5`, `recover_min_dup=3`** | **0.8612** | **+0.0022** | 0.982 | 43709 | 1.185 |

**`min_track`: 2 → 5.** The `preds_pp`-input retune's premise — "little left
to recover, so pruning at 4 deletes real short tracks" — does not hold here.
On this graph pruning at 5 still nets a gain, and 6 is already past the peak;
swept coarsely on a 16-stack `--spread` subset first, then confirmed on all
71 (`exp_edge_pp.py`, `[recover=1,min_track=N,...]`). Worth **+0.0031** on the
full fold measured with `fast_score`, positive on both halves of the fold
(+0.0017 / +0.0033) and on 57 of 71 stacks individually — a broad gain, not a
handful of stacks flipping sign.

**`recover_min_dup`: 4.0 → 3.0.** With `min_track` raised, `recover` can
afford to sit a little closer to an existing detection before it refuses to
insert a node — `min_track` is now doing more of the work of removing what
that would otherwise leave behind. The plateau is flat from 2.0 to 3.5
(0.8626–0.8631 by `fast_score`); 3.0 sits in the middle of it. This is the
*opposite* direction from `README.md`'s finding for the detector pipeline and
from this section's own "rejected" note above (`recover_min_dup=3` cost
−0.0044 on the `preds_pp` graph) — both of those were tuned on graphs with
less over-detection to begin with, where relaxing the guard mostly adds
duplicates rather than mopping up the ones `min_track=5` is about to prune
anyway. Read per-stack rather than pooled, the combined change is positive on
only 11 of 71 stacks against 60 negative, with the net gain carried by a few
large wins; both halves of the fold still land positive (+0.0027 / +0.0007),
but with less margin than `min_track` alone, and it is fair to call this the
weaker of the two changes.

**`suppress_parallel` (`dedup.py`), tried and rejected.** Wired into
`exp_edge_pp.py`'s `_chain()` as `dedup_radius` / `dedup_overlap` (0 = off),
run after `recover` and before `min_track` — after recovery both sides of a
shadowing pair are as complete as the chain will make them, so the length
comparison and overlap fraction `suppress_parallel` uses to pick a winner are
measured on the real track rather than an unfinished fragment; before pruning
because dropping a whole duplicate track can leave the other side of the pair
short, which `min_track` should then get to clean up. Swept `radius` 5–8 and
`overlap` 0.3–0.9 (`exp_dedup.py`'s own range) against the retuned base:

| radius | overlap | nodes dropped (of ~40k) | score Δ |
|---|---|---|---|
| 5–6 | 0.5–0.9 | tens to a few hundred | **+0.0002**, within noise |
| 6 | 0.4 | ~150 | +0.0008 on the subset, +0.0002 on the full 71 |
| 7 | 0.7 | ~1 000 | −0.0046 |
| 8 | 0.7 | ~3 000 | −0.0100 |

At the radii that leave real neighbouring cells alone (≤ 6 µm) the stage
drops too few nodes to move the score — this graph's over-detection is not
the "two peaks a fixed NMS radius apart, both tracked for their whole life"
shadow `dedup.py` was built for; whatever is making the node ratio 1.157 here,
it is not sitting still long enough for `suppress_parallel`'s per-track
overlap test to catch it. Past 7 µm it starts eating real detections instead
— the sharp jump in drop count between 6 and 7 µm is genuine cells, not more
duplicates. Not wired into `main_edge_postprocess.py`.

**Scoring caveat.** `exp_edge_pp.py`'s `fast_score` and
`main_edge_postprocess.py`'s own metric (`tracksdata`'s `evaluate`, the actual
competition path) agreed to 0.0000 for the chain tuned earlier in this file,
but diverge here: the retuned chain reads 0.8628–0.8631 under `fast_score`
and **0.8612** under `main_edge_postprocess.py`. The number that matters is
the latter — it is what actually gets scored — so that is what is reported
above and what `DEFAULTS` was chosen against. `fast_score` is still the right
tool for the coarse sweep (it is what makes a 71-stack sweep take seconds
instead of minutes); just re-check anything reported as a final number against
the real pipeline before trusting the third decimal.

## Dumping the candidates is what makes this possible

`main_edge_predict.py --dump_candidates` writes a `candidates` array —
`(src row, tgt row, prob, distance um, dt)` — beside the committed `edges`.
Without it the linking threshold and the assignment rule are decisions frozen
on the GPU; with it they are post-processing, the whole chain re-runs in two
seconds a stack, and `edge_pp.assign` reproduces `link_video`'s output exactly
— identical tp / fp / fn on all 71 stacks, which is the regression test for
everything else here.

It costs about 2 MB a stack — 255k candidates against 37k committed edges.

## Where the score actually goes

`diag_edge.py` puts every ground-truth link in one bucket. All 71 stacks,
19 826 links, before and after the chain:

| | raw assign | | after the chain | | what could win it back |
|---|---|---|---|---|---|
| tp | 17 528 | 88.4% | 18 300 | 92.3% | — |
| lost | 1 066 | 5.4% | 720 | 3.6% | an endpoint has no detection within 7 µm — only `recover` |
| declined | 851 | 4.3% | 336 | 1.7% | the linker left an endpoint free |
| stolen | 381 | 1.9% | 470 | 2.4% | the linker gave an endpoint to somebody else |

`lost` is the biggest bucket, `declined` the second, and both are consequences
of a missing node — which is why `recover` is the stage that pays and why it
halves `declined` as well: filling a hole lets the assignment get *both* sides
of it right. `stolen` goes up by 89, the price of the nodes it puts back.

And the ceiling, over the links whose endpoints both exist: the true child is
the parent's **rank-0 candidate 95.0%** of the time, top-3 99.2%, and absent
from the candidate set in **0.2%**. The model's ranking is very nearly right,
so the chain is a *node* stage and a *coordinate* stage and does no re-ranking
at all.

**The links it does get wrong are not recoverable by geometry either.** On the
186 failures across 12 stacks, the drift-corrected flow residual is a median
**2.30 µm for the link the assignment chose** and **6.89 µm for the true one**;
the true link is the closer of the two in **3.4%** of cases. Model and geometry
are wrong *together*, which is the same conclusion `README.md` reaches from the
other side — the endpoints matched to an annotated cell are ~8 µm apart when
the cell moved ~2 µm, so the coordinates are wrong and no linker fixes that.
Only 2 of the 186 have both endpoints free, so there is nothing for a
"commit more links" stage to pick up either.

## The stages

**1. `recover` — put back the detections that are missing.** `recover.py`
unchanged: find a chain that ends at `u` and one that starts at `u + L + 1`,
remove the field's own drift, gate on the residual as a *speed*, and
interpolate `L` nodes into the hole. What is new is the re-linking afterwards.
The model never scored the recovered nodes, so `edge_pp.recover_nodes` gives
them geometric candidates of their own — nearest neighbours in the neighbouring
frames, at a synthetic probability falling from 1 at zero distance to the
threshold at 10 µm — and re-solves the assignment over the union. Every
original pair keeps the probability the model gave it, so a recovered node can
win a link only where the model had no strong opinion.

`max_gap` 5 and 3 are within 0.0002 of each other on the fold (3 is the
detector pipeline's default). Beyond 5 the node count costs more than the links
are worth: gap ≤ 7 finds 55 more true positives and loses 0.0036, because it
also adds 1 400 nodes and 80 false positives.

**2. `prune_short` — drop chains shorter than 4 nodes.** This moves two terms.
A three-node chain is a run of spurious detections, so its edges are false
positives; and it inflates `N_pred`, which the score charges at 0.1 per unit of
relative excess. On the fold, `min_track` 4 and 5 tie at 0.8536 and 3 costs
0.0020; 8 is clearly too far (−0.006 on the subset, node recall 0.928).

Note the metric does not clamp the ratio, so under-detecting is *rewarded* —
`min_track=6` reaches a ratio of 0.938 and a factor above 1. It still loses,
because the Jaccard falls faster. Worth knowing before reading the ratio column
as if smaller were always better.

**3. `linefit_smooth` — move each node toward a line fitted along its own
track.** `pp_public` does this and it is the one part of that module worth its
runtime; `edge_pp.linefit_smooth` is the same computation in one pass over
arrays, 0.04 s a stack against 38. Worth **+0.015**, and it is the only stage
here that attacks the thing `README.md` names as the binding constraint —
localisation. On a synthetic straight track with 1 µm of noise it takes the
interior error from 1.46 to 0.46.

This is *not* in tension with `analysis_jitter.py`, which found that smoothing
a node against its temporal neighbours makes the median error worse. That was a
moving average over the raw detections; this fits a line along a *committed
track*, only over unbranched stretches, and blends part-way rather than
replacing. Both the weight and the window matter, and `pp_public`'s own
settings are past the optimum on both. All 71 stacks:

| weight | ±window | score |
|---|---|---|
| 0 (off) | — | 0.8399 |
| 0.3 | 3 | 0.8464 |
| 0.5 | 3 | 0.8524 |
| **0.6** | **3** | **0.8547** |
| 0.7 | 3 | 0.8536 |
| 0.8 | 3 | 0.8523 |
| 1.0 | 3 | 0.8477 |
| 0.5 | 2 | 0.8506 |
| 0.7 | 4 | 0.8498 |

Replacing the position outright (weight 1.0) gives back half the gain: the
detection carries real information the fit does not, and the point is to
average the two.

## Momentum: a per-track velocity in the assignment

`CompareCV.ipynb`'s linker walks the frames in order and carries a state on the
matched tracks. Strip the appearance out of that idea and the state becomes the
cell's **own motion**:

    v[track]   <- m * v[track] + (1 - m) * (pos_child - pos_parent)
    residual    = || pos_child - (pos_parent + v[track]) ||
    cost        = log-odds(p) + vel_weight * residual

This is not the neighbourhood flow `relink.py` corrects by — that is what the
cell's *neighbours* did — and `postprocessing/README.md` records per-cell
velocity as tried and worse, because "tracks fragment too often for a cell's
own history to be available when it is needed". **That premise no longer
holds.** On this graph the median track is 12 nodes and the p75 is 21, and
after `recover` closes the gaps it is longer still.

Which is exactly where the gain shows up. All 71 stacks:

| | score | Δ | tp | fp |
|---|---|---|---|---|
| raw `assign` | 0.8054 | — | 17528 | 1795 |
| raw `assign` + momentum (w=0.02, m=0.7) | 0.8056 | +0.0003 | 17497 | 1750 |
| full chain | 0.8547 | — | 18300 | 1471 |
| **full chain + momentum (w=0.02, m=0.7)** | **0.8565** | **+0.0018** | 18299 | **1431** |

On the raw graph it is worth nothing (+0.0003); inside the chain it is worth
+0.0018, and in both cases **what it moves is false positives** — 1471 → 1431
after the chain, with the true positives unmoved at 18299. That is the shape a
motion prior should have: it does not find new links, it declines the ones that
ask a cell to jump. On the raw graph it declines 31 true links to remove 45
false ones and nets nothing; after `recover` has repaired the tracks the
velocity is reliable enough that it costs one true link for forty false.

`recover` is what makes it work, and the order matters — the momentum term is
applied in the recovery re-link too, where the tracks it has just repaired are
the ones with a usable velocity.

Settings: `vel_weight` 0.02, momentum 0.7. The weight is a narrow optimum —
0.01 gives +0.0007, 0.03 gives −0.0009, 0.05 clearly loses. Momentum 0.7 and
0.9 are within 0.0001 of each other and both beat 0.5 (+0.0012) and 0.3.
Seeding a new track with the field drift rather than zero is worth +0.0006.
Costs 1.9 → 2.7 s a stack.

**Honest size.** The subset said +0.0077 and the fold says +0.0018; the subset
over-promised, as it did for `recover_min_dup`. It is a real gain, measured
twice in the same direction, and it is small.

## The other graph builder: `notebooks/CompareCV.ipynb`'s linker

That notebook links on the **contrastive embeddings** rather than on a scored
candidate graph — dense cosine matching frame to frame, an EMA track state
(`m * track + (1 - m) * detection`), and `embeddings.match.dynamic_threshold`
for the cutoff, walked over contiguous frame runs. Blending its cosine with the
*node* transformer's edge probability is worth **+0.051 cv** there
(0.734 → 0.785), which is the largest single number in that notebook and the
reason to try it here.

It is ported faithfully as `edge_pp.ascent_link` (`ascent=1`), and the cosine
is also available as a blend into this module's own linker
(`emb_weight`, `emb_momentum`), so both halves of the idea are measurable.
All 71 stacks:

| graph builder | score | Δ | tp | fp |
|---|---|---|---|---|
| `assign` (this module) | **0.8054** | — | 17528 | 1795 |
| `ascent_link`, 0.25 cos / 0.75 model | 0.8033 | −0.0020 | 17537 | 1861 |
| `ascent_link`, model only (the rule alone) | 0.8025 | −0.0028 | 17532 | 1876 |
| `ascent_link`, cosine only (the notebook's "contrastive") | 0.7493 | −0.0561 | 17129 | 2885 |
| `assign` + `emb_weight=0.25` | 0.8036 | −0.0017 | 17545 | 1862 |
| `assign` + `emb_weight=0.5` | 0.7982 | −0.0072 | 17545 | 2010 |
| `assign` + `emb_weight=0.5, emb_momentum=0.5` | 0.7976 | −0.0078 | 17544 | 2026 |

and with the full chain on top: 0.8547 for `assign`, **0.8551** for the
`ascent_link` front-end (+0.0004, and 35% slower), 0.8545 for `emb_weight=0.25`.

**It does not transfer, and the reason is that this model already reads those
embeddings.** `main_edge.py` runs with `use_embeddings: true` and the same
256-d vectors are the HOCT node features. The notebook's gain comes from giving
a graph that had *never seen* them a second, independent signal; here it is the
same signal twice. Measured directly, over the candidate sets of 12 stacks:

| ranker | true child ranked first | top-2 |
|---|---|---|
| model probability | 94.14% | 98.99% |
| cosine alone | 93.67% | 97.47% |
| 0.75 model + 0.25 cosine | 94.18% | 98.78% |
| 0.50 / 0.50 | 94.10% | 98.61% |

The cosine is an excellent ranker on its own — 0.82 median on the true
candidate against 0.17 on its rivals — and blending it moves rank-1 accuracy by
**+0.04 points**. There is nothing left in it to add.

Two things worth separating out, because they are not the same claim:

* **The signal adds nothing** (above), and pushing the blend past 0.25 actively
  hurts: `emb_weight` 0.25 → 0.5 keeps tp flat at 17545 and takes fp from 1862
  to 2010. Past that point the cosine is overriding a calibrated probability
  with a similarity that has no notion of how far a cell may move.
* **The decision rule is slightly worse too.** `w_emb=0, w_trans=1` runs the
  notebook's rule on this model's probabilities alone and scores 0.8025 against
  `assign`'s 0.8054 — so dense matching plus a dynamic threshold loses 0.003 to
  a log-odds threshold, holding the signal fixed. That is the expected
  direction: `dynamic_threshold` is a statistic of the frame's cost matrix,
  which is what you want when the scores are uncalibrated similarities, and
  strictly less than what you want when they are calibrated probabilities with
  a known break-even (0.45, see `README.md`).

`ascent_max_um` — refusing a link longer than a few microns, which the dense
cost otherwise has no reason to — is available and costs 0.06–0.08. The
assignment is a permutation, so a parent whose gated pair the solver picked
loses its link outright instead of falling back to its best near neighbour; it
drops 358 true positives to remove 39 false ones. The notebook does not gate
either.

Kept in the tree rather than deleted: it is a working implementation of the
alternative, and if the model is ever retrained with `use_embeddings: false`
the premise of this whole section changes.

## The matching rule itself

Two properties of `link_video`'s assignment were checked directly, because both
look like they could be wrong and neither is.

**The cost is a log-probability ratio, not a log-odds one.** `link_video` uses
`log(threshold) - log(p)`, where the decision-theoretically correct cost under a
one-to-one constraint is `logit(threshold) - logit(p)` — declining an edge costs
`log(1 - p)`, and dropping that term flattens the scale (a factor of 7.8 between
p = 0.45 and p = 0.99, where the odds span 24). Both cross zero at
`p = threshold`, so they accept and reject the same pairs in isolation and
differ only on contested ones. Available as `cost="logit"`; measured, it moves
**3 to 9 edges in 36 000** and the score by 0.0000. The candidate graph is
sparse enough — 5 per node — that the sign decides nearly every pair and the
magnitude almost never does. `"logprob"` stays the default so `assign` keeps
reproducing `link_video` exactly.

**One-to-one costs true links, and is still right — for the edge term only.**
The ground truth *does* have divisions (26 in 44b6, 125 in 6bba); the sentence
that stood here said it had none, and that error is what left the score's
division term at zero for this whole chain. See "Divisions" at the end of this
file. They are 0.13% of the links, so the argument below survives for the edge
term unchanged. The *predictions* are over-detected, so two detections of one
cell both legitimately want the same parent and exclusivity forces one to steal
or go unlinked. Dropping the constraint on the parent side (`match="child"`:
every child takes its best parent, a parent may keep several children) does
recover links — over 12 stacks, 2 213 of 2 375 reachable ground-truth links
against the Hungarian's 2 189:

| rule | tp | recall |
|---|---|---|
| Hungarian | 2 189 | 92.17% |
| per-parent argmax | 2 202 | 92.72% |
| per-child argmax | 2 213 | 93.18% |
| every candidate above threshold (no constraint) | 2 230 | 93.89% |

But on the metric it loses: `match="child"` gains 7 true positives and pays 15
false ones, **−0.0043**. The exclusivity is earning its keep, which is the same
conclusion `README.md` reaches for the detector's graph by a different route.
The ceiling row is worth keeping in view though — even committing *every*
above-threshold candidate reaches only 93.9%, so at this threshold 6% of the
reachable links are not in the candidate set above 0.4 at all.

## What did not work

Recorded so they are not re-tried.

- **The linking threshold.** Flat: 0.30 is +0.0005 and 0.50 is −0.0009 against
  0.40 on the subset. The break-even the metric implies is 0.45 (`README.md`,
  "What the metric actually rewards") and the model sits close enough to it
  that the assignment barely moves. Nothing to win here.
- **Mixing flow residual into the assignment cost** (`assign(flow_weight=...)`,
  `max_residual=...`). The knobs exist and are not used, because of the 2.30 vs
  6.89 µm measurement above: on the links the model gets wrong, geometry
  prefers the same wrong link. This is the learned-graph version of
  `model_assign.py`'s negative result.
- **`bridge_tracks`** — join a chain ending at `u` to one starting at `u + 1`
  on geometry when the assignment declined the link. Adds **10 edges a stack**
  at a 6 µm gate and moves the score by 0.0000. The assignment already commits
  every pair the model scores above threshold, so a free end next to a free
  start barely exists.
- **`link_through_isolated`** — the same idea over a longer gap, wiring a chain
  *through* an unlinked detection that already sits in the hole rather than
  inventing one (this is `pp_public`'s "gap_reused_existing"). It finds **0–2
  edges a stack**, for the same reason, and costs −0.0004 because the nodes it
  saves from pruning are worth less than the node count they add.
- **`drop_long(14.2 µm)`**, `pp_public`'s edge length cut: +0.0001. The model
  does not commit long links in the first place.
- **`recover_passes=2`**: +0.0007 on the subset for 60% more time. Same
  conclusion as the detector pipeline reached.
- **`recover_min_dup=3`** (let a recovered node sit closer to an existing
  detection): **+0.0017 on the 16-stack subset and −0.0044 on the fold.** The
  clearest overfit of the tuning, and the reason every candidate here was
  re-checked on all 71. `README.md` reports the opposite sign for the detector
  pipeline (4–7 µm beat forbidding, by 0.006), which is consistent: this graph
  already recovers the nodes that matter, so relaxing the guard only adds
  duplicates.
- **`pp_public` itself.** 38 s a stack, and everything in it that helps — gap
  closing, isolated-node pruning, short-track filtering, linefit — is a stage
  above. Its `motion_relink` also *replaces every edge* with a geometric
  re-link, which is exactly what should not happen to this graph.

## Files

| file | what it is |
|---|---|
| `edge_pp.py` | the stages: `assign`, `recover_nodes`, `prune_short`, `linefit_smooth`, `ascent_link` (the CompareCV linker), and the three that did not pay |
| `exp_edge_pp.py` | score named or inline chains over a preds folder; `--spread N` for a subset spanning the difficulty range |
| `diag_edge.py` | every ground-truth link as tp / stolen / declined / lost, plus the candidate-rank ceiling |
| `../main_edge_postprocess.py` | the chain end to end: npz, geff and the competition metric |

Chains are written inline, so sweeping a knob needs no edit:

```bash
python exp_edge_pp.py --preds_dir <dir> --spread 16 \
  --chains '[recover=1,min_track=4],[recover=1,min_track=4,linefit=0.5,linefit_window=3]'
```

## What is left

The buckets after the full chain still put `lost` first — 720 links, 3.6%, on
top of 470 stolen and 336 declined. Two things follow:

- **Node recall is the constraint, and `recover` only reaches gaps that are
  bracketed.** A cell missing at the edge of the field, or for a run longer
  than 5 frames, has no chain on the far side to pair with. That is a detector
  fix, not a post-processing one.
- **Localisation is worth more than anything else here**, and `linefit_smooth`
  is the cheapest possible attack on it — a straight line through detections
  the tracker already agreed belong together. `README.md`'s oracle says perfect
  localisation on the annotated cells alone is worth 0.2–0.3 on the hard
  stacks. The centroid-vote head (`--offset_head`) is the real version of this
  and is still unscored.

## Divisions — the term this chain was scoring zero on

The fold score is `adj_edge_jaccard + 0.1 * division_jaccard`, and every number
above is the first term alone. The second was **0.0**, with `division_fp` also
**0** — the chain was not proposing divisions and getting them wrong, it was not
proposing them at all. `assign` solves a one-to-one matching per frame pair, so
a fork is not something it can express.

The claim in "One-to-one costs true links, and is still right" above, that *the
ground truth has no divisions*, is **wrong**. It is nearly harmless for the edge
term — 26 links in 19 826 is 0.13% — and it is what kept anyone from looking at
the other 10% of the score. The annotation holds:

| embryo | divisions | stacks with ≥1 |
|---|---|---|
| 44b6 | 26 | 21 / 71 |
| 6bba | 125 | 66 / 128 |

`propose_divisions` gives a parent a second child. The gate stack is the public
"safe division" repair — see `PublicChanges.md`, the 946 Harmonic Fusion
notebook, the strongest of that family — with one gate that notebook cannot
have: **the model's own probability for the pair**. 946 works from geometry
alone because it has no scored candidate graph; here every pair carries a
probability, and on the true second daughters it runs a median 0.368 against an
assignment threshold of 0.45. That is precisely why they were being dropped, and
it is the sharpest single signal in the stage.

All 199 out-of-fold stacks of `logs/edge_run/2026-09-08/13/preds_0.50`, scored
with the real metric (`exp_divisions.py`, not `fast_score` — see below):

| chain | cv | Δ | 44b6 | 6bba | div tp/fp/fn |
|---|---|---|---|---|---|
| no divisions | 0.8323 | — | 0.8624 | 0.8021 | 0/0/151 |
| `min_prob=0.50` | 0.8378 | +0.0056 | 0.8657 | 0.8099 | 12/19/139 |
| `min_prob=0.45` | 0.8409 | +0.0086 | 0.8706 | 0.8112 | 18/47/133 |
| `min_prob=0.40` | 0.8432 | +0.0109 | 0.8745 | 0.8118 | 23/69/128 |
| **`min_prob=0.40, diverge=3.0`** | **0.8437** | **+0.0114** | 0.8748 | 0.8126 | 21/41/130 |
| `min_prob=0.35, diverge=3.0` | 0.8440 | +0.0117 | 0.8747 | 0.8133 | 23/47/128 |

**It gains on both embryos**, +0.0123 and +0.0105, which no other stage in this
file can say — and it costs 0.0004 of adjusted edge Jaccard, because ~2 000
added edges against 200 000 committed ones is nothing.

**What was tuned, and how flat it is.** `diverge_um` 3.0 over 946's 2.25 is the
only knob whose gain held its sign on both embryos at both thresholds
(+0.0007…+0.0009); 4.0 gives it all back. `min_prob` 0.35 and 0.40 are within
0.0003 and identical on 44b6 — the choice inside that tie is arbitrary and 0.40
was taken as the conservative end. Read the whole 0.35–0.40 × 2.25–3.0 block as
one plateau at ≈ +0.011.

**Dropping `require_divergence` is a trap.** It raises true positives from 2 to
3 on a 12-stack probe and false positives from 6 to 60, for −0.0026. The gate
that asks the two grandchildren to have separated by `t+2` is what separates a
division from two detections of one cell.

**The ceiling is not this stage, and part of it is the detector.** Sisters at
the division frame sit at median 8.98 µm (44b6) and 11.47 µm (6bba), but 22 of
the 151 pairs are closer than the detector's 7 µm NMS radius. Matching each GT
daughter to a *distinct* detection:

| | both daughters have their own node | collapsed to one by NMS |
|---|---|---|
| 44b6 | 22 / 26 (85%) | 4 (15%) |
| 6bba | 115 / 125 (92%) | 10 (8%) |

That is 9% of pairs losing a daughter — but it is measured on the *raw* node
set, and `propose_divisions` runs after `recover`, which interpolates nodes into
gaps and puts some of them back. On the chain's own output the per-cell picture
is milder, against the same one-to-one matcher `node_recall` uses:

| role | 44b6 | 6bba |
|---|---|---|
| ordinary cells | 0.983 | 0.943 |
| dividing parents | 1.000 | 0.976 |
| **daughters, division frame** | **0.942** | **0.932** |

A daughter is 4.1 points below an ordinary cell on 44b6 and 1.1 on 6bba — 3 of
52 and 17 of 250 cells. The dividing parent itself is detected essentially
perfectly, which is what one would expect: the crowding exists only after the
split. So detection is **not** the constraint. Both endpoints are available for
the large majority of divisions and the stage still converts 15% of them; the
loss is the gates and the linking, and a division-aware NMS exemption would be
chasing ~20 cells across 199 stacks. Worth knowing too that the pairs NMS does
collapse are the closest ones, which the divergence gate would reject anyway —
the two losses overlap rather than add.

**`fast_score` cannot see any of this.** It computes the edge Jaccard and
nothing else, so it is blind to the entire division term and reports 0.0000 for
every row of the table above. Every chain in this file was ranked with it, which
was safe only because nothing in those chains moved divisions. `exp_divisions.py`
scores with `tracking_cellmot.metrics.evaluate` instead — the same scorer
`main_edge_postprocess.py` reports — and caches the base chain per stack so a
sweep still costs ~35 s over all 199 stacks. **Use it, not `exp_edge_pp.py`, for
anything that touches divisions.**

### Where the division recall actually goes

`diag_divisions.py` takes every annotated division, locates the predicted parent
and both daughters, and walks `propose_divisions`' gates in their real order
against the pair that ought to have been proposed, attributing each division to
the *first* gate that rejects it. At the shipped settings, over all 151:

| first gate that rejects it | 44b6 | 6bba | all |
|---|---|---|---|
| second daughter already has a parent | 11 | 48 | **59** |
| prob < `min_prob` | 4 | 25 | 29 |
| divergence (missing grandchild / too close) | 2 | 12 | 14 |
| symmetry | 1 | 9 | 10 |
| a daughter not detected | 2 | 8 | 10 |
| not the mutual nearest neighbour | 2 | 4 | 6 |
| committed child is neither daughter | 0 | 4 | 4 |
| parent not detected / no single child / distance | 1 | 5 | 6 |
| passes every gate | 3 | 10 | 13 |

**The gates are not the constraint and neither is the detector.** Divergence
rejects 14 and symmetry 10; tightening or reshaping either is worth a handful of
events. The single largest bucket is that the second daughter is **already
somebody else's child** — 39% of all divisions — so the stage never sees a free
node to attach.

Those are not assignment mistakes. On those 59, the model gives the rival parent
a median probability of **0.719** for the daughter against this parent's
**0.016**, and in only 11 of 59 does our parent even outscore the rival. The
assignment is not close to wrong; it is confident.

**What is actually happening: the fork is split across a duplicate.** In **34 of
the 59** the rival parent is itself a detection matched to the *same annotated
parent cell*. The graph over-detects, the dividing cell is found twice, and the
two copies take one daughter each — so the division is already in the graph and
no node forks. That is 23% of all divisions, the largest recoverable bucket
here, and `merge_duplicate_parents` is the attempt at it. It does not work; see
its docstring for the numbers and for the one idea that might.

So the ranked list of what is left, by size: duplicate-split forks (34),
`min_prob` (29, and lowering it costs more in false positives than it returns —
0.35 and 0.40 are a plateau), divergence and symmetry (24 between them), missing
detections (10). Only the first is worth real effort, and it needs a duplicate
test that geometry at one frame cannot give.

### A classifier over the candidates, instead of the gates

The gates are a conjunction of thresholds — one *point* in precision/recall —
and the diagnostic above says no single one of them is doing the work. The
natural answer is to score the candidates and rank them, so
`div_dataset.py` writes every pair the loose proposer would consider, with
everything the gates test recorded as a *feature* rather than applied as a
filter, and `div_classifier.py` fits a shallow gradient boosting over it.

Two numbers frame what is possible before any model is fitted:

- **The candidate set tops out at 48% recall** (73 of 151). A candidate needs a
  *free* second daughter, and 59 divisions fail exactly that — the
  duplicate-split problem again. No ranker reaches them.
- **73 positives, two embryos.** The only honest split is leave-one-embryo-out,
  which means fitting on 61 and testing on 12, or fitting on 12 and testing on
  61.

All 199 stacks, real metric, out-of-fold scores:

| selection | cv | Δ | 44b6 | 6bba |
|---|---|---|---|---|
| **hand-set gates** | **0.8437** | **+0.0114** | +0.0123 (divJ 0.125, 4 tp / 6 fp) | +0.0105 (divJ 0.106, 17 tp / 35 fp) |
| classifier, top 20 / stack | 0.8410 | +0.0087 | +0.0134 (divJ 0.135) | +0.0040 (divJ 0.054, 229 fp) |
| classifier, top 50 / stack | 0.8398 | +0.0076 | **+0.0162** (divJ 0.167, 8 tp) | −0.0011 (divJ 0.039, 693 fp) |

**The gates win, and the reason the classifier loses is the label count, not the
idea.** The two folds are not comparable experiments. Held out on 44b6 the model
had 6bba's 61 positives to learn from and it *beat* the gates — +0.0162 against
+0.0123, twice the true divisions. Held out on 6bba it had 44b6's 12 and emitted
693 false positives. The pooled number is an average over one reasonable fit and
one hopeless one.

Worth keeping in view when this is next picked up: a model that shipped would
train on all 73, which is more than the fold that already won had. That is not a
measurement — with two embryos there is no third to check it on, and believing
the fold that agrees with you is the error this file warns about elsewhere — but
it does say where the effort goes. **More labelled divisions is the whole
problem.** The cheaper half-measure is to let the gates choose the candidates and
the classifier only rank *within* them, which caps what a bad model can cost at
the frame/global caps instead of at 693 false positives.

Not wired into `main_edge_postprocess.py`; `div_dataset.py`, `div_classifier.py`
and `exp_divisions.py --clf_scores` are measurement scaffolding.

**One harness bug worth not repeating.** The classifier path first read its
scores through a module global set in `main()` and handed to a `Pool`
initialiser. That works when this module is imported and silently loads nothing
when it is run as `__main__` — which is the only way it is ever used. Every
classifier config scored exactly +0.0000, which reads like a clean null result
and was not one. The path now travels inside the config dict. Any stage here
whose output is "it changed nothing" deserves one check that it ran at all.

### Three attacks on the division term, and what they measured

Written after the gate diagnostic above pointed at the assignment and the model
rather than at the gates. All on the 199 out-of-fold stacks of
`logs/edge_run/2026-09-08/13/preds_0.50`, real metric.

| config | cv | Δ | div tp / fp |
|---|---|---|---|
| **gates only (shipped)** | **0.8437** | **+0.0114** | 21 / 41 |
| + `merge_shadow_forks`, `shadow_frac=0.85` | 0.8437 | +0.0114 | 21 / 41 |
| + `merge_shadow_forks`, `shadow_frac=0.6` | 0.8431 | +0.0108 | 21 / 61 |
| `assign(fork_threshold=0.39)` | 0.8043 | −0.0280 | 73 / 3 477 |
| `assign(fork_threshold=0.35)` | 0.7972 | −0.0351 | 78 / 4 369 |
| `assign(fork_threshold=0.30)` | 0.7902 | −0.0421 | **85** / 5 289 |

**The single most useful number here is 85.** Letting the assignment fork on
probability alone recovers **85 of the 151** annotated divisions, four times what
the tuned gates manage, and it is the worst-scoring row in the table. So the
earlier reading of the gate diagnostic — that recall is the constraint — was
wrong. The divisions are findable. `division_jaccard = tp/(tp+fp+fn)` charges
every false one at full weight, and at 72 true positives the false ones would
have to come in under ~570 against the 4 202 actually produced. **Precision is
the whole problem, and it needs roughly a 7x improvement at fixed recall.**

`fork_threshold` is monotone across every value tried, 0.20 through 0.39, always
toward "do not fork" — there is no operating point, not a badly chosen one.

`merge_shadow_forks` is the temporal duplicate test that `merge_duplicate_parents`
lacked, and time is the better signal: the distance version cost −0.0037, this
one −0.0006 at `shadow_frac=0.6` and exactly nothing at 0.85, where it fires 88
times and changes no metric. Tight enough to be safe is tight enough to be a
no-op. Both stay off by default.

**Which leaves the model.** Nothing in post-processing can manufacture precision
that the probabilities do not carry: the true second daughter scores a median
0.368 and so do thousands of free neighbours. The one lever that attacks this at
source is `division_weight` in `edge_transformer/targets.py` — divisions are
0.13% of the annotation, and both links of a dividing parent are now priced up
in the loss (10.0 in `main_edge.py`). That is implemented and **not yet
trained**; it is the only untested item of the three, and the only one whose
mechanism addresses what the table above says is wrong.

### Do the contrastive embeddings know a dividing cell? No.

Before building a division head on the detection UNet -- the natural next step,
since every division attempt so far has re-arranged the same geometric and
probability features and precision plateaued -- the cheap version of the
question: is mitotic appearance decodable from the embeddings already on disk?

126 279 annotated nodes, 148 of them dividing (1:853), leave-one-embryo-out
logistic regression on the 256-d `embeddings_contrastive_2026-09-08_0`, with the
detector's own `node_probs` as a control. Positives *and* negatives are both
restricted to annotated nodes: an unannotated detection's division status is
unknown, and scoring it as a negative would price a guess as a fact.

| held out | features | AUC | AP vs chance | precision@k |
|---|---|---|---|---|
| 44b6 | embeddings | 0.558 | 1.3x | 0/25 |
| 44b6 | `node_prob` (control) | 0.551 | 1.3x | 0/25 |
| 6bba | embeddings | **0.435** | 0.9x | 0/123 |
| 6bba | `node_prob` (control) | 0.582 | 1.2x | 0/123 |

Three ways of saying the same thing: the AUC straddles chance and goes *below*
it on the larger embryo, which is noise and not a weak effect; precision@k is
zero on both folds, so not one dividing cell makes it into a budget the size of
the positive class; and the embeddings do not beat the control, so what little
they encode is "confidently detected", not mitosis.

**What this does and does not rule out.** It tests these embeddings, not the
images. The contrastive model is trained to recognise *the same cell across
frames*, so its objective rewards invariance to temporal appearance change --
which is exactly what mitotic rounding is. It may be discarding the signal by
construction. The honest claim is that the cheap path is closed, not that the
volumes lack the signal; testing that properly means the detection UNet's own
feature maps at annotated nodes, which needs the images and a GPU.

Probe: `scratchpad/div_embed_probe.py`. Seven embedding keys sit in those npz
files and only the one the run used was tested.
