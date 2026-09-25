# Ensemble Summary

**Best ensemble edge model as of 2026-09-24: `cv 0.8786`**

## Ranked

| node set | edge run | best raw (`preds_<th>`) | best pp | macro adj | division | node recall |
|---|---|---|---|---|---|---|
| `v1_2026-09-13_1_2026-09-23_6` | `edge_run/2026-09-23/2` | 0.8263 @0.45 | **0.8786** | 0.8645 | 0.1405 | 0.9656 |
| `v4_2026-09-13_1_2026-09-24_5_append_b0.45_a0.40_d7` | `edge_run/2026-09-24/4` | 0.8258 @0.45 | 0.8760 | 0.8631 | 0.1283 | 0.9698 |
| `v6_2026-09-13_1_2026-09-24_5_append_b0.45_a0.45_d7` | `edge_run/2026-09-24/8` | 0.8246 @0.45 | 0.8742 | 0.8624 | 0.1185 | 0.9634 |
| `v5_2026-09-14_4_2026-09-24_5_append_b0.45_a0.45_d7` | `edge_run/2026-09-24/5` | 0.8241 @0.45 | 0.8736 | 0.8620 | 0.1163 | 0.9629 |
| `v7_2026-09-13_1_2026-09-23_6_append_b0.45_a0.45_d7` | `edge_run/2026-09-24/9` | 0.8209 @0.45 | 0.8723 | 0.8602 | 0.1210 | 0.9574 |
| `v2_2026-09-13_1_2026-09-24_5_consensus_r5_th0.45` | `edge_run/2026-09-24/3` | 0.8225 @0.45 | 0.8720 | 0.8590 | 0.1305 | 0.9511 |
| `v2_2026-09-13_1_2026-09-24_5_consensus_r5_th0.40` | `edge_run/2026-09-24/2` | 0.8234 @0.45 | 0.8685 | 0.8579 | 0.1061 | 0.9640 |

Per-embryo, at the best pp row for each (`n=71` / `n=128`):

| node set | 44b6 adj | 6bba adj | 44b6 score | 6bba score | 44b6 rec | 6bba rec |
|---|---|---|---|---|---|---|
| `v1` | 0.8817 | **0.8473** | 0.8954 | 0.8617 | 0.9825 | 0.9487 |
| `v4` | 0.8810 | 0.8453 | 0.8920 | 0.8600 | **0.9878** | **0.9518** |
| `v6` | **0.8820** | 0.8427 | 0.8914 | 0.8570 | 0.9826 | 0.9441 |
| `v5` | 0.8799 | 0.8441 | 0.8896 | 0.8577 | 0.9826 | 0.9431 |
| `v7` | 0.8789 | 0.8415 | 0.8903 | 0.8543 | 0.9784 | 0.9364 |
| `v2_th0.45` | 0.8804 | 0.8375 | 0.8922 | 0.8519 | 0.9784 | 0.9238 |
| `v2_th0.40` | 0.8799 | 0.8359 | 0.8864 | 0.8507 | 0.9877 | 0.9403 |

**The ranking is decided entirely by `6bba`.** On `44b6` all seven node sets land
within 0.0031 adj (0.8789–0.8820) and the ordering there does not match the
macro one — `v6` is *first* on `44b6` and third overall. On `6bba` the spread is
0.0114 (0.8359–0.8473) and its ordering **is** the macro ordering, exactly.
Anything evaluated on `44b6` alone will read as noise; tune on `6bba`.

Note `v4` has the best node recall on *both* embryos yet loses to `v1` on both
adj — more cells found, worse links made from them.

### The append 2×2: appender identity beats node budget

`v1/v4/v6/v7` vary only the appended model and its threshold, on a fixed base
(`13/1 @0.45`, `d_add 7`). `appended` is the fraction of the node set that came
from the ADD model.

| | appender `23/6` | appender `24/5` |
|---|---|---|
| **append @0.40** | **`v1`** 6.3% → **0.8786** | `v4` 9.1% → 0.8760 |
| **append @0.45** | `v7` 4.0% → 0.8723 | `v6` 6.5% → 0.8742 |

Both questions answer cleanly, and in opposite directions to the working
hypothesis:

- **Node budget is not the explanation.** `v6` matches `v1`'s budget to within
  0.26% of total nodes (6.5% vs 6.3% appended) and still loses 0.0044. Every
  append blend built on `24/5` sits below `v1` regardless of how many nodes it
  contributes, so `v4`'s deficit was never about over-appending.
- **`v1` is at the optimum of the budget axis, not past it.** `v7` appends
  *less* than `v1` (4.0%) and is the worst append blend of the four — dropping
  0.0063 and taking node recall down to 0.9574. Appending less is as harmful as
  appending more; 6.3% is a genuine interior optimum.
- **The appender's identity is the dominant term.** `23/6` beats `24/5` at
  matched budget (`v1` 0.8786 vs `v6` 0.8742) even though `24/5` is the stronger
  standalone detector (solo cv 0.7794 vs 0.7693). This is the fourth time in
  this file that standalone quality anti-predicts ensemble value.

The threshold step is also not symmetric between the two appenders: 0.40→0.45
cuts `23/6` from 6.3%→4.0% (−37%) but `24/5` only 9.1%→6.5% (−29%), because
`23/6`'s probabilities are compressed ~0.067 lower at the median, so the same
numeric step bites deeper into its distribution.


## Pre-edge-model screen (repo metric, NN-linked edges)

| node set | config | node_rec | score | edge_jac | ratio |
|---|---|---|---|---|---|
| `13/1` alone | @0.40 | 0.9617 | 0.7658 | 0.7738 | 0.110 |
| `24/5` alone | @0.45 | 0.9656 | 0.7699 | 0.7795 | 0.114 |
| `24/5` alone | @0.50 | 0.9468 | 0.7699 | 0.7740 | 0.037 |
| **`v1`** `13/1`+`23/6` | append `b0.45 a0.40 d7` | 0.9687 | 0.7738 | 0.7830 | 0.120 |
| **`v2`** `13/1`+`24/5` | consensus `r5 th0.40` | 0.9687 | 0.7764 | 0.7864 | 0.128 |
| **`v2`** `13/1`+`24/5` | consensus `r5 th0.45` | 0.9570 | 0.7767 | 0.7819 | 0.064 |
| **`v3`** `12/0`+`24/5` | consensus `r5 th0.45` | 0.9620 | **0.7805** | **0.7876** | 0.085 |
| **`v3`** `18/0`+`24/5` | consensus `r5 th0.45` | 0.9550 | **0.7805** | 0.7856 | 0.059 |
| **`v4`** `13/1`+`24/5` | append `b0.45 a0.40 d7` | **0.9737** | 0.7720 | 0.7841 | 0.163 |
| **`v5`** `14/4`+`24/5` | append `b0.45 a0.45 d7` | 0.9677 | 0.7741 | 0.7836 | 0.126 |

Not built, but measured — the best of each family in `append_sweep.tsv`
(21 bases × both directions × `b/a ∈ {0.40, 0.45}` × `d7`, 168 configs):

| base (+`24/5`) | config | node_rec | score | edge_jac | ratio |
|---|---|---|---|---|---|
| `15/1` ⚠ | `b0.45 a0.45` | 0.9664 | 0.7746 | 0.7833 | 0.112 |
| `18/0` | `b0.45 a0.45` | 0.9685 | 0.7743 | 0.7837 | 0.122 |
| `11/3` | `b0.40 a0.40` | 0.9813 | 0.7602 | 0.7789 | 0.246 |
| `22/10` | `b0.40 a0.40` | 0.9826 | 0.7038 | 0.7300 | **0.545** |


## The ensembles — how each node set was built

All built by `scripts/ensemble_detectors.py`, fusing a base detector's node set
with a second detector's, both read from their own `preds_r6_0.30` /
`preds_r6_0.40` folders. Two fusion recipes are in use:

**`--cfg <th_base> <th_add> <d_add_um>` (append)** — keep every BASE node with
`node_probs >= th_base`, then add every ADD node with `node_probs >= th_add`
that is further than `d_add_um` from every kept BASE node, per frame. ADD only
fills gaps BASE missed; it never overrides a BASE detection.

**`--consensus <radius_um> <th>` (consensus fusion)** — cluster both models'
detections within `radius_um` (greedy, strongest-first); each cluster becomes
one node at the confidence-weighted centroid, scored as the mean of the two
models' probabilities (0 filled in for whichever model didn't fire there);
keep clusters scoring `>= th`. Unlike append, a consensus node's position and
score are blended, not inherited from one model.

| node set | base | add | recipe | n_nodes | frac from add / found by both |
|---|---|---|---|---|---|
| `v1_2026-09-13_1_2026-09-23_6` | `2026-09-13/1` | `2026-09-23/6` | append, `th 0.45/0.40, d 7` | 5,476,701 | 6.3% from add |
| `v2_..._consensus_r5_th0.40` | `2026-09-13/1` | `2026-09-24/5` | consensus, `r5, th0.40` | 5,562,613 | ~100% found by both |
| `v2_..._consensus_r5_th0.45` | `2026-09-13/1` | `2026-09-24/5` | consensus, `r5, th0.45` | 5,266,548 | 100% found by both |
| `v3_2026-09-12_0_..._th0.45` | `2026-09-12/0` | `2026-09-24/5` | consensus, `r5, th0.45` | 5,387,654 | 100% found by both |
| `v3_2026-09-18_0_..._th0.45` | `2026-09-18/0` | `2026-09-24/5` | consensus, `r5, th0.45` | 5,253,505 | 100% found by both |
| `v4_..._append_b0.45_a0.40_d7` | `2026-09-13/1` | `2026-09-24/5` | append, `th 0.45/0.40, d 7` | 5,644,322 | 9.1% from add |
| `v5_..._append_b0.45_a0.45_d7` | `2026-09-14/4` | `2026-09-24/5` | append, `th 0.45/0.45, d 7` | 5,507,243 | 7.1% from add |
| `v6_2026-09-13_1_..._a0.45_d7` | `2026-09-13/1` | `2026-09-24/5` | append, `th 0.45/0.45, d 7` | 5,490,977 | 6.5% from add |
| `v7_2026-09-13_1_..._a0.45_d7` | `2026-09-13/1` | `2026-09-23/6` | append, `th 0.45/0.45, d 7` | 5,343,330 | 4.0% from add |

Every ensemble carries its provenance forward: `annot_probs`, `coords`,
`coords_f`, `node_probs`, `edges` (NN-linked at build time), and the
`embeddings_contrastive_2026-09-09_1` array the edge model's `embeddings_key`
expects — the node set and its embeddings stay inseparable through the fusion
step, same as any single-detector node set (`Summary.md` §[3]).