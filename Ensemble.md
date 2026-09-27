# Ensemble Summary

**Best ensemble edge model as of 2026-09-27: `cv 0.8802`**
(`v1`, `edge_run/2026-09-23/2`, `preds_0.45_pp_fall_tuned_ilp`)

## Ranked

| node set | edge run | best raw | **best pp** | macro adj | div | macro rec | 44b6 adj | **6bba adj** |
|---|---|---|---|---|---|---|---|---|
| `v1_2026-09-13_1_2026-09-23_6` | `23/2` | 0.8263 @0.45 | **0.8786** | **0.8645** | 0.1405 | 0.9656 | 0.8817 | 0.8473 |
| `v12_2026-09-25_2_2026-09-23_6_append_b0.55_a0.40_d7` | `26/0` | — | 0.8767 | 0.8633 | 0.1346 | 0.9608 | 0.8743 | **0.8523** |
| `v14_2026-09-26_1_2026-09-24_5_append_b0.45_a0.40_d7` | `27/2` | 0.8294 @0.45 | 0.8773 | 0.8633 | 0.1396 | 0.9681 | 0.8783 | **0.8484** |
| `v15_2026-09-26_1_2026-09-24_5_append_b0.45_a0.45_d7` | `27/1` | **0.8301** @0.45 | 0.8770 | 0.8629 | 0.1411 | 0.9616 | 0.8791 | 0.8467 |
| `v4` *(new contrastive)* | `27/0` | 0.8253 @0.45 | 0.8768 | 0.8634 | 0.1334 | 0.9697 | 0.8816 | 0.8453 |
| `v4_2026-09-13_1_2026-09-24_5_append_b0.45_a0.40_d7` | `24/4` | 0.8258 @0.45 | 0.8760 | 0.8631 | 0.1283 | **0.9698** | 0.8810 | 0.8453 |
| `v13_2026-09-25_4_2026-09-23_6_append_b0.65_a0.40_d7` | `26/1` | — | 0.8754 | 0.8629 | 0.1255 | 0.9603 | 0.8751 | 0.8507 |
| `v12` *(seed repeat)* | `25/1` | — | 0.8752 | 0.8640 | 0.1122 | 0.9612 | 0.8760 | 0.8520 |
| `v6_2026-09-13_1_2026-09-24_5_append_b0.45_a0.45_d7` | `24/8` | 0.8246 @0.45 | 0.8742 | 0.8624 | 0.1185 | 0.9634 | **0.8820** | 0.8427 |
| `v11_2026-09-25_2_2026-09-24_5_append_b0.50_a0.45_d7` | `25/2` | — | 0.8739 | 0.8626 | 0.1133 | 0.9650 | 0.8779 | 0.8472 |
| `v5_2026-09-14_4_2026-09-24_5_append_b0.45_a0.45_d7` | `24/5` | 0.8241 @0.45 | 0.8736 | 0.8620 | 0.1163 | 0.9629 | 0.8799 | 0.8441 |
| `v7_2026-09-13_1_2026-09-23_6_append_b0.45_a0.45_d7` | `24/9` | 0.8209 @0.45 | 0.8723 | 0.8602 | 0.1210 | 0.9574 | 0.8789 | 0.8415 |
| `v2_2026-09-13_1_2026-09-24_5_consensus_r5_th0.45` | `24/3` | 0.8225 @0.45 | 0.8720 | 0.8590 | 0.1305 | 0.9511 | 0.8804 | 0.8375 |
| `v9_2026-09-24_5_2026-09-13_1_append_b0.50_a0.45_d7` | `25/0` | — | 0.8716 | 0.8591 | 0.1252 | 0.9547 | 0.8771 | 0.8410 |
| `v8_2026-09-13_1_2026-09-24_5_append_b0.50_a0.45_d7` | `24/10` | — | 0.8698 | 0.8601 | 0.0969 | 0.9596 | 0.8738 | 0.8465 |
| `v2_2026-09-13_1_2026-09-24_5_consensus_r5_th0.40` | `24/2` | 0.8234 @0.45 | 0.8685 | 0.8579 | 0.1061 | 0.9640 | 0.8799 | 0.8359 |

**Edge-run noise is ~0.0015 of pp, and most of this table sits inside it.**
`edge_run/2026-09-25/1` and `2026-09-26/0` are the same config on the same node
set — their `args.json` differ only in `output_dir` — so the pair is a seed
repeat: **0.8752 vs 0.8767 pp, 0.8640 vs 0.8633 adj**. Everything from `v12`
down to `v5` spans 0.0031, i.e. two repeats' worth. Only `v1`'s lead over the
bottom of the table is comfortably outside it, and even `v1` vs `v12` (0.0019)
is barely one repeat. Treat the ordering as indicative and re-run before
believing any single gap.

**The `26/1` blends (`v14`, `v15`) are the first to beat the field on raw
detections.** They post 0.8294 and 0.8301 at `0.45` against `v1`'s 0.8263 — the
only rows above 0.827 — which is the `26/1` base doing what the pre-edge screen
said it would. After post-processing they land 2nd and 3rd at 0.8773 and 0.8770,
inside one repeat of `v1`'s 0.8786, so the raw-detection lead does not survive
into pp intact. `v14` also takes the best 6bba adj outside the `v12`/`v13`
family at 0.8484.

`27/0` is **not** a repeat: it is `v4`'s nodes through an edge model built on
`contrastive_2026-09-26_0` instead of `contrastive_2026-09-09_1`, everything
else in `args.json` identical. It moves pp by +0.0008 and `_pp_fall_tuned_ilp`
by −0.0001 — half a repeat's worth in one direction and nothing in the other,
so on this node set the new embeddings are not measurably better or worse. That
is a useful null: it says the contrastive model is not what is limiting these
blends.

`edge run` is under `logs/edge_run/2026-09-`. `n=71` for 44b6, `n=128` for 6bba.

**6bba is the wider axis and has stopped ordering the macro.** Across these 15
node sets, 44b6 adj spans 0.0082 and 6bba adj spans 0.0164 — twice the range,
so 6bba is where blends actually separate. But the two orderings now disagree
almost everywhere: of 16 rows only three land on the same rank (`v5`, `v9`,
`v2_th0.40`), and the top is inverted — `v12` takes 1st and 2nd on 6bba while
`v1` leads the macro from 5th, and both `v4` rows sit 4th/6th on macro but
9th/10th on 6bba. The tighter-base blends (`v12`, `v13`, `v8`, `v11`) are the
ones that climb on 6bba and fall on macro; they trade 44b6 adj for 6bba adj.
Read both columns, and note that 44b6's 0.0082 span is itself under three
edge-run repeats, so much of the disagreement is the narrow axis being noise.

All rows are `scores_preds_0.45_pp.json` — the shared default-knob config — so
they are comparable. Tuned-knob variants exist for three node sets and are
*not* in the table above because they are not all available; where they are,
they change the leader's margin but not the leader:

| node set | default pp | `_pp_fall_tuned` | `_pp_fall_tuned_ilp` |  LB  | 
|---|---|---|---|---| 
| `v1` | 0.8786 | 0.8763 | **0.8802** | | 
| `v4` | 0.8760 | 0.8746 | 0.8791 | 0.924 | 
| `v4` (`27/0`, new contrastive) | 0.8768 | 0.8753 | 0.8790 | | 
| `v14` (`27/2`) | 0.8773 | 0.8734 | **0.8776** | | 
| `v15` (`27/1`) | 0.8770 | 0.8752 | 0.8770 | | 
| `v12` (`26/0`) | 0.8767 | 0.8749 | 0.8774 | | 
| `v12` (`25/1`) | 0.8752 | 0.8725 | 0.8752 | | 

The tuned knobs alone always *lose* (−0.0018 to −0.0027); it is the ILP on top
that pays, and only then. Net against default pp: `v1` +0.0016, `v4` +0.0031,
`v12` +0.0007 and +0.0000 on its two seeds, `v4` +0.0022 on its new-contrastive
run — so the gain is real for `v1` and `v4` and indistinguishable from noise for
`v12`. `v2_th0.45` also has
`_pp_f0_tuned` rows (0.8671, 0.8706), both below its own default: the fold-0
tuning did not transfer, which is why `_fall_` is the variant to use.

Not yet scored: both `v3` consensus blends.

## Pre-edge-model screen (repo metric, NN-linked edges)

| node set | config | node_rec | score | edge_jac | ratio |
|---|---|---|---|---|---|
| **`v1`** `13/1`+`23/6` | append `b0.45 a0.40 d7` | 0.9687 | 0.7738 | 0.7830 | 0.120 |
| **`v4`** `13/1`+`24/5` | append `b0.45 a0.40 d7` | **0.9737** | 0.7720 | 0.7841 | 0.163 |
| **`v5`** `14/4`+`24/5` | append `b0.45 a0.45 d7` | 0.9677 | 0.7741 | 0.7836 | 0.126 |
| **`v6`** `13/1`+`24/5` | append `b0.45 a0.45 d7` | 0.9683 | 0.7735 | 0.7828 | 0.124 |
| **`v7`** `13/1`+`23/6` | append `b0.45 a0.45 d7` | 0.9618 | 0.7727 | 0.7795 | 0.088 |
| **`v8`** `13/1`+`24/5` | append `b0.50 a0.45 d7` | 0.9633 | 0.7693 | 0.7765 | 0.095 |
| **`v9`** `24/5`+`13/1` | append `b0.50 a0.45 d7` | 0.9601 | 0.7728 | 0.7795 | 0.083 |
| **`v11`** `25/2`+`24/5` | append `b0.50 a0.45 d7` | 0.9698 | 0.7767 | 0.7885 | 0.145 |
| **`v12`** `25/2`+`23/6` | append `b0.55 a0.40 d7` | 0.9655 | **0.7806** | **0.7891** | 0.100 |
| **`v13`** `25/4`+`23/6` | append `b0.65 a0.40 d7` | 0.9657 | 0.7812 | 0.7887 | 0.088 |

- v4 -> LB 0.924
