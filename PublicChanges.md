# Two public Biohub notebooks, side by side

> LB 0.908 -> LB 0.912

Both files are variants of the same public pipeline — the same materialized
inference repo, the same chain of detection → candidate edges → ILP lineage →
gap closing → safe division → `submission.csv`. Neither is a rewrite of the
other; they differ in which optional branches are switched on, and in what each
one adds around the shared core.

| | file | cells | source |
|---|---|---|---|
| **A** | `notebooks/912-biohub-cell-tracking-two-seeds-logit-blend.ipynb` | 13 | 133k chars |
| **B** | `notebooks/clean-approach-lightweight-local-cv-no-hack/working/public_nb.ipynb` | 20 | 150k chars |

A is `dual_seed_near_balanced_center_confirmed_synthetic_gap`; B is
`biohub_132_clean_short_track_rescue_lightcv_nohack`.

## 1. The same code, opposite switches

Both notebooks *contain* the short-track-rescue branch and the DeepCenter
branch. What differs is which one is enabled — so the two are pursuing
**different recall-recovery mechanisms** over an identical detector.

| setting | B (clean / no-hack) | A (two-seeds) |
|---|---|---|
| `ADAPTIVE_SHORT_TRACK_RESCUE` | **1**, plus 6 tuning params | **0** |
| `SHORT_TRACK_RESCUE_MIN_LEN` | 5 | — |
| `SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB` | 0.90 | — |
| `SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM` | 2.75 | — |
| `SHORT_TRACK_RESCUE_MAX_NODES_FRAC` / `_ABS` | 0.006 / 60 | — |
| `USE_DEEPCENTER_VETO` / `REQUIRE_DEEPCENTER_VETO` | **0 / 0**, checkpoint `''` | **1 / 1** |
| `DEEPCENTER_CHECKPOINT` | — | `biohub-deepcenter-unet3d-center-prior-v1` |
| `DEEPCENTER_GAP_VETO` / `_GAP_THRESHOLD` | 0 / 0.20 | **1** / 0.25 |
| `DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM` | 8.0 | 8.5 |
| `DEEPCENTER_EXPECTED_EPOCH` | 0 | 500 |
| `ILP_DISAPPEARANCE_WEIGHT` | **1.575** | 1.5 |
| `MOTION_RELINK_TIGHT_UM` / `_RELAXED_UM` | 6.0 / 9.5 | not set (defaults) |
| `RUN_OUTPUT_DIAGNOSTICS` | 1 | 0 |
| `RUN_VISUAL_EDA` | present, 0 | flag removed entirely |

Identical in both: `DET_THRESHOLD` 0.96875, `ILP_APPEARANCE_WEIGHT` 0.0,
`MOTION_RELINK_LEARNED_BONUS` 1.0, `OUTPUT_FILTER_SHORT_TRACKS` 1, and every
gap-close and safe-division parameter.

B rescues high-confidence five-node components that the minimum-length-six
filter removed. A instead leaves those dropped and gates *synthetic gap nodes*
through a second center-prior UNet, which it makes mandatory
(`REQUIRE_DEEPCENTER_VETO=1`) rather than best-effort.

## 2. A adds a dual-seed ensemble — the headline difference

A resolves a second, independently seeded TemporalUNet3D
(`biohub-temporal-unet3d-seed314159-v1`) by scanning `/kaggle/input` for an
`ARTIFACT_MANIFEST.json` whose `weight_sha256` matches a pinned constant, then
re-verifies the checksum of the extracted weight file before use.

It is wired in by **runtime source patching** — six string replacements against
the materialized repo, each asserting exactly one match before substituting. B
patches nothing. The blend happens at two separate points:

**Detection logits.** The secondary is aligned to the primary by mean and
standard deviation (scale ratio clamped to `[0.5, 2.0]`), then mixed at weight
**0.475** — very nearly balanced. The full TTA stack (flips, rot90, transpose,
anti-transpose) is repeated for the secondary when `det_tta` is on, so this
roughly doubles detection cost.

**Edge logits.** Four modes exist — `raw`, `calibrated`, `adaptive`,
`low_margin_consensus` — and A runs **`low_margin_consensus`** at weight 0.15
with `LOW_MARGIN_MAX` 0.35. Under that rule the secondary contributes only
where the primary is genuinely unsure *and* the two models agree:

```
uncertainty  = clamp((0.35 - primary_top2_margin) / 0.35, 0, 1)
local_weight = 0.15 * uncertainty
local_weight = where(same_parent_argmax, local_weight, 0)
```

so a confident primary edge is untouched, and a disagreement is resolved in the
primary's favour rather than averaged. The candidate threshold also drops from
the config default to **0.48** whenever the secondary is active
(`DUAL_SEED_EDGE_THRESHOLD`), widening what reaches the ILP.

## In short

B is the conservative single-model variant plus a full local-CV and audit
harness. A strips the validation, swaps short-track rescue for a mandatory
DeepCenter gap veto, and adds a second seeded model blended into both detection
and edge logits, with multi-GPU inference.

The functions unique to each are a fair summary of that split:

| only in A | only in B |
|---|---|
| `_find_secondary_artifact_root`, `_sha256_file` | `official_spec_evaluate`, `official_spec_per_sample`, `official_spec_summarise` |
| `_merge_prediction_shards`, `_wait_for_prediction_shards`, `_visible_cuda_tokens` | `_kuhn_maximum_matching`, `_jaccard`, `_node_match`, `_components` |
| `_prediction_dir_for_method`, `candidate_roots_for_slug` | `_audit_prediction_nodes`, `resolve_cv_stems`, `paired_train_stems` |

---

# 912 → Kimi v17: the missing link

> LB 0.912 -> 0.923


## 1. Bidirectional harmonic association fusion

The headline addition. Each frame pair gets a **second** `predict_edges` call
with source and target swapped — the same model asked to associate backwards in
time. The reverse matrix is transposed back, mean/std-aligned to the forward
logits (ratio clamped to `[0.5, 2.0]`), and the two are combined in *probability*
space by a weighted harmonic mean at weight 0.30:

```
harmonic = 1 / ((1 - w) / forward_prob + w / reverse_prob)
```

then renormalised and mapped back onto the forward logit scale, so the
downstream candidate threshold and the ILP see the range they were tuned for.

The harmonic mean is the whole point: it collapses toward whichever direction
assigns the *lower* probability, so a candidate survives only with mutual
forward/reverse support. An inline comment attributes the rule to a public CC0
notebook (`yusuketogashi/no-hack-biohub-cell-another-approch-3rd` v18) and
credits it with +0.002 there. It costs one extra edge forward pass per frame
pair.

## 2. Safe-division divergence gating, and the gates it lets open

Three knobs that do not exist in 912 at all:

| knob | v17 default |
|---|---|
| `SAFE_DIV_REQUIRE_MUTUAL_NN` | on |
| `SAFE_DIV_REQUIRE_DIVERGENCE` | on |
| `SAFE_DIV_DIVERGE_UM` | 2.25 |

A division candidate must now be the mutual nearest neighbour *and* its two
daughters must actually separate by at least 2.25 µm on the following frame.
This is the mechanism the v17 → v19 section above then tunes to 4.5 µm for
almost the entire +0.015.

Because that filter exists, the purely geometric gates are opened wide:

| setting | 912 | v17 |
|---|---|---|
| `SAFE_DIV_MAX_UM` | 4.66 | **12.0** |
| `SAFE_DIV_SISTER_MAX_UM` | 8.5 | **15.0** |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | 7.65 | **10.0** |
| `DEEPCENTER_SAFE_DIV_VETO` | 0 | **1** |

The comments state the reasoning directly: an earlier test at 7.0 µm "added
noise that pure geometry couldn't filter out; divergence + mutual-NN are active
now and should do that filtering instead". The DeepCenter safe-division veto
was switched on because "funnel telemetry showed safe-div accepting 81-100% of
its own candidates".

## 3. Smaller corrections

- **DeepCenter checkpoint fixed.** `checkpoint_last.pt` (epoch 500) →
  `best.pt`, with `DEEPCENTER_EXPECTED_EPOCH` 500 → 2 and the comment
  "best.pt is epoch 2, not checkpoint_last.pt's epoch 500". 912 was loading the
  last checkpoint rather than the best one.
- **Retention guard** on the dual-seed blend, `DUAL_SEED_MIN_CANDIDATE_RETENTION`
  0.90: if blending the secondary's detection logits drops a frame's candidate
  count below 90% of the primary's, that frame reverts to primary detection —
  logged per frame to JSONL and stdout.
- **`ILP_DIVISION_WEIGHT` pinned to 1.0** (912 left it at the default), noting
  that 0.3 / 1.0 / 2.0 / 3.0 all scored 0.915 on the leaderboard.
- **Configuration guard cell** asserting six numeric and two text env values,
  raising `RuntimeError` on drift.
- **More source patching**: 912 applies one 6-entry replacement list; v17 adds
  four further standalone `replace()` calls, each asserting a unique match and
  `compile()`-checking the patched source.

Unchanged: `DET_THRESHOLD` 0.96875, ILP appearance 0.0 / disappearance 1.5, the
whole gap-close and gap-density block, `OUTPUT_MIN_TRACK_LEN` 6,
`ADAPTIVE_SHORT_TRACK_RESCUE` 0, DeepCenter gap threshold 0.25 and span 8.5.

---

# v17 → v19 of the Kimi notebook: parameter tuning only

> LB 0.923 -> LB 0.938

A second comparison, between two revisions of the same descendant notebook. The
filenames carry the public leaderboard score.

| | file | cells | source |
|---|---|---|---|
| **v17** | `notebooks/923-kimi-notebook-v17.ipynb` | 11 | 189k chars |
| **v19** | `notebooks/938-kimi-notebook-v19.ipynb` | 11 | 192k chars |

**There are no code changes.** All 69 top-level functions are byte-identical,
none added and none removed, and only cells 1, 2, 3, 4 and 7 differ at all. The
entire +0.015 is configuration constants, plus inline comments recording what
was tried — which makes this notebook unusually informative, because the
comments carry the leaderboard score of nearly every value that was rejected.

## 1. What moved

| setting | v17 (0.923) | v19 (0.938) |
|---|---|---|
| **`SAFE_DIV_DIVERGE_UM`** | **2.25** | **4.5** |
| `SAFE_DIV_REQUIRE_MUTUAL_NN` | 1 | **0** |
| `DET_THRESHOLD` | 0.96875 | 0.95 |
| `OUTPUT_GAP2_RECOVERY` | 0 | 1 |
| `BIDIRECTIONAL_EDGE_WEIGHT` | 0.30 | 0.35 |
| `ILP_DIVISION_WEIGHT` | 1.0 | 4.0 |
| `MOTION_RELINK_LEARNED_BONUS` | 1.0 | 1.5 |
| `ADAPTIVE_SHORT_TRACK_RESCUE` | 0 | 1 |
| `SHORT_TRACK_RESCUE_MAX_NODES_ABS` | 180 | 120 |

The three integrity guards that pin the bidirectional weight were updated to
match (0.30 → 0.35), and `_guard_expected_min_retention` changed from a
hardcoded `0.9` to reading `DUAL_SEED_MIN_CANDIDATE_RETENTION` — a small
robustness fix, so the guard cannot drift from the setting it checks.

## 2. Nearly all of it is one parameter

The comments record a full leaderboard sweep of the division **divergence**
gate — the rule that a proposed division's grandchildren at `t+2` must be
further apart than its sisters are at `t+1`:

| `SAFE_DIV_DIVERGE_UM` | 1.5 | 2.25 | 3 | 3.7 | 4 | **4.5** | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| score | 0.919 | 0.923 | 0.928 | 0.933 | 0.935 | **0.936** | 0.931 | 0.924 | 0.915 | 0.910 |

A clean unimodal curve accounting for ~0.013 of the ~0.015 total. Everything
else is worth a thousandth or less: `OUTPUT_GAP2_RECOVERY` 0→1 took 0.937 →
0.938, and `BIDIRECTIONAL_EDGE_WEIGHT` 0.35 was "unchanged but slightly higher"
against 0.30, with 0.40 clearly worse at 0.934.

Raising the gate from 2.25 to 4.5 µm roughly doubles the separation a candidate
division must demonstrate before it is accepted. The mutual-nearest-neighbour
gate was switched **off** in the same revision — the comment notes it scored the
same either way at this radius, "but with lower thresholds it helps". The two
gates are substitutes: a strong divergence requirement makes the mutual-NN check
redundant.

## 3. Recorded and rejected

Useful negative results, all left at their v17 values:

| knob | tried | outcome |
|---|---|---|
| `SAFE_DIV_FRAME_FRAC_CAP` | 0.015 | didn't help |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | 0.0075 | didn't help |
| `GAP2_MAX_TOTAL_UM` | 13 | decreased |
| `GAP2_MAX_STEP_UM` | 5.5 | decreased |
| `GAP2_REQUIRE_CONTEXT` | 0 | decreased |
| `SECONDARY_EDGE_WEIGHT` | 0.30 | decreased |
| `DUAL_SEED_MIN_CANDIDATE_RETENTION` | 0.85 | decreased |
| `DEEPCENTER_GAP_THRESHOLD` | 0.20 | no change |
| `DEEPCENTER_SAFE_DIV_THRESHOLD` | 0.08 (with diverge 1.5) | 0.918, slight decrease |

## 4. Two caveats

**This is leaderboard tuning, not holdout tuning.** Ten values of
`SAFE_DIV_DIVERGE_UM` were read straight off the public leaderboard, and several
decisions ("scored same, but local validator shows slight increase") break ties
using the public score. The divergence curve is smooth enough that its peak is
probably real; the last few thousandths — `OUTPUT_GAP2_RECOVERY` and the
bidirectional weight — sit inside the range where public-LB noise and selection
bias dominate, and should be treated as unverified.

**The lever is division-repair geometry, not the model.** The largest single
gain in this entire diff is one threshold governing how strictly a proposed
division is validated. No retraining, no architecture change, no new artifact —
only how hard the candidate divisions are filtered. That is worth weighing
against our own pipeline, which currently emits no divisions at all
(`division_tp = 0`, `division_fn = 26`) and therefore forfeits the whole
`0.1 × division_jaccard` term of the score.

---

# v19 → 946 Harmonic Fusion: tuning moves off the leaderboard

The last step in the lineage, and the one that changes *method* rather than
values. `946-biohub-harmonic-fusion.ipynb` is again a strict superset of v19 —
all 80 of v19's top-level functions present unchanged, plus four new ones
(`pp_apply`, `pp_restore`, `score_validator_config`, `write_test_submission`).

| | file | cells | source |
|---|---|---|---|
| **v19** | `notebooks/938-kimi-notebook-v19.ipynb` | 11 | 192k chars |
| **946** | `notebooks/946-biohub-harmonic-fusion.ipynb` | 12 | 196k chars |

946's preset is `harmonic_v3_division_wide`, its score axis "public 0.939 base
+ holdout-selected post-process configuration".

## 1. An automated post-process sweep that rewrites the submission

This is the structural change, and it directly answers the caveat raised at the
end of the previous section. v19 writes `submission.csv` once and stops. 946:

1. Writes a **base** submission from the 0.939 configuration.
2. Snapshots 20 sweepable post-process constants (`PP_SWEEP_KEYS`: the safe-div
   gates, gap-close, motion-relink, DeepCenter thresholds, short-track rescue)
   and caches the *raw* prediction graphs and ground truth for the held-out
   validator videos.
3. Scores seven single-knob candidates — `gap45`, `tight55`, `relaxed9`,
   `bonus125`, `gap2step40`, `reuse28`, `dcgap035` — by mutating the globals and
   rebuilding the graph from the cached predictions. **No re-inference**, so the
   whole sweep is cheap.
4. Builds a **combination** of every candidate that individually cleared the
   bar, and scores that too.
5. Selects the winner only if it beats base by `PPSWEEP_SELECT_MARGIN` (0.001)
   on `proxy_score` *and* gives up no more than `PPSWEEP_MAX_ADJ_LOSS` (0.0005)
   of adjusted edge Jaccard.
6. If anything wins, re-runs `write_test_submission` under the selected config
   and restores the globals afterwards.

`proxy_score = weighted_adj_edge_jaccard + VALIDATOR_DIVISION_WEIGHT ×
division_jaccard`, and the held-out set doubles from two to **four videos per
embryo prefix** (`VALIDATOR_N_PER_TYPE` 2 → 4).

The significance is the selection mechanism, not the seven knobs. v19 tuned by
reading the public leaderboard ten times over; 946 tunes against a local
holdout, with an explicit improvement margin and a guard that forbids buying
divisions with edge accuracy. Everything the previous section flagged as
unverified is exactly what this apparatus exists to verify.

## 2. Two new mechanisms

**Edge-feature TTA** (`EDGE_FEATURE_TTA=1`). v19's test-time augmentation
averaged only the *detection* logits over the eight augmented views and threw
the augmented UNet feature maps away. 946 accumulates those as well and averages
them into `unet_out`, so the edge transformer reads TTA-averaged features rather
than single-pass ones. It ships with two guards: a shape check, and a **no-op
check** that raises if the averaged features come back bit-identical to the
single pass — written explicitly so the arm cannot "read as a false null".

**Sister-symmetry gate** (`SAFE_DIV_SISTER_SYMMETRY_TAU` 0.6, absent from v19).
A proposed division is rejected when its two daughters sit asymmetrically about
the parent:

```
|child_dist - parent_dist| / mean(child_dist, parent_dist) > 0.6  ->  reject
```

## 3. v19's division tuning is substantially rolled back

| setting | v19 (938) | 946 |
|---|---|---|
| **`SAFE_DIV_DIVERGE_UM`** | **4.5** | **2.25** |
| **`SAFE_DIV_REQUIRE_MUTUAL_NN`** | 0 | **1** |
| `SAFE_DIV_MAX_UM` | 12.0 | 9.0 |
| `SAFE_DIV_SISTER_MAX_UM` | 15.0 | 14.0 |
| `ILP_DIVISION_WEIGHT` | 4.0 | 1.2 |
| `DEEPCENTER_SAFE_DIV_THRESHOLD` | 0.12 | 0.25 |
| `DET_THRESHOLD` | 0.95 | 0.965 |
| `ILP_DISAPPEARANCE_WEIGHT` | 1.5 | 2 |
| `GAP_CLOSE_UM` | 5.8 | 5.0 |
| `MOTION_RELINK_LEARNED_BONUS` | 1.5 | 1.0 |
| `BIDIRECTIONAL_EDGE_WEIGHT` | 0.35 | 0.15 |
| `SECONDARY_DETECTION_WEIGHT` | 0.475 | **0.80** |
| `SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB` | 0.82 | 0.88 |
| `SHORT_TRACK_RESCUE_MAX_NODES_FRAC` | 0.018 | 0.012 |

Read the first two rows together with the section above. `SAFE_DIV_DIVERGE_UM`
returns to **2.25** — the exact v17 value, undoing the single change that the
v17 → v19 leaderboard sweep credited with ~0.013 of its ~0.015. The two are
substitutes rather than contradictions: 946 re-enables the mutual-nearest-
neighbour gate v19 switched off, adds the symmetry gate, and tightens the raw
distance gates, so the same filtering is done by three narrow rules instead of
one wide threshold. `ILP_DIVISION_WEIGHT` 4.0 → 1.2 and the raised DeepCenter
division veto pull the same way.

It is worth being blunt about what this implies: the divergence curve in the
previous section looked clean and unimodal, and the very next notebook walked
its peak back to where it started. That is what leaderboard-fitted optima tend
to do, and it is the strongest available argument for the holdout apparatus in
§1.

Two other shifts stand out. The bidirectional harmonic weight is cut by more
than half (0.35 → 0.15), and the secondary seed's *detection* weight rises to
**0.80** — the second model now dominates detection, where v19 had it close to
balanced at 0.475.

## 4. Style

All 13 `REVIEW:` comments are gone; 946 has none. The HTML banner is replaced by
a plain module docstring ("Record edition"). Both apply four runtime source
patches. 946 closes with the same pipeline manifest plus a stricter final
assertion block over `submission.csv` — contiguous ids, dataset-set match, edge
time consistency, single parent, out-degree ≤ 2, and per-dataset division
counts.
