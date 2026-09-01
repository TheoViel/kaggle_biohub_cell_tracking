# A calibrated link classifier — built, measured, and it does not pay

This tests the hypothesis left open in `postprocessing/README.md`: that the
pipeline's third step (decline the uncertain links) fails because the signals it
gates on rank links badly, and that a small supervised model over those signals
would fix it.

**The model works. The gate does not.** It is well calibrated and it ranks wrong
links three times better than chance, but at the threshold where gating pays it
removes almost exactly the links the existing margin cut already removes, so the
end-to-end score does not move. Once `pp_public` runs afterwards the difference
disappears entirely.

```bash
cd src
python edge_classifier/build.py --jobs 24     # features + labels for 71 stacks
python edge_classifier/train.py               # grouped 5-fold, calibration report
python edge_classifier/evaluate.py --jobs 24  # the gate vs the gates it replaces
python edge_classifier/final_eval.py --jobs 24  # full chains, incl. pp_public
```

## Results

Out of fold throughout — every stack is judged by a model that never saw it.
Grouped 5-fold by stack, all within the `44b6` embryo (the accepted small leak;
a clean split would hold `6bba` out).

**The model itself** — 20,086 labelled links over 71 stacks, 89.0% correct:

| | AUC | AP(wrong links) | Brier |
|---|---|---|---|
| logistic | **0.768** | **0.344** (base rate 0.110) | 0.085 |
| gradient boosting | 0.757 | 0.344 | 0.086 |

Calibration is genuinely good — the number it emits means what it says:

| predicted | n | mean p | actually correct |
|---|---|---|---|
| [0.00, 0.50) | 407 | 0.337 | 0.415 |
| [0.50, 0.70) | 841 | 0.624 | 0.609 |
| [0.70, 0.85) | 2,935 | 0.796 | 0.795 |
| [0.85, 0.93) | 6,631 | 0.898 | 0.894 |
| [0.93, 0.97) | 6,234 | 0.950 | 0.953 |
| [0.97, 1.01) | 3,038 | 0.983 | 0.985 |

**The gate**, against the ones it replaces, on identical prepared links. A gate
must remove **1.23 FP per TP** it costs to break even:

| gate | adjJ | ΔTP | ΔFP | FP per TP |
|---|---|---|---|---|
| no gate | 0.8043 | — | — | — |
| margin ≥ 1 µm | 0.8054 | −216 | −296 | 1.37 |
| margin ≥ 2 µm | 0.8008 | −455 | −466 | 1.02 |
| **classifier p ≥ 0.30** | **0.8065** | −40 | −110 | **2.75** |
| classifier p ≥ 0.45 | 0.8057 | −123 | −189 | 1.54 |
| classifier p ≥ 0.62 | 0.8022 | −368 | −396 | 1.08 |

The classifier is the only rule that clears break-even with room to spare, which
is the hypothesis confirmed. But look at the size of the move: it drops 40 true
positives and 110 false ones out of ~20,000 links.

**End to end**, which is what matters:

| chain | adjJ | tp | fp | fn |
|---|---|---|---|---|
| pipeline (margin ≥ 0, the tuned default) | 0.8065 | 17800 | 2057 | 2026 |
| classifier(0.30) | **0.8065** | 17835 | 2101 | 1991 |
| classifier(0.45) | 0.8057 | 17752 | 2022 | 2074 |
| pipeline → pp_public | 0.8197 | 18119 | 2078 | 1707 |
| classifier(0.30) → pp_public | **0.8197** | 18119 | 2077 | 1707 |

Identical to four decimals. Per stack: 7 better, 8 worse, median 0.0000. On the
ten hardest stacks it is 0 better, 1 worse. After `pp_public`: 0 better, 0 worse
— it re-links everything itself, so any gate placed before it is overwritten.

## Why it does not pay

Two reasons, both visible in the fit.

**It learned the margin.** Standardised log-odds, largest first:

```
margin_um       +1.218      tgt_nn_um       -0.157
frame_nodes     -0.395      src_nn_um       -0.126
vel_change_um   -0.198      track_len       +0.125
residual_um     +0.183      dist_over_flow  +0.119
```

`margin_um` is three times the next feature. The rest add ranking power in
aggregate (AUC 0.77 against ~0.70 for the margin alone) but not where the
decision boundary sits, so the links it declines are largely the links the
margin cut already declines.

**There is very little to gate.** 89% of scored links are already correct. At
the break-even threshold only ~1,200 of 20,086 links are even candidates for
removal, and they are 40-60% correct — so the best achievable trade is a few
hundred edges either way against a denominator of 20,000. The remaining error is
not "links we should have declined", it is **links we never proposed**: 2,026 FN
against 2,057 FP. Precision was never the binding constraint.

## The break-even threshold was wrong in the earlier note

`postprocessing/README.md` put it at 0.54, later 0.62. The correct value is
**0.45**:

```
J = TP / (n_gt + FP);  keeping a link gives one TP with probability p and one FP
with probability 1 - p, so keep when

    p / (n_gt + FP) > (1 - p) * TP / (n_gt + FP)^2   <=>   p > TP / (TP + n_gt + FP)
```

which at the operating point (TP 17875, FP 2211, n_gt 19826) is 0.448. The
earlier figure charged a wrong decision two false positives. A ground-truth link
we get wrong does produce two false-positive *edges* — the parent linked
elsewhere and the child linked from elsewhere — but each is a separate
keep-or-drop decision worth one FP, not two. No earlier result changes: the
gates were swept empirically, never set from the formula.

Empirically 0.30 beats 0.45, because inside [0.30, 0.45) the links come true
about half the time rather than the third the calibration implies — the one
place the model is meaningfully miscalibrated, and the bins are too coarse to
have caught it.

## What this rules out, and what it points at

The hypothesis is closed: a calibrated per-link confidence is achievable and
still does not move the score, so **step 3 is not where the remaining headroom
is**. What is left, in order of expected value:

- **Detection localisation.** Not the blanket problem an earlier note here
  claimed — across all 71 stacks the median error is 1.86 µm, near the 0.81 µm
  floor the detection grid imposes by rounding. But it separates kept links
  from lost ones sharply: worst-endpoint error is 2.07 µm on links we
  reproduce against **5.28 µm** on links we miss (`analysis_localisation.py`).
- **Recall, not precision.** 2,026 FN against 2,057 FP. The links we never
  proposed outnumber the ones we proposed wrongly. That is a candidate-
  generation and scoring problem — the model's top-12 does contain every
  ground-truth link — not a filtering one.

## Files

| file | what it is |
|---|---|
| `features.py` | 19 per-link features, and labels from the annotation |
| `prepare.py` | re-link + prune, shared by training and inference so they cannot drift |
| `build.py` | the training table over all 71 stacks |
| `train.py` | grouped k-fold, calibration report, saves fold models |
| `gate.py` | `classifier_postprocess` — the pipeline with step 3 swapped |
| `evaluate.py` | the gate against the gates it replaces, out of fold |
| `final_eval.py` | full chains including `pp_public` |

Artifacts land in `output/edge_classifier/`: `dataset.npz`, `model.joblib`
(fold models + the stack→fold map, so evaluation stays out of fold), `oof.npz`,
`evaluate.json`, `final_eval.json`.
