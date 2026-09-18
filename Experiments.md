# Experiments

Only runs with **both folds** trained are listed (single-fold runs are excluded).
Scores come from each run's best `scores_*.json` covering both embryos; `_pl` (pseudo-label) score files are excluded because that pass is in-sample.

`CV = adj_edge_jaccard + 0.1 x division_jaccard`, macro-averaged (`mean(embryo_44b6, embryo_6bba)`). Best value per column in **bold**.


## Edge post-processing tuned on fold 0 (`*_f0_tuned*`)

The post-processing knobs in these runs were swept on **44b6 only**, then
applied unchanged to both embryos. So the two halves of each row mean
different things:

* **44b6 is the tuning set.** Its numbers are in-sample for the PP stage.
* **6bba is held out** and is the only honest read of what the tuning is worth.

One row per run: the ILP variant, chosen on `settings.ilp` rather than on thefile name. Every run here now has a genuine pair, and the filter wins on theheld-out embryo in all eight -- by +0.0008 to +0.0053, mean +0.0032 -- so thechoice costs nothing in fairness. Ranked on the held-out embryo, not on themacro, for the reason above.

`CV = adj_edge_jaccard + 0.1 x division_jaccard`.

| exp folder | nodes_folder | score file | 44b6 adj *(in-sample)* | 44b6 div *(in-sample)* | 44b6 CV *(in-sample)* | **6bba adj** | **6bba div** | **6bba CV** | macro adj jacc | macro CV |
|---|---|---|---|---|---|---|---|---|---|---|
| edge_run/2026-09-16/0 | 2026-09-15/1/preds_r6_0.40/ | preds_0.40_pp_f0_tuned_ilp | 0.8833 | 0.1321 | 0.8965 | 0.8260 | **0.1846** | **0.8445** | 0.8546 | 0.8705 |
| edge_run/2026-09-14/2 | 2026-09-14/0/preds_r7_0.40/ | preds_0.45_pp_f0_tuned_ilp | 0.8712 | 0.2500 | 0.8962 | **0.8272** | 0.1174 | 0.8389 | 0.8492 | 0.8676 |
| edge_run/2026-09-14/1 | 2026-09-13/2/preds_r6_0.40/ | preds_0.45_pp_f0_tuned_ilp | 0.8844 | 0.1951 | 0.9039 | 0.8251 | 0.1104 | 0.8361 | 0.8547 | 0.8700 |
| edge_run/2026-09-14/5 | 2026-09-13/1/preds_r6_0.40/ | preds_0.45_pp_f0_tuned_ilp | 0.8885 | 0.2093 | 0.9094 | 0.8251 | 0.0942 | 0.8346 | 0.8568 | 0.8720 |
| edge_run/2026-09-13/0 | 2026-09-13/1/preds_r6_0.40/ | preds_0.45_pp_f0_tuned_ilp | **0.8885** | 0.2162 | 0.9102 | 0.8265 | 0.0769 | 0.8342 | **0.8575** | **0.8722** |
| edge_run/2026-09-16/1 | 2026-09-15/0/preds_r6_0.40/ | preds_0.40_pp_f0_tuned_ilp | 0.8862 | **0.2619** | **0.9124** | 0.8196 | 0.1105 | 0.8307 | 0.8529 | 0.8715 |
| edge_run/2026-09-13/1 | 2026-09-13/1/preds_r7_0.40/ | preds_0.45_pp_f0_tuned_ilp | 0.8768 | 0.2424 | 0.9010 | 0.8238 | 0.0508 | 0.8289 | 0.8503 | 0.8650 |
| edge_run/2026-09-15/3 | 2026-09-14/7/preds_r6_0.40/ | preds_0.45_pp_f0_tuned_ilp | 0.8865 | 0.1562 | 0.9021 | 0.8100 | 0.1013 | 0.8201 | 0.8482 | 0.8611 |

Across these 8 runs the tuning embryo scores **+0.0602** on adj jacc and
**+0.1021** on division jacc against the held-out one. The adj gap is largely the
embryos differing in difficulty -- 44b6 leads 6bba by a similar margin everywhere in
this file -- but the division gap is not: it is more than twice the adj gap on a term
weighted 0.1, which is what tuning a division stage on 26 events and applying it to 125
looks like. The macro columns inherit that, so read them as half in-sample.

## Edge experiments, one shared post-processing config

All 12 rows ran `main_edge_postprocess.py --ilp --name default`, i.e.
the current `DEFAULTS` block, over each run's own candidates. The settings blocks
of every score file below are identical, so the differences are the models and
the node sets and nothing else.

That is the point of the table. The legacy one at the bottom of this file ranks
runs whose post-processing was tuned per checkpoint, to varying degrees and on
varying amounts of the data, so a run could place well there by having been
swept harder rather than by being better.

The `+cls` columns add the division classifier on top of that same chain:
`division_cls/apply_fork_gate.py` at tau 0.5, out-of-fold models from
`logs/division/2026-09-16/0`, nothing else changed. Run on the top five only.
All five gain, +0.0032 to +0.0065, and `adj` is unmoved to +0.0005 in every one
— the gate deletes second-child edges and nothing else, so the whole difference
is the division term.

**It reorders them.** `2026-09-16/1` goes 5th to 2nd and `2026-09-16/0` 2nd to
4th: both propose ~50 true divisions, but the classifier keeps 80% of the
first's and 70% of the second's. What the `+cls` column ranks is how well a
run's proposals survive gating, which the plain CV does not predict — so a
model chosen on the columns to the left is not necessarily the one to ship with
the classifier enabled.

| exp folder | nodes_folder | score file | 44b6 node rec | 44b6 jacc | 44b6 adj jacc | 44b6 div jacc | 6bba node rec | 6bba jacc | 6bba adj jacc | 6bba div jacc | **macro node rec** | **macro jacc** | **macro adj jacc** | **macro div jacc** | **macro CV** | **+cls adj jacc** | **+cls div jacc** | **+cls CV** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge_run/2026-09-13/0 | 2026-09-13/1/preds_r6_0.40/ | preds_0.45_pp_default | 0.9836 | 0.8990 | **0.8885** | 0.1667 | 0.9286 | 0.8274 | 0.8262 | 0.1579 | 0.9561 | 0.8632 | **0.8574** | **0.1623** | **0.8736** | 0.8576 | 0.2150 | **0.8791** |
| edge_run/2026-09-16/0 | 2026-09-15/1/preds_r6_0.40/ | preds_0.40_pp_default | 0.9807 | 0.8924 | 0.8831 | 0.1148 | 0.9247 | 0.8261 | 0.8266 | **0.1991** | 0.9527 | 0.8593 | 0.8549 | 0.1569 | 0.8706 | 0.8550 | 0.1876 | 0.8738 |
| edge_run/2026-09-14/5 | 2026-09-13/1/preds_r6_0.40/ | preds_0.45_pp_default | 0.9840 | 0.8986 | 0.8881 | 0.1290 | 0.9295 | 0.8266 | 0.8251 | 0.1435 | 0.9568 | 0.8626 | 0.8566 | 0.1363 | 0.8703 | 0.8568 | 0.1841 | 0.8752 |
| edge_run/2026-09-14/1 | 2026-09-13/2/preds_r6_0.40/ | preds_0.45_pp_default | 0.9837 | 0.8970 | 0.8861 | 0.1562 | 0.9267 | 0.8245 | 0.8244 | 0.1333 | 0.9552 | 0.8607 | 0.8553 | 0.1448 | 0.8697 | 0.8556 | 0.1779 | 0.8734 |
| edge_run/2026-09-16/1 | 2026-09-15/0/preds_r6_0.40/ | preds_0.40_pp_default | 0.9827 | 0.8960 | 0.8850 | **0.1695** | 0.9323 | 0.8245 | 0.8227 | 0.1434 | 0.9575 | 0.8603 | 0.8539 | 0.1564 | 0.8695 | 0.8544 | 0.2161 | 0.8760 |
| edge_run/2026-09-13/1 | 2026-09-13/1/preds_r7_0.40/ | preds_0.45_pp_default | 0.9822 | 0.8853 | 0.8785 | 0.1311 | 0.9271 | **0.8330** | **0.8338** | 0.1266 | 0.9547 | 0.8591 | 0.8561 | 0.1289 | 0.8690 | — | — | — |
| edge_run/2026-09-14/2 | 2026-09-14/0/preds_r7_0.40/ | preds_0.45_pp_default | 0.9759 | 0.8796 | 0.8728 | 0.1429 | 0.9341 | 0.8311 | 0.8314 | 0.1597 | 0.9550 | 0.8553 | 0.8521 | 0.1513 | 0.8672 | — | — | — |
| edge_run/2026-09-15/0 | 2026-09-14/4/preds_r6_0.40/ | preds_0.45_pp_default | 0.9825 | 0.8963 | 0.8862 | 0.1273 | 0.9230 | 0.8231 | 0.8227 | 0.1250 | 0.9527 | 0.8597 | 0.8545 | 0.1261 | 0.8671 | — | — | — |
| edge_run/2026-09-11/2 | 2026-09-11/1/preds_r6_0.40/ | preds_0.45_pp_default | 0.9845 | 0.8978 | 0.8832 | 0.0822 | 0.9310 | 0.8292 | 0.8268 | 0.1538 | 0.9577 | **0.8635** | 0.8550 | 0.1180 | 0.8668 | — | — | — |
| edge_run/2026-09-15/2 | 2026-09-15/0/preds_r7_0.40/ | preds_0.45_pp_default | 0.9781 | 0.8814 | 0.8744 | 0.1077 | 0.9284 | 0.8308 | 0.8312 | 0.1406 | 0.9532 | 0.8561 | 0.8528 | 0.1241 | 0.8652 | — | — | — |
| edge_run/2026-09-18/0 | 2026-09-16/0/preds_r6_0.40/ | preds_0.45_pp_default | 0.9819 | 0.8920 | 0.8813 | 0.0972 | 0.9269 | 0.8243 | 0.8244 | 0.1191 | 0.9544 | 0.8582 | 0.8528 | 0.1082 | 0.8637 | — | — | — |
| edge_run/2026-09-13/5 | 2026-09-12/0/preds_r6_0.40/ | preds_0.40_pp_default | **0.9903** | **0.9010** | 0.8823 | 0.1163 | **0.9360** | 0.8243 | 0.8196 | 0.1372 | **0.9631** | 0.8626 | 0.8509 | 0.1267 | 0.8636 | — | — | — |

## Tracking experiments (`logs/<date>/<id>`)

| exp folder | radius / threshold | changes vs reference | 44b6 node rec | 44b6 jacc | 44b6 adj jacc | 6bba node rec | 6bba jacc | 6bba adj jacc | **macro node rec** | **macro jacc** | **macro adj jacc** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-15/1 | r=6, th=0.40 | `unet_arch`: temporal → temporal_v3 | 0.9847 | 0.8282 | 0.8173 | 0.9295 | 0.7438 | 0.7414 | 0.9571 | **0.7860** | **0.7794** |
|  | r=7, th=0.40 |  | 0.9816 | 0.8096 | 0.8024 | 0.9280 | **0.7531** | **0.7527** | 0.9548 | 0.7813 | 0.7775 |
| 2026-09-14/0 | r=6, th=0.40 | `pretraining.synthetic_proportion`: 0.5 → 0.0 | 0.9875 | 0.8248 | 0.8132 | 0.9405 | 0.7357 | 0.7309 | 0.9640 | 0.7803 | 0.7721 |
|  | r=7, th=0.40 |  | 0.9849 | 0.8105 | 0.8023 | 0.9393 | 0.7494 | 0.7465 | 0.9621 | 0.7800 | 0.7744 |
| 2026-09-13/2 | r=6, th=0.40 | *repro reference* | 0.9884 | 0.8271 | 0.8147 | 0.9328 | 0.7365 | 0.7331 | 0.9606 | 0.7818 | 0.7739 |
|  | r=7, th=0.40 |  | 0.9854 | 0.8078 | 0.7991 | 0.9316 | 0.7451 | 0.7436 | 0.9585 | 0.7765 | 0.7714 |
| 2026-09-13/1 | r=6, th=0.40 | *(reference)* | 0.9886 | 0.8273 | 0.8155 | 0.9348 | 0.7360 | 0.7316 | 0.9617 | 0.7817 | 0.7735 |
|  | r=7, th=0.40 |  | 0.9861 | 0.8071 | 0.7989 | 0.9335 | 0.7462 | 0.7438 | 0.9598 | 0.7767 | 0.7713 |
| 2026-09-14/4 | r=6, th=0.40 | `pretraining.crop_size`: [32, 128, 128] → [48, 192, 192]<br>`pretraining.iterations`: 5000 → 2000<br>`pretraining.weights`: None → ../logs/2026-09-11/3/fold_*/ckpt_pretrain.pt<br>`synthetic_pretraining.enabled`: True → False<br>`unet_layers`: [64, 128, 256] → [32, 64, 128] | 0.9851 | 0.8263 | 0.8149 | 0.9297 | 0.7345 | 0.7307 | 0.9574 | 0.7804 | 0.7728 |
|  | r=7, th=0.40 |  | 0.9831 | 0.8076 | 0.7996 | 0.9286 | 0.7433 | 0.7412 | 0.9559 | 0.7754 | 0.7704 |
| 2026-09-13/0 | r=6, th=0.40 | `weak_positives.use_focus_3d`: False → True | 0.9926 | 0.8279 | 0.8116 | 0.9380 | 0.7379 | 0.7306 | 0.9653 | 0.7829 | 0.7711 |
|  | r=7, th=0.40 |  | 0.9905 | 0.8139 | 0.8017 | 0.9370 | 0.7478 | 0.7425 | 0.9637 | 0.7808 | 0.7721 |
| 2026-09-15/0 | r=6, th=0.40 | `unet_arch`: temporal → temporal_v2<br>`unet_layers`: [64, 128, 256] → [32, 64, 128] | 0.9872 | 0.8201 | 0.8076 | 0.9370 | 0.7352 | 0.7302 | 0.9621 | 0.7776 | 0.7689 |
|  | r=7, th=0.40 |  | 0.9848 | 0.8074 | 0.7988 | 0.9355 | 0.7480 | 0.7453 | 0.9602 | 0.7777 | 0.7720 |
| 2026-09-17/6 | r=6, th=0.40 | `fullfit`: True → False<br>`optimizer_config.lr`: 0.001 → 0.0001<br>`optimizer_config.lr_transformer`: 0.001 → 0.0001<br>`optimizer_config.lr_unet`: 0.001 → 0.0001<br>`optimizer_config.warmup_prop`: 0.0 → 0.5<br>`pretraining.batch_size`: 16 → 8<br>`pretraining.crop_size`: [32, 128, 128] → [48, 192, 192]<br>`pretraining.iterations`: 5000 → 2000<br>`pretraining.weights`: None → ../logs/2026-09-15/0/fold_*/ckpt_pretrain.pt<br>`synthetic_pretraining.enabled`: True → False<br>`unet_arch`: temporal → temporal_v2<br>`unet_layers`: [64, 128, 256] → [32, 64, 128] | 0.9877 | 0.8241 | 0.8120 | 0.9325 | 0.7341 | 0.7299 | 0.9601 | 0.7791 | 0.7709 |
|  | r=7, th=0.40 |  | 0.9849 | 0.8092 | 0.8010 | 0.9309 | 0.7444 | 0.7421 | 0.9579 | 0.7768 | 0.7716 |
| 2026-09-14/7 | r=6, th=0.40 | `unet_arch`: temporal → temporal_v3<br>`unet_layers`: [64, 128, 256] → [32, 64, 128] | 0.9863 | 0.8231 | 0.8104 | 0.9289 | 0.7337 | 0.7304 | 0.9576 | 0.7784 | 0.7704 |
|  | r=7, th=0.40 |  | 0.9835 | 0.8095 | 0.8006 | 0.9276 | 0.7419 | 0.7406 | 0.9555 | 0.7757 | 0.7706 |
| 2026-09-16/0 | r=6, th=0.40 | `fullfit`: True → False<br>`unet_arch`: temporal → temporal_v4<br>`unet_blocks_per_stage`: [3, 3, 3] → [2, 2, 3]<br>`unet_layers`: [64, 128, 256] → [32, 64, 128] | 0.9863 | 0.8207 | 0.8081 | 0.9331 | 0.7361 | 0.7329 | 0.9597 | 0.7784 | 0.7705 |
|  | r=7, th=0.40 |  | 0.9830 | 0.8009 | 0.7920 | 0.9317 | 0.7453 | 0.7440 | 0.9574 | 0.7731 | 0.7680 |
| 2026-09-11/1 | r=6, th=0.40 | `unet_layers`: [64, 128, 256] → [32, 64, 128]<br>`weak_positives.use_focus_3d`: False → True | 0.9906 | 0.8267 | 0.8104 | 0.9379 | 0.7353 | 0.7292 | 0.9642 | 0.7810 | 0.7698 |
|  | r=7, th=0.40 |  | 0.9883 | 0.8073 | 0.7954 | 0.9367 | 0.7462 | 0.7423 | 0.9625 | 0.7768 | 0.7688 |
| 2026-09-17/4 | r=6, th=0.40 | `fullfit`: True → False<br>`optimizer_config.lr`: 0.001 → 0.0001<br>`optimizer_config.lr_transformer`: 0.001 → 0.0001<br>`optimizer_config.lr_unet`: 0.001 → 0.0001<br>`optimizer_config.warmup_prop`: 0.0 → 0.5<br>`pretraining.aug_strength`: 3 → 1<br>`pretraining.batch_size`: 16 → 4<br>`pretraining.crop_size`: [32, 128, 128] → None<br>`pretraining.iterations`: 5000 → 1000<br>`pretraining.synthetic_proportion`: 0.5 → 0.0<br>`pretraining.weights`: None → ../logs/2026-09-13/1/fold_*/ckpt_pretrain.pt<br>`synthetic_pretraining.enabled`: True → False | 0.9883 | 0.8313 | 0.8190 | 0.9533 | 0.7256 | 0.7172 | 0.9708 | 0.7785 | 0.7681 |
|  | r=7, th=0.40 |  | 0.9851 | 0.8122 | 0.8034 | 0.9521 | 0.7367 | 0.7305 | 0.9686 | 0.7745 | 0.7670 |
| 2026-09-12/0 | r=6, th=0.40 | `offset_head`: False → True<br>`pretraining.iterations`: 5000 → 10000<br>`pretraining.synthetic_dir`: ../input/synthetic/v4/ → ../input/synthetic/v5/<br>`unet_layers`: [64, 128, 256] → [32, 64, 128]<br>`weak_positives.use_focus_3d`: False → True | 0.9936 | **0.8359** | 0.8155 | 0.9447 | 0.7250 | 0.7164 | 0.9692 | 0.7804 | 0.7660 |
|  | r=7, th=0.40 |  | 0.9923 | 0.8166 | 0.8009 | 0.9437 | 0.7387 | 0.7322 | 0.9680 | 0.7777 | 0.7665 |
| 2026-09-12/1 | r=6, th=0.40 | `pretraining.iterations`: 5000 → 7500<br>`pretraining.synthetic_dir`: ../input/synthetic/v4/ → ../input/synthetic/v5/<br>`unet_layers`: [64, 128, 256] → [32, 64, 128]<br>`weak_positives.use_focus_3d`: False → True | 0.9922 | 0.8283 | 0.8099 | 0.9339 | 0.7306 | 0.7231 | 0.9631 | 0.7794 | 0.7665 |
|  | r=7, th=0.40 |  | 0.9904 | 0.8131 | 0.7990 | 0.9329 | 0.7384 | 0.7329 | 0.9617 | 0.7757 | 0.7660 |
| 2026-09-11/3 | r=6, th=0.40 | `pretraining.iterations`: 5000 → 10000<br>`synthetic_pretraining.synthetic_dir`: ../input/synthetic/v5/ → ../input/synthetic/v4/<br>`unet_layers`: [64, 128, 256] → [32, 64, 128]<br>`weak_positives.use_focus_3d`: False → True | **0.9957** | 0.8337 | 0.8123 | 0.9382 | 0.7262 | 0.7188 | 0.9669 | 0.7800 | 0.7656 |
|  | r=7, th=0.40 |  | 0.9942 | 0.8151 | 0.7984 | 0.9369 | 0.7360 | 0.7306 | 0.9656 | 0.7756 | 0.7645 |
| 2026-09-17/3 | r=6, th=0.40 | `fullfit`: True → False<br>`pretraining.aug_strength`: 3 → 1<br>`pretraining.batch_size`: 16 → 4<br>`pretraining.crop_size`: [32, 128, 128] → None<br>`pretraining.iterations`: 5000 → 10000<br>`pretraining.weights`: None → ../logs/2026-09-13/1/fold_*/ckpt_pretrain.pt<br>`synthetic_pretraining.enabled`: True → False | 0.9901 | 0.8295 | 0.8141 | **0.9538** | 0.7143 | 0.7057 | **0.9719** | 0.7719 | 0.7599 |
|  | r=7, th=0.40 |  | 0.9879 | 0.8146 | 0.8026 | 0.9527 | 0.7262 | 0.7194 | 0.9703 | 0.7704 | 0.7610 |
| 2026-09-17/0 | r=6, th=0.40 | `fullfit`: True → False<br>`pretraining.aug_strength`: 3 → 1<br>`pretraining.iterations`: 5000 → 10000<br>`pretraining.weights`: None → ../logs/2026-09-13/1/fold_*/ckpt_pretrain.pt<br>`synthetic_pretraining.enabled`: True → False | 0.9890 | 0.8341 | **0.8215** | 0.9402 | 0.6868 | 0.6798 | 0.9646 | 0.7605 | 0.7506 |
|  | r=7, th=0.40 |  | 0.9865 | 0.8161 | 0.8070 | 0.9390 | 0.6974 | 0.6923 | 0.9628 | 0.7568 | 0.7497 |
| 2026-09-14/5 | r=6, th=0.40 | `pretraining.iterations`: 5000 → 2000<br>`unet_arch`: temporal → temporal_v2<br>`unet_layers`: [64, 128, 256] → [32, 64, 128] | 0.9783 | 0.7986 | 0.7867 | 0.9443 | 0.7073 | 0.6976 | 0.9613 | 0.7530 | 0.7421 |
|  | r=7, th=0.40 |  | 0.9743 | 0.7823 | 0.7745 | 0.9427 | 0.7207 | 0.7134 | 0.9585 | 0.7515 | 0.7440 |

## Legacy: edge experiments with per-run tuned post-processing

Kept for the record, superseded by the shared-config table above. Every row
here was scored under whatever post-processing that run happened to be tuned
with -- different knobs, tuned to different depths, some on one fold and some
on both -- so the ranking mixes "better model" with "swept harder". The
`PP tuned` column marks the worst offenders but does not quantify them. Read
it for what a run *can* reach, not for which run is best.

| exp folder            | nodes_folder                | score file          | ILP | PP tuned | 44b6 node rec | 44b6 jacc | 44b6 adj jacc | 44b6 div jacc | 6bba node rec | 6bba jacc | 6bba adj jacc | 6bba div jacc | **macro node rec** | **macro jacc** | **macro adj jacc** | **macro div jacc** | **macro CV** |
|-----------------------|-----------------------------|---------------------|-----|----|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge_run/2026-09-16/1 | 2026-09-15/0/preds_r6_0.40/ | preds_0.40_pp2 | ✅ | ✅ | 0.9806 | 0.8954 | 0.8856 | 0.2143 | 0.9306 | 0.8257 | 0.8248 | 0.1348 | 0.9556 | 0.8605 | 0.8552 | **0.1746** | **0.8726** |
| edge_run/2026-09-13/0 | 2026-09-13/1/preds_r6_0.40/ | preds_0.45_pp_ilp | ✅ | ✅ | 0.9833 | 0.8989 | **0.8885** | 0.1250 | 0.9284 | 0.8276 | 0.8264 | 0.1647 | 0.9559 | **0.8633** | **0.8575** | 0.1449 | 0.8719 |
| edge_run/2026-09-13/1 | 2026-09-13/1/preds_r7_0.40/ | preds_0.45_pp4 | ✅ | ✅ | 0.9831 | 0.8862 | 0.8788 | 0.1778 | 0.9284 | **0.8333** | **0.8336** | 0.1349 | 0.9557 | 0.8597 | 0.8562 | 0.1563 | 0.8718 |
| edge_run/2026-09-14/2 | 2026-09-14/0/preds_r7_0.40/ | preds_0.45_pp3 | ✅ | ✅ | 0.9722 | 0.8793 | 0.8733 | **0.2162** | 0.9306 | 0.8315 | 0.8329 | 0.1330 | 0.9514 | 0.8554 | 0.8531 | **0.1746** | 0.8706 |
| edge_run/2026-09-11/2 | 2026-09-11/1/preds_r6_0.40/ | preds_0.45_pp4 | ✅ | ✅ | 0.9796 | 0.8972 | 0.8853 | 0.0698 | 0.9223 | 0.8286 | 0.8292 | 0.1530 | 0.9509 | 0.8629 | 0.8572 | 0.1114 | 0.8684 |
| edge_run/2026-09-14/1 | 2026-09-13/2/preds_r6_0.40/ | preds_0.45_pp_ilp | ✅ |  | 0.9817 | 0.8931 | 0.8833 | 0.1429 | 0.9244 | 0.8246 | 0.8253 | 0.1286 | 0.9530 | 0.8589 | 0.8543 | 0.1357 | 0.8679 |
| edge_run/2026-09-13/5 | 2026-09-12/0/preds_r6_0.40/ | preds_0.40_pp3 | ✅ | ✅ | 0.9853 | **0.9010** | 0.8856 | 0.0930 | 0.9223 | 0.8223 | 0.8217 | **0.1910** | 0.9538 | 0.8616 | 0.8536 | 0.1420 | 0.8678 |
| edge_run/2026-09-15/2 | 2026-09-15/0/preds_r7_0.40/ | preds_0.45_pp_ilp | ✅ |  | 0.9765 | 0.8836 | 0.8775 | 0.1321 | 0.9264 | 0.8309 | 0.8321 | 0.1262 | 0.9514 | 0.8572 | 0.8548 | 0.1291 | 0.8677 |
| edge_run/2026-09-15/0 | 2026-09-14/4/preds_r6_0.40/ | pp_ilp_tuned | ✅ | ✅ | 0.9822 | 0.8966 | 0.8866 | 0.0938 | 0.9228 | 0.8236 | 0.8232 | 0.1605 | 0.9525 | 0.8601 | 0.8549 | 0.1271 | 0.8676 |
| edge_run/2026-09-16/0 | 2026-09-15/1/preds_r6_0.40/ | preds_0.40_pp_ilp | ✅ |  | 0.9805 | 0.8926 | 0.8833 | 0.0833 | 0.9245 | 0.8263 | 0.8269 | 0.1561 | 0.9525 | 0.8594 | 0.8551 | 0.1197 | 0.8670 |
| edge_run/2026-09-15/3 | 2026-09-14/7/preds_r6_0.40/ | preds_0.45_pp |  |  | 0.9846 | 0.8964 | 0.8836 | 0.1053 | 0.9234 | 0.8139 | 0.8132 | 0.1579 | 0.9540 | 0.8551 | 0.8484 | 0.1316 | 0.8615 |
| edge_run/2026-09-13/3 | 2026-09-13/0/preds_r6_0.40/ | preds_0.45_pp |  |  | 0.9884 | 0.8920 | 0.8772 | 0.1053 | 0.9289 | 0.8247 | 0.8216 | 0.1373 | 0.9586 | 0.8584 | 0.8494 | 0.1213 | 0.8615 |
| edge_run/2026-09-11/0 | 2026-09-11/1/preds_r7_0.40/ | preds_0.45_pp_tuned |  |  | 0.9831 | 0.8810 | 0.8702 | 0.1000 | 0.9288 | 0.8297 | 0.8296 | 0.1295 | 0.9560 | 0.8554 | 0.8499 | 0.1147 | 0.8614 |
| edge_run/2026-09-13/2 | 2026-09-13/0/preds_r7_0.40/ | preds_0.45_pp |  |  | 0.9870 | 0.8807 | 0.8697 | 0.0600 | 0.9282 | 0.8310 | 0.8296 | 0.1735 | 0.9576 | 0.8558 | 0.8496 | 0.1167 | 0.8613 |
| edge_run/2026-09-14/2 | 2026-09-14/0/preds_r7_0.40/ | preds_0.45_pp |  |  | 0.9803 | 0.8764 | 0.8691 | 0.1071 | 0.9328 | 0.8268 | 0.8273 | 0.1538 | 0.9565 | 0.8516 | 0.8482 | 0.1305 | 0.8613 |
| edge_run/2026-09-15/1 | 2026-09-14/7/preds_r7_0.40/ | preds_0.45_pp_best |  |  | 0.9816 | 0.8832 | 0.8743 | 0.0976 | 0.9223 | 0.8224 | 0.8237 | 0.1200 | 0.9520 | 0.8528 | 0.8490 | 0.1088 | 0.8599 |
| edge_run/2026-09-14/3 | 2026-09-14/0/preds_r6_0.40/ | preds_0.45_pp |  |  | 0.9847 | 0.8885 | 0.8780 | 0.0938 | 0.9329 | 0.8184 | 0.8173 | 0.1502 | 0.9588 | 0.8534 | 0.8476 | 0.1220 | 0.8598 |
| edge_run/2026-09-13/5 | 2026-09-12/0/preds_r6_0.40/ | preds_0.40_pp |  |  | 0.9896 | 0.8919 | 0.8736 | 0.0854 | 0.9336 | 0.8195 | 0.8155 | 0.1379 | 0.9616 | 0.8557 | 0.8446 | 0.1116 | 0.8557 |
| edge_run/2026-09-13/4 | 2026-09-12/1/preds_r6_0.40/ | preds_0.35_pp |  |  | 0.9876 | 0.8913 | 0.8744 | 0.1186 | 0.9162 | 0.8129 | 0.8100 | 0.1330 | 0.9519 | 0.8521 | 0.8422 | 0.1258 | 0.8548 |
| edge_run/2026-09-16/2 | 2026-09-15/0/preds_r6_0.40/ | preds_0.45_pp |  |  | 0.9845 | 0.8939 | 0.8813 | 0.0732 | 0.9276 | 0.8031 | 0.8012 | 0.1307 | 0.9561 | 0.8485 | 0.8413 | 0.1019 | 0.8515 |


