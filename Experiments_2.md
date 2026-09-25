# Experiments

Only runs with **both folds** trained are listed (single-fold runs are excluded).
Scores come from each run's best `scores_*.json` covering both embryos; `_pl` (pseudo-label) score files are excluded because that pass is in-sample.

`CV = adj_edge_jaccard + 0.1 x division_jaccard`, macro-averaged (`mean(embryo_44b6, embryo_6bba)`). Best value per column in **bold**.

Only the best current runs are kept here: the six best rows from the shared-config edge table below, plus the latest run (`edge_run/2026-09-23/1`) and its tracking model.

## Edge experiments, one shared post-processing config

All rows ran `main_edge_postprocess.py --ilp --name default`, i.e.
the current `DEFAULTS` block, over each run's own candidates. The settings blocks
of every score file below are identical, so the differences are the models and
the node sets and nothing else.

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
| edge_run/2026-09-16/1 | 2026-09-15/0/preds_r6_0.40/ | preds_0.40_pp_default | 0.9827 | 0.8960 | 0.8850 | **0.1695** | 0.9323 | 0.8245 | 0.8227 | 0.1434 | 0.9575 | 0.8603 | 0.8539 | 0.1564 | 0.8695 | 0.8544 | **0.2161** | 0.8760 |
| edge_run/2026-09-13/1 | 2026-09-13/1/preds_r7_0.40/ | preds_0.45_pp_default | 0.9822 | 0.8853 | 0.8785 | 0.1311 | 0.9271 | **0.8330** | **0.8338** | 0.1266 | 0.9547 | 0.8591 | 0.8561 | 0.1289 | 0.8690 | — | — | — |
| edge_run/2026-09-23/1 | 2026-09-23/6/preds_r6_0.40/ | preds_0.45_pp | 0.9751 | 0.8843 | 0.8719 | 0.1029 | 0.9314 | 0.8305 | 0.8324 | 0.1358 | 0.9533 | 0.8574 | 0.8521 | 0.1194 | 0.8641 | — | — | — |

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
| 2026-09-23/6 | r=6, th=0.40 | `pretraining.weights`: None → ../logs/2026-09-22/10/fold_0/ckpt_synthetic.pt (Zebrahub+synthetic pretrained backbone)<br>`pretraining.iterations`: 5000 → 2500<br>`pretraining.zebrahub_proportion`: 0.0 → 0.1<br>`pretraining.synthetic_proportion`: 0.5 → 0.4<br>`synthetic_pretraining.enabled`: True → False<br>`weak_positives.use_focus_3d`: False → True<br>`fullfit`: True → False | 0.9794 | 0.8055 | 0.7919 | 0.9360 | 0.7455 | 0.7444 | 0.9577 | 0.7755 | 0.7681 |
|  | r=7, th=0.40 |  | 0.9739 | 0.7824 | 0.7737 | 0.9342 | 0.7531 | 0.7539 | 0.9540 | 0.7678 | 0.7638 |
