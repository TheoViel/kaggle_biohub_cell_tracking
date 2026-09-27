# Experiments


## Tracking experiments — new data recipe (reference `2026-09-24/5`)

Runs on the recipe that replaced the 09-13 one: zebrahub crops mixed into
pretraining, FOCUS-3D weak positives, `border_keep`, `bg_level_p` on, and
`det_neg_weight` 0.05 everywhere (the older runs used 1 in the main stage).
Each row is at that run's **own best** radius/threshold — the recipe moved the
operating point from 0.40 to 0.45–0.50, so a fixed-threshold column would
flatter the old runs. Changes are against `2026-09-24/5`; only the ones that
alter training are listed (`fullfit`, `output_dir` and eval-only keys omitted).

| exp folder | radius / threshold | changes vs `24/5` | 44b6 node rec | 44b6 jacc | 44b6 adj jacc | 6bba node rec | 6bba jacc | 6bba adj jacc | **macro node rec** | **macro jacc** | **macro adj jacc** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25/4 | r=6, th=0.60 | `loss_config.det_sigma_anchor`: fine → **coarse**<br>`annot_head`: True → False | 0.9905 | 0.8301 | 0.8158 | 0.9349 | 0.7566 | 0.7550 | 0.9627 | 0.7933 | **0.7854** |
| 2026-09-26/1 | r=6, th=0.40 | `loss_config.det_sigma_anchor`: fine → **grid**<br>`loss_config.det_weak_weight`: 0.5 → 0.2<br>`weak_positives.use_focus_3d`: True → **False**<br>`weak_positives.pl_folder`: 23/5 → **09-01/9**<br>`pretraining.iterations`: 3000 → 5000<br>`pretraining.synthetic_proportion`: 0.4 → 0.5<br>`pretraining.zebrahub_proportion`: 0.1 → **0.0**<br>`pretraining.weights`: 22/10 → **13/1** | **0.9936** | **0.8353** | **0.8177** | 0.9341 | 0.7539 | 0.7513 | 0.9638 | **0.7946** | 0.7845 |
| 2026-09-25/2 | r=6, th=0.50 | `loss_config.det_sigma_anchor`: fine → **grid**<br>`annot_head`: True → False | 0.9886 | **0.8265** | 0.8094 | 0.9397 | **0.7605** | **0.7567** | 0.9642 | 0.7935 | **0.7831** |
| 2026-09-26/0 | r=6, th=0.50 | `loss_config.det_sigma_anchor`: fine → **grid**<br>`annot_head`: True → False | 0.9904 | 0.8274 | 0.8104 | 0.9409 | 0.7600 | 0.7557 | 0.9657 | 0.7937 | 0.7830 |
| 2026-09-25/3 | r=6, th=0.45 | `augmentations.border_keep`: 0.5 → 0.25<br>`augmentations.crop_anchor_p`: 0.9 → 0.5<br>`annot_head`: True → False | 0.9876 | 0.8218 | 0.8081 | 0.9392 | 0.7579 | 0.7543 | 0.9634 | 0.7899 | 0.7812 |
| 2026-09-24/5 | r=6, th=0.45 | *(reference)* | 0.9887 | 0.8215 | 0.8056 | 0.9426 | 0.7542 | 0.7506 | 0.9656 | 0.7878 | 0.7781 |
| 2026-09-25/5 | r=7, th=0.60 | `loss_config.det_sigma_anchor`: fine → **coarse**<br>`augmentations.crop_anchor_p`: 0.9 → **0.0**<br>`annot_head`: True → False | 0.9805 | 0.8047 | 0.7955 | 0.9321 | 0.7569 | **0.7568** | 0.9563 | 0.7808 | 0.7761 |
| 2026-09-27/0 | r=6, th=0.45 | `loss_config.det_sigma_anchor`: fine → **grid**<br>`loss_config.det_weak_weight`: 0.5 → 0.2<br>`weak_positives.use_focus_3d`: True → **False**<br>`pretraining.iterations`: 3000 → 5000<br>`pretraining.weights`: 22/10 → **13/1** | 0.9869 | 0.8242 | 0.8097 | 0.9293 | 0.7419 | 0.7405 | 0.9581 | 0.7830 | 0.7751 |
| 2026-09-24/7 | r=7, th=0.50 | `unet_n_extra_decode_steps`: 1 → 0<br>`annot_head`: True → False | 0.9930 | 0.8086 | 0.7924 | 0.9587 | 0.7547 | 0.7463 | 0.9759 | 0.7816 | 0.7693 |
| **2026-09-23/6** | r=6, th=0.40 | `augmentations.crop_anchor_p`: 0.9 → **0.0**<br>`loss_config.det_weak_weight`: 0.5 → **0.2**<br>`pretraining.iterations`: 3000 → 2500 | 0.9795 | 0.8055 | 0.7919 | 0.9360 | 0.7455 | **0.7444** | 0.9577 | 0.7755 | 0.7681 |
| 2026-09-24/3 | r=6, th=0.45 | `augmentations.crop_anchor_p`: 0.9 → 0.0<br>`loss_config.det_neg_weight`: 0.05 → 0.025<br>`loss_config.det_weak_weight`: 0.5 → 0.3 | 0.9896 | 0.8202 | 0.8003 | 0.9549 | 0.7405 | 0.7324 | 0.9723 | 0.7803 | 0.7664 |
| 2026-09-24/2 | r=7, th=0.40 | `augmentations.crop_anchor_p`: 0.9 → 0.0<br>`loss_config.det_gaussian_sigma`: 1.0 → 1.5<br>`loss_config.det_weak_weight`: 0.5 → 0.2<br>`loss_config.det_weak_smoothing`: 0.01 → 0.0 | 0.9860 | 0.8012 | 0.7883 | 0.9462 | 0.7462 | 0.7426 | 0.9661 | 0.7737 | 0.7655 |


## Edge experiments, one shared post-processing config

| exp folder | nodes_folder | score file | 44b6 node rec | 44b6 jacc | 44b6 adj jacc | 44b6 div jacc | 6bba node rec | 6bba jacc | 6bba adj jacc | 6bba div jacc | **macro node rec** | **macro jacc** | **macro adj jacc** | **macro div jacc** | **macro CV** | **+cls adj jacc** | **+cls div jacc** | **+cls CV** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge_run/2026-09-13/0 | 2026-09-13/1/preds_r6_0.40/ | preds_0.45_pp_default | 0.9836 | 0.8990 | **0.8885** | 0.1667 | 0.9286 | 0.8274 | 0.8262 | 0.1579 | 0.9561 | 0.8632 | **0.8574** | **0.1623** | **0.8736** | 0.8576 | 0.2150 | **0.8791** |
| edge_run/2026-09-16/0 | 2026-09-15/1/preds_r6_0.40/ | preds_0.40_pp_default | 0.9807 | 0.8924 | 0.8831 | 0.1148 | 0.9247 | 0.8261 | 0.8266 | **0.1991** | 0.9527 | 0.8593 | 0.8549 | 0.1569 | 0.8706 | 0.8550 | 0.1876 | 0.8738 |
| edge_run/2026-09-14/5 | 2026-09-13/1/preds_r6_0.40/ | preds_0.45_pp_default | 0.9840 | 0.8986 | 0.8881 | 0.1290 | 0.9295 | 0.8266 | 0.8251 | 0.1435 | 0.9568 | 0.8626 | 0.8566 | 0.1363 | 0.8703 | 0.8568 | 0.1841 | 0.8752 |
| edge_run/2026-09-14/1 | 2026-09-13/2/preds_r6_0.40/ | preds_0.45_pp_default | 0.9837 | 0.8970 | 0.8861 | 0.1562 | 0.9267 | 0.8245 | 0.8244 | 0.1333 | 0.9552 | 0.8607 | 0.8553 | 0.1448 | 0.8697 | 0.8556 | 0.1779 | 0.8734 |
| edge_run/2026-09-16/1 | 2026-09-15/0/preds_r6_0.40/ | preds_0.40_pp_default | 0.9827 | 0.8960 | 0.8850 | **0.1695** | 0.9323 | 0.8245 | 0.8227 | 0.1434 | 0.9575 | 0.8603 | 0.8539 | 0.1564 | 0.8695 | 0.8544 | **0.2161** | 0.8760 |
| edge_run/2026-09-13/1 | 2026-09-13/1/preds_r7_0.40/ | preds_0.45_pp_default | 0.9822 | 0.8853 | 0.8785 | 0.1311 | 0.9271 | **0.8330** | **0.8338** | 0.1266 | 0.9547 | 0.8591 | 0.8561 | 0.1289 | 0.8690 | — | — | — |
| edge_run/2026-09-23/1 | 2026-09-23/6/preds_r6_0.40/ | preds_0.45_pp | 0.9751 | 0.8843 | 0.8719 | 0.1029 | 0.9314 | 0.8305 | 0.8324 | 0.1358 | 0.9533 | 0.8574 | 0.8521 | 0.1194 | 0.8641 | — | — | — |
