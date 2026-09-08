# NETr — contrastive node embeddings

An implementation of **ASCENT** (Han, Pritchard & Lu, bioRxiv 2025,
[10.1101/2025.07.23.666425](https://doi.org/10.1101/2025.07.23.666425)):
one embedding per detected cell, learned with **no annotation**.

Positives are two augmented views of the same frame, so the supervision is
entirely the augmentation pipeline. That matters here because the real ground
truth annotates ~3% of cells — this trains on all of them.

## Running

```bash
python main_contrastive.py --coords_log_folder ../logs/<run>/preds_0.50/
```

`coords_log_folder` is the key argument: any predictions folder the tracking
pipeline wrote, holding `<sample>.npz` with a `coords` array of `(t, z, y, x)`
rows. Those detections define the crops — the nodes the matcher will actually be
asked to link, spurious ones included.

| file | role |
|---|---|
| `model.py` | NETr: patch encoder + positional MLP, summed, through a transformer |
| `patches.py` | the (3, 64, 64) patches and the paper's frame percentile normalisation |
| `augment.py` | the six augmentations; **`jitter_um` is the critical one** |
| `dataset.py` | one frame → two views |
| `losses.py` | NT-Xent at τ=0.05, plus alignment/uniformity diagnostics |
| `train.py` / `main.py` | the loop and the fold orchestration, shaped like `tracking/` |
| `match.py` | Hungarian linking, dynamic threshold, EMA track embeddings |

## What is the paper's and what is ours

Taken verbatim: patch (3, 64, 64) with z-slices as channels; frame percentile
normalisation to [0, 1] on the 1st/99.99th; d_model 256; 4 layers, 4 heads,
feedforward 512, dropout 0.1; element-wise fusion; NT-Xent at τ=0.05; Adam at
1e-4, batch 4 frames, 20 epochs; Hungarian matching with
`θ = 0.5·(mean(top-k) + mean(all))` and EMA α=0.5.

**Ours, because the paper gives no ranges:** every augmentation magnitude. The
one that matters most is `jitter_um`. The paper's ablation puts position jitter
at 37.3% → 95.1%: without it the network matches cells by coordinate and never
learns appearance. It is drawn *uniformly and bounded*, not Gaussian — cells
here sit ~8 µm apart and a Gaussian tail would recentre a patch on the
neighbour, handing the loss two different cells as a positive pair.

Also ours: rotation is restricted to y/x. z is sampled 4× coarser, so rotating
it into the plane would claim resolution the microscope never had.

## Reading the log

`top1` is the number to watch, not the loss — the loss falls as soon as the easy
negatives are pushed away, while top-1 only moves when the cells that actually
confuse the matcher come apart. `align`/`unif` are Wang & Isola's diagnostics
and catch collapse, which a falling loss will not.

## Known gaps

- **Divisions.** One-to-one assignment cannot represent a cell becoming two; a
  daughter starts a new track. Fine for neurons, wrong for embryos.
- **The dynamic threshold is relative.** It admits ~half the assignment whatever
  the frame is, so it will not reject a frame that should not have been matched.
- **Density.** The paper tracks 79–96 neurons; frames here hold 180–790 cells,
  so `max_nodes` is a real budget and the transformer is quadratic in it.
