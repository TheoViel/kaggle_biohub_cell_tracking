"""Score every detected node with "does this cell divide here", frame by frame.

The training loader reads three frames per *sample*, which costs ~69 ms and is
fine when a sample is a training example. At inference the same volume is asked
about for every cell in it, so the read is amortised instead: frames are walked
in order, each is loaded once into a rolling three-frame window, and every node
in the centre frame is cropped out of what is already in memory.

The output is one probability per node, aligned to the prediction npz's `coords`
rows, which is what `edge_pp.propose_divisions` gates on.

**Which model scores which embryo matters.** Fold `i` held out `EMBRYOS[i]`, so
a stack must be scored by the fold that did *not* train on it, exactly as the
edge transformer is applied. `load_fold_models` returns the mapping and
`score_stack` refuses a stack it has no out-of-fold model for, rather than
quietly using a model that has seen it.
"""

from __future__ import annotations


import torch

from division_cls.model import DivisionNet


def load_model(ckpt_path, device="cuda"):
    """`(model, cfg)` from a checkpoint written by `division_cls.train.fit`."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(state.get("model_config") or {})
    if not cfg:
        raise ValueError(f"{ckpt_path} carries no model_config")
    sd = state["model_state_dict"]
    # `n_features` post-dates the checkpoints that are already on disk, so it is
    # recovered from the weights rather than required in the config: the fusion
    # branch exists iff `feat_mlp` does, and its width is the BatchNorm's. A
    # config key alone would have made every checkpoint written before it
    # unloadable, which is how this was first noticed.
    # `feat_mean` is the unambiguous witness: it is (n_features,) whatever the
    # branch looks like. `feat_mlp.0.weight` is not -- it was (n_features,) when
    # the first layer was a BatchNorm and is (d_feat, n_features) now that it is
    # a Linear, so reading shape[0] silently returned 32 instead of 5 and every
    # fused checkpoint failed to load.
    if "feat_mean" in sd:
        n_features = int(sd["feat_mean"].shape[0])
    elif "feat_mlp.0.weight" in sd:
        w = sd["feat_mlp.0.weight"]
        n_features = int(w.shape[1] if w.ndim == 2 else w.shape[0])
    else:
        n_features = int(cfg.get("n_features", 0) or 0)
    model = DivisionNet(
        backbone=cfg["backbone"], pretrained=False,
        patch_yx=cfg["patch_yx"], patch_z=cfg["patch_z"], n_frames=cfg["n_frames"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
        min_tokens=cfg["min_tokens"], n_features=n_features,
    )
    model.load_state_dict(sd)
    cfg["n_features"] = n_features
    return model.eval().to(device), cfg




