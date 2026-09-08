"""Rebuilding a trained HOCT from a checkpoint, and the graph settings with it.

The geometry lives in the checkpoint's ``model_config``, written by
``main_edge`` for exactly this purpose, so nothing here needs the run's
``args.json`` or the flags it was launched with -- mirroring
``embeddings/infer.py``.

The candidate-graph settings travel in the same dict and matter as much as the
weights: a model trained against 5 neighbours within 20 um sees a different
problem if inference proposes something else. ``max_nodes_per_frame`` is the
one exception, and it is not merely overridable but *ignored* by default: it
bounds the quadratic edge attention so a training batch fits, and a node it
drops at inference gets no candidate edges and cannot be linked at all. Feeding
every node is both the better and the honest setting, so ``dataset_for``
defaults to no cap and takes one only when asked.
"""

from __future__ import annotations

from pathlib import Path

import torch

from edge_transformer.dataset import EdgeDataset
from model_zoo.hoct import HOCT


def load_hoct(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[HOCT, dict]:
    """Rebuild the model a checkpoint was trained as. Returns ``(model, config)``."""
    state = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(state.get("model_config") or {})
    if not cfg:
        raise KeyError(
            f"{ckpt_path} carries no model_config; it was not written by main_edge.py"
        )

    node_feat_dim = int(cfg["node_feat_dim"]) if "node_feat_dim" in cfg else None
    if node_feat_dim is None:
        # Written by runs before the width was recorded: it is 4 (t, z, y, x)
        # plus the embedding, and the first Linear knows how wide that was.
        node_feat_dim = int(state["model_state_dict"]["input_proj.weight"].shape[1])
    edge_feat_dim = 0
    if cfg.get("use_edge_feats", True):
        edge_feat_dim = int(
            state["model_state_dict"]["edge_input_proj.weight"].shape[1]
        )

    model = HOCT(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        d_model=int(cfg.get("d_model", 288)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_node_blocks=int(cfg.get("n_node_blocks", 4)),
        n_edge_blocks=int(cfg.get("n_edge_blocks", 4)),
        mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
        dropout=0.0,
        pos_scale=float(cfg.get("pos_scale", 20.0)),
        min_wavelength=float(cfg.get("min_wavelength_um", 4.0)),
        max_wavelength=float(cfg.get("max_wavelength_um", 200.0)),
        max_distance=float(cfg.get("attn_max_distance_um", 40.0)),
        householder=bool(cfg.get("householder", True)),
    )
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()
    if verbose:
        print(f"- HOCT [{cfg.get('n_node_blocks')} node + {cfg.get('n_edge_blocks')} "
              f"edge blocks, d={cfg.get('d_model')}]  "
              f"({model.count_parameters() / 1e6:.1f}M parameters)")
    return model, cfg


def dataset_for(
    cfg: dict,
    videos: list,
    max_nodes_per_frame: int | None = None,
    require_links: bool = False,
) -> EdgeDataset:
    """An ``EdgeDataset`` built the way the checkpoint's run built its own.

    ``max_nodes_per_frame`` defaults to no cap -- see the module docstring. Pass
    an int to reinstate one, e.g. to reproduce a run's training conditions.
    """
    return EdgeDataset(
        videos,
        window_size=int(cfg.get("window_size", 2)),
        max_nodes_per_frame=int(max_nodes_per_frame or 0),
        n_neighbors=int(cfg.get("n_neighbors", 5)),
        max_distance_um=float(cfg.get("max_distance_um", 20.0)),
        max_dt=int(cfg.get("max_dt", 1)),
        pos_scale=float(cfg.get("pos_scale", 20.0)),
        drift_correction=bool(cfg.get("edge_drift_correction", True)),
        # A checkpoint trained with the confidence features cannot be fed
        # without them -- the input width would not match -- so this follows the
        # run rather than being a choice at inference.
        use_conf_score=bool(cfg.get("use_conf_score", False)),
        require_links=require_links,
        augment=False,
    )
