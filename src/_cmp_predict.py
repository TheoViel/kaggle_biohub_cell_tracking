"""Temporary: compare tracking.predict.predict_video against output/*.geff reference."""

from pathlib import Path

import numpy as np
import torch
import tracksdata as td

from params import TRAIN_DIR
from tracking.predict import PredictConfig, load_track_model, predict_video

SAMPLE = "44b6_0113de3b"
CKPT = "../notebooks/clean-approach-lightweight-local-cv-no-hack/input/biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"

device = "cuda"
model = load_track_model(CKPT, device=device)

cfg = PredictConfig(
    det_threshold=0.99,
    det_tta=True,
    pool_kernel_um=3.0,
    edge_activation="softmax",
    threshold=0.5,
    max_parents_per_node=1,
    max_children_per_node=2,
)

coords, edges, _ = predict_video(
    model,
    Path(TRAIN_DIR) / SAMPLE,
    device=device,
    cfg=cfg,
    window_size=2,
    downsample=(1, 4, 4),
)
print(f"OURS  coords={coords.shape}  edges={len(edges)}")

ref = td.graph.IndexedRXGraph.from_geff(f"../output/{SAMPLE}.geff")[0]
rn = ref.node_attrs().to_pandas().sort_values("node_id")
re_ = ref.edge_attrs().to_pandas()
print(f"REF   nodes={rn.shape}  edges={re_.shape}")

ours_nodes = coords.astype(np.int64)
ref_nodes = rn[["t", "z", "y", "x"]].values.astype(np.int64)
print("nodes identical:", ours_nodes.shape == ref_nodes.shape and np.array_equal(ours_nodes, ref_nodes))

for t in [0, 1, 50]:
    a = ours_nodes[ours_nodes[:, 0] == t]
    b = ref_nodes[ref_nodes[:, 0] == t]
    print(f"  t={t}: ours {len(a)} ref {len(b)}")

# edges: compare as (src, tgt) index pairs after mapping ref node_id -> row idx
id2row = {nid: i for i, nid in enumerate(rn["node_id"].values)}
ref_pairs = {(id2row[s], id2row[t]) for s, t in re_[["source_id", "target_id"]].values}
our_pairs = {(int(s), int(t)) for s, t, _, _ in edges}
print(f"edge overlap: {len(ref_pairs & our_pairs)} / ref {len(ref_pairs)} / ours {len(our_pairs)}")
