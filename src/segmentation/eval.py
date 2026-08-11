import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .post_process import postprocess
from .predict import predict_volume

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODES_FOLDER = PROJECT_ROOT / "input" / "processed" / "nodes"
IMG_FOLDER = PROJECT_ROOT / "input" / "processed" / "images"
SCALE = (1.625, 0.40625, 0.40625)


def evaluate(model, args):
    """Evaluate a model against annotated frames."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    frames = np.arange(args.eval_frames, dtype=np.int32)
    nodes = pd.read_csv(NODES_FOLDER / f"{args.eval_sample_name}.csv")
    cellpose_folder = PROJECT_ROOT / "output" / "cellpose"

    total_preds = 0
    total_matched = 0
    total_gt = 0

    try:
        for frame in frames:
            data_path = cellpose_folder / f"{args.eval_sample_name}_{frame}.npz"
            with np.load(data_path) as data:
                preds = predict_volume(
                    model,
                    data,
                    use_cellpose=args.use_cellpose,
                    device=device,
                )

            gt = nodes[nodes["t"] == frame][["z", "y", "x"]].values
            _, pred_centroids = postprocess(preds)

            if len(pred_centroids) and len(gt):
                distances = np.linalg.norm(
                    (pred_centroids[:, None] - gt[None]) * np.asarray(SCALE),
                    axis=2,
                )
                adjacency = csr_matrix(distances <= 7.0)
                matching = maximum_bipartite_matching(
                    adjacency, perm_type="column"
                )
                matched_count = int(np.count_nonzero(matching >= 0))
            else:
                matched_count = 0

            total_preds += len(pred_centroids)
            total_matched += matched_count
            total_gt += len(gt)
    finally:
        model.train(was_training)

    precision = total_matched / total_preds if total_preds else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1
