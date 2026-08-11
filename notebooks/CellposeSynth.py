"""Run Cellpose-SAM on the synthetic cell volumes."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cellpose import core, models
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.plot import show_cellpose_segmentation


FOLDER = (
    ROOT
    / "notebooks"
    / "biohub-synthetic-dataset"
    / "working"
    / "synth_out_3"
    / "biohub_synthetic"
    / "static"
)
SAVE_FOLDER = ROOT / "output" / "segs_3"

FLOW_THRESHOLD = 1
CELLPROB_THRESHOLD = -1
PLOT = False
CELLPOSE = True


def main():
    sample_names = list(FOLDER.iterdir())
    print(f"Found {len(sample_names)} volumes")

    model = models.CellposeModel(gpu=core.use_gpu()) if CELLPOSE else None
    print("GPU enabled:", core.use_gpu())

    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    for sample_path in tqdm(sample_names):
        output_path = SAVE_FOLDER / sample_path.name
        if output_path.exists():
            continue

        with np.load(sample_path) as data:
            if "static" in str(FOLDER):
                img = data["volume"][..., None]
                gt_masks = data["mask"]
                gt = np.round(data["centroids"], 0).astype(np.int32)
            else:
                frame = 0
                img = data["volumes"][frame, ..., None]
                gt = np.round(data["nodes"], 0).astype(np.int32)
                gt = gt[gt[:, 0] == frame]
                gt = gt[:, [1, 2, 3]]
                gt_masks = None

        if PLOT:
            z = np.random.randint(1, img.shape[0] - 1)
            plt.figure(figsize=(15, 5))

            for i in range(3):
                plt.subplot(1, 3, i + 1)

                z_current = z - 1 + i
                plt.title(f"z = {z_current}")
                plt.imshow(img[z_current], cmap="gray")

                gt_current = gt[np.abs(gt[:, 0] - z_current) <= 3]
                plt.scatter(gt_current[:, 2], gt_current[:, 1], c="red", s=10)
                plt.axis("off")
            plt.show()

        if CELLPOSE:
            masks, flows, _ = model.eval(
                img,
                batch_size=64,
                flow_threshold=FLOW_THRESHOLD,
                cellprob_threshold=CELLPROB_THRESHOLD,
            )

            np.savez(
                output_path,
                img=img.squeeze(),
                masks=gt_masks,
                cellpose_masks=masks,
                flows=flows[0],
                probas=flows[2],
                centroids=gt,
            )
        else:
            np.savez(
                output_path,
                img=img.squeeze(),
                masks=gt_masks,
                centroids=gt,
            )

        if PLOT and CELLPOSE:
            z = 32
            plt.figure(figsize=(12, 4))
            gt_current = gt[np.abs(gt[:, 0] - z) <= 3]
            show_cellpose_segmentation(
                img[z],
                masks[z],
                flows[0][z],
                nodes=gt_current,
            )
            plt.tight_layout()
            plt.show()
            break

    print("Done!")


if __name__ == "__main__":
    main()
