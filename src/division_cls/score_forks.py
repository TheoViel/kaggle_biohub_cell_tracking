"""Score every fork candidate with the out-of-fold division classifier.

Reads the table `gen_fork_candidates.py` wrote and returns one probability per
row, aligned to it. Unlike training, **every** row is scored including the
unlabelled ones: the threshold has to be applied to every fork the chain
proposes, and only ~0.7% of them sit near an annotated cell.

The fold split is enforced rather than assumed, as in `main_division_predict`:
fold `i` trained without `EMBRYOS[i]`, so a stack is scored by the model that
held its embryo out, and a stack with no such model is skipped rather than
scored by a model that has seen it.

Crops come from `DivisionDataset` in candidate mode, which is the same code the
model trained against -- so a fused model gets its feature vector, and there is
no second copy of the cropping to drift out of step with training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader  # noqa: E402

from division_cls.dataset import DivisionDataset  # noqa: E402
from division_cls.infer import load_fold_models  # noqa: E402
from params import TRAIN_DIR  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True, help="gen_fork_candidates .npz")
    p.add_argument("--run_dir", required=True,
                   help="a division_cls run holding fold_<n>/ckpt_final.pt")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    blob = np.load(args.candidates, allow_pickle=True)
    stems = np.asarray([str(s) for s in blob["stem"]])
    table = blob["table"]
    models = load_fold_models(args.run_dir, args.device)
    print(f"{len(table)} forks, models for {sorted(models)}", flush=True)

    # Score through `DivisionDataset` itself rather than re-deriving the crops.
    # Two reasons, both learned the hard way: a fused model needs the proposal's
    # feature vector, which a coords-only scorer has no way to supply; and any
    # second implementation of the cropping is a chance for inference to
    # normalise or centre differently from training, which would show up as a
    # mysteriously weak model rather than as an error.
    scores = np.full(len(table), np.nan, np.float32)
    for embryo, (model, cfg) in sorted(models.items()):
        # A candidates file restricted to one fold -- which is how a sweep on
        # fold 0 is generated -- has nothing for the other embryo's model. The
        # length check below passes (0 == 0) and the empty loader then reaches
        # `torch.cat([])`, so say so here instead.
        if not any(s.startswith(embryo) for s in stems):
            print(f"  skip {embryo}: no rows in this candidates file", flush=True)
            continue
        ds = DivisionDataset(
            args.data_dir, embryos=[embryo], candidates=args.candidates,
            candidates_labelled_only=False, augment=False,
            patch_yx=cfg["patch_yx"], patch_z=cfg["patch_z"],
            n_frames=cfg["n_frames"],
        )
        rows = np.flatnonzero(np.asarray([s.startswith(embryo) for s in stems]))
        if len(ds.samples) != len(rows):
            raise RuntimeError(
                f"{embryo}: dataset has {len(ds.samples)} rows, candidates table "
                f"{len(rows)} -- they must be the same rows in the same order for "
                "the scores to align")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        got, done = [], 0
        with torch.no_grad():
            for batch in loader:
                clip = batch["clip"].to(args.device, non_blocking=True)
                feat = batch.get("feat")
                feat = (feat.to(args.device, non_blocking=True)
                        if feat is not None and feat.numel() else None)
                with torch.autocast("cuda", torch.float16,
                                    enabled=args.device.startswith("cuda")):
                    got.append(torch.sigmoid(model(clip, feat)).float().cpu())
                done += len(clip)
                if done % (args.batch_size * 40) == 0:
                    print(f"  {embryo}: {done}/{len(ds.samples)}", flush=True)
        scores[rows] = torch.cat(got).numpy().astype(np.float32)
        print(f"  {embryo}: {len(rows)} forks scored", flush=True)
    for embryo in ("44b6", "6bba"):
        if embryo not in models:
            print(f"  skip {embryo}: no out-of-fold model", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, stem=stems, score=scores, label=table[:, 4].astype(np.int8),
                        coords=table[:, :4].astype(np.float32),
                        features=table[:, 5:].astype(np.float32))
    lab = table[:, 4].astype(int)
    ok = np.isfinite(scores)
    print(f"\nwrote {out}  ({ok.sum()}/{len(scores)} scored)")
    for name, m in (("positive", lab == 1), ("negative", lab == 0),
                    ("unlabelled", lab == -1)):
        m = m & ok
        if m.any():
            print(f"  {name:<11} n={m.sum():6d}  score mean {scores[m].mean():.3f} "
                  f"median {np.median(scores[m]):.3f}")


if __name__ == "__main__":
    main()
