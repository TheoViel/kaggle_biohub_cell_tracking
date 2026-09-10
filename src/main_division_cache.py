"""Pre-extract the synthetic division crops the classifier trains on.

    python main_division_cache.py --synthetic_dir ../input/synthetic/v5 \
                                  --out ../output/division/synth_v5.npz

A sequence is a 50 MB npz whose `volumes` decompress as a block, so serving a
random crop from a random sequence would pay that on every sample. This walks
each sequence once. The result is one array of `(n_frames, patch_z, H, W)`
clips with their labels, which `division_cls.synthetic.SyntheticClipDataset`
reads directly.

The crop geometry must match what the model was built for -- pass the same
`--patch_yx/--patch_z/--n_frames` as the training run, or the clips will not
fit the network.
"""

import argparse

from division_cls.synthetic import build_cache


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic_dir", default="../input/synthetic/v5")
    p.add_argument("--out", default="../output/division/synth_v5.npz")
    p.add_argument("--patch_yx", type=int, default=32)
    p.add_argument("--patch_z", type=int, default=3)
    p.add_argument("--n_frames", type=int, default=3)
    p.add_argument("--exclude_window", type=int, default=1)
    p.add_argument("--neg_per_pos", type=float, default=12.0)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build_cache(a.synthetic_dir, a.out, patch_yx=a.patch_yx, patch_z=a.patch_z,
                n_frames=a.n_frames, exclude_window=a.exclude_window,
                neg_per_pos=a.neg_per_pos, max_sequences=a.max_sequences, seed=a.seed)


if __name__ == "__main__":
    main()
