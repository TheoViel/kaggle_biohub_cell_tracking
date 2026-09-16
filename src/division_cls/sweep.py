"""Generate the job list for an overnight division-classifier sweep.

Writes one job per line as `name<TAB>shell-quoted args`, consumed by
`scratchpad/worker.sh`. Ordered by what is worth knowing first, because the
queue is drained head-first and an 8-hour budget may not reach the end.

**The first block is the noise floor, and it is not optional.** Everything after
it is a single run per config, which is affordable only because the replicates
tell us what to ignore.

It has since been measured, and it is wide: four identical runs span 0.663-0.767
on fold 0 and 0.314-0.462 on fold 1. Of the 71 jobs below, exactly one landed
outside it in a way that survived checking (`bb_scratch`, which collapses). If
you extend this file, budget for replicates of anything you intend to believe --
a single run cannot resolve less than about 0.10 AP here, and most of the axes
in this sweep move less than that.

Deliberately *not* swept: `patch_yx`, `patch_z` and `n_frames`. Ten
configurations already tested those and every enlargement lost, monotonically in
how much input it added, on both folds -- see `README.md`. Re-running them would
spend a third of the budget re-deriving a known answer.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

# Jobs that came back with both folds on the night of 2026-09-09. `--resume`
# writes the queue with these removed, which puts the augmentation arm at the
# head -- it is the block the quoting bug in `worker.sh` dropped, and the one
# this sweep was built to answer. `layers6`, `layers8`, `heads2` and `heads8`
# are deliberately *not* here: they were killed part-way through fold 0 when the
# machine died, so they have a fold-0 curve and no fold-1 number, and a run
# scored on one fold is not comparable with the rest.
DONE_2026_09_09 = {
    "seed0", "seed1", "seed2", "seed3",
    "mult2", "mult12", "mult24", "mult32",
    "bb_convnext_atto", "bb_convnext_femto", "bb_convnext_pico",
    "bb_convnext_nano", "bb_convnext_small", "bb_convnextv2_nano",
    "bb_convnextv2_tiny", "bb_resnet18", "bb_resnet34", "bb_efficientnet_b0",
    "bb_maxvit_nano_rw_256", "bb_regnety_016", "bb_seresnext26d_32x4d",
    "bb_repvgg_a2", "bb_densenet121", "bb_scratch",
    "d128", "d384", "d512", "layers2",
}

# Backbones that build and run at a 32px crop, verified before queueing. The
# parameter counts barely differ (2.3-3.7M) because `FrameEncoder` cuts at the
# deepest stage still >= min_tokens wide, which for a 32px input is an early
# one -- so this axis varies the *inductive bias*, not the capacity.
BACKBONES = [
    "convnext_atto", "convnext_femto", "convnext_pico", "convnext_nano",
    "convnext_small", "convnextv2_nano", "convnextv2_tiny",
    "resnet18", "resnet34", "efficientnet_b0", "maxvit_nano_rw_256",
    "regnety_016", "seresnext26d_32x4d", "repvgg_a2", "densenet121",
]

# The clip-level knobs added in `division_cls/augment.py`, at their defaults.
# An ablation sets one to 0 and changes nothing else.
CLIP_KNOBS = {
    "flip_p": 0.5, "rot90_p": 0.5, "z_jitter": 1, "frame_shift_um": 0.4,
    "frame_brightness": 0.06, "frame_contrast": 0.06, "shot_noise": 0.04,
    "blur_p": 0.25, "cutout_p": 0.25,
}


def _aug(**over) -> list[str]:
    return ["--augmentations", json.dumps(over)]


def jobs() -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []

    def add(name, *args):
        out.append((name, [str(a) for a in args]))

    # --- 1. the noise floor -------------------------------------------------
    # Same config, four seeds. Everything below is read against the spread these
    # produce, not against each other.
    for s in (0, 1, 2, 3):
        add(f"seed{s}", "--seed", s)

    # --- 2. is the new augmentation earning its place? ----------------------
    # All clip-level knobs off = the augmentation this pipeline had before, so
    # this is the direct before/after.
    add("aug_none", *_aug(**{k: 0 for k in CLIP_KNOBS}))
    # Leave-one-out: which knob is carrying the effect, if any.
    for knob in CLIP_KNOBS:
        add(f"no_{knob}", *_aug(**{knob: 0}))

    # --- 3. augmentation strength ------------------------------------------
    add("aug_light", *_aug(rot_p=0.5, elastic_p=0.3, scale_p=0.3, brightness=0.12,
                           contrast=0.12, gamma=0.12, shot_noise=0.02, blur_p=0.1,
                           cutout_p=0.1, frame_shift_um=0.2))
    add("aug_strong", *_aug(elastic_p=0.8, elastic_um=2.0, scale_p=0.8,
                            scale_range=(0.8, 1.25), brightness=0.35, contrast=0.35,
                            gamma=0.35, shot_noise=0.07, blur_p=0.4,
                            blur_sigma=(0.4, 1.6), cutout_p=0.4,
                            cutout_frac=(0.1, 0.4), frame_shift_um=0.6,
                            frame_brightness=0.1, frame_contrast=0.1, z_jitter=1))
    add("aug_extreme", *_aug(elastic_p=0.9, elastic_um=2.6, scale_p=0.9,
                             scale_range=(0.75, 1.35), brightness=0.45, contrast=0.45,
                             gamma=0.45, shot_noise=0.10, blur_p=0.5,
                             blur_sigma=(0.5, 2.0), cutout_p=0.5,
                             cutout_frac=(0.15, 0.45), frame_shift_um=0.8,
                             frame_brightness=0.15, frame_contrast=0.15, z_jitter=2))

    # --- 4. the label-changing knobs ---------------------------------------
    # A reversed division is two cells becoming one: a hard negative that shares
    # the positive's appearance and differs only in direction. Off-distribution
    # (real cells do not merge), which is why it is measured rather than assumed.
    for p in (0.15, 0.3, 0.5):
        add(f"reverse{p}", *_aug(time_reverse_p=p))
    add("drop_ctx0.3", *_aug(drop_context_p=0.3))
    add("reverse0.3_strong", *_aug(time_reverse_p=0.3, elastic_p=0.8, scale_p=0.8,
                                   cutout_p=0.4, shot_noise=0.07))

    # --- 5. oversampling, now that augmentation is stronger ----------------
    for m in (2, 12, 24, 32):
        add(f"mult{m}", "--positive_multiplier", m)
    # The interaction the README flagged: a repeat is only worth having if it
    # comes back different, so the multiplier is meaningless without the
    # augmentation strength beside it.
    add("mult24_strong", "--positive_multiplier", 24,
        *_aug(elastic_p=0.8, scale_p=0.8, cutout_p=0.4, shot_noise=0.07,
              brightness=0.35, contrast=0.35))
    add("mult4_light", "--positive_multiplier", 4,
        *_aug(rot_p=0.5, elastic_p=0.3, scale_p=0.3, cutout_p=0.1, blur_p=0.1))

    # --- 6. architecture ----------------------------------------------------
    for b in BACKBONES:
        add(f"bb_{b}", "--backbone", b)
    add("bb_scratch", "--no-pretrained")

    # --- 7. the transformer head over the frames ---------------------------
    for d in (128, 384, 512):
        add(f"d{d}", "--d_model", d, "--dim_feedforward", d * 2)
    for n in (2, 6, 8):
        add(f"layers{n}", "--n_layers", n)
    for h in (2, 8):
        add(f"heads{h}", "--n_heads", h)
    for dr in (0.0, 0.2, 0.35):
        add(f"drop{dr}", "--dropout", dr)
    # How much spatial map the transformer gets: 8 -> more tokens per frame.
    for t in (2, 8):
        add(f"tokens{t}", "--min_tokens", t)

    # --- 8. the negatives ---------------------------------------------------
    for n in (6, 24, 48):
        add(f"neg{n}", "--neg_per_pos", n)
    add("neg_stack", "--negatives", "stack", "--neg_per_pos", 24)
    for w in (0, 2):
        add(f"exclude{w}", "--exclude_window", w)

    # --- 9. optimisation ----------------------------------------------------
    for lr in (1e-4, 6e-4):
        add(f"lr{lr:g}", "--optimizer_config", json.dumps({"lr": lr}))
    add("enc_lr1e-4", "--optimizer_config", json.dumps({"lr_encoder": 1e-4}))
    add("wd0.05", "--optimizer_config", json.dumps({"weight_decay": 0.05}))
    add("bs16", "--batch_size", 16)
    add("bs64", "--batch_size", 64)
    # Longer, at roughly twice the cost -- last, so it only runs if there is time.
    add("iter4000", "--iterations", 4000)
    add("iter4000_strong", "--iterations", 4000,
        *_aug(elastic_p=0.8, scale_p=0.8, cutout_p=0.4, shot_noise=0.07))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true",
                    help="drop the jobs already scored on both folds (see DONE_2026_09_09)")
    opts = ap.parse_args()

    js = jobs()
    name = "sweep_jobs.txt"
    if opts.resume:
        js = [(n, a) for n, a in js if n not in DONE_2026_09_09]
        name = "sweep_jobs_resume.txt"

    lines = [f"{n}\t{' '.join(shlex.quote(a) for a in args)}" for n, args in js]
    path = Path(__file__).with_name(name)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(js)} jobs -> {path}")
    for n, args in js[:6]:
        print(f"  {n:<18} {' '.join(args)}")
    print("  ...")


if __name__ == "__main__":
    main()
