"""Benchmark the tracking detector's production window-forward path.

The default shape matches the full validation videos: one two-frame window of
64 x 256 x 256 voxels.  Since inference advances overlapping windows by
``window_size - 1``, ``video_frames_per_s`` is the useful steady-state rate,
not the number of tensors passed through the backbone.

Run from the repository root:

    PYTHONPATH=src .venv/bin/python scripts/benchmark_tracking_throughput.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from model_zoo.models_track import define_track_model


ARCHITECTURES = ("temporal", "temporal_v2", "temporal_v3", "temporal_v4")
WIDTHS = {
    "1x": [32, 64, 128],
    "2x": [64, 128, 256],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spatial-shape", default="64,256,256")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def build_model(arch: str, layers: list[int]) -> torch.nn.Module:
    """Reproduce the architecture shared by the comparable logged runs."""
    return define_track_model(
        unet_arch=arch,
        unet_out_channels=32,
        unet_layers=layers,
        unet_pool_stride=(1, 2, 2),
        unet_n_anisotropic=2,
        unet_n_decode_steps=1,
        unet_n_extra_decode_steps=1,
        unet_temporal_fusion_at=("head",),
        unet_temporal_match_channels=32,
        unet_temporal_taps=(5, 3, 3),
        unet_temporal_dilation="auto",
        unet_temporal_agree="mean",
        unet_temporal_motion_prior=1.0,
        unet_temporal_flow_smooth_iters=3,
        unet_temporal_flow_channels=8,
        unet_temporal_flow_hidden_channels=32,
        unet_temporal_value_channels=8,
        unet_spatial_attention="none",
        unet_blocks_per_stage=[3, 3, 3],
        unet_stem="unshuffle",
        unet_norm="batch",
        unet_hires_fusion=True,
        transformer_hidden=128,
        transformer_heads=4,
        transformer_blocks=4,
        transformer_dropout=0.1,
        pos_feat_dim=32,
        edge_geom_feats=True,
        edge_drift_correction=True,
        unet_features=(),
        annot_head=True,
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    spatial = tuple(int(v) for v in args.spatial_shape.split(","))
    if len(spatial) != 3:
        raise ValueError("--spatial-shape must contain Z,Y,X")

    device = torch.device(args.device)
    amp_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    use_amp = args.precision != "fp32"
    x = torch.randn(1, args.window_size, *spatial, device=device)
    results = []

    for arch in ARCHITECTURES:
        for width_name, layers in WIDTHS.items():
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model(arch, layers).eval().to(device)

            def forward() -> None:
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    model.encode(x)

            with torch.inference_mode():
                for _ in range(args.warmup):
                    forward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                elapsed_ms = []
                for _ in range(args.repeats):
                    if device.type == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        forward()
                        end.record()
                        end.synchronize()
                        elapsed_ms.append(start.elapsed_time(end))
                    else:
                        import time

                        start = time.perf_counter()
                        forward()
                        elapsed_ms.append((time.perf_counter() - start) * 1000)

            mean_ms = statistics.mean(elapsed_ms)
            stride = max(args.window_size - 1, 1)
            row = {
                "architecture": arch,
                "width": width_name,
                "layers": layers,
                "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6,
                "mean_ms_per_window": mean_ms,
                "median_ms_per_window": statistics.median(elapsed_ms),
                "video_frames_per_s": 1000.0 * stride / mean_ms,
                "tensor_frames_per_s": 1000.0 * args.window_size / mean_ms,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else None
                ),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            del model

    report = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "precision": args.precision,
        "input_shape": [1, args.window_size, *spatial],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
