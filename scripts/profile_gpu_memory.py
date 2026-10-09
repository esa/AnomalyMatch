#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Profile actual GPU memory consumption at various batch sizes.

Measures peak GPU memory (allocated and reserved) for EfficientNet inference
across a sweep of batch sizes. Results are used to validate and update the
memory model coefficients in prediction_utils.estimate_batch_size.

Usage:
    python scripts/profile_gpu_memory.py --model PATH [--image-sizes 150,224] [--channels 1,3,4]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prediction_utils import MEMORY_COEFFICIENTS, estimate_batch_size


def profile_memory_at_batch_size(
    model, batch_size, image_size, num_channels, num_warmup=2, num_runs=3
):
    """Run inference at a given batch size and measure peak GPU memory.

    Returns dict with allocated_mb, reserved_mb, or None on OOM.
    """
    device = next(model.parameters()).device

    # Warmup to stabilise CUDA allocator
    for _ in range(num_warmup):
        dummy = torch.randn(batch_size, num_channels, image_size, image_size, device=device)
        with torch.no_grad():
            _ = model(dummy)
        del dummy
        torch.cuda.empty_cache()

    # Reset peak stats and measure
    peak_allocated = []
    peak_reserved = []

    for _ in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy = torch.randn(batch_size, num_channels, image_size, image_size, device=device)
        with torch.no_grad():
            logits = model(dummy)
            _ = torch.nn.functional.softmax(logits, dim=-1)

        peak_allocated.append(torch.cuda.max_memory_allocated() / 1024**2)
        peak_reserved.append(torch.cuda.max_memory_reserved() / 1024**2)

        del dummy, logits
        torch.cuda.empty_cache()

    return {
        "peak_allocated_mb": float(np.median(peak_allocated)),
        "peak_reserved_mb": float(np.median(peak_reserved)),
        "peak_allocated_std": float(np.std(peak_allocated)),
        "peak_reserved_std": float(np.std(peak_reserved)),
    }


def find_max_batch_size(model, image_size, num_channels, low=1, high=None):
    """Binary search for the maximum batch size that doesn't OOM."""
    device = next(model.parameters()).device

    # Cap upper bound to avoid 32-bit indexing overflow in CUDA kernels
    # (total tensor elements must be < 2^31)
    if high is None:
        max_elements = 2**31 - 1
        high = min(16384, max_elements // (num_channels * image_size * image_size))

    best = low

    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()
            dummy = torch.randn(mid, num_channels, image_size, image_size, device=device)
            with torch.no_grad():
                logits = model(dummy)
                _ = torch.nn.functional.softmax(logits, dim=-1)
            del dummy, logits
            torch.cuda.empty_cache()
            best = mid
            low = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            high = mid - 1

    return best


def build_model(net_name, num_channels, pretrained=False):
    """Build model for profiling (random weights are fine for memory measurement)."""
    from anomaly_match.utils.get_net_builder import get_net_builder

    net_builder = get_net_builder(net_name, pretrained=pretrained, in_channels=num_channels)
    model = net_builder(num_classes=2, in_channels=num_channels)
    model = model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Profile GPU memory for batch size estimation")
    parser.add_argument(
        "--image-sizes", default="150,224", help="Comma-separated image sizes to test"
    )
    parser.add_argument("--channels", default="1,3,4", help="Comma-separated channel counts")
    parser.add_argument(
        "--nets", default="efficientnet-lite0", help="Comma-separated network names"
    )
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated batch sizes to measure (default: auto-sweep)",
    )
    parser.add_argument(
        "--output", default="benchmarking_results/memory_profile.json", help="Output file"
    )
    parser.add_argument(
        "--find-max", action="store_true", help="Also find max batch size via binary search"
    )
    args = parser.parse_args()

    image_sizes = [int(s) for s in args.image_sizes.split(",")]
    channels_list = [int(c) for c in args.channels.split(",")]
    net_names = [n.strip() for n in args.nets.split(",")]

    # Get GPU info
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    device_props = torch.cuda.get_device_properties(0)
    total_vram_mb = device_props.total_memory / 1024**2
    print(f"GPU: {device_props.name}")
    print(f"Total VRAM: {total_vram_mb:.0f} MB")
    print()

    results = {
        "gpu_name": device_props.name,
        "total_vram_mb": total_vram_mb,
        "measurements": [],
    }

    for net_name in net_names:
        for num_channels in channels_list:
            model = build_model(net_name, num_channels)
            param_count = sum(p.numel() for p in model.parameters())
            param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            print(
                f"=== {net_name}, {num_channels}ch, {param_count / 1e6:.1f}M params ({param_mb:.1f} MB) ==="
            )

            for image_size in image_sizes:
                print(f"\n  Image size: {image_size}x{image_size}")

                # Determine batch sizes to sweep
                if args.batch_sizes:
                    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
                else:
                    # Auto-sweep: geometric progression from small to large
                    batch_sizes = sorted(
                        set(
                            [
                                1,
                                8,
                                16,
                                32,
                                64,
                                128,
                                256,
                                512,
                                768,
                                1024,
                                1536,
                                2048,
                                3072,
                                4096,
                                6144,
                                8192,
                            ]
                        )
                    )

                # Find max batch size first if requested
                max_bs = None
                if args.find_max:
                    print("  Finding max batch size...", end=" ", flush=True)
                    max_bs = find_max_batch_size(model, image_size, num_channels)
                    print(f"{max_bs}")
                    # Add max_bs and nearby points to sweep
                    batch_sizes = sorted(
                        set(batch_sizes + [max_bs, int(max_bs * 0.9), int(max_bs * 0.75)])
                    )

                # What does our current model predict?
                class _FakeCfg:
                    pass

                fake_cfg = _FakeCfg()
                fake_cfg.net = net_name
                fake_cfg.num_channels = num_channels

                class _FakeNorm:
                    pass

                fake_cfg.normalisation = _FakeNorm()
                fake_cfg.normalisation.image_size = [image_size, image_size]

                estimated_bs_30 = estimate_batch_size(
                    fake_cfg, available_vram=total_vram_mb, safety_margin=0.3
                )
                estimated_bs_20 = estimate_batch_size(
                    fake_cfg, available_vram=total_vram_mb, safety_margin=0.2
                )
                estimated_bs_10 = estimate_batch_size(
                    fake_cfg, available_vram=total_vram_mb, safety_margin=0.1
                )

                print(
                    f"  Current model estimates (30%/20%/10% margin): {estimated_bs_30} / {estimated_bs_20} / {estimated_bs_10}"
                )

                # Filter out batch sizes above max if known
                if max_bs is not None:
                    batch_sizes = [bs for bs in batch_sizes if bs <= max_bs]

                for bs in batch_sizes:
                    try:
                        mem = profile_memory_at_batch_size(model, bs, image_size, num_channels)
                        pct_used = mem["peak_reserved_mb"] / total_vram_mb * 100
                        print(
                            f"    BS={bs:>6d}: peak_alloc={mem['peak_allocated_mb']:.0f}MB, "
                            f"peak_reserved={mem['peak_reserved_mb']:.0f}MB ({pct_used:.1f}% VRAM)"
                        )
                        results["measurements"].append(
                            {
                                "net": net_name,
                                "num_channels": num_channels,
                                "image_size": image_size,
                                "batch_size": bs,
                                **mem,
                            }
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        torch.cuda.empty_cache()
                        print(f"    BS={bs:>6d}: OOM/Error!")
                        results["measurements"].append(
                            {
                                "net": net_name,
                                "num_channels": num_channels,
                                "image_size": image_size,
                                "batch_size": bs,
                                "oom": True,
                            }
                        )
                        break

                results["measurements"].append(
                    {
                        "net": net_name,
                        "num_channels": num_channels,
                        "image_size": image_size,
                        "estimated_bs_30pct_margin": estimated_bs_30,
                        "estimated_bs_20pct_margin": estimated_bs_20,
                        "estimated_bs_10pct_margin": estimated_bs_10,
                        "max_batch_size": max_bs,
                        "type": "summary",
                    }
                )

            del model
            torch.cuda.empty_cache()

    # Fit new coefficients from measurements
    print("\n\n=== Coefficient Fitting ===")
    measurements = [
        m for m in results["measurements"] if "batch_size" in m and "peak_reserved_mb" in m
    ]
    if measurements:
        _fit_and_report(measurements, results)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


def _fit_and_report(measurements, results):
    """Fit memory model coefficients from measurements and report."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for m in measurements:
        key = (m["net"], m["num_channels"], m["image_size"])
        grouped[key].append(m)

    fitted = {}
    for (net, nch, imsize), points in grouped.items():
        batch_sizes = np.array([p["batch_size"] for p in points])
        reserved = np.array([p["peak_reserved_mb"] for p in points])

        # Fit: reserved = a * bs * imsize^2 * nch + b * bs + c
        # Design matrix: [bs * imsize^2 * nch, bs, 1]
        X = np.column_stack(
            [
                batch_sizes * imsize * imsize * nch,
                batch_sizes,
                np.ones_like(batch_sizes),
            ]
        )
        # Least squares fit
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, reserved, rcond=None)
        a, b, c = coeffs

        r_squared = 1 - np.sum((reserved - X @ coeffs) ** 2) / np.sum(
            (reserved - reserved.mean()) ** 2
        )

        print(f"  {net} {nch}ch {imsize}px: a={a:.6f}, b={b:.4f}, c={c:.2f} (R²={r_squared:.6f})")

        # Compare with current coefficients
        current = MEMORY_COEFFICIENTS.get(net, MEMORY_COEFFICIENTS["efficientnet-lite0"])
        print(f"    Current: a={current['a']:.6f}, b={current['b']:.4f}, c={current['c']:.2f}")
        print(
            f"    Ratio (new/old): a={a / current['a']:.2f}x, b={b / current['b']:.2f}x, c={c / current['c']:.2f}x"
        )

        fitted[f"{net}_{nch}ch_{imsize}px"] = {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "r_squared": float(r_squared),
        }

    results["fitted_coefficients"] = fitted


if __name__ == "__main__":
    main()
