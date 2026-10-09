#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Benchmark suite for AnomalyMatch prediction pipeline via Cutana streaming.

Tests different channel configurations (1ch, 3ch, 4ch), resolutions (150x150,
224x224), normalisation methods, and VIS-only vs full multi-band catalogues.

Collects both AnomalyMatch PredictionProfiler metrics and Cutana's internal
PerformanceProfiler data (PERFORMANCE_DATA from subprocess stderr logs).

Usage:
    python scripts/benchmark_prediction.py [OPTIONS]

    --catalogue PATH      Cutana catalogue directory (default: Q1 search catalogues)
    --model PATH          Model checkpoint path
    --sources N           Number of sources per benchmark (default: 10000)
    --output-dir PATH     Output directory for results (default: benchmarking_results/)
    --configs CONFIG      Comma-separated config names to run, or "all" (default: all)
    --list-configs        List available configurations and exit
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anomaly_match as am
from anomaly_match.datasets.cutana_source import build_cutana_orchestrator_config
from anomaly_match.utils.prediction_profiler import PredictionProfiler
from prediction_utils import (
    clear_gpu_cache_if_needed,
    cutana_batch_to_model_tensor,
    load_model,
    process_batch_predictions,
)

# Default paths
DEFAULT_CATALOGUE = "/media/team_workspaces/AnomalyMatch-IDR1-Search/source_cats_q1/"
DEFAULT_MODEL = (
    "/media/team_workspaces/AnomalyMatch-IDR1-Search/q1_test_run/"
    "anomaly_match_results/sessions/Q1_search_20260211_092428/model_iteration_2.pth"
)


def build_configurations():
    """Build the matrix of benchmark configurations.

    Returns:
        dict: name → config dict for each benchmark scenario.
    """
    configs = {}

    # Channel configurations
    channel_setups = {
        "1ch": {
            "n_output_channels": 1,
            "fits_extension": [0],
            "channel_combination": None,
            "description": "VIS-only (1 channel)",
        },
        "3ch": {
            "n_output_channels": 3,
            "fits_extension": [0, 1, 2, 3],
            "channel_combination": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            "description": "VIS+NIR-H+NIR-Y → 3ch (drop NIR-J)",
        },
        "4ch": {
            "n_output_channels": 4,
            "fits_extension": [0, 1, 2, 3],
            "channel_combination": np.eye(4),
            "description": "VIS+NIR-H+NIR-Y+NIR-J → 4ch (identity)",
        },
    }

    resolutions = {
        "150": [150, 150],
        "224": [224, 224],
    }

    normalisation_methods = {
        "LOG": am.NormalisationMethod.LOG,
        "CONVERSION_ONLY": am.NormalisationMethod.CONVERSION_ONLY,
    }

    # Build full matrix
    for ch_name, ch_cfg in channel_setups.items():
        for res_name, res_size in resolutions.items():
            for norm_name, norm_method in normalisation_methods.items():
                name = f"{ch_name}_{res_name}_{norm_name}"
                configs[name] = {
                    "name": name,
                    "description": f"{ch_cfg['description']}, {res_name}x{res_name}, {norm_name}",
                    "n_output_channels": ch_cfg["n_output_channels"],
                    "fits_extension": ch_cfg["fits_extension"],
                    "channel_combination": ch_cfg["channel_combination"],
                    "image_size": res_size,
                    "normalisation_method": norm_method,
                    "vis_only": ch_name == "1ch",
                }

    return configs


def create_vis_only_catalogue(source_catalogue_dir, output_dir, max_sources=None):
    """Create a VIS-only catalogue from multi-band catalogue.

    Rewrites fits_file_paths to keep only the VIS entry.

    Returns:
        str: Path to the VIS-only parquet file (not directory).
    """
    import ast

    import pandas as pd

    cat_files = sorted(glob(os.path.join(source_catalogue_dir, "*.parquet")))
    if not cat_files:
        raise FileNotFoundError(f"No parquet files found in {source_catalogue_dir}")

    vis_cat_dir = os.path.join(output_dir, "vis_only_catalogue")
    os.makedirs(vis_cat_dir, exist_ok=True)

    # Just use the first catalogue file
    df = pd.read_parquet(cat_files[0])
    if max_sources and len(df) > max_sources:
        df = df.head(max_sources)

    # Keep only VIS paths
    def keep_vis_only(paths_str):
        if isinstance(paths_str, list):
            paths = paths_str
        elif isinstance(paths_str, str) and paths_str.startswith("["):
            paths = ast.literal_eval(paths_str)
        else:
            paths = [paths_str]
        vis_paths = [p for p in paths if "VIS" in p]
        if not vis_paths:
            return str(paths[:1])
        return str(vis_paths)

    df["fits_file_paths"] = df["fits_file_paths"].apply(keep_vis_only)

    out_path = os.path.join(vis_cat_dir, "vis_only_catalogue.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Created VIS-only catalogue: {out_path} ({len(df)} sources)")
    return out_path


def prepare_catalogue(source_catalogue_dir, output_dir, max_sources, vis_only=False):
    """Prepare the catalogue for a benchmark run.

    For VIS-only, creates a filtered catalogue file.
    Otherwise creates a truncated copy from the first parquet file.

    Returns:
        str: Path to the prepared parquet file (not directory).
    """
    import pandas as pd

    if vis_only:
        return create_vis_only_catalogue(source_catalogue_dir, output_dir, max_sources)

    cat_files = sorted(glob(os.path.join(source_catalogue_dir, "*.parquet")))
    if not cat_files:
        raise FileNotFoundError(f"No parquet files found in {source_catalogue_dir}")

    # Use first file, truncated to max_sources
    sub_cat_dir = os.path.join(output_dir, "benchmark_catalogue")
    os.makedirs(sub_cat_dir, exist_ok=True)

    df = pd.read_parquet(cat_files[0])
    if max_sources and len(df) > max_sources:
        df = df.head(max_sources)

    out_path = os.path.join(sub_cat_dir, "benchmark_catalogue.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Prepared catalogue: {out_path} ({len(df)} sources)")
    return out_path


def build_am_config(bench_cfg, model_path, catalogue_dir, output_dir):
    """Build an AnomalyMatch config for a benchmark run."""
    cfg = am.get_default_cfg()
    cfg.name = f"benchmark_{bench_cfg['name']}"
    cfg.model_path = model_path
    cfg.prediction_search_dir = catalogue_dir
    cfg.output_dir = output_dir
    cfg.normalisation.image_size = bench_cfg["image_size"]
    cfg.normalisation.n_output_channels = bench_cfg["n_output_channels"]
    cfg.num_channels = bench_cfg["n_output_channels"]
    cfg.normalisation.normalisation_method = bench_cfg["normalisation_method"]
    cfg.normalisation.fits_extension = bench_cfg["fits_extension"]
    cfg.normalisation.channel_combination = bench_cfg["channel_combination"]
    cfg.subprocess_buffer_size = 50000

    # ASINH params sized for n_output_channels
    n_ch = bench_cfg["n_output_channels"]
    cfg.normalisation.norm_asinh_scale = [0.7] * n_ch
    cfg.normalisation.norm_asinh_clip = [99.8] * n_ch

    return cfg


def collect_cutana_perf_data(cutana_output_dir):
    """Parse Cutana's PERFORMANCE_DATA from subprocess stderr logs.

    Returns:
        dict: Aggregated cutana profiler statistics.
    """
    log_dir = os.path.join(cutana_output_dir, "logs", "subprocesses")
    if not os.path.isdir(log_dir):
        return {}

    stderr_files = glob(os.path.join(log_dir, "*_stderr.log"))
    if not stderr_files:
        return {}

    aggregate = {
        "total_processes": len(stderr_files),
        "steps": {},
        "total_sources_processed": 0,
        "total_runtime": 0.0,
    }

    for stderr_file in stderr_files:
        try:
            with open(stderr_file) as f:
                for line in f:
                    if "PERFORMANCE_DATA:" not in line:
                        continue
                    json_str = line.split("PERFORMANCE_DATA:", 1)[1].strip()
                    perf = json.loads(json_str)
                    if perf.get("type") != "performance_summary":
                        continue
                    for step_name, step_data in perf.get("steps", {}).items():
                        if step_name not in aggregate["steps"]:
                            aggregate["steps"][step_name] = {
                                "times": [],
                                "count": 0,
                                "total_time": 0.0,
                            }
                        total_t = step_data.get("total_time", 0)
                        count = step_data.get("count", 0)
                        if total_t > 0:
                            aggregate["steps"][step_name]["times"].append(total_t)
                            aggregate["steps"][step_name]["count"] += count
                            aggregate["steps"][step_name]["total_time"] += total_t
                    aggregate["total_sources_processed"] += perf.get("total_sources", 0)
                    aggregate["total_runtime"] += perf.get("total_runtime", 0)
        except Exception:
            continue

    return aggregate


def run_single_benchmark(bench_cfg, cfg, catalogue_path, output_dir, batch_size=904):
    """Run a single benchmark configuration and return results."""
    import cutana
    from tqdm import tqdm

    print(f"\n{'=' * 70}")
    print(f"  Running: {bench_cfg['name']}")
    print(f"  {bench_cfg['description']}")
    print(f"{'=' * 70}")

    run_output_dir = os.path.join(output_dir, bench_cfg["name"])
    os.makedirs(run_output_dir, exist_ok=True)
    cfg.output_dir = run_output_dir

    # Setup cutana config
    cutana_config = build_cutana_orchestrator_config(catalogue_path, cfg)

    # Initialize cutana
    start_init = time.perf_counter()
    cutana_orchestrator = cutana.StreamingOrchestrator(cutana_config)
    cutana_orchestrator.init_streaming(batch_size=batch_size, write_to_disk=False)
    init_time = time.perf_counter() - start_init

    batches_count = cutana_orchestrator.get_batch_count()
    print(f"  Cutana initialised in {init_time:.1f}s, {batches_count} batches")

    # Load model
    model = load_model(cfg)
    model.eval()

    # Setup AM profiler
    profiler = PredictionProfiler(output_dir=run_output_dir, process_idx=0)

    # Process batches
    scores_list = []
    imgs_list = []
    num_images = 0

    wall_start = time.perf_counter()

    for batch_idx in tqdm(range(batches_count), desc=f"  {bench_cfg['name']}"):
        with profiler.stage("io_load"):
            loaded_batch = cutana_orchestrator.next_batch()
            batch_data = loaded_batch["cutouts"]

        if isinstance(batch_data, list):
            if len(batch_data) == 0:
                continue
            batch_data = np.array(batch_data)

        batch_size_actual = batch_data.shape[0]
        num_images += batch_size_actual

        with profiler.stage("preprocess"):
            images = cutana_batch_to_model_tensor(batch_data, cfg)

        with profiler.stage("inference"):
            batch_scores, batch_imgs = process_batch_predictions(model, images)
            del images

        profiler.end_batch(batch_size=batch_size_actual)
        scores_list.append(batch_scores)
        imgs_list.append(batch_imgs)
        clear_gpu_cache_if_needed(batch_idx)

    wall_elapsed = time.perf_counter() - wall_start

    cutana_orchestrator.cleanup()

    # Save AM profiler report
    profiler.save_partial_report()

    # Read back the partial report
    partial_path = os.path.join(
        run_output_dir,
        PredictionProfiler.PARTIAL_FILENAME_TEMPLATE.format(0),
    )
    with open(partial_path) as f:
        am_report = json.load(f)

    # Collect cutana's internal profiling data
    cutana_output_dirs = glob(os.path.join(run_output_dir, "*/"))
    cutana_perf = {}
    for d in cutana_output_dirs:
        perf = collect_cutana_perf_data(d)
        if perf:
            cutana_perf = perf
            break
    # Also check if cutana wrote logs in the main output dir
    if not cutana_perf:
        cutana_perf = collect_cutana_perf_data(run_output_dir)

    # Build result
    result = {
        "config_name": bench_cfg["name"],
        "description": bench_cfg["description"],
        "total_images": num_images,
        "wall_clock_s": round(wall_elapsed, 2),
        "throughput_images_per_sec": round(num_images / wall_elapsed, 2) if wall_elapsed > 0 else 0,
        "init_time_s": round(init_time, 2),
        "resolution": bench_cfg["image_size"],
        "n_channels": bench_cfg["n_output_channels"],
        "normalisation": str(bench_cfg["normalisation_method"]),
        "vis_only": bench_cfg.get("vis_only", False),
        "am_profiler": am_report,
        "cutana_profiler": cutana_perf,
        "batch_size": batch_size,
        "num_batches": batches_count,
    }

    throughput = result["throughput_images_per_sec"]
    print(f"  Result: {num_images} images in {wall_elapsed:.1f}s = {throughput:.1f} imgs/s")

    # Print stage breakdown
    if am_report.get("stages"):
        print("  Stage breakdown (AM profiler):")
        for stage_name, stage_data in am_report["stages"].items():
            print(
                f"    {stage_name}: {stage_data['total_s']:.1f}s ({stage_data['percentage']:.1f}%)"
            )

    if cutana_perf.get("steps"):
        print("  Cutana internal breakdown:")
        for step_name, step_data in cutana_perf["steps"].items():
            if step_data.get("total_time", 0) > 0:
                print(f"    {step_name}: {step_data['total_time']:.1f}s")

    return result


def generate_projection_table(results, target_sources=1_000_000_000, target_files=4000):
    """Generate a projection table for processing 1B sources over 4000 FITS files."""
    print(f"\n{'=' * 90}")
    print(f"  Projection: {target_sources:,} sources across {target_files} FITS files")
    print(f"{'=' * 90}")

    header = f"{'Configuration':<35} {'imgs/s':>8} {'Hours':>8} {'Days':>6} {'GPU-hrs':>8}"
    print(header)
    print("-" * 90)

    projections = []
    for r in results:
        throughput = r["throughput_images_per_sec"]
        if throughput <= 0:
            continue
        total_seconds = target_sources / throughput
        hours = total_seconds / 3600
        days = hours / 24
        # Assume 1 GPU per process
        gpu_hours = hours

        row = {
            "config": r["config_name"],
            "description": r["description"],
            "throughput": throughput,
            "hours": round(hours, 1),
            "days": round(days, 1),
            "gpu_hours": round(gpu_hours, 1),
        }
        projections.append(row)
        print(
            f"{r['config_name']:<35} {throughput:>8.1f} {hours:>8.1f} {days:>6.1f} {gpu_hours:>8.1f}"
        )

    print("-" * 90)
    return projections


def generate_charts(results, output_dir):
    """Generate performance analysis charts."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping charts")
        return

    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Throughput comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [r["config_name"] for r in results]
    throughputs = [r["throughput_images_per_sec"] for r in results]
    colors = []
    for name in names:
        if "1ch" in name:
            colors.append("#4CAF50")
        elif "3ch" in name:
            colors.append("#2196F3")
        else:
            colors.append("#FF9800")

    bars = ax.bar(range(len(names)), throughputs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Images / second")
    ax.set_title("Prediction Throughput by Configuration")
    for bar, val in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "throughput_comparison.png"), dpi=150)
    plt.close()

    # 2. Stage breakdown pie charts (one per config)
    for r in results:
        am_stages = r.get("am_profiler", {}).get("stages", {})
        if not am_stages:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        stage_names = list(am_stages.keys())
        stage_times = [am_stages[s]["total_s"] for s in stage_names]
        stage_colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]
        ax.pie(
            stage_times,
            labels=stage_names,
            autopct="%1.1f%%",
            colors=stage_colors[: len(stage_names)],
            startangle=90,
        )
        ax.set_title(f"Time Distribution: {r['config_name']}")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"stages_{r['config_name']}.png"), dpi=150)
        plt.close()

    # 3. Stacked bar chart comparing all configs
    fig, ax = plt.subplots(figsize=(14, 7))
    all_stages = set()
    for r in results:
        all_stages.update(r.get("am_profiler", {}).get("stages", {}).keys())
    all_stages = sorted(all_stages)
    stage_colors = {
        "io_load": "#ff9999",
        "preprocess": "#66b3ff",
        "inference": "#99ff99",
        "result_save": "#ffcc99",
    }

    x = range(len(results))
    bottom = np.zeros(len(results))
    for stage in all_stages:
        values = []
        for r in results:
            stages = r.get("am_profiler", {}).get("stages", {})
            values.append(stages.get(stage, {}).get("total_s", 0))
        color = stage_colors.get(stage, "#cccccc")
        ax.bar(x, values, bottom=bottom, label=stage, color=color, alpha=0.8)
        bottom += np.array(values)

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r["config_name"] for r in results], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Stage Time Breakdown by Configuration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "stacked_stages.png"), dpi=150)
    plt.close()

    # 4. Resolution impact comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, ch in enumerate(["4ch", "1ch"]):
        ax = axes[ax_idx]
        ch_results = [r for r in results if ch in r["config_name"]]
        if not ch_results:
            continue
        for r in ch_results:
            am_stages = r.get("am_profiler", {}).get("stages", {})
            if am_stages:
                stage_names = list(am_stages.keys())
                stage_times = [am_stages[s]["total_s"] for s in stage_names]
                ax.bar(
                    [f"{r['config_name'].split('_')[1]}\n{r['config_name'].split('_')[2]}"],
                    [sum(stage_times)],
                    color=stage_colors.get("io_load", "#cccccc"),
                    alpha=0.8,
                )
        ax.set_ylabel("Total Time (s)")
        ax.set_title(f"{ch} - Resolution & Norm Impact")

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "resolution_impact.png"), dpi=150)
    plt.close()

    print(f"  Charts saved to {charts_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AnomalyMatch prediction pipeline")
    parser.add_argument(
        "--catalogue",
        default=DEFAULT_CATALOGUE,
        help="Path to cutana catalogue directory",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--sources",
        type=int,
        default=10000,
        help="Number of sources per benchmark run",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarking_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--configs",
        default="all",
        help="Comma-separated config names to run, or 'all'",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=904,
        help="Batch size for prediction",
    )
    args = parser.parse_args()

    all_configs = build_configurations()

    if args.list_configs:
        print("Available benchmark configurations:")
        for name, cfg in all_configs.items():
            print(f"  {name:<35} {cfg['description']}")
        return

    # Select configs to run
    if args.configs == "all":
        selected = all_configs
    else:
        selected_names = [n.strip() for n in args.configs.split(",")]
        selected = {n: all_configs[n] for n in selected_names if n in all_configs}
        missing = [n for n in selected_names if n not in all_configs]
        if missing:
            print(f"Warning: unknown configs: {missing}")

    if not selected:
        print("No configurations selected. Use --list-configs to see options.")
        return

    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("AnomalyMatch Prediction Benchmark Suite")
    print(f"  Catalogue: {args.catalogue}")
    print(f"  Model: {args.model}")
    print(f"  Sources per run: {args.sources}")
    print(f"  Output: {output_dir}")
    print(f"  Configs: {len(selected)} selected")

    # Suppress AM warnings for cleaner output
    import warnings

    warnings.filterwarnings("ignore", message="Image maximum is not larger than minimum")
    am.set_log_level("warning", am.get_default_cfg())

    results = []
    for config_name, bench_cfg in selected.items():
        try:
            # Prepare catalogue
            cat_dir = prepare_catalogue(
                args.catalogue,
                output_dir,
                max_sources=args.sources,
                vis_only=bench_cfg.get("vis_only", False),
            )

            # Build AM config
            run_output = os.path.join(output_dir, config_name)
            cfg = build_am_config(bench_cfg, args.model, cat_dir, run_output)

            result = run_single_benchmark(
                bench_cfg, cfg, cat_dir, output_dir, batch_size=args.batch_size
            )
            results.append(result)

        except Exception as e:
            print(f"  FAILED: {config_name}: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "config_name": config_name,
                    "description": bench_cfg["description"],
                    "error": str(e),
                    "throughput_images_per_sec": 0,
                }
            )

    # Save all results
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate charts
    valid_results = [r for r in results if r.get("throughput_images_per_sec", 0) > 0]
    if valid_results:
        generate_charts(valid_results, output_dir)

    # Print projection table
    if valid_results:
        projections = generate_projection_table(valid_results)

        # Save projections
        proj_path = os.path.join(output_dir, "projections.json")
        with open(proj_path, "w") as f:
            json.dump(projections, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("  Summary")
    print(f"{'=' * 70}")
    for r in results:
        status = (
            f"{r['throughput_images_per_sec']:.1f} imgs/s"
            if r.get("throughput_images_per_sec")
            else "FAILED"
        )
        print(f"  {r['config_name']:<35} {status}")


if __name__ == "__main__":
    main()
