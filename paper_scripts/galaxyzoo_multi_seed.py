#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Multi-seed GalaxyZoo benchmark script for AnomalyMatch paper experiments.

This script runs GalaxyZoo experiments with multiple predetermined seeds to generate
robust statistical results with averages and standard deviations.

The script will:
1. Run GalaxyZoo experiments for each seed using the same configurations as create_results.py
2. Collect all results into a summary folder with statistics (mean ± std)
3. Create average astronomaly comparison plots with standard deviation shading

Usage:
    # Run with default seeds and directories
    python galaxyzoo_multi_seed.py

    # Custom output directory
    python galaxyzoo_multi_seed.py --output_dir my_results

    # Custom seeds
    python galaxyzoo_multi_seed.py --seeds 42 123 456

    # Custom input directory for datasets
    python galaxyzoo_multi_seed.py --input_dir /path/to/datasets

Output Structure:
    results_YYYYMMDD_HHMMSS/
    ├── seed_42/
    │   └── galaxyzoo/
    │       ├── zoo_class1_n200_ratio0.500/
    │       ├── zoo_class1_n400_ratio0.500/
    │       └── zoo_class1_n400_ratio0.015/
    ├── seed_123/
    │   └── ...
    ├── summary/
    │   ├── galaxyzoo_multi_seed_summary.csv
    │   └── galaxyzoo_results.csv (compatible with results_analysis.py)
    └── average_plots/
        ├── average_astronomaly_comparison_zoo_class1_n200_ratio0.500_iter{DEFAULT_TRAINING_RUNS}.pdf
        ├── average_astronomaly_comparison_zoo_class1_n400_ratio0.500_iter{DEFAULT_TRAINING_RUNS}.pdf
        └── average_astronomaly_comparison_zoo_class1_n400_ratio0.015_iter{DEFAULT_TRAINING_RUNS}.pdf
"""

import sys
import time
import datetime
import subprocess
import pickle
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Import plotting constants from paper_plots
sys.path.append(str(Path(__file__).parent))
try:
    from paper_plots import (
        BLUE,
        GREEN,
        ORANGE,
        PURPLE,
        RED,
        REFERENCE_LINE_COLOR,
        REFERENCE_LINE_STYLE,
        PERFECT_LINE_STYLE,
        DEFAULT_DPI,
    )
except ImportError:
    # Fallback colors if import fails
    BLUE = "#1f77b4"
    GREEN = "#2ca02c"
    ORANGE = "#ff7f0e"
    PURPLE = "#9467bd"
    RED = "#d62728"
    REFERENCE_LINE_COLOR = "gray"
    REFERENCE_LINE_STYLE = "--"
    PERFECT_LINE_STYLE = "-."
    DEFAULT_DPI = 300

# ========== CONFIGURATION ==========
# Predetermined seeds for reproducible results
GALAXYZOO_SEEDS = [42, 76032, 730, 83209, 13798, 4538, 5923, 99271, 3762]

# Default parameters (matching create_results.py)
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TRAINING_RUNS = 3
DEFAULT_TRAIN_ITERATIONS = 100
DEFAULT_N_MISLABELED = 20
DEFAULT_N_TO_LOAD = 10000
GALAXYZOO_N_SAMPLES = 500
GALAXYZOO_ANOMALY_RATIO = 0.05
GALAXYZOO_THRESHOLDS = [0.95, 0.9, 0.8, 0.7]
GALAXYZOO_CLASSES = [1]  # Only anomaly class since it's binary classification

# Other settings
SKIP_MOCK_UI = True
SAVE_LABELED_IMAGES = False

# Define specific configurations for GalaxyZoo (matching create_results.py)
GALAXYZOO_CONFIGS = [
    (200, 0.5),  # 200 samples with 50% anomaly ratio
    (400, 0.5),  # 400 samples with 50% anomaly ratio
    (400, 0.015),  # 400 samples with 1.5% anomaly ratio
]


def run_benchmark(args: List[str], log_file: Path) -> int:
    """Run paper_benchmark.py with the given arguments and log to file."""
    benchmark_script = "paper_benchmark.py"

    # Build command with all arguments
    cmd = [sys.executable, benchmark_script] + args

    # Add save_labeled_images flag if needed
    if SAVE_LABELED_IMAGES:
        cmd.append("--save_labeled_images")

    # Print command for logging
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")

    # Open log file for this run
    with open(log_file, "w") as log:
        # Write command to log
        log.write(f"Command: {cmd_str}\n\n")
        log.flush()

        # Run process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        # Wait for process to complete
        process.wait()

        return process.returncode


def run_single_seed_experiment(
    seed: int, output_dir: Path, input_dir: Path, plot_only=False
) -> Path:
    """Run GalaxyZoo experiments for a single seed."""
    print(f"\n======= Running GalaxyZoo Experiments with Seed {seed} =======")

    seed_output_dir = output_dir / f"seed_{seed}"
    zoo_output_dir = seed_output_dir / "galaxyzoo"
    zoo_output_dir.mkdir(parents=True, exist_ok=True)
    if not plot_only:
        # Test anomaly detection on Galaxy Zoo dataset (binary classification)
        for cls in GALAXYZOO_CLASSES:
            for n_samples, anomaly_ratio in GALAXYZOO_CONFIGS:
                experiment_name = f"zoo_class{cls}_n{n_samples}_ratio{anomaly_ratio:.3f}"
                print(f"\nRunning experiment: {experiment_name} (seed {seed})")
                exp_start_time = time.time()

                args = [
                    "--dataset",
                    "galaxyzoo",
                    "--anomaly_classes",
                    str(cls),
                    "--n_samples",
                    str(n_samples),
                    "--anomaly_ratio",
                    str(anomaly_ratio),
                    "--train_iterations",
                    str(DEFAULT_TRAIN_ITERATIONS),
                    "--n_mislabeled",
                    str(DEFAULT_N_MISLABELED),
                    "--output_dir",
                    str(zoo_output_dir / experiment_name),
                    "--input_dir",
                    str(input_dir),
                    "--seed",
                    str(seed),
                    "--size",
                    str(DEFAULT_IMAGE_SIZE),
                    "--n_to_load",
                    str(DEFAULT_N_TO_LOAD),
                    "--training_runs",
                    str(DEFAULT_TRAINING_RUNS),
                    "--create_galaxy_plots",
                ]

                if SKIP_MOCK_UI:
                    args.append("--skip_mock_ui")

                log_file = zoo_output_dir / f"{experiment_name}.log"
                return_code = run_benchmark(args, log_file)

                if return_code != 0:
                    print(
                        f"Warning: Experiment {experiment_name} with seed {seed} failed with return code {return_code}"
                    )

                exp_duration = time.time() - exp_start_time
                print(f"Experiment completed in {exp_duration:.2f} seconds")

    return seed_output_dir


def collect_results_from_seeds(output_dir: Path, seeds: List[int]) -> Dict[str, pd.DataFrame]:
    """Collect and organize results from all seed experiments."""
    print("\n======= Collecting Results from All Seeds =======")

    all_results = {}

    # For each configuration, collect results from all seeds
    for cls in GALAXYZOO_CLASSES:
        for n_samples, anomaly_ratio in GALAXYZOO_CONFIGS:
            config_key = f"zoo_class{cls}_n{n_samples}_ratio{anomaly_ratio:.3f}"
            config_results = []
            sub_config_key = (
                f"galaxyzoo_anomaly{cls}_n{n_samples-int(n_samples*anomaly_ratio)}"
                + f"_a{int(n_samples*anomaly_ratio)}"
            )
            for seed in seeds:
                seed_dir = output_dir / f"seed_{seed}" / "galaxyzoo" / config_key / sub_config_key
                summary_file = seed_dir / "results_summary.csv"

                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file)
                        df["seed"] = seed
                        df["config"] = config_key
                        config_results.append(df)
                        print(f"Loaded results for {config_key}, seed {seed}")
                    except Exception as e:
                        print(f"Error loading {summary_file}: {e}")
                else:
                    print(f"Warning: Results file not found: {summary_file}")

            if config_results:
                all_results[config_key] = pd.concat(config_results, ignore_index=True)
            else:
                print(f"No results found for configuration: {config_key}")

    return all_results


def create_summary_statistics(
    all_results: Dict[str, pd.DataFrame], summary_dir: Path
) -> pd.DataFrame:
    """Create summary statistics with means and standard deviations."""
    print("\n======= Creating Summary Statistics =======")

    summary_rows = []

    for config_key, df in all_results.items():
        if df.empty:
            continue

        # Calculate statistics for numeric columns
        numeric_cols = [
            "final_auroc",
            "final_auprc",
            "top_0.1pct_anomalies_found",
            "top_0.1pct_precision",
            "top_1.0pct_anomalies_found",
            "top_1.0pct_precision",
        ]

        stats = {}
        stats["config"] = config_key
        stats["dataset"] = "galaxyzoo"
        stats["n_seeds"] = len(df["seed"].unique())

        # Extract configuration details
        if "anomaly_class" in df.columns:
            stats["anomaly_class"] = df["anomaly_class"].iloc[0]

        # Calculate mean and std for each metric
        for col in numeric_cols:
            if col in df.columns:
                stats[f"{col}_mean"] = df[col].mean()
                stats[f"{col}_std"] = df[col].std()
                stats[col] = stats[f"{col}_mean"]  # For compatibility

        summary_rows.append(stats)

        # Print summary for this configuration
        print(f"\nConfiguration: {config_key}")
        print(f"  Number of seeds: {stats['n_seeds']}")
        for col in numeric_cols:
            if col in df.columns:
                print(f"  {col}: {stats[f'{col}_mean']:.4f} ± {stats[f'{col}_std']:.4f}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Save summary
    summary_file = summary_dir / "galaxyzoo_multi_seed_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to {summary_file}")

    # Also create a format compatible with results_analysis.py
    compatible_df = summary_df.copy()
    compatible_df = compatible_df.rename(columns={"config": "run_dir"})
    compatible_file = summary_dir / "galaxyzoo_results.csv"
    compatible_df.to_csv(compatible_file, index=False)
    print(f"Saved compatible results to {compatible_file}")

    return summary_df


def load_plot_data_from_seeds(
    output_dir: Path, seeds: List[int], iteration: int = DEFAULT_TRAINING_RUNS
) -> Dict[str, List[Dict]]:
    """Load plot data from all seeds for creating average plots."""
    print(f"\n======= Loading Plot Data from All Seeds (iteration {iteration}) =======")

    plot_data_by_config = {}

    for cls in GALAXYZOO_CLASSES:
        for n_samples, anomaly_ratio in GALAXYZOO_CONFIGS:
            config_key = f"zoo_class{cls}_n{n_samples}_ratio{anomaly_ratio:.3f}"
            sub_config_key = (
                f"galaxyzoo_anomaly{cls}_n{n_samples-int(n_samples*anomaly_ratio)}"
                + f"_a{int(n_samples*anomaly_ratio)}"
            )

            plot_data_list = []

            for seed in seeds:
                seed_dir = output_dir / f"seed_{seed}" / "galaxyzoo" / config_key / sub_config_key
                plot_data_dir = seed_dir / "plots" / "plot_data"
                plot_data_file = (
                    plot_data_dir / f"data_for_astronomaly_comparison_iter{iteration}.pkl"
                )

                if plot_data_file.exists():
                    try:
                        with open(plot_data_file, "rb") as f:
                            plot_data = pickle.load(f)
                        plot_data["seed"] = seed
                        plot_data_list.append(plot_data)
                        print(f"Loaded plot data for {config_key}, seed {seed}")
                    except Exception as e:
                        print(f"Error loading plot data from {plot_data_file}: {e}")
                else:
                    print(f"Warning: Plot data file not found: {plot_data_file}")

            if plot_data_list:
                plot_data_by_config[config_key] = plot_data_list
            else:
                print(f"No plot data found for configuration: {config_key}")

    return plot_data_by_config


def create_average_astronomaly_comparison_plot(
    plot_data_by_config: Dict[str, List[Dict]],
    output_dir: Path,
    iteration: int = DEFAULT_TRAINING_RUNS,
):
    """Create average astronomaly comparison plots with std deviation shading."""
    print("\n======= Creating Average Astronomaly Comparison Plots =======")

    plots_dir = output_dir / "average_plots"
    plots_dir.mkdir(exist_ok=True)

    # Load the Astronomaly Figure 5a reference data
    try:
        astronomaly_data = pd.read_csv(
            Path(__file__).parent / "AstronomalyFigure5a.csv", skiprows=[0]
        )
        astronomaly_x = astronomaly_data["xaxis"].values
        astronomaly_y = astronomaly_data["yaxis"].values
    except Exception as e:
        print(f"Error loading Astronomaly reference data: {e}")
        astronomaly_x = astronomaly_y = None

    for config_key, plot_data_list in plot_data_by_config.items():
        print(f"\nCreating average plot for {config_key}")

        # Create the figure
        plt.figure(figsize=(8, 8))

        # Plot perfect prediction curve
        plt.plot(
            np.linspace(0, 2000, 10),
            np.linspace(0, 2000, 10),
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            linewidth=4,
            label="1:1 limit",
        )

        # Plot lines for each threshold with average and std deviation
        colors = [BLUE, GREEN, ORANGE, PURPLE]
        thresholds = GALAXYZOO_THRESHOLDS

        for i, threshold in enumerate(thresholds):
            # Collect data for this threshold from all seeds
            all_anomalies_found = []

            for plot_data in plot_data_list:
                try:
                    # Extract the data for this seed
                    scores = np.array(plot_data["scores"])
                    filenames = plot_data["filenames"]
                    true_labels_df = plot_data["true_labels_df"]

                    # Create scores dataframe and merge with true labels
                    scores_df = pd.DataFrame({"filename": filenames, "score": scores})
                    merged_df = pd.merge(scores_df, true_labels_df, on="filename")

                    if len(merged_df) == 0:
                        continue

                    # Create binary labels based on threshold
                    merged_df["true_anomaly"] = (
                        merged_df["anomaly_score_raw"] >= threshold
                    ).astype(int)

                    # Sort by model scores
                    filtered_df = merged_df.sort_values("score", ascending=False).reset_index(
                        drop=True
                    )

                    # Calculate cumulative sum of anomalies found
                    filtered_df["cum_anomalies"] = filtered_df["true_anomaly"].cumsum()

                    # Create x-axis values and corresponding anomalies found
                    inspection_indices = np.arange(0, min(2000, len(filtered_df)))
                    anomalies_found = []
                    for idx in inspection_indices:
                        if idx == 0:
                            anomalies_found.append(0)
                        else:
                            anomalies_found.append(filtered_df.loc[idx - 1, "cum_anomalies"])

                    # Pad or truncate to ensure consistent length
                    if len(anomalies_found) < 2000:
                        # Pad with the last value
                        last_val = anomalies_found[-1] if anomalies_found else 0
                        anomalies_found.extend([last_val] * (2000 - len(anomalies_found)))
                    else:
                        anomalies_found = anomalies_found[:2000]

                    all_anomalies_found.append(anomalies_found)

                except Exception as e:
                    print(
                        f"Error processing plot data for seed {plot_data.get('seed', 'unknown')}: {e}"
                    )
                    continue

            if all_anomalies_found:
                # Convert to numpy array for easier calculation
                all_anomalies_found = np.array(all_anomalies_found)

                # Calculate mean and std
                mean_anomalies = np.mean(all_anomalies_found, axis=0)
                std_anomalies = np.std(all_anomalies_found, axis=0)

                x_values = np.arange(2000)

                # Plot the mean line
                plt.plot(
                    x_values,
                    mean_anomalies,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=f"AnomalyMatch (t={threshold})",
                )

                # Add std deviation shading
                plt.fill_between(
                    x_values,
                    mean_anomalies - std_anomalies,
                    mean_anomalies + std_anomalies,
                    color=colors[i % len(colors)],
                    alpha=0.3,
                )

        # Plot Astronomaly reference curve if available
        if astronomaly_x is not None and astronomaly_y is not None:
            plt.plot(
                astronomaly_x,
                astronomaly_y,
                color=RED,
                linestyle=PERFECT_LINE_STYLE,
                linewidth=2,
                label="Astronomaly (t=0.9)",
            )

        # Add labels and legend
        plt.xlabel("Index in ranked list")
        plt.ylabel("Number of anomalies detected")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right", frameon=True, framealpha=0.7)

        # Set axis limits
        plt.xlim([0, 2000])
        plt.ylim([0, 300])

        # Add title with configuration info
        # plt.title(
        #    f"Astronomaly Comparison - {config_key}\n(Average over {len(plot_data_list)} seeds)"
        # )
        plt.figtext(
            0.5,
            -0.05,  # (x, y) position in figure coordinates (0.5 = centered, -0.05 = below axis)
            f"(Average over {len(plot_data_list)} seeds)",
            ha="center",
            va="top",
            fontsize=10,
        )
        # Save figure
        output_path = (
            plots_dir / f"average_astronomaly_comparison_{config_key}_cycle{iteration}.pdf"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=DEFAULT_DPI)
        plt.close()

        print(f"Average plot saved to {output_path}")


def main():
    """Main function to run multi-seed GalaxyZoo experiments."""
    parser = argparse.ArgumentParser(description="Run GalaxyZoo experiments with multiple seeds")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="galaxyzoo_multi_seed_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets",
        help="Input directory containing prepared datasets",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=GALAXYZOO_SEEDS,
        help="List of seeds to use for experiments",
    )
    # add --plot only argument that when set wil set plot_only in run_single_seed_experiment to True
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only create plots from existing results without running experiments",
    )
    # add argument to include a timestamp string
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Enforce a timestamp/description in the output (sub)directory name",
    )

    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Create timestamped output directory
    timestamp = args.timestamp  # datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting GalaxyZoo multi-seed experiments")
    print(f"Seeds: {args.seeds}")
    print(f"Output directory: {output_dir}")
    print(f"Input directory: {input_dir}")

    start_time = time.time()

    # Run experiments for each seed
    for i, seed in enumerate(args.seeds):
        print(f"\n{'='*60}")
        print(f"Running experiments for seed {seed} ({i+1}/{len(args.seeds)})")
        print(f"{'='*60}")

        try:
            seed_output_dir = run_single_seed_experiment(
                seed, output_dir, input_dir, args.plot_only
            )
            print(f"Completed experiments for seed {seed} -> saved at {seed_output_dir}")
        except Exception as e:
            print(f"Error running experiments for seed {seed}: {e}")
            continue

    # Collect and analyze results
    print(f"\n{'='*60}")
    print("Collecting and analyzing results from all seeds")
    print(f"{'='*60}")

    # Create summary directory
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Collect results
    all_results = collect_results_from_seeds(output_dir, args.seeds)

    if not all_results:
        print("No results found from any seed experiments!")
        return 1

    # Create summary statistics
    _ = create_summary_statistics(all_results, summary_dir)
    # summary_df
    # Load plot data and create average plots
    plot_data_by_config = load_plot_data_from_seeds(output_dir, args.seeds)

    if plot_data_by_config:
        create_average_astronomaly_comparison_plot(plot_data_by_config, output_dir)
    else:
        print("Warning: No plot data found for creating average plots")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Seeds processed: {len(args.seeds)}")
    print(f"Configurations per seed: {len(GALAXYZOO_CONFIGS)}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary statistics: {summary_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
