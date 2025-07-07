#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
# !/usr/bin/env python3
"""
Plot recreation script for AnomalyMatch benchmark results.

This script recreates plots from saved data and creates new comparative plots
for active learning runs.

Usage:
    python recreate_plots.py --results-dir /path/to/results

Author: GitHub Copilot
Date: April 14, 2025
"""

import os
import sys
import argparse
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

# Add parent directory to path to import paper_plots
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_scripts.paper_plots import (
    plot_score_histogram,
    plot_roc_prc_curves,
    plot_top_n_anomaly_detection,
    plot_metrics_over_time,
    plot_combined_anomaly_detection,
    plot_top_n_with_thresholds,
    plot_roc_with_thresholds,
    plot_astronomaly_comparison,
    plot_score_vs_user_score_grid,
    FONT_SCALE,
)
from paper_scripts.plot_colors import (
    BLUE,
    ORANGE,
    PURPLE,
    PERFECT_LINE_COLOR,
    PERFECT_LINE_STYLE,
    PERFECT_LINE_ALPHA,
    VLINE_COLOR,
    VLINE_STYLE,
    VLINE_ALPHA,
)

# Set matplotlib parameters for publication-quality plots (similar to paper_plots.py)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12 * FONT_SCALE,
        "axes.titlesize": 16 * FONT_SCALE,
        "axes.labelsize": 12 * FONT_SCALE,
        "xtick.labelsize": 11 * FONT_SCALE,
        "ytick.labelsize": 11 * FONT_SCALE,
        "legend.fontsize": 10 * FONT_SCALE,
        "figure.figsize": (8, 8),  # Square figures
        "figure.dpi": 300,
        "savefig.dpi": 300,  # High-res output for publications
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    }
)

# Use a publication-friendly style
sns.set_style("whitegrid")

# Default DPI for saving figures
DEFAULT_DPI = 600


def recreate_plots_from_data(results_dir, plot_type=None, custom_thresholds=None):
    """
    Recreate plots from saved data in the results directory.

    Args:
        results_dir (str): Path to the results directory
        plot_type (str, optional): If provided, only recreate plots of this type
        custom_thresholds (list, optional): If provided, use these thresholds instead of the ones in the plot data
    """
    results_dir = Path(results_dir)

    # Find all plot data files
    all_plot_files = []
    for plot_data_dir in glob.glob(str(results_dir / "**" / "plot_data"), recursive=True):
        plot_files = glob.glob(os.path.join(plot_data_dir, "*.pkl"))
        all_plot_files.extend(plot_files)

    logger.info(f"Found {len(all_plot_files)} plot data files")

    # Group files by plot type
    plot_type_files = {}
    for file_path in all_plot_files:
        file_name = os.path.basename(file_path)

        # Determine plot type from filename
        if "score_histogram" in file_name:
            plot_type_name = "score_histogram"
        elif "roc_prc_curves" in file_name:
            plot_type_name = "roc_prc_curves"
        elif "top_n_anomaly_detection" in file_name:
            plot_type_name = "top_n_anomaly_detection"
        elif "top_mispredicted" in file_name:
            plot_type_name = "top_mispredicted"
        elif "metrics_over_time" in file_name:
            plot_type_name = "metrics_over_time"
        elif "combined_anomaly_detection" in file_name:
            plot_type_name = "combined_anomaly_detection"
        elif "top_n_with_thresholds" in file_name:
            plot_type_name = "top_n_with_thresholds"
        elif "roc_with_thresholds" in file_name:
            plot_type_name = "roc_with_thresholds"
        elif "astronomaly_comparison" in file_name:
            plot_type_name = "astronomaly_comparison"
        elif "score_vs_user_score_grid" in file_name:
            plot_type_name = "score_vs_user_score_grid"
        else:
            plot_type_name = "unknown"

        if plot_type is None or plot_type == plot_type_name:
            if plot_type_name not in plot_type_files:
                plot_type_files[plot_type_name] = []
            plot_type_files[plot_type_name].append(file_path)

    # Process files by plot type
    for plot_type_name, files in plot_type_files.items():
        logger.info(f"Processing {len(files)} files of type {plot_type_name}")

        for file_path in files:
            try:
                # Load plot data
                with open(file_path, "rb") as f:
                    plot_data = pickle.load(f)

                # Determine output directory
                output_dir = Path(os.path.dirname(os.path.dirname(file_path)))
                plots_dir = output_dir / "plots_recreated"
                plots_dir.mkdir(parents=True, exist_ok=True)

                # Extract iteration number if present
                iteration = 0
                if "iteration" in plot_data:
                    iteration = plot_data["iteration"]
                elif "iter" in os.path.basename(file_path):
                    iteration = int(os.path.basename(file_path).split("iter")[1].split(".")[0])

                # Recreate plot based on type
                if plot_type_name == "score_histogram":
                    logger.info(f"Recreating score histogram for iteration {iteration}")
                    plot_score_histogram(
                        plot_data["anomaly_scores"],
                        plot_data["normal_scores"],
                        iteration,
                        plots_dir,
                    )
                elif plot_type_name == "roc_prc_curves":
                    logger.info(f"Recreating ROC/PRC curves for iteration {iteration}")
                    # The function expects a metrics dictionary
                    plot_roc_prc_curves(plot_data, iteration, plots_dir)
                elif plot_type_name == "top_n_anomaly_detection":
                    logger.info(f"Recreating top-N anomaly detection for iteration {iteration}")
                    plot_top_n_anomaly_detection(
                        plot_data["scores"],
                        plot_data["filenames"],
                        plot_data["true_labels_df"],
                        plot_data["anomaly_class"],
                        iteration,
                        plots_dir,
                    )
                elif plot_type_name == "metrics_over_time":
                    logger.info("Recreating metrics over time")
                    plot_metrics_over_time(
                        plot_data["metrics_history"],
                        plots_dir,
                        batch_size=plot_data.get("batch_size"),
                    )
                elif plot_type_name == "combined_anomaly_detection":
                    logger.info("Recreating combined anomaly detection")
                    plot_combined_anomaly_detection(
                        plot_data["detection_curves"],
                        plots_dir,
                        plot_data.get("anomaly_prevalence", None),
                    )
                elif plot_type_name == "top_n_with_thresholds":
                    logger.info(f"Recreating top-N with thresholds for iteration {iteration}")
                    # Use custom thresholds if provided, otherwise use the ones from plot data
                    thresholds_to_use = (
                        custom_thresholds
                        if custom_thresholds is not None
                        else plot_data["thresholds"]
                    )
                    if custom_thresholds is not None:
                        logger.info(f"Using custom thresholds: {thresholds_to_use}")
                    plot_top_n_with_thresholds(
                        plot_data["scores"],
                        plot_data["filenames"],
                        plot_data["true_labels_df"],
                        iteration,
                        plots_dir,
                        thresholds=thresholds_to_use,
                    )
                elif plot_type_name == "roc_with_thresholds":
                    logger.info(f"Recreating ROC with thresholds for iteration {iteration}")
                    # Extract necessary parameters from the plot data
                    if (
                        "scores" in plot_data
                        and "filenames" in plot_data
                        and "true_labels_df" in plot_data
                        and "anomaly_class" in plot_data
                    ):
                        # Use custom thresholds if provided, otherwise use the ones from plot data
                        thresholds_to_use = (
                            custom_thresholds
                            if custom_thresholds is not None
                            else plot_data["thresholds"]
                        )
                        if custom_thresholds is not None:
                            logger.info(f"Using custom thresholds: {thresholds_to_use}")
                        plot_roc_with_thresholds(
                            plot_data["scores"],
                            plot_data["filenames"],
                            plot_data["true_labels_df"],
                            plot_data["anomaly_class"],
                            iteration,
                            plots_dir,
                            thresholds=thresholds_to_use,
                        )
                    else:
                        logger.warning(
                            f"Missing required data for roc_with_thresholds in {file_path}"
                        )
                elif plot_type_name == "astronomaly_comparison":
                    logger.info(f"Recreating Astronomaly comparison for iteration {iteration}")
                    # Extract necessary parameters from the plot data

                    if (
                        "scores" in plot_data
                        and "filenames" in plot_data
                        and "true_labels_df" in plot_data
                        and "anomaly_class" in plot_data
                    ):
                        thresholds_to_use = (
                            custom_thresholds
                            if custom_thresholds is not None
                            else plot_data["thresholds"]
                        )
                        if custom_thresholds is not None:
                            logger.info(f"Using custom thresholds: {thresholds_to_use}")
                        plot_astronomaly_comparison(
                            plot_data["scores"],
                            plot_data["filenames"],
                            plot_data["true_labels_df"],
                            plot_data["anomaly_class"],
                            iteration,
                            plots_dir,
                            thresholds=thresholds_to_use,
                        )
                    else:
                        logger.warning(
                            f"Missing required data for astronomaly_comparison in {file_path}"
                        )
                elif plot_type_name == "score_vs_user_score_grid":
                    logger.info(f"Recreating score vs user score grid for iteration {iteration}")
                    if (
                        "scores" in plot_data
                        and "filenames" in plot_data
                        and "true_labels_df" in plot_data
                    ):
                        plot_score_vs_user_score_grid(
                            plot_data["scores"],
                            plot_data["filenames"],
                            plot_data["true_labels_df"],
                            iteration,
                            plots_dir,
                            plot_data.get("data_dir", ""),
                        )
                    else:
                        logger.warning(
                            f"Missing required data for score_vs_user_score_grid in {file_path}"
                        )
                elif plot_type_name == "top_mispredicted":
                    logger.info(
                        f"Skipping top_mispredicted for iteration {iteration} (needs image data)"
                    )
                else:
                    logger.warning(f"Unknown plot type: {plot_type_name}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info("Plot recreation completed")


def plot_active_learning_comparison(results_dir):
    """
    Create a comparative top-N anomaly detection plot for active learning experiments.

    Args:
        results_dir (str): Path to the results directory containing active learning experiments
    """
    results_dir = Path(results_dir)

    # Find active learning directories
    al_dir = results_dir / "active_learning"
    if not al_dir.exists():
        # Try finding an active_learning directory anywhere in the results
        al_dirs = list(results_dir.glob("**/active_learning"))
        if not al_dirs:
            logger.error("No active learning directory found")
            return
        al_dir = al_dirs[0]

    logger.info(f"Found active learning directory: {al_dir}")

    # Find the three experiment directories
    experiment_dirs = {"with_al": None, "without_al": None, "total_samples": None}

    # Look for directories matching the expected names
    for exp_name in experiment_dirs.keys():
        matching_dirs = list(al_dir.glob(f"**/*{exp_name}*"))
        if matching_dirs:
            experiment_dirs[exp_name] = matching_dirs[0]
            logger.info(f"Found {exp_name} directory: {matching_dirs[0]}")

    # Check if we have all three experiment types
    missing_experiments = [exp for exp, path in experiment_dirs.items() if path is None]
    if missing_experiments:
        logger.warning(
            f"Missing experiments: {missing_experiments}"
        )  # Find the final iteration data for each experiment
    iteration_data = {}
    for exp_name, exp_dir in experiment_dirs.items():
        if exp_dir is None:
            continue

        # Check if there's a subdirectory level (e.g., miniimagenet_anomaly48_n485_a35)
        subdirs = list(exp_dir.glob("**/"))
        if subdirs:
            # Find the first directory that looks like a dataset directory
            dataset_dirs = [
                d for d in subdirs if d.name.startswith(("miniimagenet", "galaxymnist"))
            ]
            if dataset_dirs:
                logger.info(f"Found dataset directory in {exp_name}: {dataset_dirs[0]}")
                exp_dir = dataset_dirs[0]

        # Find the highest iteration directory (could be directly in exp_dir or in subdirectories)
        iteration_dirs = sorted(
            list(exp_dir.glob("**/iteration_*")), key=lambda x: int(x.name.split("_")[1])
        )
        if not iteration_dirs:
            # Try to find plots directory with plot_data directly (without iteration directories)
            plot_dirs = list(exp_dir.glob("**/plots/plot_data"))
            if plot_dirs:
                # Use parent of plot_data dir as the "iteration" directory
                iteration_dirs = [plot_dirs[0].parent.parent]
                logger.info(f"Found plots directory without iterations: {iteration_dirs[0]}")
            else:
                logger.warning(f"No iteration directories found in {exp_dir}")
                continue

        final_iter_dir = iteration_dirs[-1]
        logger.info(f"Using final iteration from {exp_name}: {final_iter_dir}")

        # Find plot data for top_n_anomaly_detection
        plot_data_file = None

        # Look for plot_data directory in different possible locations
        plot_data_locations = [final_iter_dir / "plot_data", final_iter_dir / "plots" / "plot_data"]

        for plot_data_dir in plot_data_locations:
            if plot_data_dir.exists():
                logger.info(f"Found plot data directory: {plot_data_dir}")
                plot_data_files = list(plot_data_dir.glob("*top_n_anomaly_detection*.pkl"))
                if plot_data_files:
                    plot_data_file = plot_data_files[0]
                    break

        if plot_data_file is None:
            logger.warning(f"No plot data found for {exp_name}")
            continue

        # Load the plot data
        try:
            with open(plot_data_file, "rb") as f:
                plot_data = pickle.load(f)

            # Extract scores, filenames, etc.
            iteration_data[exp_name] = {
                "scores": plot_data["scores"],
                "filenames": plot_data["filenames"],
                "true_labels_df": plot_data["true_labels_df"],
                "anomaly_class": plot_data["anomaly_class"],
                "iteration": plot_data["iteration"],
            }
            logger.info(f"Loaded plot data for {exp_name}")
        except Exception as e:
            logger.error(f"Error loading plot data for {exp_name}: {e}")

    if not iteration_data:
        logger.error("No valid plot data found for any experiment")
        return

    # Create output directory for the comparative plot
    output_dir = al_dir / "comparative_plots"
    output_dir.mkdir(exist_ok=True)

    # Create the comparative plot
    create_active_learning_comparative_plot(iteration_data, output_dir)


def create_active_learning_comparative_plot(iteration_data, output_dir):
    """
    Create a comparative plot showing top-N anomaly detection curves for different active learning approaches.

    Args:
        iteration_data (dict): Dictionary with data for each experiment
        output_dir (Path): Directory to save the plot
    """
    # Get one of the datasets to determine true anomaly prevalence
    any_exp = next(iter(iteration_data.values()))
    true_labels_df = any_exp["true_labels_df"]
    anomaly_class = any_exp["anomaly_class"]

    # Calculate true anomaly prevalence in the dataset
    anomaly_count = len(true_labels_df[true_labels_df["label_idx"] == anomaly_class])
    total_count = len(true_labels_df)
    true_anomaly_prevalence = anomaly_count / total_count

    logger.info(f"Creating comparative plot with anomaly prevalence: {true_anomaly_prevalence:.4f}")

    # Create figure with log scale for x-axis
    plt.figure(figsize=(8, 8))

    # Colors and styles for different experiment types
    styles = {
        "with_al": {
            "color": BLUE,
            "linestyle": "-",
            "label": "With Active Learning",
            "zorder": 3,
        },
        "without_al": {
            "color": ORANGE,
            "linestyle": ":",  # Dotted
            "label": "Without Active Learning",
            "zorder": 2,
        },
        "total_samples": {
            "color": PURPLE,
            "linestyle": "-.",  # Dash-dot
            "label": "Total Samples from Start",
            "zorder": 1,
        },
    }

    # Process each experiment type
    for exp_name, data in iteration_data.items():
        # Calculate detection curves
        x_vals, y_vals = calculate_detection_curve(
            data["scores"], data["filenames"], data["true_labels_df"], data["anomaly_class"]
        )

        # Plot the curve with appropriate style
        style = styles.get(
            exp_name, {"color": "gray", "linestyle": ":", "label": exp_name, "zorder": 0}
        )
        plt.plot(x_vals, y_vals, linewidth=2, **style)

    # Add perfect detection line
    x_perfect = np.concatenate(
        [
            np.logspace(np.log10(0.0001), np.log10(0.1), 500),  # More points in the lower range
            np.linspace(0.1, 100, 500),  # Linear in higher range
        ]
    )
    x_perfect = np.unique(x_perfect)  # Remove duplicates

    # Calculate perfect detection line
    detection_rate_factor = 1.0 / true_anomaly_prevalence if true_anomaly_prevalence > 0 else 1.0
    y_perfect = np.minimum(x_perfect * detection_rate_factor, 100)

    # Plot perfect detection line
    plt.plot(
        x_perfect,
        y_perfect,
        color=PERFECT_LINE_COLOR,
        linestyle=PERFECT_LINE_STYLE,
        alpha=PERFECT_LINE_ALPHA,
        linewidth=1.5,
        label="Perfect detection",
        zorder=0,
    )

    # Add key inspection points if we have the "with_al" experiment
    if "with_al" in iteration_data:
        x_vals, y_vals = calculate_detection_curve(
            iteration_data["with_al"]["scores"],
            iteration_data["with_al"]["filenames"],
            iteration_data["with_al"]["true_labels_df"],
            iteration_data["with_al"]["anomaly_class"],
        )

        # Add vertical lines at key inspection points
        plt.axvline(x=0.1, color=VLINE_COLOR, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)
        plt.text(
            0.07,
            50,
            "0.1% inspected",
            rotation=90,
            va="bottom",
            fontsize=8 * FONT_SCALE,
        )

        plt.axvline(x=1.0, color=VLINE_COLOR, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)
        plt.text(
            0.7,
            50,
            "1% inspected",
            rotation=90,
            va="bottom",
            fontsize=8 * FONT_SCALE,
        )

    # Set up axes
    plt.xscale("log")
    plt.xlim(0.01, 100)
    plt.ylim(0, 100)

    # Add x-axis ticks for log scale
    plt.xticks(
        [0.01, 0.1, 1, 10, 100],
        ["0.01%", "0.1%", "1%", "10%", "100%"],
    )

    # Add labels and legend (no title)
    plt.xlabel("% of top-scoring predictions inspected")
    plt.ylabel("% of Total Anomalies Found")
    plt.grid(True)
    plt.legend(loc="lower right", frameon=True, framealpha=0.7)

    # Save the plot
    output_path = os.path.join(output_dir, "active_learning_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"Active learning comparison plot saved to {output_path}")


def calculate_detection_curve(scores, filenames, true_labels_df, anomaly_class):
    """
    Calculate the detection curve for top-N anomaly detection.

    Args:
        scores: Anomaly scores
        filenames: Corresponding filenames
        true_labels_df: DataFrame with ground truth labels
        anomaly_class: Index of the anomaly class

    Returns:
        tuple: (x_values, y_values) for plotting
    """
    # Create a DataFrame with scores and filenames
    pred_df = pd.DataFrame({"filename": filenames, "score": scores})

    # Merge with true labels
    merged_df = pd.merge(pred_df, true_labels_df, on="filename")

    # Create true binary labels (1 for anomaly class, 0 for others)
    merged_df["true_anomaly"] = (merged_df["label_idx"] == anomaly_class).astype(int)

    # Sort by score (highest to lowest)
    sorted_df = merged_df.sort_values("score", ascending=False).reset_index(drop=True)

    # Calculate total number of anomalies in the dataset
    total_anomalies = sorted_df["true_anomaly"].sum()

    # Calculate cumulative sum of anomalies found
    sorted_df["cum_anomalies"] = sorted_df["true_anomaly"].cumsum()

    # Calculate percentage of total anomalies found
    sorted_df["percent_anomalies_found"] = 100 * sorted_df["cum_anomalies"] / total_anomalies

    # Create x-axis values for number of inspected samples
    # Use more points in the low percentage range to avoid interpolation issues
    total_samples = len(sorted_df)

    # More points in the lower ranges of the log scale
    log_space_points = np.concatenate(
        [
            np.linspace(0.0001, 0.001, 20),  # 0.0001% to 0.001% (very fine-grained)
            np.linspace(0.001, 0.01, 30),  # 0.001% to 0.01% (more fine-grained)
            np.linspace(0.01, 0.1, 30),  # 0.01% to 0.1% (more fine-grained)
            np.linspace(0.1, 1, 20),  # 0.1% to 1% (fine-grained)
            np.linspace(1, 10, 10),  # 1% to 10%
            np.linspace(10, 100, 10),  # 10% to 100%
        ]
    )

    # Convert percentages to sample counts and ensure we get unique values
    inspection_points = np.unique(np.round((log_space_points * total_samples / 100)).astype(int))
    inspection_points = inspection_points[inspection_points > 0]  # Remove zero
    inspection_points = np.insert(inspection_points, 0, 0)  # Add zero at the beginning

    if total_samples not in inspection_points:
        inspection_points = np.append(inspection_points, total_samples)  # Add last point if needed

    # Calculate percentage of anomalies found at each point
    anomalies_found = []
    for i in inspection_points:
        if i == 0:
            anomalies_found.append(0)
        else:
            anomalies_found.append(
                sorted_df.loc[min(i - 1, len(sorted_df) - 1), "percent_anomalies_found"]
            )

    # Return x and y values for plotting
    x_values = inspection_points / total_samples * 100
    y_values = anomalies_found

    return x_values, y_values


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Recreate plots from AnomalyMatch benchmark results"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True, help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=[
            "score_histogram",
            "roc_prc_curves",
            "top_n_anomaly_detection",
            "metrics_over_time",
            "combined_anomaly_detection",
            "top_n_with_thresholds",
            "roc_with_thresholds",
            "astronomaly_comparison",
            "score_vs_user_score_grid",
        ],
        help="Specific plot type to recreate (optional)",
    )
    parser.add_argument(
        "--active-learning-only",
        action="store_true",
        help="Only create the active learning comparison plot",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        help="Custom thresholds to use for plots (comma-separated list of floats, e.g., '0.95,0.7,0.5')",
    )
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Setup logger
    logger.remove()
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.add(f"recreate_plots_{os.path.basename(args.results_dir)}.log", level="DEBUG")

    logger.info(
        f"Processing results directory: {args.results_dir}"
    )  # Parse custom thresholds if provided
    custom_thresholds = None
    if args.thresholds:
        try:
            # Parse comma-separated list of floats
            custom_thresholds = [float(t) for t in args.thresholds.split(",")]
            logger.info(f"Using custom thresholds: {custom_thresholds}")
        except ValueError:
            logger.error(
                f"Invalid threshold format: {args.thresholds}. Should be comma-separated floats."
            )
            return

    # Only create active learning comparison plot if requested
    if args.active_learning_only:
        logger.info("Creating active learning comparison plot only")
        plot_active_learning_comparison(args.results_dir)
        return

    # Recreate plots from saved data
    recreate_plots_from_data(
        results_dir=args.results_dir, plot_type=args.plot_type, custom_thresholds=custom_thresholds
    )

    # Create active learning comparison plot
    plot_active_learning_comparison(args.results_dir)

    logger.info("All plots recreation completed")


if __name__ == "__main__":
    main()
