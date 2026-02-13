#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Plotting utilities for AnomalyMatch benchmarking

This module contains functions for creating various plots to visualize
the performance of AnomalyMatch models during benchmarking.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from paper_utils import save_plot_data
from sklearn.metrics import roc_curve

from paper_scripts.create_results import GALAXYZOO_THRESHOLDS
from paper_scripts.plot_colors import (
    ANOMALY_COLOR,
    BLUE,
    COLORMAP_NAME,
    GREEN,
    HIST_ALPHA,
    HLINE_ALPHA,
    HLINE_COLOR,
    HLINE_STYLE,
    LAST_ITER_COLOR,
    NORMAL_COLOR,
    ORANGE,
    PERFECT_LINE_ALPHA,
    PERFECT_LINE_COLOR,
    PERFECT_LINE_STYLE,
    PURPLE,
    RED,
    REFERENCE_LINE_ALPHA,
    REFERENCE_LINE_COLOR,
    REFERENCE_LINE_STYLE,
    VLINE_ALPHA,
    VLINE_COLOR,
    VLINE_STYLE,
)

# Scaling factor for all font sizes (adjust this to make all text larger or smaller)
FONT_SCALE = 1.75

# Set matplotlib parameters for publication-quality plots
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
        "lines.linewidth": 2.0,  # Increased from 1.5
        "lines.markersize": 8,  # Increased from 6
    }
)

# Use a publication-friendly style
sns.set_style("whitegrid")

# Default DPI for saving figures
DEFAULT_DPI = 450


def plot_score_histogram(anomaly_scores, normal_scores, iteration, plots_dir):
    """Plot histogram of model scores for normal and anomalous images."""
    # Save plot data for later recreation
    plot_data = {
        "anomaly_scores": anomaly_scores,
        "normal_scores": normal_scores,
        "iteration": iteration,
    }
    save_plot_data(plot_data, "score_histogram", iteration, plots_dir)

    # Convert inputs to plain NumPy arrays and flatten them
    anomaly_scores = np.array(anomaly_scores).flatten()
    normal_scores = np.array(normal_scores).flatten()

    # Create figure with square aspect ratio for publication
    plt.figure(figsize=(8, 6 * 0.8))

    # Plot histograms with density=True for normalization
    sns.histplot(
        normal_scores,
        color=NORMAL_COLOR,
        alpha=HIST_ALPHA,
        label="Nominal",
        kde=True,
        bins=30,
        stat="density",
    )
    sns.histplot(
        anomaly_scores,
        color=ANOMALY_COLOR,
        alpha=HIST_ALPHA,
        label="Anomaly",
        kde=True,
        bins=30,
        stat="density",
    )

    # Add labels (no title for publication)
    plt.xlabel("AnomalyMatch score")
    plt.ylabel("Density")
    plt.legend(frameon=True, framealpha=0.7)

    # Save figure with high DPI for publication
    output_path = os.path.join(plots_dir, f"score_histogram_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"Score histogram saved to {output_path}")


def plot_metrics_over_time(metrics_history, plots_dir, batch_size=None):
    """Plot AUROC and AUPRC over training iterations for paper publication."""
    # Save plot data for later recreation
    plot_data = {"metrics_history": metrics_history, "batch_size": batch_size}
    save_plot_data(plot_data, "metrics_over_time", 0, plots_dir)

    iterations = range(len(metrics_history))
    auroc_values = [m["auroc"] for m in metrics_history]
    auprc_values = [m["auprc"] for m in metrics_history]

    # Create square figure for publication
    plt.figure(figsize=(8, 8))

    # Use training batches for x-axis if provided
    if batch_size is not None:
        x_values = [i * batch_size for i in iterations]
        x_label = "Training Batches"
    else:
        x_values = iterations
        x_label = "Training Cycle"  # Plot metrics with emphasis on data points
    plt.plot(x_values, auroc_values, "-", color=BLUE, marker="o", label="AUROC", markersize=8)
    plt.plot(x_values, auprc_values, "-", color=RED, marker="o", label="AUPRC", markersize=8)

    # Add labels (no title for publication)
    plt.xlabel(x_label)
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, framealpha=0.7)

    # Set y-axis limits to highlight differences
    plt.ylim(
        max(0, min(auroc_values + auprc_values) - 0.05),
        min(1.0, max(auroc_values + auprc_values) + 0.05),
    )

    # Save figure with high DPI for publication
    output_path = os.path.join(plots_dir, "metrics_over_time.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"Performance metrics plot saved to {output_path}")

    # Also save the metrics as CSV for further analysis
    metrics_df = pd.DataFrame(
        {
            "iteration": iterations,
            "training_batch": x_values if batch_size is not None else iterations,
            "auroc": auroc_values,
            "auprc": auprc_values,
        }
    )
    metrics_df.to_csv(os.path.join(plots_dir, "metrics_history.csv"), index=False)
    logger.info(f"Metrics history saved to {os.path.join(plots_dir, 'metrics_history.csv')}")


def plot_roc_prc_curves(metrics, iteration, plots_dir):
    """Plot ROC and Precision-Recall curves for paper publication."""
    # Save plot data for later recreation
    plot_data = {"metrics": metrics, "iteration": iteration}
    save_plot_data(plot_data, "roc_prc_curves", iteration, plots_dir)

    # Plot each curve separately as square figures

    # Calculate ROC curve points
    y_true = np.concatenate(
        [np.ones(len(metrics["anomaly_scores"])), np.zeros(len(metrics["normal_scores"]))]
    )
    y_scores = np.concatenate([metrics["anomaly_scores"], metrics["normal_scores"]])
    fpr, tpr, _ = roc_curve(y_true, y_scores)  # 1. ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"AUROC = {metrics['auroc']:.3f}")
    plt.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        linewidth=1.5,
        alpha=REFERENCE_LINE_ALPHA,
        label="Random",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True, framealpha=0.7)

    # Equal aspect ratio for ROC curve
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Save the ROC curve
    roc_path = os.path.join(plots_dir, f"roc_curve_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=DEFAULT_DPI)
    plt.close()  # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 8))
    plt.plot(
        metrics["recall"],
        metrics["precision"],
        color=RED,
        linewidth=2,
        label=f"AUPRC = {metrics['auprc']:.3f}",
    )

    # Note: We're removing the baseline from the PR curve as requested
    # The baseline would be the prevalence of positive class (n_pos / (n_pos + n_neg))
    # but it's not necessary for paper publication    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", frameon=True, framealpha=0.7)

    # Set axis limits for PR curve
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Save the PR curve
    pr_path = os.path.join(plots_dir, f"pr_curve_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=DEFAULT_DPI)
    plt.close()

    # 3. Combined figure (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Plot ROC curve
    ax1.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"AUROC = {metrics['auroc']:.3f}")
    ax1.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        linewidth=1.5,
        alpha=REFERENCE_LINE_ALPHA,
        label="Random",
    )
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot PR curve (without baseline)
    ax2.plot(
        metrics["recall"],
        metrics["precision"],
        color=RED,
        linewidth=2,
        label=f"AUPRC = {metrics['auprc']:.3f}",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.7)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    # Save the combined figure
    combined_path = os.path.join(plots_dir, f"roc_prc_curves_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(combined_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"ROC and PR curves saved to {plots_dir}")
    return combined_path


def plot_top_mispredicted(
    scores, filenames, true_labels_df, anomaly_class, iteration, plots_dir, data_dir, n=10
):
    """Plot top mispredicted images from the dataset for paper publication."""
    # Save plot data for later recreation
    plot_data = {
        "scores": scores,
        "filenames": filenames,
        "true_labels_df": true_labels_df,
        "anomaly_class": anomaly_class,
        "iteration": iteration,
        "data_dir": data_dir,
        "n": n,
    }
    save_plot_data(plot_data, "top_mispredicted", iteration, plots_dir)

    from PIL import Image

    logger.info(f"Generating plot of top {n} mispredicted images")

    # Create a DataFrame with scores and filenames
    pred_df = pd.DataFrame({"filename": filenames, "score": scores})

    # Merge with true labels
    merged_df = pd.merge(pred_df, true_labels_df, on="filename")

    # Create true binary labels (1 for anomaly class, 0 for others)
    merged_df["true_anomaly"] = (merged_df["label_idx"] == anomaly_class).astype(int)

    # Get false positives and false negatives
    merged_df["predicted_anomaly"] = (merged_df["score"] > 0.5).astype(int)
    false_positives = merged_df[
        (merged_df["true_anomaly"] == 0) & (merged_df["predicted_anomaly"] == 1)
    ]
    false_negatives = merged_df[
        (merged_df["true_anomaly"] == 1) & (merged_df["predicted_anomaly"] == 0)
    ]

    # Sort by score (most confident FPs and most missed FNs)
    top_fps = false_positives.sort_values("score", ascending=False).head(n // 2)
    top_fns = false_negatives.sort_values("score", ascending=True).head(n // 2)

    # Create figure
    n_rows = 2  # FP and FN rows
    n_cols = min(n // 2, 5)  # Max 5 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    # No fig title for publication quality

    # Plot false positives
    for i, (_, row) in enumerate(top_fps.iterrows()):
        if i >= n_cols:
            break
        try:
            img_path = os.path.join(data_dir, row["filename"])
            img = np.array(Image.open(img_path))
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                img = np.repeat(img[..., None], 3, axis=2)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"FP: {row['score']:.2f}", fontsize=12)
            axes[0, i].axis("off")
        except Exception as e:
            logger.warning(f"Error plotting FP ({row['filename']}): {e}")
            axes[0, i].text(0.5, 0.5, "Error", ha="center", va="center")
            axes[0, i].axis("off")

    # Plot false negatives
    for i, (_, row) in enumerate(top_fns.iterrows()):
        if i >= n_cols:
            break
        try:
            img_path = os.path.join(data_dir, row["filename"])
            img = np.array(Image.open(img_path))
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                img = np.repeat(img[..., None], 3, axis=2)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"FN: {row['score']:.2f}", fontsize=12)
            axes[1, i].axis("off")
        except Exception as e:
            logger.warning(f"Error plotting FN ({row['filename']}): {e}")
            axes[1, i].text(0.5, 0.5, "Error", ha="center", va="center")
            axes[1, i].axis("off")

    # Add row labels
    fig.text(0.01, 0.75, "False Positives", ha="left", va="center", fontsize=14, rotation=90)
    fig.text(0.01, 0.25, "False Negatives", ha="left", va="center", fontsize=14, rotation=90)

    output_path = os.path.join(plots_dir, f"mispredicted_images_iter{iteration}.jpg")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"Mispredicted images saved to {output_path}")


def plot_top_n_anomaly_detection(
    scores, filenames, true_labels_df, anomaly_class, iteration, plots_dir
):
    """Plot percentage of anomalies found in top N predictions for paper publication.

    This function creates a curve showing what percentage of all anomalies
    would be caught if a user inspected the top N predictions based on anomaly score.

    Addresses the issue with the perfect line interpolation in the 0.001% to 0.1% range
    by using more data points in the low percentage range.
    """
    # Save plot data for later recreation
    plot_data = {
        "scores": scores,
        "filenames": filenames,
        "true_labels_df": true_labels_df,
        "anomaly_class": anomaly_class,
        "iteration": iteration,
    }
    save_plot_data(plot_data, "top_n_anomaly_detection", iteration, plots_dir)

    logger.info("Creating Top-N anomaly detection plot")

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

    # Create square figure with log scale for publication (8x8 inches)
    plt.figure(figsize=(8, 8))  # Plot the actual detection curve
    plt.plot(
        inspection_points / total_samples * 100,
        anomalies_found,
        color=BLUE,
        linewidth=2,
        label=f"Anomaly detection rate (Cycle {iteration})",
    )

    # Add reference line (perfect detection - if all anomalies come first)
    # Use very high resolution for the perfect line to avoid interpolation issues
    x_perfect = np.concatenate(
        [
            np.logspace(np.log10(0.0001), np.log10(0.1), 500),  # More points in the lower range
            np.linspace(0.1, 100, 500),  # Linear in higher range
        ]
    )
    x_perfect = np.unique(x_perfect)  # Remove duplicates

    # Calculate perfect detection line
    anomaly_prevalence = total_anomalies / total_samples
    detection_rate_factor = 1.0 / anomaly_prevalence if anomaly_prevalence > 0 else 1.0
    y_perfect = np.minimum(x_perfect * detection_rate_factor, 100)

    plt.plot(
        x_perfect,
        y_perfect,
        color=PERFECT_LINE_COLOR,
        linestyle=PERFECT_LINE_STYLE,
        alpha=PERFECT_LINE_ALPHA,
        linewidth=3,
        label="Perfect detection",
    )

    # Calculate percentage of anomalies found at key inspection points
    # remove the values which are interpolated - they can be taken from the tables
    # percent_at_0_1pct = np.interp(0.1, inspection_points / total_samples * 100, anomalies_found)
    # percent_at_1pct = np.interp(
    #    1, inspection_points / total_samples * 100, anomalies_found
    # )  # Add vertical line at 0.1% inspection
    plt.axvline(x=0.1, color=VLINE_COLOR, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)
    plt.text(
        0.07,
        45,
        f"0.1% inspected = {int(total_samples * 0.001)} samples",
        rotation=90,
        va="center",
        fontsize=8 * FONT_SCALE,
    )
    # plt.text(
    #    0.095,
    #    102,
    #    f"found {percent_at_0_1pct:.1f}% \n of anomalies",
    #    ha="center",
    #    fontsize=8 * FONT_SCALE,
    # )

    # Add vertical line at 1% inspection
    plt.axvline(x=1, color=VLINE_COLOR, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)
    plt.text(
        0.7,
        45,
        f"1% inspected = {int(total_samples * 0.01)} samples",
        rotation=90,
        va="center",
        fontsize=8 * FONT_SCALE,
    )
    # plt.text(
    #    0.95,
    #    102,
    #    f"found {percent_at_1pct:.1f}% \n of anomalies",
    #    ha="center",
    #    fontsize=8 * FONT_SCALE,
    # )  # Add labels - NO TITLE for publication
    plt.xlabel("% of top-scoring predictions inspected")
    plt.ylabel("% of Total Anomalies Found")
    plt.grid(True)
    plt.legend(loc="upper left", frameon=True, framealpha=0.7, fontsize=8 * FONT_SCALE)

    # Set axis limits and log scale
    plt.xscale("log")
    plt.xlim(0.008, 100)
    plt.ylim(0, 100)

    # Add x-axis ticks for log scale with larger fontsize
    plt.xticks(
        [0.01, 0.1, 1, 10, 100],
        ["0.01%", "0.1%", "1%", "10%", "100%"],
    )

    # Save figure with high resolution for publication
    output_path = os.path.join(plots_dir, f"top_n_detection_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Top-N anomaly detection plot saved to {output_path}")

    # Return the data for potential future combined plots
    return inspection_points / total_samples * 100, anomalies_found


def plot_combined_anomaly_detection(detection_curves, plots_dir, anomaly_prevalence=None):
    """Create combined plot of Top-N anomaly detection curves for paper publication."""
    # Save plot data for later recreation
    plot_data = {"detection_curves": detection_curves, "anomaly_prevalence": anomaly_prevalence}
    save_plot_data(plot_data, "combined_anomaly_detection", 0, plots_dir)

    logger.info("Creating combined Top-N anomaly detection plot for all iterations")

    # Create a square figure for publication
    plt.figure(
        figsize=(8, 8)
    )  # Define color map for iterations - use viridis for colorblind-friendly palette
    colors = plt.cm.get_cmap(COLORMAP_NAME)(np.linspace(0, 1, len(detection_curves)))

    # Plot each iteration's curve, with the last iteration highlighted in our consistent blue
    iterations = sorted(list(detection_curves.keys()))
    for i, iteration in enumerate(iterations):
        x, y = detection_curves[iteration]
        # Use special color for last iteration to match anomaly detection rate
        if iteration == iterations[-1]:  # if this is the last iteration
            plt.plot(x, y, color=LAST_ITER_COLOR, linewidth=2.5, label=f"Cycle {iteration}")
        else:
            plt.plot(x, y, color=colors[i], linewidth=2, label=f"Cycle {iteration}")

    # Add reference line (perfect detection - if all anomalies come first)
    # Use very high resolution for the perfect line to avoid interpolation issues
    x_perfect = np.concatenate(
        [
            np.logspace(np.log10(0.0001), np.log10(0.1), 500),  # More points in the lower range
            np.linspace(0.1, 100, 500),  # Linear in higher range
        ]
    )
    x_perfect = np.unique(x_perfect)  # Remove duplicates

    if anomaly_prevalence is not None and anomaly_prevalence > 0:
        # Calculate perfect detection curve based on the prevalence
        detection_rate_factor = 1.0 / anomaly_prevalence
        y_perfect = np.minimum(x_perfect * detection_rate_factor, 100)
        logger.info(
            f"Using provided anomaly prevalence: {anomaly_prevalence:.2%} for perfect detection curve"
        )
    else:
        # Fallback to a simple diagonal line if prevalence isn't provided
        y_perfect = x_perfect
        logger.warning(
            "Anomaly prevalence not provided, using simple diagonal for perfect detection curve"
        )

    plt.plot(
        x_perfect,
        y_perfect,
        color=PERFECT_LINE_COLOR,
        linestyle=PERFECT_LINE_STYLE,
        alpha=PERFECT_LINE_ALPHA,
        linewidth=3,
        label="Perfect detection",
    )

    # Add vertical line at 0.1% inspection
    plt.axvline(x=0.1, color=VLINE_COLOR, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)

    # Add vertical line at 1% inspection
    plt.axvline(x=1, color=VLINE_COLOR, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)

    # Add labels (no title for publication)
    plt.xlabel("% of top-scoring predictions inspected")
    plt.ylabel("% of Total Anomalies Found")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", frameon=True, framealpha=0.7, fontsize=8 * FONT_SCALE)

    # Set axis limits and log scale
    plt.xscale("log")
    plt.xlim(0.008, 100)
    plt.ylim(0, 100)

    # Add x-axis ticks for log scale
    plt.xticks(
        [0.01, 0.1, 1, 10, 100],
        ["0.01%", "0.1%", "1%", "10%", "100%"],
    )

    # Save figure with high resolution for publication
    output_path = os.path.join(plots_dir, "combined_top_n_detection.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Combined Top-N anomaly detection plot saved to {output_path}")


def plot_comparative_anomaly_detection(detection_curves, output_dir):
    """Create a comparative plot of Top-N anomaly detection curves for paper publication."""
    logger.info("Creating comparative Top-N anomaly detection plot across anomaly classes")

    # Create square figure for publication
    plt.figure(
        figsize=(8, 8)
    )  # Define color map for different classes - use viridis for colorblind-friendly palette
    colors = plt.cm.get_cmap(COLORMAP_NAME)(np.linspace(0, 1, len(detection_curves)))

    # Plot each class's curve
    for i, (anomaly_class, (x, y)) in enumerate(detection_curves.items()):
        plt.plot(x, y, color=colors[i], linewidth=2, label=f"Anomaly Class {anomaly_class}")

    # Add reference line (random detection)
    plt.plot(
        [0, 100],
        [0, 100],
        REFERENCE_LINE_STYLE,
        color=REFERENCE_LINE_COLOR,
        alpha=REFERENCE_LINE_ALPHA,
        linewidth=1.5,
        label="Random detection",
    )  # Add vertical line at 1% inspection
    plt.axvline(x=1, color=VLINE_COLOR, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)
    plt.text(1.1, 50, "1% inspection", rotation=90, va="center", fontsize=10)

    # Add horizontal lines at 50%, 80%, and 90% detection
    plt.axhline(y=50, color=HLINE_COLOR, linestyle=HLINE_STYLE, alpha=HLINE_ALPHA)
    plt.text(50, 51, "50% of anomalies", ha="center", fontsize=10)

    plt.axhline(y=80, color=HLINE_COLOR, linestyle=HLINE_STYLE, alpha=HLINE_ALPHA)
    plt.text(50, 81, "80% of anomalies", ha="center", fontsize=10)

    plt.axhline(y=90, color=HLINE_COLOR, linestyle=HLINE_STYLE, alpha=HLINE_ALPHA)
    plt.text(50, 91, "90% of anomalies", ha="center", fontsize=10)

    # Add labels (no title for publication)
    plt.xlabel("% of top-scoring predictions inspected")
    plt.ylabel("% of Total Anomalies Found")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True, framealpha=0.7)

    # Set axis limits
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Create a data table showing percentage found at key inspection points
    class_data = {}
    for anomaly_class, (x, y) in detection_curves.items():
        # Calculate percentage found at different inspection thresholds
        found_at_1pct = np.interp(1, x, y)
        found_at_5pct = np.interp(5, x, y)
        found_at_10pct = np.interp(10, x, y)

        # Find inspection needed for different detection levels
        inspection_for_50pct = np.interp(50, y, x) if max(y) >= 50 else float("inf")
        inspection_for_80pct = np.interp(80, y, x) if max(y) >= 80 else float("inf")
        inspection_for_95pct = np.interp(95, y, x) if max(y) >= 95 else float("inf")

        class_data[anomaly_class] = {
            "found_at_1pct": found_at_1pct,
            "found_at_5pct": found_at_5pct,
            "found_at_10pct": found_at_10pct,
            "inspection_for_50pct": inspection_for_50pct,
            "inspection_for_80pct": inspection_for_80pct,
            "inspection_for_95pct": inspection_for_95pct,
        }

    # Create dataframe for results table
    df_data = []
    for anomaly_class, metrics in class_data.items():
        df_data.append(
            {
                "anomaly_class": anomaly_class,
                "found_at_1pct": f"{metrics['found_at_1pct']:.1f}%",
                "found_at_5pct": f"{metrics['found_at_5pct']:.1f}%",
                "found_at_10pct": f"{metrics['found_at_10pct']:.1f}%",
                "inspection_for_50pct": (
                    f"{metrics['inspection_for_50pct']:.1f}%"
                    if metrics["inspection_for_50pct"] != float("inf")
                    else "N/A"
                ),
                "inspection_for_80pct": (
                    f"{metrics['inspection_for_80pct']:.1f}%"
                    if metrics["inspection_for_80pct"] != float("inf")
                    else "N/A"
                ),
                "inspection_for_95pct": (
                    f"{metrics['inspection_for_95pct']:.1f}%"
                    if metrics["inspection_for_95pct"] != float("inf")
                    else "N/A"
                ),
            }
        )

    results_df = pd.DataFrame(df_data)

    # Save results to CSV
    csv_path = os.path.join(output_dir, "comparative_detection_metrics.csv")
    results_df.to_csv(csv_path, index=False)

    # Save figure with high resolution for publication
    output_path = os.path.join(output_dir, "comparative_top_n_detection.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"Comparative Top-N anomaly detection plot saved to {output_path}")
    logger.info(f"Comparative detection metrics saved to {csv_path}")

    return results_df


def plot_comparative_metrics(class_metrics, output_dir):
    """Plot comparative metrics for different anomaly classes for paper publication."""
    # Extract class indices and metrics
    class_indices = sorted(list(class_metrics.keys()))

    # Use first iteration (index 1) as the baseline instead of untrained model (index 0)
    # This compares performance after initial training with performance after active learning
    first_iter_auroc = [class_metrics[cls][1]["auroc"] for cls in class_indices]
    final_auroc = [class_metrics[cls][-1]["auroc"] for cls in class_indices]
    first_iter_auprc = [class_metrics[cls][1]["auprc"] for cls in class_indices]
    final_auprc = [class_metrics[cls][-1]["auprc"] for cls in class_indices]

    # Create a figure for AUROC and AUPRC comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Prepare x-axis
    x = np.arange(len(class_indices))
    width = 0.35

    # Plot AUROC comparison
    rects1 = ax1.bar(
        x - width / 2,
        first_iter_auroc,
        width,
        label="First Cycle",
        color="lightblue",
        edgecolor="blue",
    )
    rects2 = ax1.bar(
        x + width / 2, final_auroc, width, label="Final", color="skyblue", edgecolor="darkblue"
    )

    # Add labels (no title for publication)
    ax1.set_xlabel("Anomaly Class")
    ax1.set_ylabel("AUROC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_indices)
    ax1.legend(frameon=True, framealpha=0.7)
    ax1.grid(True, alpha=0.3)

    # Add value annotations
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{:.3f}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # Plot AUPRC comparison
    rects3 = ax2.bar(
        x - width / 2,
        first_iter_auprc,
        width,
        label="First Cycle",
        color="lightpink",
        edgecolor="red",
    )
    rects4 = ax2.bar(
        x + width / 2, final_auprc, width, label="Final", color="lightcoral", edgecolor="darkred"
    )

    # Add labels (no title for publication)
    ax2.set_xlabel("Anomaly Class")
    ax2.set_ylabel("AUPRC")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_indices)
    ax2.legend(frameon=True, framealpha=0.7)
    ax2.grid(True, alpha=0.3)

    autolabel(rects3, ax2)
    autolabel(rects4, ax2)

    # Set y-axis limits to start from 0
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    # Save the figure with high resolution for publication
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparative_metrics.pdf")
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"Comparative metrics plot saved to {output_path}")

    # Create a summary table
    summary_data = []
    for cls in class_indices:
        summary_data.append(
            {
                "anomaly_class": cls,
                "first_iter_auroc": class_metrics[cls][1]["auroc"],
                "final_auroc": class_metrics[cls][-1]["auroc"],
                "improvement_auroc": class_metrics[cls][-1]["auroc"]
                - class_metrics[cls][1]["auroc"],
                "first_iter_auprc": class_metrics[cls][1]["auprc"],
                "final_auprc": class_metrics[cls][-1]["auprc"],
                "improvement_auprc": class_metrics[cls][-1]["auprc"]
                - class_metrics[cls][1]["auprc"],
            }
        )

    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "comparative_results_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Comparative results summary saved to {csv_path}")

    return summary_df


# Galaxy Zoo specialized plotting functions
def plot_top_n_with_thresholds(
    scores, filenames, true_labels_df, iteration, plots_dir, thresholds=GALAXYZOO_THRESHOLDS
):
    """
    Plot percentage of anomalies found in top N predictions with different anomaly thresholds.

    Args:
        scores: Anomaly scores for all predictions
        filenames: Corresponding filenames
        true_labels_df: DataFrame with ground truth labels
        iteration: Current training iteration
        plots_dir: Directory to save plots
        thresholds: List of anomaly score thresholds to use
    """
    # Save plot data for later recreation
    plot_data = {
        "scores": scores,
        "filenames": filenames,
        "true_labels_df": true_labels_df,
        "iteration": iteration,
        "thresholds": thresholds,
    }
    save_plot_data(plot_data, "top_n_with_thresholds", iteration, plots_dir)

    logger.info("Creating Top-N anomaly detection plot with different thresholds")

    # Create a DataFrame with scores and filenames
    pred_df = pd.DataFrame({"filename": filenames, "score": scores})

    # Merge with true labels
    merged_df = pd.merge(pred_df, true_labels_df, on="filename")

    # Sort by score (highest to lowest)
    sorted_df = merged_df.sort_values("score", ascending=False).reset_index(drop=True)

    # Create square figure with log scale for publication
    plt.figure(figsize=(8, 8))

    # Set up colors for different thresholds
    colors = [BLUE, GREEN, ORANGE]

    for i, threshold in enumerate(thresholds):
        # Create true binary labels based on the current threshold
        merged_df["true_anomaly"] = (merged_df["anomaly_score_raw"] >= threshold).astype(int)

        # Calculate total number of anomalies in the dataset for this threshold
        total_anomalies = merged_df["true_anomaly"].sum()

        if total_anomalies == 0:
            logger.warning(f"No anomalies found with threshold {threshold}, skipping")
            continue

        # Sort by model score (highest to lowest)
        sorted_df = merged_df.sort_values("score", ascending=False).reset_index(drop=True)

        # Calculate cumulative sum of anomalies found
        sorted_df["cum_anomalies"] = sorted_df["true_anomaly"].cumsum()

        # Calculate percentage of total anomalies found
        sorted_df["percent_anomalies_found"] = 100 * sorted_df["cum_anomalies"] / total_anomalies

        # Create x-axis values for number of inspected samples
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
        inspection_points = np.unique(
            np.round((log_space_points * total_samples / 100)).astype(int)
        )
        inspection_points = inspection_points[inspection_points > 0]  # Remove zero
        inspection_points = np.insert(inspection_points, 0, 0)  # Add zero at the beginning

        if total_samples not in inspection_points:
            inspection_points = np.append(
                inspection_points, total_samples
            )  # Add last point if needed

        # Calculate percentage of anomalies found at each point
        anomalies_found = []
        for i_point in inspection_points:
            if i_point == 0:
                anomalies_found.append(0)
            else:
                anomalies_found.append(
                    sorted_df.loc[min(i_point - 1, len(sorted_df) - 1), "percent_anomalies_found"]
                )

        # Plot the detection curve for this threshold
        plt.plot(
            inspection_points / total_samples * 100,
            anomalies_found,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"Threshold = {threshold:.1f}",
        )

        # Calculate anomaly prevalence for this threshold
        anomaly_prevalence = total_anomalies / total_samples
        logger.info(f"Anomaly prevalence with threshold {threshold}: {anomaly_prevalence:.4f}")

    # Add perfect detection line based on the highest threshold
    # Use very high resolution for the perfect line to avoid interpolation issues
    threshold = thresholds[0]  # Use the highest threshold for perfect line
    merged_df["true_anomaly"] = (merged_df["anomaly_score_raw"] >= threshold).astype(int)
    total_anomalies = merged_df["true_anomaly"].sum()
    anomaly_prevalence = total_anomalies / len(merged_df)

    x_perfect = np.concatenate(
        [
            np.logspace(np.log10(0.0001), np.log10(0.1), 500),  # More points in the lower range
            np.linspace(0.1, 100, 500),  # Linear in higher range
        ]
    )
    x_perfect = np.unique(x_perfect)  # Remove duplicates

    # Calculate perfect detection line
    detection_rate_factor = 1.0 / anomaly_prevalence if anomaly_prevalence > 0 else 1.0
    y_perfect = np.minimum(x_perfect * detection_rate_factor, 100)

    plt.plot(
        x_perfect,
        y_perfect,
        color=PERFECT_LINE_COLOR,
        linestyle=PERFECT_LINE_STYLE,
        alpha=PERFECT_LINE_ALPHA,
        linewidth=3,
        label="Perfect detection",
    )

    # Add labels - NO TITLE for publication
    plt.xlabel("% of top-scoring predictions inspected")
    plt.ylabel("% of Total Anomalies Found")
    plt.grid(True)
    plt.legend(loc="lower right", frameon=True, framealpha=0.7)

    # Set axis limits and log scale
    plt.xscale("log")
    plt.xlim(0.008, 100)
    plt.ylim(0, 100)

    # Add x-axis ticks for log scale with larger fontsize
    plt.xticks(
        [0.01, 0.1, 1, 10, 100],
        ["0.01%", "0.1%", "1%", "10%", "100%"],
    )

    # Save figure with high resolution for publication
    output_path = os.path.join(plots_dir, f"top_n_detection_thresholds_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Top-N anomaly detection plot with thresholds saved to {output_path}")


def plot_roc_with_thresholds(
    scores,
    filenames,
    true_labels_df,
    anomaly_class,
    iteration,
    plots_dir,
    thresholds=GALAXYZOO_THRESHOLDS,
):
    """
    Plot ROC curves with different anomaly thresholds.

    Args:
        scores: Anomaly scores for all predictions
        filenames: Corresponding filenames
        true_labels_df: DataFrame with ground truth labels
        anomaly_class: The class index to treat as anomaly
        iteration: Current training iteration
        plots_dir: Directory to save plots
        thresholds: List of anomaly score thresholds to use
    """
    # Save plot data for later recreation
    plot_data = {
        "scores": scores,
        "filenames": filenames,
        "true_labels_df": true_labels_df,
        "anomaly_class": anomaly_class,
        "iteration": iteration,
        "thresholds": thresholds,
    }
    save_plot_data(plot_data, "roc_with_thresholds", iteration, plots_dir)

    # Create a dataframe with scores and filenames
    scores_df = pd.DataFrame({"filename": filenames, "score": scores})

    # Check if the true_labels_df has the required columns
    if "anomaly_score_raw" not in true_labels_df.columns:
        logger.info(f"No anomaly_score_raw in true_labels_df columns: {true_labels_df.columns}")
        return

    # Merge with true labels
    merged_df = pd.merge(scores_df, true_labels_df, on="filename")

    # Check if there are matching filenames
    if len(merged_df) == 0:
        logger.warning("No matching filenames between scores and true labels")
        return  # Set up colors for different thresholds
    colors = [BLUE, GREEN, ORANGE]

    # Create ROC plot
    plt.figure(figsize=(8, 8))

    for i, threshold in enumerate(thresholds):
        # Create binary labels based on threshold
        merged_df["true_label"] = (merged_df["anomaly_score_raw"] >= threshold).astype(int)

        # Calculate ROC curve
        y_true = merged_df["true_label"].values
        y_scores = merged_df["score"].values
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        # Calculate AUROC
        auroc = np.trapz(tpr, fpr)

        # Plot ROC curve
        plt.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"Threshold = {threshold:.1f}, AUROC = {auroc:.3f}",
        )

    # Add reference line
    plt.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        linewidth=1.5,
        alpha=REFERENCE_LINE_ALPHA,
        label="Random",
    )

    # Add labels and legend
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True, framealpha=0.7)

    # Equal aspect ratio for ROC curve
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Save the ROC curve
    output_path = os.path.join(plots_dir, f"roc_curve_thresholds_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"ROC curves with thresholds saved to {output_path}")


def plot_astronomaly_comparison(
    scores,
    filenames,
    true_labels_df,
    anomaly_class,
    iteration,
    plots_dir,
    thresholds=GALAXYZOO_THRESHOLDS,
):
    """
    Create a plot comparing our results with the Astronomaly Figure 5a reference curve.

    Args:
        scores: Anomaly scores for all predictions
        filenames: Corresponding filenames
        true_labels_df: DataFrame with ground truth labels
        anomaly_class: The class index to treat as anomaly
        iteration: Current training iteration
        plots_dir: Directory to save plots
        thresholds: List of threshold values to plot (default=GALAXYZOO_THRESHOLDS)
    """
    # Save plot data for later recreation
    plot_data = {
        "scores": scores,
        "filenames": filenames,
        "true_labels_df": true_labels_df,
        "anomaly_class": anomaly_class,
        "iteration": iteration,
        "thresholds": thresholds,
    }
    save_plot_data(plot_data, "astronomaly_comparison", iteration, plots_dir)

    logger.info("Creating Astronomaly comparison plot")

    # Check if the true_labels_df has the required columns
    if "anomaly_score_raw" not in true_labels_df.columns:
        logger.error(
            "anomaly_score_raw not found in true_labels_df columns, cannot create Astronomaly comparison plot"
        )
        return

    # Create a dataframe with scores and filenames
    scores_df = pd.DataFrame({"filename": filenames, "score": scores})

    # Merge with true labels
    merged_df = pd.merge(scores_df, true_labels_df, on="filename")

    # Check if there are matching filenames
    if len(merged_df) == 0:
        logger.warning("No matching filenames between scores and true labels")
        return  # Load the Astronomaly Figure 5a data
    try:
        astronomaly_data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "AstronomalyFigure5a.csv"), skiprows=[0]
        )
        astronomaly_x = astronomaly_data["xaxis"].values
        astronomaly_y = astronomaly_data["yaxis"].values
    except Exception as e:
        logger.error(f"Error loading Astronomaly data: {e}")
        return

    # Create the figure
    plt.figure(figsize=(8, 8))

    # plot perfect prediction curve
    plt.plot(
        np.linspace(0, 2000, 10),
        np.linspace(0, 2000, 10),
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        linewidth=4,
        label="1:1 limit",
    )

    # Plot lines for each threshold
    colors = [BLUE, GREEN, ORANGE, PURPLE]
    for i, threshold in enumerate(thresholds):
        # Create binary labels based on threshold
        merged_df["true_anomaly"] = (merged_df["anomaly_score_raw"] >= threshold).astype(int)

        # Sort by model scores for inspection
        filtered_df = merged_df.sort_values("score", ascending=False).reset_index(drop=True)

        # Calculate cumulative sum of anomalies found
        filtered_df["cum_anomalies"] = filtered_df["true_anomaly"].cumsum()

        # Create x-axis values (0 to min(2000, len(filtered_df)))
        inspection_indices = np.arange(0, min(2000, len(filtered_df)))

        # Calculate number of anomalies found at each inspection point
        anomalies_found = []
        for idx in inspection_indices:
            if idx == 0:
                anomalies_found.append(0)
            else:
                anomalies_found.append(filtered_df.loc[idx - 1, "cum_anomalies"])

        # Plot the line for this threshold
        plt.plot(
            inspection_indices,
            anomalies_found,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"AnomalyMatch (t={threshold})",
        )

    # Plot Astronomaly reference curve
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

    # Save figure
    output_path = os.path.join(plots_dir, f"astronomaly_comparison_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close()

    logger.info(f"Astronomaly comparison plot saved to {output_path}")


def plot_score_vs_user_score_grid(
    scores, filenames, true_labels_df, iteration, plots_dir, data_dir, n_grid=15
):
    """
    Create a grid plot that shows machine learning scores versus user scores with
    representative image cutouts.

    Args:
        scores: Machine learning model anomaly scores for each image
        filenames: Corresponding filenames
        true_labels_df: DataFrame with ground truth labels including anomaly_score_raw
        iteration: Current training iteration
        plots_dir: Directory to save plots
        data_dir: Directory containing the image files
        n_grid: Number of grid cells in each dimension (default=15, producing 225 cells)
    """
    # Save plot data for later recreation
    plot_data = {
        "scores": scores,
        "filenames": filenames,
        "true_labels_df": true_labels_df,
        "iteration": iteration,
        "data_dir": data_dir,
        "n_grid": n_grid,
    }
    save_plot_data(plot_data, "score_vs_user_score_grid", iteration, plots_dir)

    logger.info("Creating score vs user score grid plot")

    # Check if the true_labels_df has the required column
    if "anomaly_score_raw" not in true_labels_df.columns:
        logger.error(
            "anomaly_score_raw not found in true_labels_df columns, cannot create score grid plot"
        )
        return

    # Create a dataframe with scores and filenames
    scores_df = pd.DataFrame({"filename": filenames, "ml_score": scores})

    # Merge with true labels
    merged_df = pd.merge(scores_df, true_labels_df, on="filename")

    # Check if there are matching filenames
    if len(merged_df) == 0:
        logger.warning("No matching filenames between scores and true labels")
        return

    # Calculate percentile thresholds
    user_score_80th = merged_df["anomaly_score_raw"].quantile(
        0.80
    )  # 80th percentile for user scores
    user_score_50th = merged_df["anomaly_score_raw"].quantile(
        0.50
    )  # 50th percentile for user scores
    ml_score_80th = merged_df["ml_score"].quantile(0.80)  # 80th percentile for ML scores
    ml_score_50th = merged_df["ml_score"].quantile(0.50)  # 50th percentile for ML scores

    # MAIN PLOT - Full grid view
    create_grid_plot(
        merged_df,
        12,
        data_dir,
        plots_dir,
        iteration,
        "full",
        fig_title="ML Scores vs User Scores - Full Grid",
    )
    if False:
        # PLOT 1 - Top Left Corner: High user scores (top 20%), low ML scores (bottom 50%)
        filtered_df1 = merged_df[
            (merged_df["anomaly_score_raw"] > user_score_80th)
            & (merged_df["ml_score"] <= ml_score_50th)
        ]
        if len(filtered_df1) > 0:
            create_grid_plot(
                filtered_df1,
                7,
                data_dir,
                plots_dir,
                iteration,
                "topleft",
                fig_title="High User Scores (>80th percentile), Low ML Scores (<50th percentile)",
            )
        else:
            logger.warning("No data points for top-left quadrant plot")

        # PLOT 2 - Bottom Right Corner: Low user scores (bottom 50%), high ML scores (top 20%)
        filtered_df2 = merged_df[
            (merged_df["anomaly_score_raw"] <= user_score_50th)
            & (merged_df["ml_score"] > ml_score_80th)
        ]
        if len(filtered_df2) > 0:
            create_grid_plot(
                filtered_df2,
                7,
                data_dir,
                plots_dir,
                iteration,
                "bottomright",
                fig_title="Low User Scores (<50th percentile), High ML Scores (>80th percentile)",
            )
    else:
        logger.warning("No data points for bottom-right quadrant plot")

    # PLOT 3 - Hexbin density plot with anomalies overlaid
    create_rank_comparison_plot(merged_df, plots_dir, iteration)


def create_grid_plot(merged_df, n_grid, data_dir, plots_dir, iteration, suffix, fig_title=None):
    """
    Create a grid plot for the given data with percentile-based sampling and visual indicators.

    Args:
        merged_df: DataFrame with merged scores and filenames
        n_grid: Number of grid cells in each dimension
        data_dir: Directory containing the image files
        plots_dir: Directory to save plots
        iteration: Current training iteration
        suffix: Suffix for the output filename
        fig_title: Optional title for the figure
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    from PIL import Image

    # Number of user score bins (fewer than ML score bins)
    n_user_grid = 6
    # Number of rows for nominal and anomalies
    n_nominal_rows = 3
    n_anomaly_rows = 3

    # Create figure with square aspect ratio for grid cells
    # Calculate better figure size to ensure square grid cells
    # Add some margin for labels and spacing
    fig_width = 11
    # Adjust height to ensure grid cells are square
    # We need to account for the fact that the grid is n_grid columns by n_user_grid rows
    # The 0.9 factor helps account for the extra space taken by labels
    fig_height = (fig_width * n_user_grid / n_grid) * 1.13
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create grid layout with space for axis labels, histograms, and colorbar
    gs = gridspec.GridSpec(
        n_user_grid + 2,  # Add 1 for x-axis labels + 1 for histogram
        n_grid + 1,  # Add 1 for y-axis labels
        width_ratios=[1] + [1] * n_grid,  # Extra column for labels
        height_ratios=[1] + [1] * n_user_grid + [1],  # Extra row for labels
        wspace=0.0,
        hspace=0.0,
    )

    # We'll use the original ML scores for percentile calculation
    # but we'll have separate bins for nominal and anomalous classes

    # Calculate ranks for both scores (1 -> lowest value)
    ml_ranks = merged_df["ml_score"].rank(method="first")
    user_ranks = merged_df["anomaly_score_raw"].rank(method="first")

    # Calculate rank differences with sign (positive = higher in AM, negative = lower in AM)
    signed_rank_differences = ml_ranks - user_ranks  # eg 10 - 5 = 5 (AM score higher)
    rank_differences = np.abs(signed_rank_differences)
    merged_df["rank_diff"] = rank_differences
    merged_df["signed_rank_diff"] = signed_rank_differences

    # Create empty grid to store axes
    axes = np.empty((n_user_grid, n_grid), dtype=object)

    # Dictionary to store the selected images for each cell
    selected_images = {}

    # Split data into nominal and anomalous samples
    threshold = 0.9
    anomalous_df = merged_df[merged_df["anomaly_score_raw"] >= threshold].copy()
    nominal_df = merged_df[merged_df["anomaly_score_raw"] < threshold].copy()

    # Calculate separate percentile bin edges for anomalies and nominal samples
    # This ensures each class has its own independent distribution
    anomaly_ml_percentiles = np.percentile(
        anomalous_df["ml_score"], np.linspace(0, 100, n_grid + 1)
    )
    nominal_ml_percentiles = np.percentile(nominal_df["ml_score"], np.linspace(0, 100, n_grid + 1))

    # Record the actual ML score values at each percentile for axis labels
    anomaly_score_values = []
    nominal_score_values = []
    for x in range(n_grid + 1):
        anomaly_score_values.append(anomaly_ml_percentiles[x])
        nominal_score_values.append(nominal_ml_percentiles[x])

    # For each ML score percentile bin
    for x in range(n_grid):
        # Define ML score bin boundaries using separate percentiles for each class
        if x == n_grid - 1:
            ml_mask_anomaly = (anomalous_df["ml_score"] >= anomaly_ml_percentiles[x]) & (
                anomalous_df["ml_score"] <= anomaly_ml_percentiles[x + 1]
            )
            ml_mask_nominal = (nominal_df["ml_score"] >= nominal_ml_percentiles[x]) & (
                nominal_df["ml_score"] <= nominal_ml_percentiles[x + 1]
            )
        else:
            ml_mask_anomaly = (anomalous_df["ml_score"] >= anomaly_ml_percentiles[x]) & (
                anomalous_df["ml_score"] < anomaly_ml_percentiles[x + 1]
            )
            ml_mask_nominal = (nominal_df["ml_score"] >= nominal_ml_percentiles[x]) & (
                nominal_df["ml_score"] < nominal_ml_percentiles[x + 1]
            )

        # Get samples in this ML score bin for each class
        anomaly_bin_samples = anomalous_df[ml_mask_anomaly].copy()
        nominal_bin_samples = nominal_df[ml_mask_nominal].copy()

        # Process nominal samples for top rows
        if len(nominal_bin_samples) > 0:
            # Sort by absolute rank difference (lowest to highest)
            nominal_bin_samples = nominal_bin_samples.sort_values("rank_diff")

            # Sample images with increasing rank differences: closest, median, furthest
            if len(nominal_bin_samples) <= n_nominal_rows:
                # If we have 2 or fewer images, just use what we have
                nominal_indices = np.arange(len(nominal_bin_samples))
            else:
                # For 3 or more images: get lowest, median, and highest rank difference samples
                if n_nominal_rows == 3:
                    # Get closest (top), median (middle), and furthest (bottom) samples
                    median_idx = len(nominal_bin_samples) // 2
                    nominal_indices = [
                        0,  # Closest - smallest rank diff
                        median_idx,  # Median rank diff
                        len(nominal_bin_samples) - 1,  # Furthest - largest rank diff
                    ]
                else:
                    # Sample with increasing density toward higher rank differences
                    sample_points = np.linspace(0, 1, n_nominal_rows)
                    nominal_indices = (sample_points * (len(nominal_bin_samples) - 1)).astype(int)

            # Store selected nominal images
            for y_idx, idx in enumerate(nominal_indices):
                if y_idx >= n_nominal_rows:
                    continue

                if idx < len(nominal_bin_samples):
                    row = nominal_bin_samples.iloc[idx]
                    selected_images[(y_idx, x)] = {
                        "filename": row["filename"],
                        "rank_diff": row["rank_diff"],
                        "signed_rank_diff": row["signed_rank_diff"],
                        "is_anomaly": False,
                        "ml_score": row["ml_score"],
                        "user_score": row["anomaly_score_raw"],
                    }

        # Process anomaly samples for bottom rows
        if len(anomaly_bin_samples) > 0:
            # Sort by absolute rank difference (lowest to highest)
            anomaly_bin_samples = anomaly_bin_samples.sort_values("rank_diff")

            # Sample images with increasing rank differences: closest, median, furthest
            if len(anomaly_bin_samples) <= n_anomaly_rows:
                # If we have 2 or fewer images, just use what we have
                anomaly_indices = np.arange(len(anomaly_bin_samples))
            else:
                # For 3 or more images: get lowest, median, and highest rank difference samples
                if n_anomaly_rows == 3:
                    # Get closest (top), median (middle), and furthest (bottom) samples
                    median_idx = len(anomaly_bin_samples) // 2
                    anomaly_indices = [
                        0,  # Closest - smallest rank diff
                        median_idx,  # Median rank diff
                        len(anomaly_bin_samples) - 1,  # Furthest - largest rank diff
                    ]
                else:
                    # Sample with increasing density toward higher rank differences
                    sample_points = np.linspace(0, 1, n_anomaly_rows)
                    anomaly_indices = (sample_points * (len(anomaly_bin_samples) - 1)).astype(int)

            # Store selected anomaly images (in bottom rows)
            for y_idx, idx in enumerate(anomaly_indices):
                if y_idx >= n_anomaly_rows:
                    continue

                # Position in bottom half of grid
                y_position = n_nominal_rows + y_idx

                if idx < len(anomaly_bin_samples):
                    row = anomaly_bin_samples.iloc[idx]
                    selected_images[(y_position, x)] = {
                        "filename": row["filename"],
                        "rank_diff": row["rank_diff"],
                        "signed_rank_diff": row["signed_rank_diff"],
                        "is_anomaly": True,
                        "ml_score": row["ml_score"],
                        "user_score": row["anomaly_score_raw"],
                    }

    # Plot the grid
    for y in range(n_user_grid):
        for x in range(n_grid):
            # Create subplot at the right position (add 1 to account for labels)
            ax = plt.subplot(gs[y + 1, x + 1])
            axes[y, x] = ax

            # If we have an image for this cell, display it
            bin_key = (y, x)
            if bin_key in selected_images:
                image_data = selected_images[bin_key]
                filename = image_data["filename"]
                try:
                    img_path = os.path.join(data_dir, os.path.basename(filename))
                    img = np.array(Image.open(img_path))

                    # Make sure image is RGB
                    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                        img = np.repeat(img[..., None], 3, axis=2)

                    # Display the image
                    ax.imshow(img)

                    # Add frame based on anomaly status - Fix the red box positioning
                    if image_data["is_anomaly"]:
                        # Create a properly positioned rectangle that covers the whole image
                        # Use the axes coordinates instead of data coordinates
                        rect = patches.Rectangle(
                            (0, 0),  # Start at the bottom left
                            1,
                            1,  # Full width and height in axes coordinates
                            transform=ax.transAxes,  # Use axes coordinates
                            linewidth=3,  # Slightly thinner line for better appearance
                            edgecolor="red",
                            facecolor="none",
                            alpha=0.8,
                            zorder=10,  # Ensure the box is drawn on top
                        )
                        ax.add_patch(rect)

                        # Add a second rectangle slightly smaller to create a thicker border effect
                        rect_inner = patches.Rectangle(
                            (0.01, 0.01),  # Slightly inset
                            0.98,
                            0.98,  # Slightly smaller
                            transform=ax.transAxes,
                            linewidth=2,
                            edgecolor="red",
                            facecolor="none",
                            alpha=0.6,
                            zorder=10,
                        )
                        ax.add_patch(rect_inner)

                    # Add small text showing signed rank difference
                    signed_diff = int(image_data["signed_rank_diff"])
                    sign = "+" if signed_diff > 0 else ""  # Plus sign for positive values
                    text_color = "white"
                    # Use different background colors to indicate direction
                    bg_color = (
                        "black" if signed_diff > 0 else "black" if signed_diff < 0 else "black"
                    )

                    ax.text(
                        0.95,
                        0.05,
                        f"{sign}{signed_diff}",
                        transform=ax.transAxes,
                        color=text_color,
                        fontsize=6,
                        fontweight="bold",
                        ha="right",
                        va="bottom",
                        bbox=dict(facecolor=bg_color, alpha=0.8, pad=2),
                        zorder=20,  # Ensure text is on top
                    )

                except Exception as e:
                    logger.warning(f"Error loading image {filename}: {e}")
                    # Create a black empty image for error cases
                    empty_img = np.zeros((224, 224, 3), dtype=np.uint8)
                    ax.imshow(empty_img)
            else:
                # Empty cell - create a light gray empty image
                empty_img = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Light gray (200,200,200)
                ax.imshow(empty_img)

            # Remove axis ticks and labels for grid cells
            ax.set_xticks([])
            ax.set_yticks([])

            # Add a white separator line at the boundary between nominal and anomalies
            if y == n_nominal_rows - 1:
                # Add a white border at the bottom of the cell
                ax.spines["bottom"].set_visible(True)
                ax.spines["bottom"].set_color("white")
                ax.spines["bottom"].set_linewidth(4)
                ax.spines["bottom"].set_zorder(250)
            if y == n_nominal_rows:
                # Add a white border at the bottom of the cell
                ax.spines["top"].set_visible(True)
                ax.spines["top"].set_color("white")
                ax.spines["top"].set_linewidth(2)
                ax.spines["top"].set_zorder(250)

    # Font size for axis labels
    if fig_title and not ("Low" in fig_title or "High" in fig_title):
        ticks_fontsize = 8
        tick_rotation = 0
    else:
        ticks_fontsize = 12
        tick_rotation = 90

    # Add row markers
    for y in range(n_user_grid):
        ax = plt.subplot(gs[y + 1, 0])
        # Add text indicators for sections
        if y == 0:
            ax.text(
                0.5,
                0.5,
                "Normal\nImages",
                ha="center",
                va="center",
                rotation=tick_rotation,
                fontsize=ticks_fontsize,
                color="black",
                fontweight="bold",
            )
        elif y == n_nominal_rows:
            ax.text(
                0.5,
                0.5,
                "Anomaly\nImages",
                ha="center",
                va="center",
                rotation=tick_rotation,
                fontsize=ticks_fontsize,
                color="red",
                fontweight="bold",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add column markers to show ML score percentiles
    for x in range(n_grid):
        ax = plt.subplot(gs[n_user_grid + 1, x + 1])
        # If it's a specific column, add percentile indicator
        if x == 0:
            ax.text(0.5, 0.5, "Low AM", ha="center", va="center", fontsize=ticks_fontsize)
        elif x == n_grid - 1:
            ax.text(0.5, 0.5, "High AM", ha="center", va="center", fontsize=ticks_fontsize)
        # You could also show specific percentile values here if needed
        # e.g., ax.text(0.5, 0.5, f"{x * 100/n_grid:.0f}%", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add horizontal histogram (top) for anomaly distribution
    if fig_title and not ("Low" in fig_title or "High" in fig_title) and False:
        ax_hist_top = plt.subplot(gs[0, 1 : n_grid + 1])

        # Create anomaly mask based on threshold
        threshold = 0.9
        anomaly_mask = merged_df["anomaly_score_raw"] >= threshold

        # Calculate global percentile bin edges for ML scores for the histogram
        global_ml_percentiles = np.percentile(
            merged_df["ml_score"], np.linspace(0, 100, n_grid + 1)
        )

        # Create histogram showing anomaly counts per ML score percentile bin
        anomaly_counts = np.zeros(n_grid)
        for i in range(n_grid):
            # Find samples in this ML score percentile bin
            if i == n_grid - 1:
                bin_mask = (merged_df["ml_score"] >= global_ml_percentiles[i]) & (
                    merged_df["ml_score"] <= global_ml_percentiles[i + 1]
                )
            else:
                bin_mask = (merged_df["ml_score"] >= global_ml_percentiles[i]) & (
                    merged_df["ml_score"] < global_ml_percentiles[i + 1]
                )

            # Count anomalies in this bin
            anomaly_counts[i] = np.sum(anomaly_mask & bin_mask)

            # Count anomalies in this bin
            anomaly_counts[i] = np.sum(anomaly_mask & bin_mask)

        bar_positions = np.arange(n_grid) + 0.5  # Center of each grid cell
        bar_width = 0.8  # Width of each bar

        ax_hist_top.bar(
            bar_positions,
            anomaly_counts,
            width=bar_width,
            align="center",
            color="orange",
            edgecolor="darkorange",
            alpha=0.7,
        )

        ax_hist_top.spines["top"].set_visible(False)
        ax_hist_top.spines["right"].set_visible(False)
        ax_hist_top.set_ylabel("Anomalies", fontsize=10)
        ax_hist_top.yaxis.set_tick_params(labelsize=10)
        ax_hist_top.set_title(
            f"Distribution of AM Scores of Anomalies (GZ Scores > {threshold})",
            fontsize=12,
        )
        ax_hist_top.grid(alpha=0.3)
        ax_hist_top.set_xticks([])  # Remove xticks

        # Set x-axis limits to align with grid
        ax_hist_top.set_xlim(0, n_grid)

        # Set y-axis limits based on figure title
        if fig_title and "Low" in fig_title:
            ax_hist_top.set_ylim(0, 1)
        ax_hist_top.yaxis.set_label_position("right")
        ax_hist_top.yaxis.tick_right()

    # Add axis titles
    if fig_title and ("Low User Scores" in fig_title):
        fig.text(
            0.5, 0.01, "AnomalyMatch Score (higher than 80th percentile)", ha="center", fontsize=16
        )
        fig.text(
            0.01,
            0.5,
            "GZ Score (lower than 50th percentile)",
            va="center",
            rotation=90,
            fontsize=ticks_fontsize + 2,
        )
    elif fig_title and ("High User Scores" in fig_title):
        fig.text(
            0.5, 0.01, "AnomalyMatch Score (lower than 50th percentile)", ha="center", fontsize=16
        )
        fig.text(
            0.01,
            0.5,
            "GZ Score (higher than 80th percentile)",
            va="center",
            rotation=90,
            fontsize=ticks_fontsize + 2,
        )
    else:
        fig.text(
            0.5,
            0.09,
            "AnomalyMatch Score (class-specific percentile bins)",
            ha="center",
            fontsize=ticks_fontsize + 3,
        )
        fig.text(
            0.1,
            0.5,
            "Sample Type & Rank Difference",
            va="center",
            rotation=90,
            fontsize=ticks_fontsize + 3,
        )

    # Add a legend for the visual indicators in the bottom-left corner
    ax_legend = plt.subplot(gs[n_user_grid + 1, 0])
    legend_text = "+/-n: AM ranks\nhigher/lower\nscore than GZ\n"
    ax_legend.text(
        0.35,
        0.5,
        legend_text,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        linespacing=1.3,
    )
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.set_facecolor("none")

    for spine in ax_legend.spines.values():
        spine.set_visible(False)

    # Save figure with high resolution
    output_path = os.path.join(plots_dir, f"score_vs_user_score_grid_{suffix}_iter{iteration}.jpg")
    # plt.tight_layout()
    if False:
        if fig_title:
            # Add title with proper padding
            plt.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.98)
            plt.subplots_adjust(top=0.92)  # Make room for the title
        else:
            plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Score vs user score grid plot ({suffix}) saved to {output_path}")


def create_rank_comparison_plot(merged_df, plots_dir, iteration):
    """
    Create a histogram showing the relative occurrence of absolute rank deviations
    between ML scores and user scores, with an overlaid histogram for anomalies only.
    """
    plt.figure(figsize=(10, 6))

    # Calculate ranks for both scores using 'first' method to handle ties consistently

    ml_ranks = merged_df["ml_score"].rank(method="first")
    user_ranks = merged_df["anomaly_score_raw"].rank(method="first")

    # Calculate absolute relative rank deviation
    n_samples = len(merged_df)
    rank_deviation = np.abs((ml_ranks - user_ranks))
    n_bins = 50

    # Create mask for anomalies (user_score > 0.9)
    anomaly_mask = merged_df["anomaly_score_raw"] > 0.9
    n_anomalies = anomaly_mask.sum()

    # Plot histogram for all samples
    counts, bins = np.histogram(rank_deviation, bins=n_bins)
    percentages = (counts / n_samples) * 100
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bar_width = bins[1] - bins[0]

    plt.bar(
        bin_centers,
        percentages,
        width=bar_width,
        color=BLUE,
        alpha=0.5,  # Reduced alpha for better overlay visibility
        edgecolor="black",
        linewidth=1,
        label=f"All Objects, N={n_samples}",
    )

    # Plot histogram for anomalies
    anomaly_counts, _ = np.histogram(rank_deviation[anomaly_mask], bins=bins)
    anomaly_percentages = (anomaly_counts / n_anomalies) * 100

    plt.bar(
        bin_centers,
        anomaly_percentages,
        width=bar_width,
        color=RED,
        alpha=0.5,
        edgecolor="darkred",
        linewidth=1,
        label=f"Anomalies (GZ Score > 0.9, N={n_anomalies})",
    )

    # Calculate statistics for both distributions
    mean_deviation = np.mean(rank_deviation)
    median_deviation = np.median(rank_deviation)
    anomaly_mean = np.mean(rank_deviation[anomaly_mask])
    anomaly_median = np.median(rank_deviation[anomaly_mask])

    # Add vertical lines for statistics
    plt.axvline(
        x=mean_deviation,
        color=BLUE,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean (All) = {mean_deviation:.0f}",
    )
    plt.axvline(
        x=median_deviation,
        color=BLUE,
        linestyle=":",
        linewidth=1.5,
        label=f"Median (All) = {median_deviation:.0f}",
    )
    plt.axvline(
        x=anomaly_mean,
        color=RED,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean (Anomalies) = {anomaly_mean:.0f}",
    )
    plt.axvline(
        x=anomaly_median,
        color=RED,
        linestyle=":",
        linewidth=1.5,
        label=f"Median (Anomalies) = {anomaly_median:.0f}",
    )

    # Add grid and labels
    plt.grid(alpha=0.3)
    plt.xlabel("$|$Rank$_{{AM}} - $Rank$_{{GZ}}|$", fontsize=24)
    plt.ylabel("Relative Occurrence [%]", fontsize=24)

    # Add legend with better placement
    plt.legend(loc="upper right", frameon=True, framealpha=0.7, bbox_to_anchor=(1.0, 0.95))

    # Save the figure
    output_path = os.path.join(plots_dir, f"score_rank_correlation_iter{iteration}.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    # Log statistics
    logger.info(f"Mean rank deviation (all): {mean_deviation:.3f}")
    logger.info(f"Median rank deviation (all): {median_deviation:.3f}")
    logger.info(f"Mean rank deviation (anomalies): {anomaly_mean:.3f}")
    logger.info(f"Median rank deviation (anomalies): {anomaly_median:.3f}")
    logger.info(f"Rank correlation plot saved to {output_path}")
