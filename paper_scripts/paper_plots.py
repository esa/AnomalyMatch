#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""
Plotting utilities for AnomalyMatch benchmarking

This module contains functions for creating various plots to visualize
the performance of AnomalyMatch models during benchmarking.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from loguru import logger
from paper_utils import save_plot_data

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
DEFAULT_DPI = 600


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
    plt.figure(figsize=(8, 8))

    # Plot histograms with density=True for normalization
    sns.histplot(
        normal_scores, color="blue", alpha=0.5, label="Normal", kde=True, bins=30, stat="density"
    )
    sns.histplot(
        anomaly_scores, color="red", alpha=0.5, label="Anomaly", kde=True, bins=30, stat="density"
    )

    # Add labels (no title for publication)
    plt.xlabel("Model Anomaly Score")
    plt.ylabel("Density")
    plt.legend(frameon=True, framealpha=0.7)

    # Save figure with high DPI for publication
    output_path = os.path.join(plots_dir, f"score_histogram_iter{iteration}.png")
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
        x_label = "Training Iteration"

    # Plot metrics with emphasis on data points
    plt.plot(x_values, auroc_values, "bo-", label="AUROC", markersize=8)
    plt.plot(x_values, auprc_values, "ro-", label="AUPRC", markersize=8)

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
    output_path = os.path.join(plots_dir, "metrics_over_time.png")
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
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # 1. ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f'AUROC = {metrics["auroc"]:.3f}')
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True, framealpha=0.7)

    # Equal aspect ratio for ROC curve
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Save the ROC curve
    roc_path = os.path.join(plots_dir, f"roc_curve_iter{iteration}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=DEFAULT_DPI)
    plt.close()

    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 8))
    plt.plot(
        metrics["recall"],
        metrics["precision"],
        "r-",
        linewidth=2,
        label=f'AUPRC = {metrics["auprc"]:.3f}',
    )

    # Note: We're removing the baseline from the PR curve as requested
    # The baseline would be the prevalence of positive class (n_pos / (n_pos + n_neg))
    # but it's not necessary for paper publication

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", frameon=True, framealpha=0.7)

    # Set axis limits for PR curve
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Save the PR curve
    pr_path = os.path.join(plots_dir, f"pr_curve_iter{iteration}.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=DEFAULT_DPI)
    plt.close()

    # 3. Combined figure (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot ROC curve
    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f'AUROC = {metrics["auroc"]:.3f}')
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
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
        "b-",
        linewidth=2,
        label=f'AUPRC = {metrics["auprc"]:.3f}',
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.7)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    # Save the combined figure
    combined_path = os.path.join(plots_dir, f"roc_prc_curves_iter{iteration}.png")
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

    output_path = os.path.join(plots_dir, f"mispredicted_images_iter{iteration}.png")
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
    plt.figure(figsize=(8, 8))

    # Plot the actual detection curve
    plt.plot(
        inspection_points / total_samples * 100,
        anomalies_found,
        "b-",
        linewidth=2,
        label="Anomaly detection rate",
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

    plt.plot(x_perfect, y_perfect, "r--", alpha=0.7, linewidth=1.5, label="Perfect detection")

    # Calculate percentage of anomalies found at key inspection points
    percent_at_0_1pct = np.interp(0.1, inspection_points / total_samples * 100, anomalies_found)
    percent_at_1pct = np.interp(1, inspection_points / total_samples * 100, anomalies_found)

    # Add vertical line at 0.1% inspection
    plt.axvline(x=0.1, color="g", linestyle="--", alpha=0.7)
    plt.text(
        0.07,
        50,
        f"0.1% inspected = {int(total_samples * 0.001)} samples",
        rotation=90,
        va="bottom",
        fontsize=8 * FONT_SCALE,
    )
    plt.text(
        0.095,
        102,
        f"found {percent_at_0_1pct:.1f}% \n of anomalies",
        ha="center",
        fontsize=8 * FONT_SCALE,
    )

    # Add vertical line at 1% inspection
    plt.axvline(x=1, color="g", linestyle="--", alpha=0.7)
    plt.text(
        0.7,
        50,
        f"1% inspected = {int(total_samples * 0.01)} samples",
        rotation=90,
        va="bottom",
        fontsize=8 * FONT_SCALE,
    )
    plt.text(
        0.95,
        102,
        f"found {percent_at_1pct:.1f}% \n of anomalies",
        ha="center",
        fontsize=8 * FONT_SCALE,
    )

    # Add labels - NO TITLE for publication
    plt.xlabel("% of Data Inspected (ranked by anomaly score)")
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
    output_path = os.path.join(plots_dir, f"top_n_detection_iter{iteration}.png")
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
    plt.figure(figsize=(8, 8))

    # Define color map for iterations
    colors = plt.cm.viridis(np.linspace(0, 1, len(detection_curves)))

    # Plot each iteration's curve
    for i, (iteration, (x, y)) in enumerate(detection_curves.items()):
        plt.plot(x, y, color=colors[i], linewidth=2, label=f"Iteration {iteration}")

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

    plt.plot(x_perfect, y_perfect, "r--", alpha=0.7, linewidth=1.5, label="Perfect detection")

    # Add vertical line at 0.1% inspection
    plt.axvline(x=0.1, color="g", linestyle="--", alpha=0.7)
    plt.text(0.07, 70, "0.1% inspection", rotation=90, ha="left", fontsize=8 * FONT_SCALE)

    # Add vertical line at 1% inspection
    plt.axvline(x=1, color="g", linestyle="--", alpha=0.7)
    plt.text(0.7, 70, "1% inspection", rotation=90, ha="left", fontsize=8 * FONT_SCALE)

    # Add labels (no title for publication)
    plt.xlabel("% of Data Inspected (ranked by anomaly score)")
    plt.ylabel("% of Total Anomalies Found")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True, framealpha=0.7)

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
    output_path = os.path.join(plots_dir, "combined_top_n_detection.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Combined Top-N anomaly detection plot saved to {output_path}")


def plot_comparative_anomaly_detection(detection_curves, output_dir):
    """Create a comparative plot of Top-N anomaly detection curves for paper publication."""
    logger.info("Creating comparative Top-N anomaly detection plot across anomaly classes")

    # Create square figure for publication
    plt.figure(figsize=(8, 8))

    # Define color map for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(detection_curves)))

    # Plot each class's curve
    for i, (anomaly_class, (x, y)) in enumerate(detection_curves.items()):
        plt.plot(x, y, color=colors[i], linewidth=2, label=f"Anomaly Class {anomaly_class}")

    # Add reference line (random detection)
    plt.plot([0, 100], [0, 100], "k:", alpha=0.7, linewidth=1.5, label="Random detection")

    # Add vertical line at 1% inspection
    plt.axvline(x=1, color="g", linestyle="--", alpha=0.7)
    plt.text(1.1, 50, "1% inspection", rotation=90, va="center", fontsize=10)

    # Add horizontal lines at 50%, 80%, and 90% detection
    plt.axhline(y=50, color="m", linestyle="--", alpha=0.7)
    plt.text(50, 51, "50% of anomalies", ha="center", fontsize=10)

    plt.axhline(y=80, color="m", linestyle="--", alpha=0.7)
    plt.text(50, 81, "80% of anomalies", ha="center", fontsize=10)

    plt.axhline(y=90, color="m", linestyle="--", alpha=0.7)
    plt.text(50, 91, "90% of anomalies", ha="center", fontsize=10)

    # Add labels (no title for publication)
    plt.xlabel("% of Data Inspected (ranked by anomaly score)")
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
    output_path = os.path.join(output_dir, "comparative_top_n_detection.png")
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
        label="First Iteration",
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
        label="First Iteration",
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
    output_path = os.path.join(output_dir, "comparative_metrics.png")
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
