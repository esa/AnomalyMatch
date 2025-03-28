#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""
Test script for visualizing paper plot improvements.
Generates plots with dummy/mock data to quickly check visual changes.

Usage:
    python test_plots.py          # Generate test plots with mock data
    python test_plots.py --reload # Test loading and recreating plots from saved data
"""

import os
import argparse
import numpy as np
import pandas as pd
from paper_plots import (
    plot_score_histogram,
    plot_metrics_over_time,
    plot_roc_prc_curves,
    plot_top_n_anomaly_detection,
    plot_combined_anomaly_detection,
)

# Create output directory for test plots
output_dir = "test_plots_output"
os.makedirs(output_dir, exist_ok=True)

# ============================
# Generate mock data for plots
# ============================


# For score histograms
def generate_score_data():
    # Generate normal scores (lower mean)
    normal_scores = np.random.beta(2, 5, 500)
    # Generate anomaly scores (higher mean)
    anomaly_scores = np.random.beta(5, 2, 100)
    return normal_scores, anomaly_scores


# For metrics over time
def generate_metrics_history(iterations=5):
    metrics_history = []
    # Start with lower values, gradually improve
    auroc_start, auroc_end = 0.65, 0.95
    auprc_start, auprc_end = 0.30, 0.70

    for i in range(iterations):
        # Add some randomness to the progression
        progress_ratio = i / (iterations - 1) if iterations > 1 else 1
        noise = np.random.normal(0, 0.02)

        auroc = auroc_start + (auroc_end - auroc_start) * progress_ratio + noise
        auroc = min(max(auroc, 0.5), 1.0)  # Clip to valid range

        auprc = auprc_start + (auprc_end - auprc_start) * progress_ratio + noise
        auprc = min(max(auprc, 0.0), 1.0)  # Clip to valid range

        metrics_history.append({"auroc": auroc, "auprc": auprc})

    return metrics_history


# For ROC and PR curves
def generate_roc_prc_data():
    # Generate some mock precision-recall values
    recall = np.linspace(0, 1, 100)
    # Create a concave precision curve
    precision = 1 - recall**0.7

    # Add noise
    precision = np.maximum(0, precision + np.random.normal(0, 0.03, size=precision.shape))

    # Calculate AUPRC
    auprc = np.trapz(precision, recall)

    # For AUROC, we'll just set a value
    auroc = 0.85

    # Generate normal and anomaly scores for plotting
    normal_scores, anomaly_scores = generate_score_data()

    return {
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "auprc": auprc,
        "normal_scores": normal_scores,
        "anomaly_scores": anomaly_scores,
    }


# For top-n anomaly detection
def generate_detection_data(total_samples=10000, total_anomalies=100):
    # Create mock scores for normal and anomaly samples
    anomaly_scores = np.random.beta(5, 2, total_anomalies)
    normal_scores = np.random.beta(2, 5, total_samples - total_anomalies)

    # Combine all scores
    all_scores = np.concatenate([anomaly_scores, normal_scores])

    # Create mock filenames
    anomaly_filenames = [f"anomaly_{i}.jpg" for i in range(total_anomalies)]
    normal_filenames = [f"normal_{i}.jpg" for i in range(total_samples - total_anomalies)]
    all_filenames = anomaly_filenames + normal_filenames

    # Create mock true_labels_df
    data = {
        "filename": all_filenames,
        "label_idx": [1] * total_anomalies + [0] * (total_samples - total_anomalies),
    }
    true_labels_df = pd.DataFrame(data)

    # Also create a sparse version with fewer samples (to test interpolation issue)
    sparse_total_samples = 1000
    sparse_total_anomalies = 10

    sparse_anomaly_scores = np.random.beta(5, 2, sparse_total_anomalies)
    sparse_normal_scores = np.random.beta(2, 5, sparse_total_samples - sparse_total_anomalies)
    sparse_all_scores = np.concatenate([sparse_anomaly_scores, sparse_normal_scores])

    sparse_anomaly_filenames = [f"sparse_anomaly_{i}.jpg" for i in range(sparse_total_anomalies)]
    sparse_normal_filenames = [
        f"sparse_normal_{i}.jpg" for i in range(sparse_total_samples - sparse_total_anomalies)
    ]
    sparse_all_filenames = sparse_anomaly_filenames + sparse_normal_filenames

    sparse_data = {
        "filename": sparse_all_filenames,
        "label_idx": [1] * sparse_total_anomalies
        + [0] * (sparse_total_samples - sparse_total_anomalies),
    }
    sparse_true_labels_df = pd.DataFrame(sparse_data)

    return (
        all_scores,
        all_filenames,
        true_labels_df,
        sparse_all_scores,
        sparse_all_filenames,
        sparse_true_labels_df,
        total_anomalies / total_samples,
    )


# ===========================
# Test all plotting functions
# ===========================


def test_all_plots():
    print("Generating test plots...")

    # Test score histogram
    normal_scores, anomaly_scores = generate_score_data()
    plot_score_histogram(anomaly_scores, normal_scores, 1, output_dir)
    print("✓ Generated score histogram")

    # Test metrics over time
    metrics_history = generate_metrics_history(iterations=10)
    plot_metrics_over_time(metrics_history, output_dir, batch_size=100)
    print("✓ Generated metrics over time plot")

    # Test ROC and PR curves
    metrics = generate_roc_prc_data()
    plot_roc_prc_curves(metrics, 1, output_dir)
    print("✓ Generated ROC and PR curves")

    # Test top-n anomaly detection (with dense data)
    (
        scores,
        filenames,
        true_labels_df,
        sparse_scores,
        sparse_filenames,
        sparse_true_labels_df,
        anomaly_prevalence,
    ) = generate_detection_data()

    # Use the actual plot_top_n_anomaly_detection function (not the test version)
    x1, y1 = plot_top_n_anomaly_detection(scores, filenames, true_labels_df, 1, 1, output_dir)
    print("✓ Generated top-n anomaly detection plot (dense)")

    # Test with sparse data (to show the issue with interpolation)
    x2, y2 = plot_top_n_anomaly_detection(
        sparse_scores, sparse_filenames, sparse_true_labels_df, 1, 2, output_dir
    )
    print("✓ Generated top-n anomaly detection plot (sparse)")

    # Convert the returned data to numpy arrays to ensure they support numerical operations
    x1 = np.array(x1)
    y1 = np.array(y1)

    # Create mock detection curves data for iterations
    detection_curves = {
        0: (x1, y1 * 0.7),  # Baseline - now using numpy array
        1: (x1, y1 * 0.8),  # Iteration 1
        2: (x1, y1 * 0.9),  # Iteration 2
        3: (x1, y1),  # Iteration 3
    }
    plot_combined_anomaly_detection(detection_curves, output_dir, anomaly_prevalence)
    print("✓ Generated combined anomaly detection plot")

    print(f"\nAll test plots generated in: {os.path.abspath(output_dir)}")


def test_reload_plots(plot_data_dir):
    """Test loading saved plot data and recreating plots from it."""
    import pickle
    import os
    import glob

    # Directory for reloaded plots
    reload_dir = os.path.join(output_dir, "reloaded_plots")
    os.makedirs(reload_dir, exist_ok=True)

    print(f"Loading plot data from {plot_data_dir}")

    # Find all plot data files
    data_files = glob.glob(os.path.join(plot_data_dir, "*.pkl"))

    if not data_files:
        print(f"No plot data files found in {plot_data_dir}")
        return

    print(f"Found {len(data_files)} plot data files")

    # Process each data file
    for data_file in data_files:
        try:
            # Load the data
            with open(data_file, "rb") as f:
                data = pickle.load(f)

            # Get the filename without extension to determine the plot type
            filename = os.path.basename(data_file)
            print(f"Processing {filename}")

            # Recreate the plot based on the plot type
            if "score_histogram" in filename:
                iteration = int(filename.split("iter")[1].split(".")[0])
                plot_score_histogram(
                    data["anomaly_scores"], data["normal_scores"], iteration, reload_dir
                )
                print(f"✓ Recreated score histogram from saved data (iteration {iteration})")

            elif "roc_prc_curves" in filename:
                iteration = int(filename.split("iter")[1].split(".")[0])
                plot_roc_prc_curves(data["metrics"], iteration, reload_dir)
                print(f"✓ Recreated ROC/PRC curves from saved data (iteration {iteration})")

            elif "top_n_anomaly_detection" in filename:
                iteration = int(filename.split("iter")[1].split(".")[0])
                plot_top_n_anomaly_detection(
                    data["scores"],
                    data["filenames"],
                    data["true_labels_df"],
                    data["anomaly_class"],
                    iteration,
                    reload_dir,
                )
                print(
                    f"✓ Recreated top-n anomaly detection plot from saved data (iteration {iteration})"
                )

            elif "metrics_over_time" in filename:
                plot_metrics_over_time(
                    data["metrics_history"], reload_dir, batch_size=data.get("batch_size")
                )
                print("✓ Recreated metrics over time plot from saved data")

            elif "combined_anomaly_detection" in filename:
                plot_combined_anomaly_detection(
                    data["detection_curves"], reload_dir, data.get("anomaly_prevalence")
                )
                print("✓ Recreated combined anomaly detection plot from saved data")

        except Exception as e:
            print(f"Error processing {data_file}: {e}")

    print(f"\nAll reloaded plots saved to: {os.path.abspath(reload_dir)}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test plotting functions for AnomalyMatch")
    parser.add_argument(
        "--reload", action="store_true", help="Test loading and recreating plots from saved data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(output_dir, "plot_data"),
        help="Directory containing saved plot data files (used with --reload)",
    )
    args = parser.parse_args()

    if args.reload:
        # Test loading and recreating plots from saved data
        test_reload_plots(args.data_dir)
    else:
        # Generate test plots with mock data (also saves plot data)
        test_all_plots()
