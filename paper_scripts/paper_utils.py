#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import argparse
import os

from pathlib import Path
import ipywidgets as widgets
import numpy as np
import pandas as pd
import h5py
from loguru import logger
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import sys

sys.path.append("/media/home/AnomalyMatch")
sys.path.append("../")
import anomaly_match as am


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark AnomalyMatch on prepared datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["galaxymnist", "miniimagenet"],
        default="galaxymnist",
        help="Dataset to use for benchmarking",
    )
    parser.add_argument(
        "--anomaly_classes",
        type=str,
        default="1",
        help="Class indices to treat as anomalies, comma-separated (e.g. '0,1,2')",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Total number of labeled samples to start with (normal + anomaly)",
    )
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.1, help="Ratio of anomaly to total samples (0-1)"
    )
    parser.add_argument(
        "--train_iterations",
        type=int,
        default=100,
        help="Number of iterations for each training run",
    )
    parser.add_argument(
        "--n_mislabeled",
        type=int,
        default=20,
        help="Number of mislabeled samples to correct after each training run",
    )
    parser.add_argument(
        "--save_labeled_images",
        action="store_true",
        help="Whether to save labelled images",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../benchmark_results", help="Directory to save results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--size", type=int, default=224, help="Image size for model input")
    parser.add_argument(
        "--n_to_load", type=int, default=10000, help="Number of images to load for prediction"
    )
    parser.add_argument(
        "--training_runs", type=int, default=3, help="Number of training runs to perform"
    )
    parser.add_argument(
        "--skip_mock_ui", action="store_true", help="Skip mock UI and direct logging to terminal"
    )
    args = parser.parse_args()

    # Convert anomaly_classes from string to list of integers
    args.anomaly_classes = [int(cls) for cls in args.anomaly_classes.split(",")]

    return args


def setup_directories(args):
    """Create necessary directories for the benchmark."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate nominal and anomaly counts for directory naming
    n_anomaly = int(args.n_samples * args.anomaly_ratio)
    n_nominal = args.n_samples - n_anomaly

    # Create subdirectories for specific benchmark run
    run_name = f"{args.dataset}_anomaly{args.anomaly_classes[0]}_n{n_nominal}_a{n_anomaly}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create model directory
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, model_dir, plots_dir


def load_dataset_info(args):
    """Load dataset information from CSV files."""
    base_path = "datasets"
    if not os.path.exists(base_path):
        base_path = os.path.join("paper_scripts", "datasets")

    if args.dataset == "galaxymnist":
        labels_path = os.path.join(base_path, "labels_galaxymnist.csv")
        data_dir = os.path.join(base_path, "galaxymnist")
        hdf5_path = os.path.join(base_path, "galaxymnist_224.hdf5")
    else:  # miniimagenet
        labels_path = os.path.join(base_path, "labels_miniimagenet.csv")
        data_dir = os.path.join(base_path, "miniimagenet")
        hdf5_path = os.path.join(base_path, "miniimagenet_224.hdf5")

    # Load full labels dataset
    all_labels_df = pd.read_csv(labels_path)

    # Print dataset statistics
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Total images: {len(all_labels_df)}")
    class_dist = all_labels_df["label_idx"].value_counts()
    logger.info(f"Class distribution: {class_dist.to_dict()}")

    return all_labels_df, data_dir, hdf5_path


def create_initial_labeled_data(
    all_labels_df, anomaly_class, output_path, seed=42, n_samples=60, anomaly_ratio=0.1
):
    """Create initial labeled data CSV for model training."""
    # Set random seed
    np.random.seed(seed)

    # Filter dataframe for anomaly and normal classes
    anomaly_samples = all_labels_df[all_labels_df["label_idx"] == anomaly_class]
    normal_samples = all_labels_df[all_labels_df["label_idx"] != anomaly_class]

    # Calculate number of anomaly and normal samples based on ratio
    n_anomaly = int(n_samples * anomaly_ratio)
    n_nominal = n_samples - n_anomaly

    logger.info(f"Creating initial labeled dataset with {n_samples} samples")
    logger.info(
        f"Using anomaly ratio {anomaly_ratio} - {n_anomaly} anomalies, {n_nominal} normal samples"
    )

    # Randomly select samples
    if len(anomaly_samples) < n_anomaly:
        logger.warning(f"Only {len(anomaly_samples)} anomaly samples available")
        selected_anomalies = anomaly_samples
    else:
        selected_anomalies = anomaly_samples.sample(n=n_anomaly, random_state=seed)

    if len(normal_samples) < n_nominal:
        logger.warning(f"Only {len(normal_samples)} normal samples available")
        selected_normal = normal_samples
    else:
        selected_normal = normal_samples.sample(n=n_nominal, random_state=seed)

    # Create labeled data dataframe
    labeled_data = []

    # Add anomaly samples
    for _, row in selected_anomalies.iterrows():
        labeled_data.append({"filename": row["filename"], "label": "anomaly"})

    # Add normal samples
    for _, row in selected_normal.iterrows():
        labeled_data.append({"filename": row["filename"], "label": "normal"})

    # Create and save dataframe
    labeled_df = pd.DataFrame(labeled_data)
    labeled_df.to_csv(output_path, index=False)

    logger.info(
        f"Created initial labeled dataset with {len(selected_anomalies)} anomalies and {len(selected_normal)} normal samples"
    )
    logger.info(f"Saved to {output_path}")

    return labeled_df


def setup_mock_ui():
    """Create mock UI for running model without actual UI."""
    out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid white", height="400px", background_color="black", overflow="auto"
        ),
        style={"color": "white"},
    )
    progress_bar = widgets.FloatProgress(
        value=0.0,
        min=0.0,
        max=1.0,
    )
    return out, progress_bar


def setup_pipeline(args, data_dir, labeled_data_path, run_dir, output_widget, progress_bar):
    """Set up the AnomalyMatch pipeline for training."""
    model_path = os.path.join(run_dir, "models", "model.pth")

    # Create configuration
    cfg = am.get_default_cfg()
    cfg.name = f"benchmark_{args.dataset}_anomaly{args.anomaly_classes[0]}"
    cfg.model_path = model_path
    cfg.data_dir = data_dir
    cfg.label_file = labeled_data_path
    cfg.size = [args.size, args.size]
    cfg.N_to_load = args.n_to_load  # Use parameter for number of images to load
    cfg.test_ratio = 0.0  # No test evaluation within the session
    cfg.output_dir = str(run_dir)
    cfg.num_train_iter = args.train_iterations
    cfg.progress_bar = progress_bar
    cfg.num_workers = 0  # Avoid multiprocessing issues
    cfg.pin_memory = False

    # Configure logging
    am.set_log_level("info", cfg)

    # Create session
    session = am.Session(cfg)

    # Always set terminal output, but use None when skipping mock UI
    # This ensures session.out is always initialized
    if args.skip_mock_ui:
        session.set_terminal_out(None)
    else:
        session.set_terminal_out(output_widget)

    return session, cfg


def load_full_dataset(hdf5_path):
    """Load the full dataset from HDF5 file for evaluation."""
    logger.info(f"Loading full dataset from {hdf5_path} for evaluation")

    with h5py.File(hdf5_path, "r") as f:
        # Get data from HDF5
        filenames = [name.decode("utf-8") for name in f["filenames"][:]]

        # We don't need to load images, just filenames for evaluation
        logger.info(f"Loaded {len(filenames)} images from HDF5 file")

    return filenames


def find_mislabeled(scores, filenames, true_labels_df, anomaly_class, n_mislabeled):
    """Find samples to add to labeled dataset based on prediction scores.

    Instead of just finding mislabeled samples, this function:
    1. Finds normal samples with highest anomaly scores (potential false positives)
    2. Finds anomaly samples with highest anomaly scores (true positives)

    This simulates a more realistic active learning scenario where users would
    be shown the most anomalous samples regardless of ground truth.

    Args:
        scores (array): Anomaly scores for unlabeled samples
        filenames (list): List of filenames corresponding to scores
        true_labels_df (DataFrame): DataFrame with ground truth labels
        anomaly_class (int): Class index for anomaly
        n_mislabeled (int): Number of samples to select (half FP, half TP)

    Returns:
        DataFrame: Selected samples with filenames and labels
    """
    # Create a DataFrame with scores and filenames
    pred_df = pd.DataFrame({"filename": filenames, "score": scores})

    # Merge with true labels
    merged_df = pd.merge(pred_df, true_labels_df, on="filename")

    # Create true binary labels (1 for anomaly class, 0 for others)
    merged_df["true_anomaly"] = (merged_df["label_idx"] == anomaly_class).astype(int)

    # For normal samples (true_anomaly=0), select those with highest scores (false positives)
    normal_samples = merged_df[merged_df["true_anomaly"] == 0].sort_values("score", ascending=False)
    n_each = n_mislabeled // 2
    top_fp = normal_samples.head(n_each)

    # For anomaly samples (true_anomaly=1), select those with highest scores (true positives)
    anomaly_samples = merged_df[merged_df["true_anomaly"] == 1].sort_values(
        "score", ascending=False
    )
    top_tp = anomaly_samples.head(n_each)

    # Combine and create correction labels
    corrections = pd.concat([top_fp, top_tp])
    corrections["label"] = corrections["true_anomaly"].map({0: "normal", 1: "anomaly"})

    return corrections[["filename", "label"]]


def calculate_top_percentile_metrics(merged_df, anomaly_class, percentiles=[0.1, 1.0]):
    """Calculate metrics for anomalies found in top percentiles of data.

    Args:
        merged_df (DataFrame): DataFrame with scores, filenames and ground truth
        anomaly_class (int): Class index for anomaly
        percentiles (list): Percentiles to calculate metrics for (e.g. [0.1, 1.0] for top 0.1% and 1%)

    Returns:
        dict: Dictionary with percentile metrics
    """
    # Add true anomaly column if not already present
    if "true_anomaly" not in merged_df.columns:
        merged_df["true_anomaly"] = (merged_df["label_idx"] == anomaly_class).astype(int)

    # Sort by score in descending order
    sorted_df = merged_df.sort_values("score", ascending=False)

    # Total number of anomalies in dataset
    total_anomalies = merged_df["true_anomaly"].sum()

    # Calculate metrics for each percentile
    percentile_metrics = {}
    for percentile in percentiles:
        # Calculate number of samples to include for this percentile
        n_samples = max(1, int(len(sorted_df) * percentile / 100))

        # Get top n_samples
        top_samples = sorted_df.head(n_samples)

        # Count anomalies in top samples
        anomalies_found = top_samples["true_anomaly"].sum()

        # Calculate percentage of anomalies found and precision
        pct_anomalies_found = (
            (anomalies_found / total_anomalies) * 100 if total_anomalies > 0 else 0
        )
        precision = (anomalies_found / n_samples) * 100 if n_samples > 0 else 0

        # Store metrics
        percentile_metrics[f"top_{percentile:.1f}pct_anomalies_found"] = pct_anomalies_found
        percentile_metrics[f"top_{percentile:.1f}pct_precision"] = precision

    return percentile_metrics


def evaluate_performance(scores, filenames, true_labels_df, anomaly_class):
    """Evaluate model performance on the dataset."""
    # Create a DataFrame with scores and filenames
    pred_df = pd.DataFrame({"filename": filenames, "score": scores})

    # Merge with true labels
    merged_df = pd.merge(pred_df, true_labels_df, on="filename")

    # Create binary labels (1 for anomaly class, 0 for others)
    y_true = (merged_df["label_idx"] == anomaly_class).astype(int).values
    y_score = merged_df["score"].values

    # Calculate metrics
    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    # Group scores by true class for histogram
    anomaly_scores = merged_df[merged_df["label_idx"] == anomaly_class]["score"].values
    normal_scores = merged_df[merged_df["label_idx"] != anomaly_class]["score"].values

    # Calculate top percentile metrics (top 0.1% and 1%)
    percentile_metrics = calculate_top_percentile_metrics(merged_df, anomaly_class, [0.1, 1.0])

    # Combine all metrics
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "anomaly_scores": anomaly_scores,
        "normal_scores": normal_scores,
        **percentile_metrics,  # Add top percentile metrics
    }

    return metrics


def copy_labeled_images(labeled_df, data_dir, output_dir):
    """Copy labeled images to a folder in the output directory.

    Args:
        labeled_df (DataFrame): DataFrame with labeled data
        data_dir (str): Directory with source images
        output_dir (Path): Output directory path
    """
    import shutil

    # Create labeled_images directory
    labeled_images_dir = os.path.join(output_dir, "labeled_images")
    os.makedirs(labeled_images_dir, exist_ok=True)

    # Create subdirectories for normal and anomaly
    normal_dir = os.path.join(labeled_images_dir, "normal")
    anomaly_dir = os.path.join(labeled_images_dir, "anomaly")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)

    # Count copied files
    copied_count = 0

    # Copy each file to the appropriate directory
    for _, row in labeled_df.iterrows():
        src_path = os.path.join(data_dir, row["filename"])
        if row["label"] == "normal":
            dst_path = os.path.join(normal_dir, row["filename"])
        else:
            dst_path = os.path.join(anomaly_dir, row["filename"])

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            logger.warning(f"Source file not found: {src_path}")

    logger.info(f"Copied {copied_count} labeled images to {labeled_images_dir}")
    return labeled_images_dir


def train_with_progress_bar(session, cfg):
    """Train the model with a progress bar.

    Args:
        session (am.Session): The AnomalyMatch session
        cfg (DotMap): Configuration for training
    """
    import time
    import datetime
    from tqdm import tqdm

    # Create a tqdm progress bar
    progress_bar = tqdm(
        total=cfg.num_train_iter,
        desc="Training",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    # Track start time for ETA calculation
    start_time = time.time()

    def update_progress(current, total):
        """Callback function to update progress bar."""
        # Update tqdm progress bar
        progress_bar.update(1)

        # Calculate ETA and other stats
        elapsed_time = time.time() - start_time
        if current > 0:
            time_per_iteration = elapsed_time / current
            remaining_iterations = total - current
            eta_seconds = time_per_iteration * remaining_iterations

            # Update progress bar description with ETA
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            progress_bar.set_description(f"Training (ETA: {eta_str})")

    # Train the model with progress callback
    session.train(cfg, progress_callback=update_progress)

    # Close the progress bar
    progress_bar.close()


def collect_and_save_metrics(metrics_history, run_info, output_path):
    """Collect all metrics and save to CSV.

    Args:
        metrics_history (list): List of metrics dictionaries
        run_info (dict): Dictionary with run information
        output_path (str): Path to save CSV to
    """
    # Get baseline, first iteration, and final metrics
    baseline_metrics = metrics_history[0]
    first_iter_metrics = metrics_history[1] if len(metrics_history) > 1 else baseline_metrics
    final_metrics = metrics_history[-1]

    # Combine metrics with run info
    summary = {
        **run_info,
        "baseline_auroc": baseline_metrics["auroc"],
        "baseline_auprc": baseline_metrics["auprc"],
        "first_iter_auroc": first_iter_metrics["auroc"],
        "first_iter_auprc": first_iter_metrics["auprc"],
        "final_auroc": final_metrics["auroc"],
        "final_auprc": final_metrics["auprc"],
        "improvement_auroc": final_metrics["auroc"] - first_iter_metrics["auroc"],
        "improvement_auprc": final_metrics["auprc"] - first_iter_metrics["auprc"],
    }

    # Add top percentile metrics
    if "top_0.1pct_anomalies_found" in final_metrics:
        summary["top_0.1pct_anomalies_found"] = final_metrics["top_0.1pct_anomalies_found"]
        summary["top_0.1pct_precision"] = final_metrics["top_0.1pct_precision"]

    if "top_1.0pct_anomalies_found" in final_metrics:
        summary["top_1.0pct_anomalies_found"] = final_metrics["top_1.0pct_anomalies_found"]
        summary["top_1.0pct_precision"] = final_metrics["top_1.0pct_precision"]

    # Create and save DataFrame
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path, index=False)

    return summary_df


def save_plot_data(data, plot_type, iteration, output_dir):
    """Save plotting data to allow recreation of plots without rerunning the benchmark.

    Args:
        data: Dictionary or tuple containing data used to create the plot
        plot_type: String identifier for the type of plot
        iteration: Iteration number (0 for baseline)
        output_dir: Directory to save data to
    """
    import pickle
    import os

    # Create plot_data directory if it doesn't exist
    plot_data_dir = os.path.join(output_dir, "plot_data")
    os.makedirs(plot_data_dir, exist_ok=True)

    # Create more descriptive filename with plot type and iteration
    if plot_type == "score_histogram":
        filename = f"data_for_score_histogram_iter{iteration}.pkl"
    elif plot_type == "roc_prc_curves":
        filename = f"data_for_roc_prc_curves_iter{iteration}.pkl"
    elif plot_type == "top_mispredicted":
        filename = f"data_for_top_mispredicted_images_iter{iteration}.pkl"
    elif plot_type == "top_n_anomaly_detection":
        filename = f"data_for_top_n_anomaly_detection_iter{iteration}.pkl"
    elif plot_type == "combined_anomaly_detection":
        filename = "data_for_combined_anomaly_detection.pkl"
    elif plot_type == "metrics_over_time":
        filename = "data_for_metrics_over_time.pkl"
    else:
        # Generic fallback
        filename = f"data_for_{plot_type}_iter{iteration}.pkl"

    filepath = os.path.join(plot_data_dir, filename)

    # Save data using pickle
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Saved {plot_type} plot data to {filepath}")
    return filepath
