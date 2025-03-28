#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
#!/usr/bin/env python3
# Disable flake8
# flake8: noqa
# flake8: skip-file

"""
Benchmark script for AnomalyMatch

This script evaluates AnomalyMatch on prepared datasets (GalaxyMNIST or MiniImageNet)
by running multiple training iterations with active learning feedback and measuring performance.

Usage:
    python paper_benchmark.py [--dataset {galaxymnist,miniimagenet}]
    [--anomaly_classes ANOMALY_CLASSES]
    [--n_samples N_SAMPLES] [--anomaly_ratio ANOMALY_RATIO]
    [--train_iterations TRAIN_ITERATIONS] [--n_mislabeled N_MISLABELED]
    [--output_dir OUTPUT_DIR] [--seed SEED] [--size SIZE]
    [--n_to_load N_TO_LOAD] [--training_runs TRAINING_RUNS]
    [--skip_mock_ui]
"""

import os

import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm

from pathlib import Path
from loguru import logger

import sys


sys.path.append("..")

from paper_plots import (
    plot_score_histogram,
    plot_roc_prc_curves,
    plot_top_mispredicted,
    plot_top_n_anomaly_detection,
    plot_metrics_over_time,
    plot_combined_anomaly_detection,
    plot_comparative_anomaly_detection,
    plot_comparative_metrics,
)
from paper_utils import (
    parse_arguments,
    setup_directories,
    load_dataset_info,
    create_initial_labeled_data,
    copy_labeled_images,
    setup_pipeline,
    load_full_dataset,
    find_mislabeled,
    evaluate_performance,
    train_with_progress_bar,
    setup_mock_ui,
    collect_and_save_metrics,
)

import io
import torch
from PIL import Image
import torchvision.transforms as transforms
from anomaly_match.datasets.SSL_Dataset import get_transform
from concurrent.futures import ThreadPoolExecutor
import time

# Try to import TurboJPEG for faster JPEG decoding
try:
    from turbojpeg import TurboJPEG

    jpeg_decoder = TurboJPEG()
    USE_TURBOJPEG = True
    logger.info("Using TurboJPEG for faster image decoding")
except ImportError:
    USE_TURBOJPEG = False
    logger.info("TurboJPEG not available, using PIL for image decoding")


def read_and_decode_image(image_data):
    """Decode image data from HDF5 with optimized handling."""
    try:
        # Convert from vlen array back to bytes
        image_bytes = bytes(image_data)

        # Use TurboJPEG if available (much faster for JPEGs)
        if USE_TURBOJPEG:
            try:
                return jpeg_decoder.decode(image_bytes)
            except Exception:
                # Fall back to PIL if TurboJPEG fails
                return np.array(Image.open(io.BytesIO(image_bytes)))
        else:
            # Standard PIL decoding
            return np.array(Image.open(io.BytesIO(image_bytes)))
    except Exception as e:
        logger.warning(f"Error decoding image: {e}")
        return None


def load_and_process_image(args):
    """Load and preprocess a single image for batch processing."""
    image_data, transform = args
    try:
        # Decode image
        image = read_and_decode_image(image_data)
        if image is None:
            return None

        # Handle grayscale images
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        # Handle RGBA images
        elif len(image.shape) > 2 and image.shape[2] > 3:
            image = image[:, :, :3]  # Keep only RGB

        # Apply transform and convert to tensor
        return transform(Image.fromarray(image))
    except Exception as e:
        logger.warning(f"Error processing image: {e}")
        return None


def get_prediction_scores(session, labeled_filenames, hdf5_path, progress_bar=None):
    """Get prediction scores for all images in HDF5 and filter out labeled samples."""
    logger.info("Getting prediction scores for all data in HDF5 file")
    start_time = time.time()

    # Ensure the model is loaded and in evaluation mode
    if not hasattr(session.model, "eval_model") or session.model.eval_model is None:
        session.model.load_model(session.cfg)

    # Get the appropriate model
    use_ema = hasattr(session.model, "eval_model")
    model = session.model.eval_model if use_ema else session.model.train_model
    model.eval()

    # Import the transform directly from the module - use evaluation transform (train=False)
    transform = get_transform(train=False)

    # Number of worker threads for parallel processing
    num_workers = 1
    logger.info(f"Using {num_workers} worker threads for image decoding")

    # Increased batch size for better GPU utilization
    batch_size = 1000  # Lowered batch size to avoid 32-bit indexing error from PyTorch

    all_scores = []
    all_filenames = []

    # Process HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        # Get dimensions
        total_images = len(f["filenames"])
        logger.info(f"Processing predictions for {total_images} images from HDF5")

        # Get all filenames first
        all_filenames = [name.decode("utf-8") for name in f["filenames"][:]]

        # Calculate number of batches
        num_batches = (total_images + batch_size - 1) // batch_size

        # Set up progress tracking
        if progress_bar is not None:
            progress_bar.max = num_batches
            progress_bar.value = 0

        # Create tqdm progress bar
        pbar = tqdm(total=num_batches, desc="Processing images")

        # Track performance metrics
        processed_images = 0
        batch_times = []

        # Process each batch
        for batch_idx in range(num_batches):
            batch_start_time = time.time()

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)

            # Load batch of image data
            batch_data = f["images"][start_idx:end_idx]

            # Process images in parallel using ThreadPoolExecutor
            valid_tensors = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Create arguments for parallel processing
                process_args = [(img_data, transform) for img_data in batch_data]

                # Process images in parallel and collect results
                for img_tensor in executor.map(load_and_process_image, process_args):
                    if img_tensor is not None:
                        valid_tensors.append(img_tensor)

            # Check if we have any valid images
            if not valid_tensors:
                logger.warning(f"No valid images in batch {batch_idx}")
                continue

            # Convert to batch tensor
            try:
                batch_tensor = torch.stack(valid_tensors)
                batch_size_actual = len(valid_tensors)

                # Move to GPU if available
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()

                # Get predictions
                with torch.no_grad():
                    outputs = model(batch_tensor)
                    batch_scores = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                # Store results
                all_scores.extend(batch_scores)

                # Update progress tracking
                processed_images += batch_size_actual

                # Calculate and log batch performance
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                images_per_sec = batch_size_actual / batch_time

                if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                    logger.info(
                        f"Batch {batch_idx+1}/{num_batches}: Processed {batch_size_actual} images "
                        f"in {batch_time:.2f}s ({images_per_sec:.1f} img/s)"
                    )

                # Update progress bars
                if progress_bar is not None:
                    progress_bar.value = batch_idx + 1
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Close progress bar
        pbar.close()

    # Check if we have any predictions
    if not all_scores:
        logger.error("No valid predictions were generated")
        return np.array([]), []

    # Convert to numpy arrays
    all_scores = np.array(all_scores)

    # Log performance statistics
    total_time = time.time() - start_time
    avg_time_per_image = total_time / processed_images if processed_images > 0 else 0
    logger.info(
        f"Processed {processed_images} images in {total_time:.2f}s "
        f"({processed_images/total_time:.1f} img/s, {avg_time_per_image*1000:.2f}ms/img)"
    )

    # Handle mismatched length between scores and filenames
    if len(all_scores) < len(all_filenames):
        logger.warning(
            f"Mismatch between scores ({len(all_scores)}) and filenames ({len(all_filenames)})"
        )
        all_filenames = all_filenames[: len(all_scores)]

    # Filter out labeled files
    labeled_set = set(labeled_filenames)
    unlabeled_indices = [i for i, fname in enumerate(all_filenames) if fname not in labeled_set]

    logger.info(f"Filtered out {len(all_filenames) - len(unlabeled_indices)} labeled samples")
    logger.info(f"Remaining unlabeled samples: {len(unlabeled_indices)}")

    unlabeled_scores = all_scores[unlabeled_indices]
    unlabeled_filenames = [all_filenames[i] for i in unlabeled_indices]

    return unlabeled_scores, unlabeled_filenames


def get_available_filenames(session):
    """Get set of filenames currently available in the session's dataset.

    Args:
        session: The AnomalyMatch session object

    Returns:
        set: Set of available filenames in the current dataset
    """
    # Get filenames from labeled and unlabeled datasets
    available_filenames = set()

    # Add filenames from labeled dataset
    if hasattr(session, "labeled_train_dataset") and session.labeled_train_dataset is not None:
        if hasattr(session.labeled_train_dataset, "filenames"):
            available_filenames.update(session.labeled_train_dataset.filenames)

    # Add filenames from unlabeled dataset
    if hasattr(session, "unlabeled_train_dataset") and session.unlabeled_train_dataset is not None:
        if hasattr(session.unlabeled_train_dataset, "filenames"):
            available_filenames.update(session.unlabeled_train_dataset.filenames)

    # Add filenames from original dataset if available
    if hasattr(session, "train_dset") and session.train_dset is not None:
        if hasattr(session.train_dset, "dset") and hasattr(session.train_dset.dset, "filenames"):
            available_filenames.update(session.train_dset.dset.filenames)

    logger.info(f"Found {len(available_filenames)} available filenames in current dataset")
    return available_filenames


def run_benchmark(args):
    """Run the full benchmarking process."""
    # Set up directories
    run_dir, model_dir, plots_dir = setup_directories(args)
    logger.info(f"Results will be saved to {run_dir}")

    # Calculate nominal and anomaly counts based on ratio for use in various places
    n_anomaly = int(args.n_samples * args.anomaly_ratio)
    n_nominal = args.n_samples - n_anomaly

    # Load dataset information
    all_labels_df, data_dir, hdf5_path = load_dataset_info(args)

    # Create initial labeled data
    labeled_data_path = os.path.join(run_dir, "labeled_data.csv")
    labeled_df = create_initial_labeled_data(
        all_labels_df,
        args.anomaly_classes[0],
        labeled_data_path,
        args.seed,
        args.n_samples,
        args.anomaly_ratio,
    )

    # Get anomaly prevalence from df
    anomaly_prevalence = labeled_df["label"].value_counts().get("anomaly", 0) / len(labeled_df)
    logger.info(f"Anomaly prevalence in initial labeled data: {anomaly_prevalence:.2%}")

    # Check if we should save labeled images
    if hasattr(args, "save_labeled_images") and args.save_labeled_images:
        # Copy initial labeled images to output directory
        copy_labeled_images(labeled_df, data_dir, run_dir)
        logger.info("Saved labeled images to output directory")
    else:
        logger.info("Skipping saving labeled images to save disk space")

    # Set up mock UI or use terminal directly
    if args.skip_mock_ui:
        output_widget = None
        logger.info("Skipping mock UI, logging directly to terminal")
    else:
        output_widget, progress_bar = setup_mock_ui()
        progress_bar = progress_bar

    # Set up pipeline
    session, cfg = setup_pipeline(
        args,
        data_dir,
        labeled_data_path,
        run_dir,
        output_widget,
        None if args.skip_mock_ui else progress_bar,
    )

    # Initialize metrics history
    metrics_history = []

    # Initialize dictionary to store top-N detection curves for each iteration
    detection_curves = {}

    # Create iteration-specific directories
    logger.info("Creating iteration-specific directories for results")
    for i in range(args.training_runs + 1):  # +1 for baseline
        iter_dir = os.path.join(run_dir, f"iteration_{i}")
        os.makedirs(iter_dir, exist_ok=True)
        iter_plots_dir = os.path.join(iter_dir, "plots")
        os.makedirs(iter_plots_dir, exist_ok=True)

    # Evaluate baseline model before training (iteration 0)
    logger.info("\n======= Evaluating baseline model before training =======")
    labeled_filenames = labeled_df["filename"].tolist()
    unlabeled_scores, unlabeled_filenames = get_prediction_scores(
        session, labeled_filenames, hdf5_path, None if args.skip_mock_ui else progress_bar
    )
    # Assign filenames to session for later lookup
    session.filenames = np.array(unlabeled_filenames)

    # Evaluate baseline performance
    baseline_metrics = evaluate_performance(
        unlabeled_scores, unlabeled_filenames, all_labels_df, args.anomaly_classes[0]
    )
    metrics_history.append(baseline_metrics)

    logger.info(
        f"Baseline AUROC: {baseline_metrics['auroc']:.4f}, AUPRC: {baseline_metrics['auprc']:.4f}"
    )

    # Log top percentile metrics
    if "top_0.1pct_anomalies_found" in baseline_metrics:
        logger.info(
            f"Baseline - Anomalies in top 0.1%: {baseline_metrics['top_0.1pct_anomalies_found']:.2f}%, "
            f"Precision: {baseline_metrics['top_0.1pct_precision']:.2f}%"
        )
    if "top_1.0pct_anomalies_found" in baseline_metrics:
        logger.info(
            f"Baseline - Anomalies in top 1.0%: {baseline_metrics['top_1.0pct_anomalies_found']:.2f}%, "
            f"Precision: {baseline_metrics['top_1.0pct_precision']:.2f}%"
        )

    # Save baseline visualizations
    baseline_plots_dir = os.path.join(run_dir, "iteration_0", "plots")

    # Plot standard visualizations
    plot_score_histogram(
        baseline_metrics["anomaly_scores"], baseline_metrics["normal_scores"], 0, baseline_plots_dir
    )
    plot_roc_prc_curves(baseline_metrics, 0, baseline_plots_dir)
    plot_top_mispredicted(
        unlabeled_scores,
        unlabeled_filenames,
        all_labels_df,
        args.anomaly_classes[0],
        0,
        baseline_plots_dir,
        data_dir,
    )

    # Plot Top-N anomaly detection curve for baseline
    x, y = plot_top_n_anomaly_detection(
        unlabeled_scores,
        unlabeled_filenames,
        all_labels_df,
        args.anomaly_classes[0],
        0,
        baseline_plots_dir,
    )
    detection_curves[0] = (x, y)

    # Run training and evaluation loop
    for iteration in range(args.training_runs):
        # Create iteration-specific directory
        iter_dir = os.path.join(run_dir, f"iteration_{iteration+1}")
        iter_plots_dir = os.path.join(iter_dir, "plots")

        logger.info(
            f"\n======= Starting training iteration {iteration+1}/{args.training_runs} ======="
        )

        # Train model
        logger.info(f"Training for {args.train_iterations} iterations")
        train_with_progress_bar(session, cfg)

        # Save model after training
        model_save_path = os.path.join(model_dir, f"model_iter{iteration+1}.pth")
        iter_model_path = os.path.join(iter_dir, f"model.pth")
        cfg.model_path = model_save_path
        session.save_model()

        # Copy model to iteration-specific directory
        import shutil

        shutil.copy2(model_save_path, iter_model_path)

        logger.info(f"Model saved to {model_save_path}")

        # Get prediction scores (filtering out already labeled samples)
        labeled_filenames = labeled_df["filename"].tolist()
        unlabeled_scores, unlabeled_filenames = get_prediction_scores(
            session, labeled_filenames, hdf5_path, None if args.skip_mock_ui else progress_bar
        )
        # Update session.filenames for labeling corrections
        session.filenames = np.array(unlabeled_filenames)

        # Evaluate performance
        logger.info("Evaluating model performance")
        metrics = evaluate_performance(
            unlabeled_scores, unlabeled_filenames, all_labels_df, args.anomaly_classes[0]
        )
        metrics_history.append(metrics)

        logger.info(f"AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")

        # Log top percentile metrics
        if "top_0.1pct_anomalies_found" in metrics:
            logger.info(
                f"Iter {iteration+1} - Anomalies in top 0.1%: {metrics['top_0.1pct_anomalies_found']:.2f}%, "
                f"Precision: {metrics['top_0.1pct_precision']:.2f}%"
            )
        if "top_1.0pct_anomalies_found" in metrics:
            logger.info(
                f"Iter {iteration+1} - Anomalies in top 1.0%: {metrics['top_1.0pct_anomalies_found']:.2f}%, "
                f"Precision: {metrics['top_1.0pct_precision']:.2f}%"
            )

        # Plot standard visualizations
        # Plot score histogram in iteration-specific directory
        plot_score_histogram(
            metrics["anomaly_scores"], metrics["normal_scores"], iteration + 1, iter_plots_dir
        )
        # Also save to main plots directory for consistency
        plot_score_histogram(
            metrics["anomaly_scores"], metrics["normal_scores"], iteration + 1, plots_dir
        )

        # Plot ROC and PR curves in iteration-specific directory
        plot_roc_prc_curves(metrics, iteration + 1, iter_plots_dir)
        plot_roc_prc_curves(metrics, iteration + 1, plots_dir)  # Also in main plots dir

        # Plot top mispredicted images in iteration-specific directory
        plot_top_mispredicted(
            unlabeled_scores,
            unlabeled_filenames,
            all_labels_df,
            args.anomaly_classes[0],
            iteration + 1,
            iter_plots_dir,
            data_dir,
        )
        plot_top_mispredicted(
            unlabeled_scores,
            unlabeled_filenames,
            all_labels_df,
            args.anomaly_classes[0],
            iteration + 1,
            plots_dir,
            data_dir,
        )

        # Plot Top-N anomaly detection curve for this iteration
        x, y = plot_top_n_anomaly_detection(
            unlabeled_scores,
            unlabeled_filenames,
            all_labels_df,
            args.anomaly_classes[0],
            iteration + 1,
            iter_plots_dir,
        )
        # Also save to main plots directory for consistency
        plot_top_n_anomaly_detection(
            unlabeled_scores,
            unlabeled_filenames,
            all_labels_df,
            args.anomaly_classes[0],
            iteration + 1,
            plots_dir,
        )

        # Save data for combined plot
        detection_curves[iteration + 1] = (x, y)

        # Get the filenames currently in the training dataset
        if (
            hasattr(session, "unlabeled_train_dataset")
            and session.unlabeled_train_dataset is not None
        ):
            training_filenames = set(session.unlabeled_train_dataset.filenames)
            logger.info(f"Found {len(training_filenames)} files in current training batch")

            # Create mask for files that are in the training dataset
            in_training_mask = np.array(
                [fname in training_filenames for fname in unlabeled_filenames]
            )
            training_scores = unlabeled_scores[in_training_mask]
            training_filenames = np.array(unlabeled_filenames)[in_training_mask]

            logger.info(
                f"Filtered to {len(training_filenames)} files that are in current training batch"
            )
        else:
            logger.warning("Could not access unlabeled training dataset, using all files")
            training_scores = unlabeled_scores
            training_filenames = unlabeled_filenames

        # Find mislabeled samples using only files from the current training batch
        corrections = find_mislabeled(
            training_scores,
            training_filenames,
            all_labels_df,
            args.anomaly_classes[0],
            args.n_mislabeled,
        )

        logger.info(f"Found {len(corrections)} mislabeled samples to correct")

        # Apply corrections by labeling the samples in the session
        for _, row in corrections.iterrows():
            filename = row["filename"]
            label = row["label"]

            # Find index of this filename in session.filenames
            try:
                idx = session.filenames.tolist().index(filename)
                session.label_image(idx, label)
                logger.debug(f"Labeled {filename} as {label}")
            except ValueError:
                logger.warning(f"Could not find {filename} in session filenames")

        # Save updated labels
        session.save_labels()

        # Update labeled_df with new labels
        labeled_df = pd.read_csv(labeled_data_path)
        logger.info(f"Updated labels saved, now have {len(labeled_df)} labeled samples")

        # Copy labeled data CSV to iteration dir
        iteration_label_path = os.path.join(iter_dir, "labeled_data.csv")
        shutil.copy2(labeled_data_path, iteration_label_path)

        # Copy newly labeled images to output directory after each iteration
        copy_labeled_images(labeled_df, data_dir, iter_dir)

    # Plot metrics over time with training batches as x-axis
    plot_metrics_over_time(metrics_history, plots_dir, batch_size=args.train_iterations)

    # Calculate true anomaly prevalence in the full dataset, not just labeled data
    # This is the correct value to use for perfect detection curve
    anomaly_count = len(all_labels_df[all_labels_df["label_idx"] == args.anomaly_classes[0]])
    total_count = len(all_labels_df)
    true_anomaly_prevalence = anomaly_count / total_count
    logger.info(
        f"True anomaly prevalence in full dataset: {true_anomaly_prevalence:.4%} (used for perfect detection curve)"
    )

    # Create the combined anomaly detection plot with curves from all iterations
    plot_combined_anomaly_detection(detection_curves, plots_dir, true_anomaly_prevalence)

    # Prepare run information for summary
    run_info = {
        "dataset": args.dataset,
        "anomaly_class": args.anomaly_classes[0],
        "n_samples": args.n_samples,
        "anomaly_ratio": args.anomaly_ratio,
        "n_anomaly_initial": n_anomaly,
        "n_nominal_initial": n_nominal,
        "n_total_labeled": len(labeled_df),
        "training_iterations": args.train_iterations,
        "n_mislabeled": args.n_mislabeled,
    }

    # Save complete metrics summary using the new utility function
    summary_path = os.path.join(run_dir, "results_summary.csv")
    summary_df = collect_and_save_metrics(metrics_history, run_info, summary_path)

    # Log final results
    logger.info(f"\n======= Benchmark completed =======")
    final_metrics = metrics_history[-1]
    first_iter_metrics = metrics_history[1]  # Use first iteration metrics (index 1) as baseline
    logger.info(
        f"First Iteration AUROC: {first_iter_metrics['auroc']:.4f}, AUPRC: {first_iter_metrics['auprc']:.4f}"
    )
    logger.info(f"Final AUROC: {final_metrics['auroc']:.4f}, AUPRC: {final_metrics['auprc']:.4f}")
    logger.info(
        f"Improvement: AUROC +{final_metrics['auroc'] - first_iter_metrics['auroc']:.4f}, "
        f"AUPRC +{final_metrics['auprc'] - first_iter_metrics['auprc']:.4f}"
    )

    # Log top percentile metrics
    if "top_0.1pct_anomalies_found" in final_metrics:
        logger.info(
            f"Final - Anomalies in top 0.1%: {final_metrics['top_0.1pct_anomalies_found']:.2f}%, "
            f"Precision: {final_metrics['top_0.1pct_precision']:.2f}%"
        )
    if "top_1.0pct_anomalies_found" in final_metrics:
        logger.info(
            f"Final - Anomalies in top 1.0%: {final_metrics['top_1.0pct_anomalies_found']:.2f}%, "
            f"Precision: {final_metrics['top_1.0pct_precision']:.2f}%"
        )

    logger.info(f"Results saved to {run_dir}")

    return summary_df


def run_multi_class_benchmark(args):
    """Run benchmarks for multiple anomaly classes and produce comparative results.

    Args:
        args: Command line arguments
    """
    # Store metrics for each anomaly class
    all_class_metrics = {}

    # Store detection curves for each anomaly class
    all_detection_curves = {}

    # Get base output directory
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory for comparative results
    dataset_name = args.dataset
    comparative_dir = base_output_dir / f"{dataset_name}_comparative_results"
    comparative_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark for each anomaly class
    logger.info(
        f"Running benchmarks for {len(args.anomaly_classes)} anomaly classes: {args.anomaly_classes}"
    )

    for anomaly_class in args.anomaly_classes:
        logger.info(f"\n\n{'='*50}")
        logger.info(f"Starting benchmark for anomaly class {anomaly_class}")
        logger.info(f"{'='*50}\n")

        # Update args for this specific class
        args.anomaly_class = anomaly_class

        # Set up directories specific to this anomaly class
        n_anomaly = int(args.n_samples * args.anomaly_ratio)
        n_nominal = args.n_samples - n_anomaly

        # Create subdirectories for specific benchmark run
        run_name = f"{args.dataset}_anomaly{args.anomaly_class}_n{n_nominal}_a{n_anomaly}"
        run_dir = base_output_dir / run_name
        model_dir = run_dir / "models"
        plots_dir = run_dir / "plots"

        # Ensure directories exist
        run_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset information
        all_labels_df, data_dir, hdf5_path = load_dataset_info(args)

        # Create initial labeled data for this anomaly class
        labeled_data_path = os.path.join(run_dir, "labeled_data.csv")
        labeled_df = create_initial_labeled_data(
            all_labels_df,
            anomaly_class,
            labeled_data_path,
            args.seed,
            args.n_samples,
            args.anomaly_ratio,
        )

        # Copy initial labeled images to output directory
        copy_labeled_images(labeled_df, data_dir, run_dir)

        # Set up mock UI or use terminal directly
        if args.skip_mock_ui:
            output_widget = None
            logger.info("Skipping mock UI, logging directly to terminal")
        else:
            output_widget, progress_bar = setup_mock_ui()
            progress_bar = progress_bar

        # Set up pipeline
        session, cfg = setup_pipeline(
            args,
            data_dir,
            labeled_data_path,
            run_dir,
            output_widget,
            None if args.skip_mock_ui else progress_bar,
        )

        # Load full dataset for evaluation
        all_filenames = load_full_dataset(hdf5_path)

        # Initialize metrics history for this class
        metrics_history = []

        # Initialize detection curves for this class
        detection_curves = {}

        # Create iteration-specific directories
        logger.info("Creating iteration-specific directories for results")
        for i in range(args.training_runs + 1):  # +1 for baseline
            iter_dir = os.path.join(run_dir, f"iteration_{i}")
            os.makedirs(iter_dir, exist_ok=True)
            iter_plots_dir = os.path.join(iter_dir, "plots")
            os.makedirs(iter_plots_dir, exist_ok=True)

        # Evaluate baseline model before training (iteration 0)
        logger.info(
            f"\n======= Evaluating baseline model for anomaly class {anomaly_class} ======="
        )
        session.update_predictions()
        labeled_filenames = labeled_df["filename"].tolist()
        unlabeled_scores, unlabeled_filenames = get_prediction_scores(
            session, labeled_filenames, hdf5_path, None if args.skip_mock_ui else progress_bar
        )
        # Assign filenames to session for later lookup
        session.filenames = np.array(unlabeled_filenames)

        # Evaluate baseline performance
        baseline_metrics = evaluate_performance(
            unlabeled_scores, unlabeled_filenames, all_labels_df, anomaly_class
        )
        metrics_history.append(baseline_metrics)

        logger.info(
            f"Baseline AUROC: {baseline_metrics['auroc']:.4f}, AUPRC: {baseline_metrics['auprc']:.4f}"
        )

        # Save baseline visualizations
        baseline_plots_dir = os.path.join(run_dir, "iteration_0", "plots")

        # Standard visualizations
        plot_score_histogram(
            baseline_metrics["anomaly_scores"],
            baseline_metrics["normal_scores"],
            0,
            baseline_plots_dir,
        )
        plot_roc_prc_curves(baseline_metrics, 0, baseline_plots_dir)
        plot_top_mispredicted(
            unlabeled_scores,
            unlabeled_filenames,
            all_labels_df,
            anomaly_class,
            0,
            baseline_plots_dir,
            data_dir,
        )

        # Plot Top-N anomaly detection curve for baseline
        x, y = plot_top_n_anomaly_detection(
            unlabeled_scores,
            unlabeled_filenames,
            all_labels_df,
            anomaly_class,
            0,
            baseline_plots_dir,
        )
        detection_curves[0] = (x, y)

        # Run training and evaluation loop
        for iteration in range(args.training_runs):
            # Create iteration-specific directory
            iter_dir = os.path.join(run_dir, f"iteration_{iteration+1}")
            iter_plots_dir = os.path.join(iter_dir, "plots")

            logger.info(
                f"\n======= Starting training iteration {iteration+1}/{args.training_runs} for anomaly class {anomaly_class} ======="
            )

            # Train model with progress bar
            logger.info(f"Training for {args.train_iterations} iterations")
            train_with_progress_bar(session, cfg)

            # Save model after training
            model_save_path = os.path.join(model_dir, f"model_iter{iteration+1}.pth")
            iter_model_path = os.path.join(iter_dir, f"model.pth")
            cfg.model_path = model_save_path
            session.save_model()

            # Copy model to iteration-specific directory
            import shutil

            shutil.copy2(model_save_path, iter_model_path)

            logger.info(f"Model saved to {model_save_path}")

            # Get prediction scores (filtering out already labeled samples)
            labeled_filenames = labeled_df["filename"].tolist()
            unlabeled_scores, unlabeled_filenames = get_prediction_scores(
                session, labeled_filenames, hdf5_path, None if args.skip_mock_ui else progress_bar
            )
            # Update session.filenames for labeling corrections
            session.filenames = np.array(unlabeled_filenames)

            # Evaluate performance
            logger.info("Evaluating model performance")
            metrics = evaluate_performance(
                unlabeled_scores, unlabeled_filenames, all_labels_df, anomaly_class
            )
            metrics_history.append(metrics)

            logger.info(f"AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")

            # Plot standard visualizations
            plot_score_histogram(
                metrics["anomaly_scores"], metrics["normal_scores"], iteration + 1, iter_plots_dir
            )
            plot_score_histogram(
                metrics["anomaly_scores"], metrics["normal_scores"], iteration + 1, plots_dir
            )
            plot_roc_prc_curves(metrics, iteration + 1, iter_plots_dir)
            plot_roc_prc_curves(metrics, iteration + 1, plots_dir)
            plot_top_mispredicted(
                unlabeled_scores,
                unlabeled_filenames,
                all_labels_df,
                anomaly_class,
                iteration + 1,
                iter_plots_dir,
                data_dir,
            )
            plot_top_mispredicted(
                unlabeled_scores,
                unlabeled_filenames,
                all_labels_df,
                anomaly_class,
                iteration + 1,
                plots_dir,
                data_dir,
            )

            # Plot Top-N anomaly detection curve for this iteration
            x, y = plot_top_n_anomaly_detection(
                unlabeled_scores,
                unlabeled_filenames,
                all_labels_df,
                anomaly_class,
                iteration + 1,
                iter_plots_dir,
            )
            # Also save to main plots directory for consistency
            plot_top_n_anomaly_detection(
                unlabeled_scores,
                unlabeled_filenames,
                all_labels_df,
                anomaly_class,
                iteration + 1,
                plots_dir,
            )

            # Save data for combined plot
            detection_curves[iteration + 1] = (x, y)

            # Get the filenames currently in the training dataset
            if (
                hasattr(session, "unlabeled_train_dataset")
                and session.unlabeled_train_dataset is not None
            ):
                training_filenames = set(session.unlabeled_train_dataset.filenames)
                logger.info(f"Found {len(training_filenames)} files in current training batch")

                # Create mask for files that are in the training dataset
                in_training_mask = np.array(
                    [fname in training_filenames for fname in unlabeled_filenames]
                )
                training_scores = unlabeled_scores[in_training_mask]
                training_filenames = np.array(unlabeled_filenames)[in_training_mask]

                logger.info(
                    f"Filtered to {len(training_filenames)} files that are in current training batch"
                )
            else:
                logger.warning("Could not access unlabeled training dataset, using all files")
                training_scores = unlabeled_scores
                training_filenames = unlabeled_filenames

            # Find mislabeled samples using only files from the current training batch
            corrections = find_mislabeled(
                training_scores,
                training_filenames,
                all_labels_df,
                anomaly_class,
                args.n_mislabeled,
            )

            logger.info(f"Found {len(corrections)} mislabeled samples to correct")

            # Apply corrections by labeling the samples in the session
            for _, row in corrections.iterrows():
                filename = row["filename"]
                label = row["label"]

                # Find index of this filename in session.filenames
                try:
                    idx = session.filenames.tolist().index(filename)
                    session.label_image(idx, label)
                    logger.debug(f"Labeled {filename} as {label}")
                except ValueError:
                    logger.warning(f"Could not find {filename} in session filenames")

            # Save updated labels
            session.save_labels()

            # Update labeled_df with new labels
            labeled_df = pd.read_csv(labeled_data_path)
            logger.info(f"Updated labels saved, now have {len(labeled_df)} labeled samples")

            # Copy labeled data CSV to iteration dir
            iteration_label_path = os.path.join(iter_dir, "labeled_data.csv")
            shutil.copy2(labeled_data_path, iteration_label_path)

            # Copy newly labeled images to output directory after each iteration
            copy_labeled_images(labeled_df, data_dir, iter_dir)

        # Plot metrics over time with training batches as x-axis
        plot_metrics_over_time(metrics_history, plots_dir, batch_size=args.train_iterations)

        # Calculate true anomaly prevalence in the full dataset for this class
        anomaly_count = len(all_labels_df[all_labels_df["label_idx"] == anomaly_class])
        total_count = len(all_labels_df)
        true_anomaly_prevalence = anomaly_count / total_count
        logger.info(
            f"True anomaly prevalence in full dataset for class {anomaly_class}: {true_anomaly_prevalence:.4%}"
        )

        # Create the combined anomaly detection plot with curves from all iterations for this class
        plot_combined_anomaly_detection(detection_curves, plots_dir, true_anomaly_prevalence)

        # Save final results summary for this class
        final_metrics = metrics_history[-1]
        first_iter_metrics = metrics_history[1]  # Use first iteration metrics as baseline
        summary = {
            "dataset": args.dataset,
            "anomaly_class": anomaly_class,
            "n_samples": args.n_samples,
            "anomaly_ratio": args.anomaly_ratio,
            "n_anomaly_initial": n_anomaly,
            "n_nominal_initial": n_nominal,
            "n_total_labeled": len(labeled_df),
            "first_iter_auroc": first_iter_metrics["auroc"],
            "first_iter_auprc": first_iter_metrics["auprc"],
            "final_auroc": final_metrics["auroc"],
            "final_auprc": final_metrics["auprc"],
            "improvement_auroc": final_metrics["auroc"] - first_iter_metrics["auroc"],
            "improvement_auprc": final_metrics["auprc"] - first_iter_metrics["auprc"],
        }

        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(run_dir, "results_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        # Log results for this anomaly class
        logger.info(f"\n======= Benchmark for anomaly class {anomaly_class} completed =======")
        logger.info(
            f"First Iteration AUROC: {first_iter_metrics['auroc']:.4f}, AUPRC: {first_iter_metrics['auprc']:.4f}"
        )
        logger.info(
            f"Final AUROC: {final_metrics['auroc']:.4f}, AUPRC: {final_metrics['auprc']:.4f}"
        )
        logger.info(
            f"Improvement: AUROC +{final_metrics['auroc'] - first_iter_metrics['auroc']:.4f}, AUPRC +{final_metrics['auprc'] - first_iter_metrics['auprc']:.4f}"
        )

        # Store metrics for this anomaly class for later comparison
        all_class_metrics[anomaly_class] = metrics_history

        # Store the final iteration detection curves for comparative visualization
        all_detection_curves[anomaly_class] = detection_curves[args.training_runs]

    # After processing all anomaly classes, create comparative plots
    if len(args.anomaly_classes) > 1:
        logger.info("\n\n======= Creating comparative plots for all anomaly classes =======")
        summary_df = plot_comparative_metrics(all_class_metrics, str(comparative_dir))

        # Create comparative Top-N anomaly detection plot across classes
        plot_comparative_anomaly_detection(all_detection_curves, str(comparative_dir))

        # Log comparative results
        logger.info("\nComparative results summary:")
        for _, row in summary_df.iterrows():
            logger.info(
                f"Class {row['anomaly_class']}: AUROC {row['final_auroc']:.4f} (+{row['improvement_auroc']:.4f}), "
                f"AUPRC {row['final_auprc']:.4f} (+{row['improvement_auprc']:.4f})"
            )

    logger.info(f"\nAll benchmarks completed. Results saved to {base_output_dir}")

    return all_class_metrics


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Set up logger
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end="\n"), colorize=True)

    # Use more descriptive log file name based on dataset and anomaly classes
    if len(args.anomaly_classes) > 1:
        classes_str = "-".join(map(str, args.anomaly_classes))
        log_filename = f"benchmark_{args.dataset}_anomaly{classes_str}.log"
    else:
        log_filename = f"benchmark_{args.dataset}_anomaly{args.anomaly_classes[0]}.log"

    logger.add(log_filename, rotation="10 MB")

    # Run appropriate benchmark function based on number of anomaly classes
    if len(args.anomaly_classes) > 1:
        logger.info(f"Running multi-class benchmark with anomaly classes: {args.anomaly_classes}")
        run_multi_class_benchmark(args)
    else:
        logger.info(f"Running single-class benchmark with anomaly class: {args.anomaly_classes[0]}")
        run_benchmark(args)
