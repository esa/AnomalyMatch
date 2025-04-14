#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""
Results creation script for AnomalyMatch paper experiments.

This script automates running multiple benchmark configurations to produce results for:
- Multiple datasets (miniImageNet and galaxyMNIST)
- Different settings (classes, anomaly ratios)
- Training iterations analysis
- Active learning impact analysis

Usage:
    python create_results.py
"""

import sys
import time
import datetime
import subprocess
from pathlib import Path
import argparse
import pandas as pd
import glob

# ========== CONFIGURATION ==========
# Toggle which experiment sets to run
RUN_MINIIMAGENET = False
RUN_GALAXYMNIST = False
RUN_TRAINING_ITERATIONS_STUDY = False  # Different training iterations
RUN_ACTIVE_LEARNING_ABLATION = False  # With/without active learning
RUN_N_SAMPLES_ABLATION = False  # New ablation study with varying sample sizes

# Default parameters
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TRAINING_RUNS = 3
DEFAULT_TRAIN_ITERATIONS = 100
DEFAULT_N_MISLABELED = 20
# Dataset-specific sample sizes
MINIIMAGENET_N_SAMPLES = 500
GALAXYMNIST_N_SAMPLES = 40  # Different sample size for GalaxyMNIST
# Dataset-specific anomaly ratios
MINIIMAGENET_ANOMALY_RATIO = 0.01  # Fixed ratio for miniImageNet
GALAXYMNIST_ANOMALY_RATIO = 0.25  # Fixed ratio for GalaxyMNIST
DEFAULT_N_TO_LOAD = 10000  # Number of images to load for prediction
DEFAULT_OUTPUT_DIR = (
    "/media/team_workspaces/AnomalyMatch/paper_results"  # Default base directory for results
    # "benchmark_results/"
)

# MiniImageNet experiment configurations
MINIIMAGENET_CLASSES = [48, 57, 68, 85, 95]  # Classes to use as anomalies
# GalaxyMNIST experiment configurations
GALAXYMNIST_CLASSES = [0, 1, 2, 3]  # Classes to use as anomalies

# Training iterations study configurations
TRAINING_ITERATIONS_VALUES = [50, 100, 250, 500]  # Different numbers of training iterations

# N_SAMPLES ablation study configurations
N_SAMPLES_VALUES = [100, 500, 1000]  # Different number of samples
N_MISLABELED_VALUES = [10, 20, 40]  # Corresponding number of mislabeled samples

# Other settings
SKIP_MOCK_UI = True  # Skip mock UI for cleaner logs
SAVE_LABELED_IMAGES = False  # Whether to copy labeled images (set to False to save disk space)
# ====================================

# Time tracking variables
total_experiments = 0
completed_experiments = 0
experiment_times = []
start_time = None


def calculate_total_experiments():
    """Calculate the total number of experiments to be run."""
    total = 0

    if RUN_MINIIMAGENET:
        total += len(MINIIMAGENET_CLASSES)

    if RUN_GALAXYMNIST:
        total += len(GALAXYMNIST_CLASSES)

    if RUN_TRAINING_ITERATIONS_STUDY:
        total += len(TRAINING_ITERATIONS_VALUES)

    if RUN_ACTIVE_LEARNING_ABLATION:
        total += 3  # With AL, without AL, and total samples from start

    if RUN_N_SAMPLES_ABLATION:
        total += len(N_SAMPLES_VALUES)

    return total


def update_time_estimate(experiment_duration):
    """Update time tracking and display estimated remaining time."""
    global completed_experiments, experiment_times # noqa

    completed_experiments += 1
    experiment_times.append(experiment_duration)

    # Calculate average time per experiment
    avg_time = sum(experiment_times) / len(experiment_times)

    # Estimate remaining time
    remaining_experiments = total_experiments - completed_experiments
    estimated_remaining_seconds = avg_time * remaining_experiments

    # Convert to hours, minutes, seconds
    hours, remainder = divmod(estimated_remaining_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Calculate elapsed time
    elapsed = time.time() - start_time
    elapsed_hours, remainder = divmod(elapsed, 3600)
    elapsed_minutes, elapsed_seconds = divmod(remainder, 60)

    # Print progress and time estimate
    print(f"\nProgress: {completed_experiments}/{total_experiments} experiments completed")
    print(f"Elapsed time: {int(elapsed_hours)}h {int(elapsed_minutes)}m {int(elapsed_seconds)}s")
    print(f"Average experiment time: {avg_time:.2f} seconds")
    print(f"Estimated remaining time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(
        f"Estimated completion time: {datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_seconds)}"
    )
    print("=" * 60)


def parse_arguments():
    """Parse command line arguments to override configuration."""
    parser = argparse.ArgumentParser(description="Run AnomalyMatch benchmark experiments")
    parser.add_argument(
        "--miniimagenet", action="store_true", help="Run only miniImageNet experiments"
    )
    parser.add_argument(
        "--galaxymnist", action="store_true", help="Run only GalaxyMNIST experiments"
    )
    parser.add_argument(
        "--training-study", action="store_true", help="Run only training iterations study"
    )
    parser.add_argument(
        "--active-learning", action="store_true", help="Run only active learning ablation"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, help="Custom output directory (defaults to timestamped dir)"
    )

    args = parser.parse_args()

    # Create a config dictionary to return instead of modifying globals
    config = {
        "run_miniimagenet": RUN_MINIIMAGENET,
        "run_galaxymnist": RUN_GALAXYMNIST,
        "run_training_iterations_study": RUN_TRAINING_ITERATIONS_STUDY,
        "run_active_learning_ablation": RUN_ACTIVE_LEARNING_ABLATION,
        "run_n_samples_ablation": RUN_N_SAMPLES_ABLATION,
        "seed": DEFAULT_SEED,
    }

    # If specific flags are set, only run those experiments
    if args.miniimagenet or args.galaxymnist or args.training_study or args.active_learning:
        config["run_miniimagenet"] = args.miniimagenet
        config["run_galaxymnist"] = args.galaxymnist
        config["run_training_iterations_study"] = args.training_study
        config["run_active_learning_ablation"] = args.active_learning

    # If --all is specified, run everything
    if args.all:
        config["run_miniimagenet"] = True
        config["run_galaxymnist"] = True
        config["run_training_iterations_study"] = True
        config["run_active_learning_ablation"] = True
        config["run_n_samples_ablation"] = True

    # Update seed
    config["seed"] = args.seed

    return args, config


def create_output_dir(custom_dir=None):
    """Create timestamped output directory for results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if custom_dir:
        output_dir = Path(custom_dir)
    else:
        # Use the default output directory with a timestamped subfolder
        output_dir = Path(DEFAULT_OUTPUT_DIR) / f"results_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different experiment types
    (output_dir / "miniimagenet").mkdir(exist_ok=True)
    (output_dir / "galaxymnist").mkdir(exist_ok=True)
    (output_dir / "training_iterations").mkdir(exist_ok=True)
    (output_dir / "active_learning").mkdir(exist_ok=True)
    (output_dir / "n_samples_ablation").mkdir(exist_ok=True)

    return output_dir


def run_benchmark(args, log_file):
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


def run_miniimagenet_experiments(output_dir, seed):
    """Run miniImageNet experiments with various configurations."""
    print("\n======= Running miniImageNet Experiments =======")
    mini_output_dir = output_dir / "miniimagenet"

    # Test different anomaly classes with fixed anomaly ratio
    for cls in MINIIMAGENET_CLASSES:
        experiment_name = f"mini_class{cls}_ratio{MINIIMAGENET_ANOMALY_RATIO:.2f}"
        print(f"\nRunning experiment: {experiment_name}")
        exp_start_time = time.time()

        args = [
            "--dataset",
            "miniimagenet",
            "--anomaly_classes",
            str(cls),
            "--n_samples",
            str(MINIIMAGENET_N_SAMPLES),  # Use miniImageNet specific sample size
            "--anomaly_ratio",
            str(MINIIMAGENET_ANOMALY_RATIO),
            "--train_iterations",
            str(DEFAULT_TRAIN_ITERATIONS),
            "--n_mislabeled",
            str(DEFAULT_N_MISLABELED),
            "--output_dir",
            str(mini_output_dir / experiment_name),
            "--seed",
            str(seed),
            "--size",
            str(DEFAULT_IMAGE_SIZE),
            "--n_to_load",
            str(DEFAULT_N_TO_LOAD),
            "--training_runs",
            str(DEFAULT_TRAINING_RUNS),
        ]

        if SKIP_MOCK_UI:
            args.append("--skip_mock_ui")

        log_file = mini_output_dir / f"{experiment_name}.log"
        run_benchmark(args, log_file)

        # Update time estimate
        exp_duration = time.time() - exp_start_time
        update_time_estimate(exp_duration)


def run_galaxymnist_experiments(output_dir, seed):
    """Run GalaxyMNIST experiments with various configurations."""
    print("\n======= Running GalaxyMNIST Experiments =======")
    galaxy_output_dir = output_dir / "galaxymnist"

    # Test different anomaly classes with fixed anomaly ratio
    for cls in GALAXYMNIST_CLASSES:
        experiment_name = f"galaxy_class{cls}_ratio{GALAXYMNIST_ANOMALY_RATIO:.2f}"
        print(f"\nRunning experiment: {experiment_name}")
        exp_start_time = time.time()

        args = [
            "--dataset",
            "galaxymnist",
            "--anomaly_classes",
            str(cls),
            "--n_samples",
            str(GALAXYMNIST_N_SAMPLES),  # Use GalaxyMNIST specific sample size
            "--anomaly_ratio",
            str(GALAXYMNIST_ANOMALY_RATIO),
            "--train_iterations",
            str(DEFAULT_TRAIN_ITERATIONS),
            "--n_mislabeled",
            str(DEFAULT_N_MISLABELED),
            "--output_dir",
            str(galaxy_output_dir / experiment_name),
            "--seed",
            str(seed),
            "--size",
            str(DEFAULT_IMAGE_SIZE),
            "--n_to_load",
            str(DEFAULT_N_TO_LOAD),
            "--training_runs",
            str(DEFAULT_TRAINING_RUNS),
        ]

        if SKIP_MOCK_UI:
            args.append("--skip_mock_ui")

        log_file = galaxy_output_dir / f"{experiment_name}.log"
        run_benchmark(args, log_file)

        # Update time estimate
        exp_duration = time.time() - exp_start_time
        update_time_estimate(exp_duration)


def training_iterations_study(output_dir, seed):
    """Run study on effect of different training iterations."""
    print("\n======= Running Training Iterations Study =======")
    training_output_dir = output_dir / "training_iterations"

    # Use first miniImageNet class for this study
    cls = MINIIMAGENET_CLASSES[0]

    for iterations in TRAINING_ITERATIONS_VALUES:
        experiment_name = f"mini_class{cls}_iterations{iterations}"
        print(f"\nRunning training study: {experiment_name}")
        exp_start_time = time.time()

        args = [
            "--dataset",
            "miniimagenet",
            "--anomaly_classes",
            str(cls),
            "--n_samples",
            str(MINIIMAGENET_N_SAMPLES),  # Use miniImageNet specific sample size
            "--anomaly_ratio",
            str(MINIIMAGENET_ANOMALY_RATIO),
            "--train_iterations",
            str(iterations),
            "--n_mislabeled",
            str(DEFAULT_N_MISLABELED),
            "--output_dir",
            str(training_output_dir / experiment_name),
            "--seed",
            str(seed),
            "--size",
            str(DEFAULT_IMAGE_SIZE),
            "--n_to_load",
            str(DEFAULT_N_TO_LOAD),
            "--training_runs",
            str(DEFAULT_TRAINING_RUNS),
        ]

        if SKIP_MOCK_UI:
            args.append("--skip_mock_ui")

        log_file = training_output_dir / f"{experiment_name}.log"
        run_benchmark(args, log_file)

        # Update time estimate
        exp_duration = time.time() - exp_start_time
        update_time_estimate(exp_duration)


def active_learning_ablation(output_dir, seed):
    """Run experiment to compare with and without active learning."""
    print("\n======= Running Active Learning Ablation Study =======")
    al_output_dir = output_dir / "active_learning"

    # Use first miniImageNet class for this study
    cls = MINIIMAGENET_CLASSES[0]

    # Calculate total samples after active learning
    initial_samples = MINIIMAGENET_N_SAMPLES
    initial_anomalies = int(initial_samples * MINIIMAGENET_ANOMALY_RATIO)
    initial_nominal = initial_samples - initial_anomalies

    # DEFAULT_N_MISLABELED is total added samples (half nominal, half anomalies) each run
    added_nominal = (DEFAULT_N_MISLABELED // 2) * DEFAULT_TRAINING_RUNS
    added_anomalies = (DEFAULT_N_MISLABELED // 2) * DEFAULT_TRAINING_RUNS

    # Calculate total samples after active learning
    total_samples = initial_samples + DEFAULT_N_MISLABELED
    total_anomalies = initial_anomalies + added_anomalies
    total_nominal = initial_nominal + added_nominal
    total_anomaly_ratio = total_anomalies / total_samples

    # Logging the calculated values
    print("Simulated active learning run stats:")
    print(
        f"Initial samples: {initial_samples}, Anomalies: {initial_anomalies}, Nominal: {initial_nominal}"
    )
    print(
        f"Added samples: {DEFAULT_N_MISLABELED}, Anomalies: {added_anomalies}, Nominal: {added_nominal}"
    )
    print(f"Total samples: {total_samples}, Anomalies: {total_anomalies}, Nominal: {total_nominal}")
    print(f"Final anomaly ratio: {total_anomaly_ratio:.3f}")

    # With active learning (default)
    experiment_name = f"mini_class{cls}_with_al"
    print(f"\nRunning with active learning: {experiment_name}")
    exp_start_time = time.time()

    args = [
        "--dataset",
        "miniimagenet",
        "--anomaly_classes",
        str(cls),
        "--n_samples",
        str(MINIIMAGENET_N_SAMPLES),
        "--anomaly_ratio",
        str(MINIIMAGENET_ANOMALY_RATIO),
        "--train_iterations",
        str(DEFAULT_TRAIN_ITERATIONS),
        "--n_mislabeled",
        str(DEFAULT_N_MISLABELED),
        "--output_dir",
        str(al_output_dir / experiment_name),
        "--seed",
        str(seed),
        "--size",
        str(DEFAULT_IMAGE_SIZE),
        "--n_to_load",
        str(DEFAULT_N_TO_LOAD),
        "--training_runs",
        str(DEFAULT_TRAINING_RUNS),
    ]

    if SKIP_MOCK_UI:
        args.append("--skip_mock_ui")

    log_file = al_output_dir / f"{experiment_name}.log"
    run_benchmark(args, log_file)

    # Update time estimate
    exp_duration = time.time() - exp_start_time
    update_time_estimate(exp_duration)

    # Without active learning (n_mislabeled=0)
    experiment_name = f"mini_class{cls}_without_al"
    print(f"\nRunning without active learning: {experiment_name}")
    exp_start_time = time.time()

    args = [
        "--dataset",
        "miniimagenet",
        "--anomaly_classes",
        str(cls),
        "--n_samples",
        str(MINIIMAGENET_N_SAMPLES),
        "--anomaly_ratio",
        str(MINIIMAGENET_ANOMALY_RATIO),
        "--train_iterations",
        str(DEFAULT_TRAIN_ITERATIONS),
        "--n_mislabeled",
        "0",  # No active learning
        "--output_dir",
        str(al_output_dir / experiment_name),
        "--seed",
        str(seed),
        "--size",
        str(DEFAULT_IMAGE_SIZE),
        "--n_to_load",
        str(DEFAULT_N_TO_LOAD),
        "--training_runs",
        str(DEFAULT_TRAINING_RUNS),
    ]

    if SKIP_MOCK_UI:
        args.append("--skip_mock_ui")

    log_file = al_output_dir / f"{experiment_name}.log"
    run_benchmark(args, log_file)

    # Update time estimate
    exp_duration = time.time() - exp_start_time
    update_time_estimate(exp_duration)

    # With total samples from beginning (new scenario)
    experiment_name = f"mini_class{cls}_total_samples"
    print(f"\nRunning with total samples from start: {experiment_name}")
    print(
        f"Total samples: {total_samples} (Nominal: {total_nominal}, Anomalies: {total_anomalies})"
    )
    print(f"Final anomaly ratio: {total_anomaly_ratio:.3f}")
    exp_start_time = time.time()

    args = [
        "--dataset",
        "miniimagenet",
        "--anomaly_classes",
        str(cls),
        "--n_samples",
        str(total_samples),
        "--anomaly_ratio",
        str(total_anomaly_ratio),
        "--train_iterations",
        str(DEFAULT_TRAIN_ITERATIONS),
        "--n_mislabeled",
        "0",  # No active learning needed as we start with all samples
        "--output_dir",
        str(al_output_dir / experiment_name),
        "--seed",
        str(seed),
        "--size",
        str(DEFAULT_IMAGE_SIZE),
        "--n_to_load",
        str(DEFAULT_N_TO_LOAD),
        "--training_runs",
        str(DEFAULT_TRAINING_RUNS),
    ]

    if SKIP_MOCK_UI:
        args.append("--skip_mock_ui")

    log_file = al_output_dir / f"{experiment_name}.log"
    run_benchmark(args, log_file)

    # Update time estimate
    exp_duration = time.time() - exp_start_time
    update_time_estimate(exp_duration)


def n_samples_ablation(output_dir, seed):
    """Run ablation study with varying sample sizes."""
    print("\n======= Running N_SAMPLES Ablation Study =======")
    n_samples_output_dir = output_dir / "n_samples_ablation"

    # Use first miniImageNet class for this study
    cls = MINIIMAGENET_CLASSES[0]

    for n_samples, n_mislabeled in zip(N_SAMPLES_VALUES, N_MISLABELED_VALUES):
        experiment_name = f"mini_class{cls}_nsamples{n_samples}"
        print(f"\nRunning N_SAMPLES ablation: {experiment_name}")
        exp_start_time = time.time()

        args = [
            "--dataset",
            "miniimagenet",
            "--anomaly_classes",
            str(cls),
            "--n_samples",
            str(n_samples),
            "--anomaly_ratio",
            str(MINIIMAGENET_ANOMALY_RATIO),
            "--train_iterations",
            str(DEFAULT_TRAIN_ITERATIONS),
            "--n_mislabeled",
            str(n_mislabeled),
            "--output_dir",
            str(n_samples_output_dir / experiment_name),
            "--seed",
            str(seed),
            "--size",
            str(DEFAULT_IMAGE_SIZE),
            "--n_to_load",
            str(DEFAULT_N_TO_LOAD),
            "--training_runs",
            str(DEFAULT_TRAINING_RUNS),
        ]

        if SKIP_MOCK_UI:
            args.append("--skip_mock_ui")

        log_file = n_samples_output_dir / f"{experiment_name}.log"
        run_benchmark(args, log_file)

        # Update time estimate
        exp_duration = time.time() - exp_start_time
        update_time_estimate(exp_duration)


def collect_ablation_results(output_dir):
    """Collects and aggregates results from ablation studies into summary CSVs.

    Args:
        output_dir (Path): The base output directory containing all results
    """

    # Create summary directory
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Initialize dictionaries to hold DataFrames for each run type
    run_type_dfs = {
        "miniimagenet": [],
        "galaxymnist": [],
        "training_iterations": [],
        "active_learning": [],
        "n_samples_ablation": [],
    }

    # Function to determine run type from directory path
    def determine_run_type(dir_path):
        path_str = str(dir_path)
        if "n_samples_ablation" in path_str:
            return "n_samples_ablation"
        elif "active_learning" in path_str:
            return "active_learning"
        elif "training_iterations" in path_str:
            return "training_iterations"
        elif "miniimagenet" in path_str:
            return "miniimagenet"
        elif "galaxymnist" in path_str:
            return "galaxymnist"
        else:
            return None

    # Find all results_summary.csv files
    summary_files = glob.glob(str(output_dir / "**" / "results_summary.csv"), recursive=True)

    print(f"Found {len(summary_files)} summary files")

    for summary_file in summary_files:
        file_path = Path(summary_file)
        run_dir = file_path.parent
        run_type = determine_run_type(run_dir)

        if run_type is None:
            print(f"Could not determine run type for {run_dir}, skipping")
            continue

        try:
            # Read the CSV file
            df = pd.read_csv(summary_file)

            # Add directory information
            df["run_dir"] = str(run_dir)

            # Append to the correct list
            run_type_dfs[run_type].append(df)
        except Exception as e:
            print(f"Error processing {summary_file}: {e}")

    # Concatenate and save results for each run type
    for run_type, dfs in run_type_dfs.items():
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            output_file = summary_dir / f"{run_type}_results.csv"
            combined_df.to_csv(output_file, index=False)
            print(f"Saved combined results for {run_type} to {output_file}")

            # Print some statistics
            if "n_samples" in combined_df.columns and "final_auroc" in combined_df.columns:
                print(f"\nSummary statistics for {run_type}:")
                for n_samples, group in combined_df.groupby("n_samples"):
                    print(f"n_samples={n_samples}:")
                    print(f"  Mean AUROC: {group['final_auroc'].mean():.4f}")
                    print(f"  Mean AUPRC: {group['final_auprc'].mean():.4f}")
                    if "top_0.1pct_anomalies_found" in group.columns:
                        print(
                            f"  Mean top 0.1% anomalies found: {group['top_0.1pct_anomalies_found'].mean():.2f}%"
                        )
                        print(
                            f"  Mean top 0.1% precision: {group['top_0.1pct_precision'].mean():.2f}%"
                        )
                    if "top_1.0pct_anomalies_found" in group.columns:
                        print(
                            f"  Mean top 1.0% anomalies found: {group['top_1.0pct_anomalies_found'].mean():.2f}%"
                        )
                        print(
                            f"  Mean top 1.0% precision: {group['top_1.0pct_precision'].mean():.2f}%"
                        )
        else:
            print(f"No results found for {run_type}")

    print(f"\nAll result summaries saved to {summary_dir}")
    return summary_dir


def main():
    """Main function to run all experiments."""
    global start_time, total_experiments
    start_time = time.time()

    # Parse arguments
    args, config = parse_arguments()

    # Extract config values
    run_miniimagenet = config["run_miniimagenet"]
    run_galaxymnist = config["run_galaxymnist"]
    run_training_iterations_study = config["run_training_iterations_study"]
    run_active_learning_ablation = config["run_active_learning_ablation"]
    run_n_samples_ablation = config["run_n_samples_ablation"]
    seed = config["seed"]

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Results will be saved to: {output_dir}")

    # Calculate total experiments to run
    total_experiments = calculate_total_experiments()
    print(f"Total experiments to run: {total_experiments}")

    # Log experiment configuration
    with open(output_dir / "experiment_config.txt", "w") as f:
        f.write("Experiment configuration:\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Running miniImageNet: {run_miniimagenet}\n")
        f.write(f"Running GalaxyMNIST: {run_galaxymnist}\n")
        f.write(f"Running training iterations study: {run_training_iterations_study}\n")
        f.write(f"Running active learning ablation: {run_active_learning_ablation}\n")
        f.write(f"Running N_SAMPLES ablation: {run_n_samples_ablation}\n")
        f.write(f"\nMiniImageNet classes: {MINIIMAGENET_CLASSES}\n")
        f.write(f"GalaxyMNIST classes: {GALAXYMNIST_CLASSES}\n")
        f.write(f"Training iterations values: {TRAINING_ITERATIONS_VALUES}\n")
        f.write(f"N_SAMPLES values: {N_SAMPLES_VALUES}\n")
        f.write(f"N_MISLABELED values: {N_MISLABELED_VALUES}\n")

    # Run selected experiments
    if run_miniimagenet:
        run_miniimagenet_experiments(output_dir, seed)

    if run_galaxymnist:
        run_galaxymnist_experiments(output_dir, seed)

    if run_training_iterations_study:
        training_iterations_study(output_dir, seed)

    if run_active_learning_ablation:
        active_learning_ablation(output_dir, seed)

    if run_n_samples_ablation:
        n_samples_ablation(output_dir, seed)

    # Collect and aggregate results
    summary_dir = collect_ablation_results(output_dir)

    # Create a README file with experiment details
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# AnomalyMatch Benchmark Results\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Experiment Summary\n\n")
        # Using ASCII compatible characters instead of Unicode symbols
        f.write(f"- MiniImageNet experiments: {'Yes' if run_miniimagenet else 'No'}\n")
        f.write(f"- GalaxyMNIST experiments: {'Yes' if run_galaxymnist else 'No'}\n")
        f.write(
            f"- Training iterations study: {'Yes' if run_training_iterations_study else 'No'}\n"
        )
        f.write(f"- Active learning ablation: {'Yes' if run_active_learning_ablation else 'No'}\n")
        f.write(f"- N_SAMPLES ablation: {'Yes' if run_n_samples_ablation else 'No'}\n\n")

        f.write("## Summary Files\n\n")
        f.write(
            f"Combined result summaries can be found in the `{summary_dir.relative_to(output_dir)}` directory.\n\n"
        )

        f.write("## Configuration\n\n")
        f.write(f"- Random seed: {seed}\n")
        f.write(f"- Default training runs: {DEFAULT_TRAINING_RUNS}\n")
        f.write(f"- Default training iterations: {DEFAULT_TRAIN_ITERATIONS}\n")
        f.write(f"- Default N_MISLABELED: {DEFAULT_N_MISLABELED}\n\n")

        if run_n_samples_ablation:
            f.write("## N_SAMPLES Ablation Study\n\n")
            f.write("This study examines the effect of varying the number of labeled samples.\n\n")
            f.write("| N_SAMPLES | N_MISLABELED |\n")
            f.write("|-----------|-------------|\n")
            for n_samples, n_mislabeled in zip(N_SAMPLES_VALUES, N_MISLABELED_VALUES):
                f.write(f"| {n_samples} | {n_mislabeled} |\n")

    # Calculate and print total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nAll experiments completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results are saved in: {output_dir}")
    print(f"Summary results are in: {summary_dir}")

    return output_dir


if __name__ == "__main__":
    main()
