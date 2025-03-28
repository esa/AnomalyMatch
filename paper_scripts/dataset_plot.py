#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""
Dataset visualization script for GalaxyMNIST and MiniImageNet.

This script creates a compact, high-DPI grid visualization of sample images
from the GalaxyMNIST and MiniImageNet datasets for the AnomalyMatch paper.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image

# Figure settings for paper-quality output
FIGURE_DPI = 300
FIGURE_WIDTH = 12  # Wider to accommodate 12 columns
FIGURE_HEIGHT = 5  # Adjusted for 1x12 + 3x12 layout

# Class name mappings for both datasets
GALAXYMNIST_CLASS_NAMES = {
    0: "Smooth \n Round",
    1: "Smooth \n Cigar-shaped",
    2: "Edge-on \n Disk",
    3: "Unbarred \n Spiral",
}

MINIIMAGENET_CLASS_NAMES = {48: "Guitar", 57: "Hourglass", 68: "Printer", 85: "Piano", 95: "Orange"}


def load_image(file_path):
    """
    Load and return an image from the specified file path.

    Args:
        file_path (str): Path to the image file

    Returns:
        PIL.Image: Loaded image object
    """
    try:
        img = Image.open(file_path)
        img = img.convert("RGB")  # Ensure RGB format
        return img
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


def get_galaxymnist_samples(df, data_dir, samples_per_class=3):
    """
    Get sample images from the GalaxyMNIST dataset with equal class representation.

    Args:
        df (pandas.DataFrame): DataFrame with GalaxyMNIST metadata
        data_dir (str): Directory containing the images
        samples_per_class (int): Number of samples to get per class

    Returns:
        list: Sample dictionaries with path and class information
    """
    samples = []
    samples_by_class = {0: [], 1: [], 2: [], 3: []}  # Initialize empty list for each class

    # Process each class
    for class_idx in range(4):  # 4 classes in GalaxyMNIST
        # Get samples for this class
        class_df = df[df["label_idx"] == class_idx]
        # Use a larger sample size initially to ensure we have enough valid images
        class_samples = class_df.sample(min(samples_per_class * 2, len(class_df)), random_state=42)

        # Process each sample
        for _, row in class_samples.iterrows():
            img_path = os.path.join(data_dir, row["filename"])
            if os.path.exists(img_path):
                samples_by_class[class_idx].append(
                    {
                        "path": img_path,
                        "class_name": GALAXYMNIST_CLASS_NAMES[row["label_idx"]],
                        "class_idx": row["label_idx"],
                        "dataset": "GalaxyMNIST",
                    }
                )
                # Break once we have enough valid samples
                if len(samples_by_class[class_idx]) >= samples_per_class:
                    break

    # Ensure we have exactly samples_per_class images for each class
    for class_idx in range(4):
        if len(samples_by_class[class_idx]) < samples_per_class:
            print(
                f"Warning: Only found {len(samples_by_class[class_idx])} samples for GalaxyMNIST class {class_idx}"
            )
            # If we don't have enough samples, duplicate the last one to fill
            while len(samples_by_class[class_idx]) < samples_per_class:
                if samples_by_class[class_idx]:  # If at least one sample exists
                    samples_by_class[class_idx].append(samples_by_class[class_idx][-1])
                else:
                    # Extreme fallback - create a placeholder
                    print(f"Error: No samples found for GalaxyMNIST class {class_idx}")
                    break

        # Add exactly samples_per_class to the final samples list
        samples.extend(samples_by_class[class_idx][:samples_per_class])

    return samples


def get_miniimagenet_samples(df, data_dir, anomaly_samples_per_class=2, normal_samples_total=30):
    """
    Get sample images from the MiniImageNet dataset with both anomaly and normal classes.

    Args:
        df (pandas.DataFrame): DataFrame with MiniImageNet metadata
        data_dir (str): Directory containing the images
        anomaly_samples_per_class (int): Number of samples to get per anomaly class
        normal_samples_total (int): Total number of normal class samples

    Returns:
        list: Sample dictionaries with path and class information
    """
    samples = []

    # First include samples from anomaly classes
    for class_idx, class_name in MINIIMAGENET_CLASS_NAMES.items():
        class_df = df[df["label_idx"] == class_idx]
        if len(class_df) > 0:
            class_samples = class_df.sample(
                min(anomaly_samples_per_class, len(class_df)), random_state=42
            )

            for _, row in class_samples.iterrows():
                img_path = os.path.join(data_dir, row["filename"])
                if os.path.exists(img_path):
                    samples.append(
                        {
                            "path": img_path,
                            "class_name": class_name,
                            "class_idx": class_idx,
                            "is_anomaly_class": True,
                            "dataset": "MiniImageNet",
                        }
                    )

    # Then include normal samples (not from anomaly classes)
    normal_df = df[~df["label_idx"].isin(MINIIMAGENET_CLASS_NAMES.keys())]
    if len(normal_df) > 0:
        normal_samples = normal_df.sample(
            min(normal_samples_total, len(normal_df)), random_state=42
        )

        for _, row in normal_samples.iterrows():
            img_path = os.path.join(data_dir, row["filename"])
            if os.path.exists(img_path):
                samples.append(
                    {
                        "path": img_path,
                        "class_name": "Nominal",  # Just use "Nominal" for all non-anomaly classes
                        "class_idx": row["label_idx"],
                        "is_anomaly_class": False,
                        "dataset": "MiniImageNet",
                    }
                )

    return samples


def create_compact_figure(
    galaxymnist_samples,
    miniimagenet_samples,
    figwidth=FIGURE_WIDTH,
    figheight=FIGURE_HEIGHT,
    dpi=FIGURE_DPI,
):
    """
    Create a compact figure with sample images from both datasets with in-image annotations.

    Args:
        galaxymnist_samples (list): List of dictionaries containing GalaxyMNIST image data
        miniimagenet_samples (list): List of dictionaries containing MiniImageNet image data
        figwidth (float): Width of the figure in inches
        figheight (float): Height of the figure in inches
        dpi (int): DPI for the output figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Set font properties for in-image annotations
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,  # Increased font size
        }
    )

    # Create figure
    fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi)

    # Define grid layout
    n_cols = 12  # 12 columns as requested
    n_galaxy_rows = 1  # 1 row for GalaxyMNIST
    n_mini_rows = 3  # 3 rows for MiniImageNet

    # Create GridSpec for layout
    gs = gridspec.GridSpec(
        n_galaxy_rows + n_mini_rows,
        n_cols,
        figure=fig,
        wspace=0.02,
        hspace=0.0,  # Remove vertical white space between rows
        left=0.01,
        right=0.99,
        top=0.95,  # Adjusted to make room for titles
        bottom=0.01,
    )

    # Organize GalaxyMNIST samples by class for layout
    galaxy_by_class = {}
    for sample in galaxymnist_samples:
        class_idx = sample["class_idx"]
        if class_idx not in galaxy_by_class:
            galaxy_by_class[class_idx] = []
        galaxy_by_class[class_idx].append(sample)

    # Assign GalaxyMNIST samples to grid positions
    # For 1x12 layout, distribute each class evenly across the row
    for class_idx, samples in galaxy_by_class.items():
        for i, sample in enumerate(samples):
            # Calculate column - distribute evenly
            col = class_idx * 3 + i  # 3 samples per class, 4 classes = 12 columns

            ax = fig.add_subplot(gs[0, col])

            # Load and display image
            img = np.array(Image.open(sample["path"]))
            ax.imshow(img)

            # Add class name as annotation inside the image
            # Create a semi-transparent background rectangle for text
            rect = Rectangle(
                (0, 0), img.shape[1], 20, color="black", alpha=0.6
            )  # Taller rect for larger font
            ax.add_patch(rect)

            # Add text (white on black background)
            # For GalaxyMNIST, use first word of class name to save space
            ax.text(
                img.shape[1] / 2,
                36,  # Adjusted position for centering with taller rect
                sample["class_name"],
                color="white",
                fontsize=8,  # Increased font size
                ha="center",
                va="center",
            )

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add thin border
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["right"].set_linewidth(0.5)
            ax.spines["top"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)

    # Organize MiniImageNet samples
    anomaly_samples = [s for s in miniimagenet_samples if s.get("is_anomaly_class", False)]
    normal_samples = [s for s in miniimagenet_samples if not s.get("is_anomaly_class", False)]

    # First allocate space for anomaly samples - 10 total (5 classes, 2 per class)
    # We'll allocate them to the first row of MiniImageNet samples
    for i, sample in enumerate(anomaly_samples):
        if i >= 10:  # Only want 10 anomaly samples (5 classes × 2 samples)
            break

        col = i % n_cols
        row = 1  # First row of MiniImageNet (after GalaxyMNIST row)

        ax = fig.add_subplot(gs[row, col])

        # Load and display image
        img = np.array(Image.open(sample["path"]))
        ax.imshow(img)

        # Add class name as annotation inside the image
        rect = Rectangle(
            (0, 0), img.shape[1], 20, color="red", alpha=0.6
        )  # Taller rect for larger font
        ax.add_patch(rect)

        # Add text (white on red background)
        ax.text(
            img.shape[1] / 2,
            10,  # Adjusted position for centering with taller rect
            sample["class_name"],
            color="white",
            fontsize=8,  # Increased font size
            ha="center",
            va="center",
        )

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Red border for anomaly classes
        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(0.5)

    # Fill the remaining MiniImageNet grid with normal samples
    normal_idx = 0
    for row in range(1, n_galaxy_rows + n_mini_rows):  # Start from row 1 (after GalaxyMNIST)
        # Skip first row anomaly positions
        start_col = 10 if row == 1 else 0  # Skip the anomaly samples in the first row

        for col in range(start_col, n_cols):
            if normal_idx < len(normal_samples):
                sample = normal_samples[normal_idx]

                ax = fig.add_subplot(gs[row, col])

                # Load and display image
                img = np.array(Image.open(sample["path"]))
                ax.imshow(img)

                # Add class name as annotation inside the image
                rect = Rectangle(
                    (0, 0), img.shape[1], 20, color="black", alpha=0.6
                )  # Taller rect for larger font
                ax.add_patch(rect)

                # Add text (white on black background)
                ax.text(
                    img.shape[1] / 2,
                    10,  # Adjusted position for centering with taller rect
                    "Nominal",
                    color="white",
                    fontsize=8,  # Increased font size
                    ha="center",
                    va="center",
                )

                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Add thin border
                ax.spines["left"].set_linewidth(0.5)
                ax.spines["right"].set_linewidth(0.5)
                ax.spines["top"].set_linewidth(0.5)
                ax.spines["bottom"].set_linewidth(0.5)

                normal_idx += 1

    # Add dataset labels as text at top of figure
    fig.text(0.01, 0.98, "GalaxyMNIST", fontsize=12, fontweight="bold", ha="left")
    fig.text(0.01, 0.70, "MiniImageNet", fontsize=12, fontweight="bold", ha="left")

    # Add "Anomaly Classes in Red" annotation at top right
    fig.text(0.99, 0.98, "Anomaly Classes in Red", fontsize=10, color="red", ha="right")

    return fig


def main():
    """Main function to create and save the dataset visualization."""
    # Define base paths
    datasets_dir = os.path.join("datasets/")

    # Define paths for dataset files
    galaxymnist_csv_path = os.path.join(datasets_dir, "labels_galaxymnist.csv")
    miniimagenet_csv_path = os.path.join(datasets_dir, "labels_miniimagenet.csv")

    galaxymnist_image_dir = os.path.join(datasets_dir, "galaxymnist")
    miniimagenet_image_dir = os.path.join(datasets_dir, "miniimagenet")

    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Load CSV files
    try:
        galaxymnist_df = pd.read_csv(galaxymnist_csv_path)
        print(f"GalaxyMNIST data loaded. Shape: {galaxymnist_df.shape}")
    except Exception as e:
        print(f"Error loading GalaxyMNIST data: {e}")
        galaxymnist_df = None

    try:
        miniimagenet_df = pd.read_csv(miniimagenet_csv_path)
        print(f"MiniImageNet data loaded. Shape: {miniimagenet_df.shape}")
    except Exception as e:
        print(f"Error loading MiniImageNet data: {e}")
        miniimagenet_df = None

    # Get samples from each dataset
    galaxymnist_samples = []
    miniimagenet_samples = []

    if galaxymnist_df is not None:
        galaxymnist_samples = get_galaxymnist_samples(
            galaxymnist_df,
            galaxymnist_image_dir,
            samples_per_class=3,  # Get 3 samples per class (12 total for 1x12 grid)
        )
        print(f"Selected {len(galaxymnist_samples)} GalaxyMNIST samples")

    if miniimagenet_df is not None:
        miniimagenet_samples = get_miniimagenet_samples(
            miniimagenet_df,
            miniimagenet_image_dir,
            anomaly_samples_per_class=2,  # 2 samples per anomaly class (10 total)
            normal_samples_total=26,  # Need 26 normal samples to fill the 3x12 grid (after 10 anomaly samples)
        )
        print(f"Selected {len(miniimagenet_samples)} MiniImageNet samples")

    # Create and save figure
    if galaxymnist_samples and miniimagenet_samples:
        fig = create_compact_figure(galaxymnist_samples, miniimagenet_samples)

        # Save figure with high DPI
        output_path = os.path.join(output_dir, "dataset_samples_compact.png")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")

        # Also save a PDF version for vector graphics
        pdf_path = os.path.join(output_dir, "dataset_samples_compact.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")

        print(f"Dataset visualization saved to {output_path} and {pdf_path}")
        plt.close(fig)
    else:
        print("Unable to create figure: missing dataset samples")


if __name__ == "__main__":
    main()
