#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Dataset visualization script for GalaxyMNIST, MiniImageNet, and GalaxyZoo.

This script creates a compact, high-DPI grid visualization of sample images
from the GalaxyMNIST, MiniImageNet, and GalaxyZoo datasets for the AnomalyMatch paper.
"""

import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

# Figure settings for paper-quality output
FIGURE_DPI = 300
FIGURE_WIDTH = 12  # Wider to accommodate 12 columns
FIGURE_HEIGHT = 7  # Increased to accommodate additional GalaxyZoo row

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


def get_galaxyzoo_samples(df, data_dir, anomaly_samples=6, normal_samples=6):
    """
    Get sample images from the GalaxyZoo dataset with both anomaly and normal galaxies.

    Args:
        df (pandas.DataFrame): DataFrame with GalaxyZoo metadata
        data_dir (str): Directory containing the images
        anomaly_samples (int): Number of anomaly samples to get
        normal_samples (int): Number of normal samples to get

    Returns:
        list: Sample dictionaries with path and class information
    """
    samples = []

    # Get anomaly samples (label_idx == 1)
    anomaly_df = df[df["label_idx"] == 1]
    if len(anomaly_df) > 0:
        anomaly_sample_rows = anomaly_df.sample(
            min(anomaly_samples, len(anomaly_df)), random_state=42
        )

        for _, row in anomaly_sample_rows.iterrows():
            img_path = os.path.join(data_dir, row["original_filename"])
            if os.path.exists(img_path):
                samples.append(
                    {
                        "path": img_path,
                        "class_name": "Anomaly",
                        "class_idx": row["label_idx"],
                        "is_anomaly_class": True,
                        "dataset": "GalaxyZoo",
                        "anomaly_score": row.get("anomaly_score_raw", 0.0),
                    }
                )

    # Get normal samples (label_idx == 0)
    normal_df = df[df["label_idx"] == 0]
    if len(normal_df) > 0:
        normal_sample_rows = normal_df.sample(min(normal_samples, len(normal_df)), random_state=42)

        for _, row in normal_sample_rows.iterrows():
            img_path = os.path.join(data_dir, row["original_filename"])
            if os.path.exists(img_path):
                samples.append(
                    {
                        "path": img_path,
                        "class_name": "Normal",
                        "class_idx": row["label_idx"],
                        "is_anomaly_class": False,
                        "dataset": "GalaxyZoo",
                        "anomaly_score": row.get("anomaly_score_raw", 0.0),
                    }
                )

    return samples


def create_compact_figure(
    galaxymnist_samples,
    miniimagenet_samples,
    galaxyzoo_samples,
    figwidth=FIGURE_WIDTH,
    figheight=FIGURE_HEIGHT,
    dpi=FIGURE_DPI,
):
    """
    Create a compact figure with sample images from all three datasets with in-image annotations.

    Args:
        galaxymnist_samples (list): List of dictionaries containing GalaxyMNIST image data
        miniimagenet_samples (list): List of dictionaries containing MiniImageNet image data
        galaxyzoo_samples (list): List of dictionaries containing GalaxyZoo image data
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
    n_zoo_rows = 1  # 1 row for GalaxyZoo
    n_mini_rows = 3  # 3 rows for MiniImageNet

    # Create GridSpec for layout
    gs = gridspec.GridSpec(
        n_galaxy_rows + n_zoo_rows + n_mini_rows,
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

    # Assign GalaxyMNIST samples to grid positions (row 0)
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
            rect = Rectangle((0, 0), img.shape[1], img.shape[1] * 0.25, color="black", alpha=0.6)
            ax.add_patch(rect)

            ax.text(
                img.shape[1] / 2,
                img.shape[1] * 0.005,
                sample["class_name"],
                color="white",
                fontsize=8,
                ha="center",
                va="top",
            )

            # Remove ticks and add border
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # Assign GalaxyZoo samples to grid positions (row 1)
    # Show anomaly and normal samples
    galaxyzoo_anomaly = [s for s in galaxyzoo_samples if s.get("is_anomaly_class", False)]
    galaxyzoo_normal = [s for s in galaxyzoo_samples if not s.get("is_anomaly_class", False)]

    # Fill first 6 columns with anomaly samples
    for i, sample in enumerate(galaxyzoo_anomaly[:6]):
        ax = fig.add_subplot(gs[1, i])

        img = np.array(Image.open(sample["path"]))
        ax.imshow(img)

        rect = Rectangle((0, 0), img.shape[1], img.shape[1] * 0.13, color="red", alpha=0.6)
        ax.add_patch(rect)

        ax.text(
            img.shape[1] / 2,
            img.shape[1] * 0.01,
            "Anomaly",
            color="white",
            fontsize=8,
            ha="center",
            va="top",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(0.5)

    # Fill remaining columns with normal samples
    for i, sample in enumerate(galaxyzoo_normal[:6]):
        col = i + 6  # Start after anomaly samples
        ax = fig.add_subplot(gs[1, col])

        img = np.array(Image.open(sample["path"]))
        ax.imshow(img)

        rect = Rectangle((0, 0), img.shape[1], img.shape[1] * 0.13, color="black", alpha=0.6)
        ax.add_patch(rect)

        ax.text(
            img.shape[1] / 2,
            img.shape[1] * 0.01,
            "Normal",
            color="white",
            fontsize=8,
            ha="center",
            va="top",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # Organize MiniImageNet samples for rows 2-4
    mini_anomaly = [s for s in miniimagenet_samples if s.get("is_anomaly_class", False)]
    mini_normal = [s for s in miniimagenet_samples if not s.get("is_anomaly_class", False)]

    # Fill first row of MiniImageNet (row 2) with anomaly samples
    for i, sample in enumerate(mini_anomaly[:10]):
        ax = fig.add_subplot(gs[2, i])

        img = np.array(Image.open(sample["path"]))
        ax.imshow(img)

        rect = Rectangle((0, 0), img.shape[1], img.shape[1] * 0.13, color="red", alpha=0.6)
        ax.add_patch(rect)

        ax.text(
            img.shape[1] / 2,
            img.shape[1] * 0.01,
            sample["class_name"],
            color="white",
            fontsize=8,
            ha="center",
            va="top",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(0.5)

    # Fill remaining anomaly samples and normal samples in row 2
    normal_idx = 0
    for col in range(10, n_cols):
        if normal_idx < len(mini_normal):
            sample = mini_normal[normal_idx]
            ax = fig.add_subplot(gs[2, col])

            img = np.array(Image.open(sample["path"]))
            ax.imshow(img)

            rect = Rectangle((0, 0), img.shape[1], img.shape[1] * 0.13, color="black", alpha=0.6)
            ax.add_patch(rect)

            ax.text(
                img.shape[1] / 2,
                img.shape[1] * 0.01,
                "Nominal",
                color="white",
                fontsize=8,
                ha="center",
                va="top",
            )

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            normal_idx += 1

    # Fill remaining rows (3-4) with normal samples
    for row in range(3, 5):
        for col in range(n_cols):
            if normal_idx < len(mini_normal):
                sample = mini_normal[normal_idx]
                ax = fig.add_subplot(gs[row, col])

                img = np.array(Image.open(sample["path"]))
                ax.imshow(img)

                rect = Rectangle(
                    (0, 0), img.shape[1], img.shape[1] * 0.13, color="black", alpha=0.6
                )
                ax.add_patch(rect)

                ax.text(
                    img.shape[1] / 2,
                    img.shape[1] * 0.01,
                    "Nominal",
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="top",
                )

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)

                normal_idx += 1

    # Add dataset labels
    fig.text(0.01, 0.94, "GalaxyMNIST", fontsize=12, fontweight="bold", ha="left")
    fig.text(0.01, 0.75, "Galaxy Zoo 2", fontsize=12, fontweight="bold", ha="left")
    fig.text(0.01, 0.56, "MiniImageNet", fontsize=12, fontweight="bold", ha="left")

    # Add "Anomaly Classes in Red" annotation at top right
    fig.text(0.99, 0.75, "Anomaly Classes in Red", fontsize=10, color="red", ha="right")
    fig.text(0.99, 0.56, "Anomaly Classes in Red", fontsize=10, color="red", ha="right")

    return fig


def main():
    """Main function to create and save the dataset visualization."""
    # Define base paths
    datasets_dir = os.path.join("/media/team_workspaces/AnomalyMatch/paper_datasets/")

    # Define paths for dataset files
    galaxymnist_csv_path = os.path.join(datasets_dir, "labels_galaxymnist.csv")
    miniimagenet_csv_path = os.path.join(datasets_dir, "labels_miniimagenet.csv")
    galaxyzoo_csv_path = os.path.join(datasets_dir, "galaxyzoo_labels.csv")

    galaxymnist_image_dir = os.path.join(datasets_dir, "galaxymnist")
    miniimagenet_image_dir = os.path.join(datasets_dir, "miniimagenet")
    galaxyzoo_image_dir = os.path.join(datasets_dir, "galaxyzoo")

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

    try:
        galaxyzoo_df = pd.read_csv(galaxyzoo_csv_path)
        print(f"GalaxyZoo data loaded. Shape: {galaxyzoo_df.shape}")
    except Exception as e:
        print(f"Error loading GalaxyZoo data: {e}")
        galaxyzoo_df = None

    # Get samples from each dataset
    galaxymnist_samples = []
    miniimagenet_samples = []
    galaxyzoo_samples = []

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
            normal_samples_total=26,  # Need 26 normal samples to fill the remaining grid
        )
        print(f"Selected {len(miniimagenet_samples)} MiniImageNet samples")

    if galaxyzoo_df is not None:
        galaxyzoo_samples = get_galaxyzoo_samples(
            galaxyzoo_df,
            galaxyzoo_image_dir,
            anomaly_samples=6,  # 6 anomaly samples
            normal_samples=6,  # 6 normal samples for 1x12 grid
        )
        print(f"Selected {len(galaxyzoo_samples)} GalaxyZoo samples")

    # Create and save figure
    if galaxymnist_samples and miniimagenet_samples and galaxyzoo_samples:
        fig = create_compact_figure(galaxymnist_samples, miniimagenet_samples, galaxyzoo_samples)

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
        if not galaxymnist_samples:
            print("- Missing GalaxyMNIST samples")
        if not miniimagenet_samples:
            print("- Missing MiniImageNet samples")
        if not galaxyzoo_samples:
            print("- Missing GalaxyZoo samples")


if __name__ == "__main__":
    main()
