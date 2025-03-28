#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""
Generate example images with weak and strong augmentations from MiniImageNet.

This script creates examples of weakly and strongly augmented images from the
MiniImageNet dataset, focusing on 'hourglass' class (anomaly) and random other
classes (nominal). Images are saved with and without annotations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

sys.path.append("/media/home/AnomalyMatch")
sys.path.append("../")
from anomaly_match.datasets.augmentation.randaugment import RandAugment

# Constants
HOURGLASS_CLASS_IDX = 57  # Class ID for hourglass images (anomaly)
NUM_EXAMPLES = 10  # Number of examples to generate for each class
IMAGE_SIZE = (224, 224)  # Size to resize images to (if needed)
FIGURE_DPI = 100


def setup_output_directories():
    """Create output directories for the example images."""
    output_dir = Path("figures/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_dataset_info(csv_path):
    """
    Load dataset information from CSV file.

    Args:
        csv_path (str): Path to the CSV file with image information

    Returns:
        pandas.DataFrame: DataFrame with dataset information
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset info loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return None


def get_sample_images(
    df, image_dir, anomaly_class_idx=HOURGLASS_CLASS_IDX, num_examples=NUM_EXAMPLES
):
    """
    Get sample images from both anomaly and normal classes.

    Args:
        df (pandas.DataFrame): DataFrame with dataset information
        image_dir (str): Directory containing the images
        anomaly_class_idx (int): Class index for anomaly class
        num_examples (int): Number of examples to get for each category

    Returns:
        tuple: Lists of anomaly and normal samples
    """
    # Get anomaly samples (hourglass)
    anomaly_df = df[df["label_idx"] == anomaly_class_idx]
    if len(anomaly_df) > 0:
        anomaly_samples = anomaly_df.sample(min(num_examples, len(anomaly_df)), random_state=42)
    else:
        print(f"No samples found for anomaly class {anomaly_class_idx}")
        anomaly_samples = pd.DataFrame()

    # Get normal samples (not hourglass)
    normal_df = df[df["label_idx"] != anomaly_class_idx]
    if len(normal_df) > 0:
        normal_samples = normal_df.sample(min(num_examples, len(normal_df)), random_state=42)
    else:
        print("No samples found for normal classes")
        normal_samples = pd.DataFrame()

    # Convert to list of dicts with path and metadata
    anomaly_list = []
    for _, row in anomaly_samples.iterrows():
        img_path = os.path.join(image_dir, row["filename"])
        if os.path.exists(img_path):
            anomaly_list.append(
                {
                    "path": img_path,
                    "filename": row["filename"],
                    "class_idx": row["label_idx"],
                    "is_anomaly": True,
                    "class_name": "Anomaly",
                }
            )

    normal_list = []
    for _, row in normal_samples.iterrows():
        img_path = os.path.join(image_dir, row["filename"])
        if os.path.exists(img_path):
            normal_list.append(
                {
                    "path": img_path,
                    "filename": row["filename"],
                    "class_idx": row["label_idx"],
                    "is_anomaly": False,
                    "class_name": "Nominal",
                }
            )

    return anomaly_list, normal_list


def create_augmentations():
    """
    Create weak and strong transformations for image augmentation.

    Returns:
        tuple: (weak_transform, strong_transform)
    """
    # Define a simple weak transform (as used in BasicDataset)
    weak_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    # Create strong transform (following BasicDataset approach)
    strong_transform = transforms.Compose(
        [
            RandAugment(3, 5),  # Apply RandAugment as first step
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    return weak_transform, strong_transform


def apply_transforms(image, weak_transform, strong_transform):
    """
    Apply weak and strong transformations to an image.

    Args:
        image (PIL.Image): Image to transform
        weak_transform: Weak transformation pipeline
        strong_transform: Strong transformation pipeline

    Returns:
        tuple: (original_tensor, weak_tensor, strong_tensor)
    """
    # Convert to tensor for the original image
    to_tensor = transforms.ToTensor()
    original_tensor = to_tensor(image)

    # Apply weak and strong transforms
    weak_tensor = weak_transform(image)
    strong_tensor = strong_transform(image)

    return original_tensor, weak_tensor, strong_tensor


def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image.

    Args:
        tensor (torch.Tensor): Image tensor

    Returns:
        PIL.Image: PIL Image
    """
    # Convert tensor to numpy array and then to PIL Image
    np_image = tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC

    # Clip values to [0, 1] and then scale to [0, 255]
    np_image = np.clip(np_image, 0, 1) * 255
    np_image = np_image.astype(np.uint8)

    return Image.fromarray(np_image)


def save_examples(samples, output_dir, weak_transform, strong_transform, image_size=IMAGE_SIZE):
    """
    Save example images with their weak and strong augmentations.

    Args:
        samples (list): List of sample dictionaries
        output_dir (Path): Output directory path
        weak_transform: Weak transformation pipeline
        strong_transform: Strong transformation pipeline
        image_size (tuple): Size to resize images to
    """
    for i, sample in enumerate(samples):
        # Load image
        try:
            image = Image.open(sample["path"])
            image = image.convert("RGB")

            # Resize if needed
            if image_size:
                image = image.resize(image_size, Image.LANCZOS)

            # Apply transformations
            original_tensor, weak_tensor, strong_tensor = apply_transforms(
                image, weak_transform, strong_transform
            )

            # Convert back to PIL images for saving
            original_image = tensor_to_image(original_tensor)
            weak_image = tensor_to_image(weak_tensor)
            strong_image = tensor_to_image(strong_tensor)

            # Generate filenames
            class_type = "anomaly" if sample["is_anomaly"] else "nominal"
            base_filename = f"{class_type}_{i + 1}"

            # Save images without annotation
            original_image.save(os.path.join(output_dir, f"{base_filename}_original.png"))
            weak_image.save(os.path.join(output_dir, f"{base_filename}_weak.png"))
            strong_image.save(os.path.join(output_dir, f"{base_filename}_strong.png"))

            # Save images with annotation
            save_annotated_image(
                original_image,
                os.path.join(output_dir, f"{base_filename}_original_annotated.png"),
                sample["class_name"],
                sample["is_anomaly"],
            )
            save_annotated_image(
                weak_image,
                os.path.join(output_dir, f"{base_filename}_weak_annotated.png"),
                sample["class_name"],
                sample["is_anomaly"],
            )
            save_annotated_image(
                strong_image,
                os.path.join(output_dir, f"{base_filename}_strong_annotated.png"),
                sample["class_name"],
                sample["is_anomaly"],
            )

            print(f"Saved examples for {base_filename}")

        except Exception as e:
            print(f"Error processing {sample['path']}: {e}")


def save_annotated_image(image, output_path, class_name, is_anomaly):
    """
    Save image with class annotation overlay.

    Args:
        image (PIL.Image): Image to annotate
        output_path (str): Path to save the annotated image
        class_name (str): Class name to display
        is_anomaly (bool): Whether this is an anomaly class
    """
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIGURE_DPI)

    # Display the image
    ax.imshow(img_array)

    # Set background color based on anomaly status
    bg_color = "red" if is_anomaly else "black"

    # Create background rectangle for text
    rect = Rectangle((0, 0), img_array.shape[1], 20, color=bg_color, alpha=0.6)
    ax.add_patch(rect)

    # Add text annotation
    ax.text(
        img_array.shape[1] / 2,
        10,
        class_name,
        color="white",
        fontsize=40,
        ha="center",
        va="center",
    )

    # Remove axes and white space
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save and close
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    """Main function to generate example images."""
    # Setup paths
    datasets_dir = os.path.join("datasets")
    miniimagenet_csv_path = os.path.join(datasets_dir, "labels_miniimagenet.csv")
    miniimagenet_image_dir = os.path.join(datasets_dir, "miniimagenet")

    # Create output directory
    output_dir = setup_output_directories()

    # Load dataset info
    df = load_dataset_info(miniimagenet_csv_path)
    if df is None:
        print("Could not load dataset information. Exiting.")
        return

    # Get sample images
    anomaly_samples, normal_samples = get_sample_images(df, miniimagenet_image_dir)
    print(f"Found {len(anomaly_samples)} anomaly samples and {len(normal_samples)} normal samples")

    # Create transformations
    weak_transform, strong_transform = create_augmentations()

    # Generate and save example images
    save_examples(anomaly_samples, output_dir, weak_transform, strong_transform)
    save_examples(normal_samples, output_dir, weak_transform, strong_transform)

    print(f"All example images saved to {output_dir}")


if __name__ == "__main__":
    main()
