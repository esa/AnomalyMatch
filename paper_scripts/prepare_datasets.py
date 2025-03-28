#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""
Dataset preparation script for AnomalyMatch paper experiments.

This script prepares both miniImageNet and galaxyMNIST datasets:
1. Downloads and processes data (if needed)
2. Saves images in a consistent format in the datasets folder
3. Creates label CSV files with original class labels
4. Creates HDF5 files for quick loading

Usage:
    python prepare_datasets.py [--dataset {miniimagenet,galaxymnist,all}]

Requirements:
    - galaxy-datasets package for GalaxyMNIST dataset
    - miniImageNet data files (parquet format) in mini-imagenet/data/
"""

import os
import io
import argparse
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import torch
from tqdm import tqdm
from loguru import logger
import concurrent.futures
import pyarrow.parquet as pq
import gc  # Add garbage collection

# Configure basic logging
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
logger.add("dataset_preparation.log", rotation="10 MB", level="DEBUG")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare datasets for AnomalyMatch paper")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["miniimagenet", "galaxymnist", "all"],
        default="all",
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Target image size (square images)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Directory to store prepared datasets",
    )
    parser.add_argument(
        "--miniimagenet_dir",
        type=str,
        default="mini-imagenet/data/",
        help="Directory containing miniImageNet parquet files",
    )
    parser.add_argument(
        "--galaxymnist_dir",
        type=str,
        default=None,
        help="Directory to store galaxy mnist data downloads (defaults to output_dir/galaxy_downloads)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for image processing",
    )
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality for storage")
    return parser.parse_args()


def save_image(image, filename, output_path, size, quality=90):
    """Save a single image to disk with specified parameters"""
    if isinstance(image, np.ndarray):
        if image.dtype == np.float32 and image.max() <= 1.0:
            # Convert from float [0,1] to uint8 [0,255]
            image = (image * 255).astype(np.uint8)

        # Handle grayscale images
        if len(image.shape) == 2:
            img = Image.fromarray(image).convert("RGB")
        else:
            img = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            # Convert CHW to HWC if needed
            if image.shape[0] == 3 and image.shape[1] != 3:
                image = image.permute(1, 2, 0)

        # Convert to numpy for PIL
        image_np = image.cpu().numpy()

        if image_np.dtype == np.float32 and image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        img = Image.fromarray(image_np.astype(np.uint8))
    else:
        img = image

    # Resize if needed
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)

    # Save the image
    img.save(output_path, format="JPEG", quality=quality)


def create_hdf5_file(images, filenames, output_path, img_size=224, quality=90):
    """Create a single HDF5 file containing all images and their filenames."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create dataset types
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    vlen_str = h5py.special_dtype(vlen=str)  # for variable-length strings

    # Create HDF5 file
    with h5py.File(output_path, "w") as hf:
        # Create datasets for images and filenames
        images_dset = hf.create_dataset(
            "images", shape=(len(images),), dtype=vlen_uint8, compression="gzip"
        )
        names_dset = hf.create_dataset("filenames", shape=(len(filenames),), dtype=vlen_str)

        # Process each image
        for i, (image, filename) in enumerate(
            tqdm(
                zip(images, filenames),
                total=len(images),
                desc=f"Creating HDF5: {os.path.basename(output_path)}",
            )
        ):
            try:
                # Convert to PIL Image if needed
                if isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
                    # Use a BytesIO buffer to store JPEG data
                    with io.BytesIO() as buffer:
                        # Convert and save to buffer
                        save_image(image, filename, buffer, img_size, quality)
                        buffer.seek(0)
                        jpeg_bytes = buffer.getvalue()
                else:
                    # For PIL images
                    with io.BytesIO() as buffer:
                        if image.size != (img_size, img_size):
                            image = image.resize((img_size, img_size), Image.LANCZOS)
                        image.save(buffer, format="JPEG", quality=quality)
                        buffer.seek(0)
                        jpeg_bytes = buffer.getvalue()

                # Store JPEG bytes and filename
                images_dset[i] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                names_dset[i] = filename

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")

    logger.info(f"Created HDF5 file at {output_path} with {len(images)} images")


def create_hdf5_file_incrementally(
    output_path, total_images, img_size=224, quality=90, batch_size=1000
):
    """Create an HDF5 file incrementally without storing all images in memory.

    Returns:
        A function that can be called to add batches of images to the file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create dataset types
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    vlen_str = h5py.special_dtype(vlen=str)  # for variable-length strings

    # Create the HDF5 file with placeholder datasets
    hf = h5py.File(output_path, "w")
    images_dset = hf.create_dataset(
        "images", shape=(total_images,), dtype=vlen_uint8, compression="gzip"
    )
    names_dset = hf.create_dataset("filenames", shape=(total_images,), dtype=vlen_str)

    # Track current position in the file
    current_idx = 0

    def add_batch(images, filenames):
        """Add a batch of images to the HDF5 file."""
        nonlocal current_idx

        for i, (image, filename) in enumerate(
            tqdm(
                zip(images, filenames),
                total=len(images),
                desc=f"Adding batch to HDF5 (position {current_idx})",
            )
        ):
            try:
                # For PIL images
                with io.BytesIO() as buffer:
                    if image.size != (img_size, img_size):
                        image = image.resize((img_size, img_size), Image.LANCZOS)
                    image.save(buffer, format="JPEG", quality=quality)
                    buffer.seek(0)
                    jpeg_bytes = buffer.getvalue()

                # Store JPEG bytes and filename
                images_dset[current_idx] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                names_dset[current_idx] = filename
                current_idx += 1

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")

        # Force sync to disk
        hf.flush()
        return current_idx

    def finalize():
        """Close the HDF5 file."""
        hf.close()
        logger.info(f"Completed HDF5 file at {output_path} with {current_idx} images")

    # Return functions to add batches and finalize
    return add_batch, finalize


def prepare_galaxymnist(output_dir, download_dir=None, img_size=224):
    """
    Download and prepare the GalaxyMNIST dataset using galaxy-datasets package.

    Args:
        output_dir: Base output directory
        download_dir: Directory to store downloaded files
        img_size: Target image size

    Returns:
        Tuple of (images, filenames, labels_df, output_directory)
    """
    logger.info("Preparing GalaxyMNIST dataset using galaxy-datasets package")

    try:
        # Import galaxy-datasets package instead of the old galaxy_mnist
        from galaxy_datasets import galaxy_mnist
    except ImportError:
        logger.error(
            "galaxy-datasets package not found. Please install it with: pip install galaxy-datasets"
        )
        return None, None, None, None

    # Set download directory
    if download_dir is None:
        download_dir = os.path.join(output_dir, "galaxy_downloads")

    # Create output directories
    galaxy_dir = os.path.join(output_dir, "galaxymnist")
    os.makedirs(galaxy_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # Download and load the data
    logger.info(f"Downloading GalaxyMNIST dataset to {download_dir}")

    # Get the training catalog and images
    train_catalog, label_cols = galaxy_mnist(
        root=download_dir,
        train=True,
        download=True,
    )

    # Get the test catalog and images
    test_catalog, _ = galaxy_mnist(
        root=download_dir,
        train=False,
        download=True,
    )

    logger.info(
        f"Downloaded GalaxyMNIST dataset with {len(train_catalog)} training and {len(test_catalog)} test images"
    )

    # Define class names for GalaxyMNIST
    class_names = ["smooth_round", "smooth_cigar", "edge_on_disk", "unbarred_spiral"]

    # Process and collect all images
    all_images = []
    all_filenames = []
    all_labels = []
    all_splits = []

    # Process training data
    logger.info(f"Processing {len(train_catalog)} training images")
    for i, row in tqdm(
        train_catalog.iterrows(), total=len(train_catalog), desc="Processing GalaxyMNIST train"
    ):
        try:
            # Load the image
            img_path = row["file_loc"]
            img = Image.open(img_path).convert("RGB")

            # Create filename
            label_idx = int(row["label"])
            filename = f"galaxymnist_train_{i:05d}_{label_idx}.jpeg"

            # Add to lists
            all_images.append(img)
            all_labels.append(label_idx)
            all_filenames.append(filename)
            all_splits.append("train")
        except Exception as e:
            logger.error(f"Error processing train image at index {i}: {e}")

    # Process test data
    logger.info(f"Processing {len(test_catalog)} test images")
    for i, row in tqdm(
        test_catalog.iterrows(), total=len(test_catalog), desc="Processing GalaxyMNIST test"
    ):
        try:
            # Load the image
            img_path = row["file_loc"]
            img = Image.open(img_path).convert("RGB")

            # Create filename
            label_idx = int(row["label"])
            filename = f"galaxymnist_test_{i:05d}_{label_idx}.jpeg"

            # Add to lists
            all_images.append(img)
            all_labels.append(label_idx)
            all_filenames.append(filename)
            all_splits.append("test")
        except Exception as e:
            logger.error(f"Error processing test image at index {i}: {e}")

    # Create a mapping of index to class name
    label_maps = {i: name for i, name in enumerate(class_names)}

    # Create labels dataframe
    labels_df = pd.DataFrame(
        {
            "filename": all_filenames,
            "label": [label_maps[label] for label in all_labels],
            "label_idx": all_labels,
            "split": all_splits,
        }
    )

    logger.info(
        f"Prepared GalaxyMNIST dataset with {len(all_images)} images and {len(class_names)} classes"
    )

    return all_images, all_filenames, labels_df, galaxy_dir


def prepare_miniimagenet(miniimagenet_dir, output_dir, img_size=224, batch_size=1000):
    """
    Process miniImageNet dataset from parquet files with reduced memory usage.

    Args:
        miniimagenet_dir: Directory containing miniImageNet parquet files
        output_dir: Base output directory
        img_size: Target image size
        batch_size: Number of images to process at once

    Returns:
        Tuple of (total_images, output_directory, labels_df)
    """
    logger.info(f"Preparing miniImageNet dataset from {miniimagenet_dir}")

    # Check if directory exists
    if not os.path.exists(miniimagenet_dir):
        logger.error(f"miniImageNet directory {miniimagenet_dir} not found")
        return 0, None, None

    # Create output directory for images
    miniimagenet_dir_out = os.path.join(output_dir, "miniimagenet")
    os.makedirs(miniimagenet_dir_out, exist_ok=True)

    # Find parquet files for each split
    parquet_files = []
    for split in ["train", "validation", "test"]:
        pattern = f"{split}-"
        files = [
            f
            for f in os.listdir(miniimagenet_dir)
            if f.startswith(pattern) and f.endswith(".parquet")
        ]
        parquet_files.extend([os.path.join(miniimagenet_dir, f) for f in files])
        logger.info(f"Found {len(files)} {split} parquet files")

    if not parquet_files:
        logger.error("No parquet files found for miniImageNet")
        return 0, None, None

    # First pass to count images and collect labels
    total_images = 0
    all_labels = []
    all_filenames = []
    all_splits = []

    logger.info("Counting total images and collecting metadata...")
    for pf in tqdm(parquet_files, desc="Scanning parquet files"):
        try:
            split_name = os.path.basename(pf).split("-")[0]
            table = pq.read_table(pf)
            df = table.to_pandas()

            for i, row in df.iterrows():
                label_str = str(row["label"]).replace("n", "")
                filename = f"miniimagenet_{split_name}_{label_str}_{i:05d}.jpeg"

                all_filenames.append(filename)
                all_labels.append(str(row["label"]))
                all_splits.append(split_name)

            total_images += len(df)
            # Clear memory
            del table, df
            gc.collect()

        except Exception as e:
            logger.error(f"Error scanning parquet file {pf}: {e}")

    # Create labels dataframe
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    labels_df = pd.DataFrame(
        {
            "filename": all_filenames,
            "label": all_labels,
            "label_idx": [label_to_idx[label] for label in all_labels],
            "split": all_splits,
        }
    )

    logger.info(f"Found {total_images} images with {len(unique_labels)} classes")

    # Second pass to process images and save them
    return total_images, miniimagenet_dir_out, labels_df


def save_images_to_folder(images, filenames, output_dir, img_size=224, quality=90, num_workers=8):
    """Save all images to the output directory with parallel processing"""
    os.makedirs(output_dir, exist_ok=True)

    def process_image(args):
        i, img, filename = args
        try:
            output_path = os.path.join(output_dir, filename)
            save_image(img, filename, output_path, img_size, quality)
            return True
        except Exception as e:
            logger.error(f"Error saving image {filename}: {e}")
            return False

    # Create argument list for parallel processing
    args_list = [(i, img, filename) for i, (img, filename) in enumerate(zip(images, filenames))]

    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_image, args_list),
                total=len(args_list),
                desc=f"Saving images to {output_dir}",
            )
        )

    success_count = sum(results)
    logger.info(f"Saved {success_count} images to {output_dir}")


def process_miniimagenet_files(
    miniimagenet_dir,
    output_dir,
    hdf5_path,
    img_size=224,
    quality=90,
    num_workers=8,
    batch_size=1000,
):
    """Process miniImageNet files in batches to reduce memory usage."""
    # Find parquet files for each split
    parquet_files = []
    for split in ["train", "validation", "test"]:
        pattern = f"{split}-"
        files = [
            f
            for f in os.listdir(miniimagenet_dir)
            if f.startswith(pattern) and f.endswith(".parquet")
        ]
        parquet_files.extend([os.path.join(miniimagenet_dir, f) for f in files])

    # Get total image count first to pre-allocate HDF5 file
    total_images = 0
    for pf in parquet_files:
        table = pq.read_table(pf)
        total_images += len(table)
        del table
        gc.collect()

    # Create HDF5 file incrementally
    add_batch, finalize_hdf5 = create_hdf5_file_incrementally(
        output_path=hdf5_path,
        total_images=total_images,
        img_size=img_size,
        quality=quality,
        batch_size=batch_size,
    )

    # Process each parquet file
    images_processed = 0
    for pf in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            split_name = os.path.basename(pf).split("-")[0]
            logger.info(f"Processing {pf}")

            # Read the parquet file
            table = pq.read_table(pf)
            df = table.to_pandas()

            # Process in batches
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]

                batch_images = []
                batch_filenames = []

                # Process each row in the batch
                for i, row in tqdm(
                    batch_df.iterrows(),
                    total=len(batch_df),
                    desc=f"Batch {batch_start // batch_size + 1}",
                ):
                    try:
                        # Extract image bytes and convert to PIL
                        image_bytes = row["image"]["bytes"]
                        img = Image.open(io.BytesIO(image_bytes))
                        img = img.convert("RGB")  # Ensure RGB format

                        # Create filename
                        label_str = str(row["label"]).replace("n", "")
                        filename = f"miniimagenet_{split_name}_{label_str}_{i:05d}.jpeg"

                        # Save individual image
                        output_path = os.path.join(output_dir, filename)
                        save_image(img, filename, output_path, img_size, quality)

                        # Add to batch
                        batch_images.append(img)
                        batch_filenames.append(filename)

                    except Exception as e:
                        logger.error(f"Error processing image at index {i}: {e}")

                # Add batch to HDF5 file
                images_processed = add_batch(batch_images, batch_filenames)

                # Clear memory
                del batch_images, batch_filenames, batch_df
                gc.collect()

            # Clear memory after each parquet file
            del table, df
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing parquet file {pf}: {e}")

    # Finalize HDF5 file
    finalize_hdf5()
    return images_processed


def process_dataset(dataset_name, args):
    """Process a specific dataset"""
    img_size = args.img_size
    output_dir = args.output_dir

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "galaxymnist":
        # Prepare GalaxyMNIST dataset
        images, filenames, labels_df, dataset_dir = prepare_galaxymnist(
            output_dir=output_dir, download_dir=args.galaxymnist_dir, img_size=img_size
        )

        if images is None:
            return False

        # Define output paths
        hdf5_path = os.path.join(output_dir, f"galaxymnist_{img_size}.hdf5")
        csv_path = os.path.join(output_dir, "labels_galaxymnist.csv")

        # Save images to individual files
        save_images_to_folder(
            images=images,
            filenames=filenames,
            output_dir=dataset_dir,
            img_size=img_size,
            quality=args.jpeg_quality,
            num_workers=args.num_workers,
        )

        # Create HDF5 file
        logger.info(f"Creating {dataset_name} HDF5 file at {hdf5_path}")
        create_hdf5_file(
            images=images,
            filenames=filenames,
            output_path=hdf5_path,
            img_size=img_size,
            quality=args.jpeg_quality,
        )

    elif dataset_name == "miniimagenet":
        # Prepare miniImageNet dataset with reduced memory usage
        total_images, dataset_dir, labels_df = prepare_miniimagenet(
            miniimagenet_dir=args.miniimagenet_dir,
            output_dir=output_dir,
            img_size=img_size,
            batch_size=1000,
        )

        if dataset_dir is None:
            return False

        # Define output paths
        hdf5_path = os.path.join(output_dir, f"miniimagenet_{img_size}.hdf5")
        csv_path = os.path.join(output_dir, "labels_miniimagenet.csv")

        # Process files and build HDF5 incrementally
        process_miniimagenet_files(
            miniimagenet_dir=args.miniimagenet_dir,
            output_dir=dataset_dir,
            hdf5_path=hdf5_path,
            img_size=img_size,
            quality=args.jpeg_quality,
            num_workers=args.num_workers,
            batch_size=1000,
        )

    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False

    # Save labels CSV
    logger.info(f"Saving {dataset_name} labels to {csv_path}")
    labels_df.to_csv(csv_path, index=False)

    logger.success(f"Completed processing {dataset_name} dataset")
    return True


def main():
    """Main function to prepare datasets"""
    args = parse_arguments()

    logger.info("Starting dataset preparation for AnomalyMatch paper experiments")
    logger.info(f"Target image size: {args.img_size}x{args.img_size}")
    logger.info(f"Output directory: {args.output_dir}")

    # Process datasets based on command-line arguments
    if args.dataset in ["all", "galaxymnist"]:
        process_dataset("galaxymnist", args)

    if args.dataset in ["all", "miniimagenet"]:
        process_dataset("miniimagenet", args)

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
