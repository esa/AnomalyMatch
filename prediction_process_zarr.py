#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import os
import sys
import pickle

from dotmap import DotMap
import torch
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import time
import zarr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from prediction_utils import (
    load_model,
    save_results,
    process_batch_predictions,
)

from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from anomaly_match.data_io.load_images import process_image_array

# Configure logging
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Remove default handler and set up file logging
logger.remove()
logger.add(
    os.path.join(logs_dir, "prediction_zarr_{time}.log"),
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
)
logger.add(sys.stderr, level="INFO")


def read_and_preprocess_image_from_zarr(image_data, cfg):
    """Read and preprocess image data from Zarr array using standardized functions."""
    try:
        # Convert Zarr data to numpy array if it's not already
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)

        # Convert from CHW to HWC format
        image = image_data.transpose(1, 2, 0)

        # Use the centralized processing function - this handles RGB conversion,
        # normalization, and resizing efficiently without temporary files
        processed_image = process_image_array(image, cfg, convert_to_rgb=True, image_source="zarr")
        return processed_image

    except Exception as e:
        logger.error(f"Error processing image from Zarr: {e}")
        raise


def load_and_preprocess_zarr(args):
    """Load and preprocess a single image from Zarr."""
    image_data, transform, cfg = args
    image = read_and_preprocess_image_from_zarr(image_data, cfg)
    image = transform(image)
    return image


def evaluate_images_in_zarr(zarr_path, cfg, top_n=1000, batch_size=1000, max_workers=4):
    """Evaluate images inside a Zarr file and return top N scores."""
    logger.info(f"Opening Zarr file {zarr_path}")

    zarr_path = Path(zarr_path)

    # Open Zarr store
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
    except Exception as e:
        logger.error(f"Failed to open Zarr store: {e}")
        raise

    if "images" not in root:
        raise ValueError(f"No 'images' array found in Zarr store {zarr_path}")

    images_array = root["images"]
    num_images = images_array.shape[0]
    logger.info(f"Found {num_images} images in the Zarr file")
    logger.info(f"Image array shape: {images_array.shape}")
    logger.info(f"Image array dtype: {images_array.dtype}")

    # Try to load metadata
    filenames = []
    metadata_file = None

    # Check for metadata file in Zarr attributes
    if "metadata_file" in root.attrs:
        metadata_file = Path(root.attrs["metadata_file"])
        if not metadata_file.is_absolute():
            # Try relative to zarr file
            metadata_file = zarr_path.parent / metadata_file.name

    # Fallback: look for metadata parquet file next to zarr
    if metadata_file is None or not metadata_file.exists():
        potential_metadata = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        if potential_metadata.exists():
            metadata_file = potential_metadata

    if metadata_file and metadata_file.exists():
        logger.info(f"Loading metadata from {metadata_file}")
        try:
            metadata_df = pd.read_parquet(metadata_file)
            if "original_filename" in metadata_df.columns:
                filenames = metadata_df["original_filename"].tolist()
            elif "filename" in metadata_df.columns:
                filenames = metadata_df["filename"].tolist()
            else:
                logger.warning("No filename column found in metadata, using indices")
                filenames = [f"image_{i:06d}" for i in range(num_images)]
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            filenames = [f"image_{i:06d}" for i in range(num_images)]
    else:
        logger.info("No metadata file found, using image indices as filenames")
        filenames = [f"image_{i:06d}" for i in range(num_images)]

    # Ensure we have the right number of filenames
    if len(filenames) != num_images:
        logger.warning(
            f"Filename count ({len(filenames)}) doesn't match image count ({num_images})"
        )
        filenames = [f"image_{i:06d}" for i in range(num_images)]

    model = load_model(cfg)
    model.eval()
    transform = get_prediction_transforms()

    # Process images in batches
    scores_list = []
    imgs_list = []

    start_time = time.time()
    last_log_time = start_time
    processed_since_last_log = 0

    for batch_start in tqdm(range(0, num_images, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, num_images)
        batch_size_actual = batch_end - batch_start

        # Read batch data from Zarr
        batch_data = images_array[batch_start:batch_end]

        # Process batch in parallel
        batch_process_start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_args = [(batch_data[i], transform, cfg) for i in range(batch_size_actual)]
            batch_images = list(executor.map(load_and_preprocess_zarr, batch_args))

        # Stack images into a batch tensor and get predictions
        stack_start = time.time()
        images = torch.stack(batch_images, dim=0)
        batch_scores, batch_imgs = process_batch_predictions(model, images)

        scores_list.append(batch_scores)
        imgs_list.append(batch_imgs)

        processed_since_last_log += batch_size_actual
        current_time = time.time()

        # Log performance every 10000 images or 60 seconds
        if processed_since_last_log >= 10000 or (current_time - last_log_time) >= 60:
            elapsed = current_time - last_log_time
            rate = processed_since_last_log / elapsed
            batch_time = current_time - batch_process_start
            logger.info(
                f"Performance: {rate:.1f} images/sec "
                f"(batch {batch_size_actual}: {batch_time:.2f}s, "
                f"load: {stack_start - batch_process_start:.2f}s, "
                f"inference: {current_time - stack_start:.2f}s)"
            )
            last_log_time = current_time
            processed_since_last_log = 0

    total_time = time.time() - start_time
    logger.info(
        f"Total processing time: {total_time:.1f}s, "
        f"Average rate: {num_images / total_time:.1f} images/sec"
    )

    # Concatenate results
    all_scores = np.concatenate(scores_list)
    all_imgs = np.concatenate(imgs_list)
    all_filenames = np.array(filenames)

    return save_results(cfg, all_scores, all_imgs, all_filenames, top_n)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing images")
    parser.add_argument("top_n", type=int, default=1000, help="Number of top scores to keep")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config_path}")
    # Load cfg from pkl
    try:
        with open(args.config_path, "rb") as f:
            cfg = pickle.load(f)
            cfg = DotMap(cfg)
    except Exception as e:
        logger.error(f"Failed to load config from {args.config_path}: {e}")
        sys.exit(1)

    # Log key configuration parameters
    logger.debug("Configuration loaded with parameters:")
    logger.debug(f"  Save file: {cfg.save_file}")
    logger.debug(f"  Save path: {cfg.save_path}")
    logger.debug(f"  Model path: {cfg.model_path}")
    logger.debug(f"  Output directory: {cfg.output_dir}")
    logger.debug(f"  Image size: {cfg.size}")

    # Log full configuration
    logger.debug("Full configuration:")
    logger.debug(f"{cfg.toDict()}")

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info(f"Processing Zarr file: {args.zarr_path}")

    try:
        evaluate_images_in_zarr(args.zarr_path, cfg, top_n=args.top_n)
        elapsed_time = time.time() - start_time
        logger.success(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
