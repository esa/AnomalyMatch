#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from loguru import logger
from tqdm import tqdm

from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from prediction_utils import (
    clear_gpu_cache_if_needed,
    load_and_preprocess_zarr,
    load_model,
    load_prediction_config,
    process_batch_predictions,
    save_results,
    setup_prediction_logging,
)


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

    # Generate a unique prefix for this zarr file to avoid filename collisions
    # Use the parent directory name for batch folders, or the zarr file name itself
    if zarr_path.name == "images.zarr":
        # For batch folders, use the parent directory name
        zarr_prefix = zarr_path.parent.name
    else:
        # For direct zarr files, use the zarr file name
        zarr_prefix = zarr_path.stem

    # Check for metadata file in Zarr attributes
    if "metadata_file" in root.attrs:
        metadata_file = Path(root.attrs["metadata_file"])
        if not metadata_file.is_absolute():
            # Try relative to zarr file
            metadata_file = zarr_path.parent / metadata_file.name

    # Fallback: look for metadata parquet file next to zarr
    if metadata_file is None or not metadata_file.exists():
        # First try: <zarr_name>_metadata.parquet next to zarr file
        potential_metadata = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        if potential_metadata.exists():
            metadata_file = potential_metadata
        # Second try: For batch folders with images.zarr subdirectory,
        # look for images_metadata.parquet in parent directory
        elif zarr_path.name == "images.zarr":
            potential_metadata = zarr_path.parent / "images_metadata.parquet"
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
            elif "source_id" in metadata_df.columns:
                # Use source_id as filename if available
                filenames = metadata_df["source_id"].tolist()
                logger.info("Using source_id column as filenames")
            else:
                logger.warning(
                    "No filename column found in metadata, using indices with zarr prefix"
                )
                filenames = [f"{zarr_prefix}__image_{i:06d}" for i in range(num_images)]
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            logger.info("Using image indices with zarr prefix as fallback")
            filenames = [f"{zarr_prefix}__image_{i:06d}" for i in range(num_images)]
    else:
        logger.info("No metadata file found, using image indices with zarr prefix as filenames")
        filenames = [f"{zarr_prefix}__image_{i:06d}" for i in range(num_images)]

    # Ensure we have the right number of filenames
    if len(filenames) != num_images:
        logger.warning(
            f"Filename count ({len(filenames)}) doesn't match image count ({num_images}), regenerating with zarr prefix"
        )
        filenames = [f"{zarr_prefix}__image_{i:06d}" for i in range(num_images)]

    model = load_model(cfg)
    model.eval()
    transform = get_prediction_transforms()

    # Process images in batches
    scores_list = []
    imgs_list = []

    start_time = time.time()
    last_log_time = start_time
    processed_since_last_log = 0

    for batch_idx, batch_start in enumerate(
        tqdm(range(0, num_images, batch_size), desc="Processing batches")
    ):
        batch_end = min(batch_start + batch_size, num_images)
        batch_size_actual = batch_end - batch_start

        # Read batch data from Zarr
        batch_data = images_array[batch_start:batch_end]

        # I/O and preprocessing in ThreadPool (returns numpy arrays)
        # CUDA operations are kept on main thread to prevent memory fragmentation
        batch_process_start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_args = [(batch_data[i], cfg) for i in range(batch_size_actual)]
            numpy_images = list(executor.map(load_and_preprocess_zarr, batch_args))

        # Tensor conversion on main thread (not in ThreadPool)
        stack_start = time.time()
        batch_tensors = [transform(img) for img in numpy_images]
        images = torch.stack(batch_tensors, dim=0)
        del numpy_images, batch_tensors  # Free memory before CUDA ops

        # CUDA inference with explicit cleanup
        batch_scores, batch_imgs = process_batch_predictions(model, images)
        del images  # Free CUDA tensor reference

        scores_list.append(batch_scores)
        imgs_list.append(batch_imgs)

        # Periodic GPU cache clearing to prevent fragmentation
        clear_gpu_cache_if_needed(batch_idx)

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

    cfg, batch_size = load_prediction_config(args.config_path)

    logger.info(f"Processing Zarr file: {args.zarr_path}")

    try:
        evaluate_images_in_zarr(args.zarr_path, cfg, batch_size=batch_size, top_n=args.top_n)
        elapsed_time = time.time() - start_time
        logger.success(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    setup_prediction_logging("prediction_zarr")
    main()
