#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import argparse
import os
import sys
import torch
import numpy as np
from loguru import logger
from dotmap import DotMap
from concurrent.futures import ThreadPoolExecutor
import time
import toml
import h5py
from tqdm import tqdm

from prediction_utils import (
    load_model,
    save_results,
    get_transform,
    process_batch_predictions,
    jpeg_decoder,
)

# Configure logging
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Remove default handler and set up file logging
logger.remove()
logger.add(
    os.path.join(logs_dir, "prediction_thread_{time}.log"),
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
)
logger.add(sys.stderr, level="INFO")


def read_and_decode_image_from_hdf5(image_data, size):
    """Read image data from HDF5 and decode it."""
    # Convert from vlen array back to bytes
    image_bytes = bytes(image_data)

    try:
        # Try decoding with TurboJPEG first (faster for JPEG)
        try:
            image = jpeg_decoder.decode(image_bytes)
        except Exception:
            # If TurboJPEG fails, fall back to PIL
            from PIL import Image
            import io

            image = np.array(Image.open(io.BytesIO(image_bytes)))

        # If grayscale, convert to RGB
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = np.stack((image,) * 3, axis=-1)
        # Handle RGBA or other formats
        elif len(image.shape) > 2 and image.shape[2] > 3:
            image = image[:, :, :3]  # Keep only RGB channels

        return image
    except Exception as e:
        logger.error(f"Error decoding image from HDF5: {e}")
        # Return a blank image as fallback
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)


def load_and_preprocess(args):
    """Load and preprocess a single image."""
    image_data, size, transform = args
    image = read_and_decode_image_from_hdf5(image_data, size)
    image = transform(image)
    return image


def evaluate_images_in_hdf5(hdf5_path, cfg, top_n=1000, batch_size=2500, max_workers=1):
    """Evaluate images inside an HDF5 file and return top N scores."""
    logger.info(f"Opening HDF5 file {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5f:
        dataset = h5f["images"]
        filenames_dataset = h5f["filenames"]
        num_images = len(dataset)
        logger.info(f"Found {num_images} images in the HDF5 file")

        model = load_model(cfg)
        model.eval()
        transform = get_transform()

        # Process images in batches
        scores_list = []
        # Properly decode bytes strings, removing the b'' prefix
        filenames = [
            fname.decode("utf-8") if isinstance(fname, bytes) else fname
            for fname in filenames_dataset[:]
        ]
        imgs_list = []

        start_time = time.time()
        last_log_time = start_time
        processed_since_last_log = 0

        for batch_start in tqdm(range(0, num_images, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, num_images)
            batch_data = dataset[batch_start:batch_end]
            batch_size_actual = len(batch_data)

            # Process batch in parallel
            batch_process_start = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_args = [(data, cfg.size, transform) for data in batch_data]
                batch_images = list(executor.map(load_and_preprocess, batch_args))

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
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file containing images")
    parser.add_argument("top_n", type=int, default=1000, help="Number of top scores to keep")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config_path}")
    cfg = DotMap(toml.load(args.config_path))

    # Log key configuration parameters
    logger.debug("Configuration loaded with parameters:")
    logger.debug(f"  Save file: {cfg.save_file}")
    logger.debug(f"  Save path: {cfg.save_path}")
    logger.debug(f"  Model path: {cfg.model_path}")
    logger.debug(f"  Output directory: {cfg.output_dir}")
    logger.debug(f"  Image size: {cfg.size}")

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info(f"Processing HDF5 file: {args.hdf5_path}")

    try:
        evaluate_images_in_hdf5(args.hdf5_path, cfg, top_n=args.top_n)
        elapsed_time = time.time() - start_time
        logger.success(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
