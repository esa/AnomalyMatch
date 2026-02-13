#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from anomaly_match.data_io.load_images import process_single_wrapper
from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from prediction_utils import (
    clear_gpu_cache_if_needed,
    jpeg_decoder,
    load_model,
    load_prediction_config,
    process_batch_predictions,
    save_results,
    setup_prediction_logging,
)


def read_and_decode_image_from_hdf5(image_data, cfg):
    """Read image data from HDF5 and decode it using centralized processing."""
    # Convert from vlen array back to bytes
    image_bytes = bytes(image_data)

    try:
        # Try decoding with TurboJPEG first (faster for JPEG)
        try:
            image = jpeg_decoder.decode(image_bytes)
            # TurboJPEG decodes in BGR format, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, [2, 1, 0]]
        except Exception:
            # If TurboJPEG fails, fall back to PIL
            import io

            from PIL import Image

            image = np.array(Image.open(io.BytesIO(image_bytes)))

        processed_image = process_single_wrapper(image, cfg, desc="hdf5")
        return processed_image

    except Exception as e:
        logger.error(f"Error decoding image from HDF5: {e}")
        # Return a blank image as fallback
        return np.zeros(
            (cfg.normalisation.image_size[0], cfg.normalisation.image_size[1], 3),
            dtype=np.uint8,
        )


def load_and_preprocess_hdf5(args):
    """Load and preprocess a single image from HDF5.

    Note: Returns numpy array, not tensor. Tensor conversion is done on main
    thread to avoid CUDA context issues in ThreadPoolExecutor.
    """
    image_data, cfg = args
    return read_and_decode_image_from_hdf5(image_data, cfg)


def evaluate_images_in_hdf5(hdf5_path, cfg, top_n=1000, batch_size=1000, max_workers=4):
    """Evaluate images inside an HDF5 file and return top N scores."""
    logger.info(f"Opening HDF5 file {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5f:
        dataset = h5f["images"]
        filenames_dataset = h5f["filenames"]
        num_images = len(dataset)
        logger.info(f"Found {num_images} images in the HDF5 file")

        model = load_model(cfg)
        model.eval()
        transform = get_prediction_transforms()

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

        for batch_idx, batch_start in enumerate(
            tqdm(range(0, num_images, batch_size), desc="Processing batches")
        ):
            batch_end = min(batch_start + batch_size, num_images)
            batch_data = dataset[batch_start:batch_end]
            batch_size_actual = len(batch_data)

            # I/O and preprocessing in ThreadPool (returns numpy arrays)
            # CUDA operations are kept on main thread to prevent memory fragmentation
            batch_process_start = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_args = [(data, cfg) for data in batch_data]
                numpy_images = list(executor.map(load_and_preprocess_hdf5, batch_args))

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
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file containing images")
    parser.add_argument("top_n", type=int, default=1000, help="Number of top scores to keep")
    args = parser.parse_args()

    cfg, batch_size = load_prediction_config(args.config_path)

    logger.info(f"Processing HDF5 file: {args.hdf5_path}")

    try:
        evaluate_images_in_hdf5(args.hdf5_path, cfg, batch_size=batch_size, top_n=args.top_n)
        elapsed_time = time.time() - start_time
        logger.success(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    setup_prediction_logging("prediction_hdf5")
    main()
