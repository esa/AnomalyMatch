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
from tqdm import tqdm
import cutana

from anomaly_match.data_io.load_images import process_single_wrapper

from prediction_utils import (
    load_model,
    save_results,
    process_batch_predictions,
    estimate_batch_size,
    clear_gpu_cache_if_needed,
)

from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)


def read_and_preprocess_image_from_zarr(image_data, cfg):
    """Read and preprocess image data from Zarr array using standardized functions."""
    try:
        # Convert Zarr data to numpy array if it's not already
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)

        # Check if we need to transpose based on the shape
        # If last dimension is 3 (RGB channels), data is already in HWC format
        # If first dimension is 3, data is in CHW format and needs transposing
        if image_data.shape[0] == cfg.normalisation.n_output_channels:
            # In CHW format, convert to HWC
            image = image_data.transpose(1, 2, 0)
        else:
            # Assume HWC format if neither first nor last dimension is 3
            # This handles grayscale or other formats
            image = image_data

        # Use the centralized processing function - this handles RGB conversion,
        # normalization, and resizing efficiently without temporary files
        processed_image = process_single_wrapper(image, cfg, desc="zarr")
        return processed_image

    except Exception as e:
        logger.error(f"Error processing image from Zarr: {e}")
        raise


def load_and_preprocess_zarr(args):
    """Load and preprocess a single image from Zarr.

    Note: Returns numpy array, not tensor. Tensor conversion is done on main
    thread to avoid CUDA context issues in ThreadPoolExecutor.
    """
    image_data, cfg = args
    return read_and_preprocess_image_from_zarr(image_data, cfg)


def evaluate_images_from_cutana(
    cutana_sources_path, cfg, top_n=1000, batch_size=1000, max_workers=4
):
    """Evaluate images provided by Cutana stream and return top N scores."""

    cutana_config = cutana.get_default_config()

    cutana_config.target_resolution = cfg.normalisation.image_size[0]
    cutana_config.source_catalogue = cutana_sources_path

    # Configure FITS extensions from AM config, default to PRIMARY if not specified
    # fits_extension can be: None, str/int, list of str/int, or list of tuples (name, ext_type)
    fits_ext = cfg.normalisation.fits_extension
    if fits_ext is None:
        fits_ext = ["PRIMARY"]
    elif isinstance(fits_ext, (str, int)):
        fits_ext = [fits_ext]

    # Build selected_extensions - handle both simple names and (name, ext_type) tuples
    selected_extensions = []
    extension_names = []
    for ext in fits_ext:
        if isinstance(ext, tuple):
            name, ext_type = ext
            selected_extensions.append({"name": str(name), "ext": ext_type})
            extension_names.append(name)
        else:
            selected_extensions.append({"name": str(ext), "ext": "PrimaryHDU"})
            extension_names.append(ext)

    cutana_config.fits_extensions = extension_names
    cutana_config.selected_extensions = selected_extensions

    # Pass channel combination - required for multi-extension data
    if cfg.normalisation.channel_combination is not None:
        cutana_config.channel_weights = cfg.normalisation.channel_combination
    elif len(fits_ext) > 1:
        raise ValueError(
            "cfg.normalisation.channel_combination must be set when using multiple FITS extensions. "
            "This defines how extensions are combined into RGB channels."
        )

    # Pass AnomalyMatch's fitsbolt_cfg directly to cutana for normalization
    # This ensures cutana uses the exact same normalization settings as training
    if hasattr(cfg, "fitsbolt_cfg") and cfg.fitsbolt_cfg is not None:
        cutana_config.external_fitsbolt_cfg = cfg.fitsbolt_cfg
        logger.debug("Passed fitsbolt_cfg to cutana for normalization")

    try:
        logger.info(f"Creating Cutana orchestrator, streaming from {cutana_sources_path}")
        logger.debug(
            f"Cutana config: target_resolution={cutana_config.target_resolution}, "
            f"fits_extensions={cutana_config.fits_extensions}, "
            f"selected_extensions={cutana_config.selected_extensions}"
        )

        cutana_orchestrator = cutana.StreamingOrchestrator(cutana_config)

        cutana_orchestrator.init_streaming(
            batch_size=batch_size, write_to_disk=False, synchronised_loading=False
        )
    except Exception as e:
        logger.error(f"Failed to initialize Cutana orchestrator: {e}")
        raise

    logger.info("Cutana orchestrator streaming mode initalized")

    logger.info(f"Available batches in cutana: {cutana_orchestrator.get_batch_count()}")

    model = load_model(cfg)
    model.eval()
    transform = get_prediction_transforms()

    # Process images in batches
    scores_list = []
    imgs_list = []

    start_time = time.time()
    last_log_time = start_time
    processed_since_last_log = 0

    # Require fitsbolt config from model checkpoint for consistent predictions
    # Note: DotMap auto-creates empty DotMaps when accessing missing keys
    # So we check for 'size' key which must exist in a valid fitsbolt config
    fitsbolt_cfg = cfg.fitsbolt_cfg
    if fitsbolt_cfg is None or (isinstance(fitsbolt_cfg, DotMap) and "size" not in fitsbolt_cfg):
        raise ValueError(
            "fitsbolt_cfg not found in model checkpoint. "
            "Models must be saved with fitsbolt config for prediction. "
            "Please retrain and save the model to include fitsbolt config."
        )
    logger.debug("Using fitsbolt config loaded from model checkpoint")

    batches_count = cutana_orchestrator.get_batch_count()

    num_images = 0
    filenames = []

    for batch_idx in tqdm(range(batches_count), desc="Processing batches"):

        loaded_batch = cutana_orchestrator.next_batch()
        batch_data = loaded_batch["cutouts"]

        # Debug: Log what we received
        logger.debug(
            f"Batch {batch_idx}: cutouts type={type(batch_data).__name__}, "
            f"metadata count={len(loaded_batch.get('metadata', []))}"
        )

        # Handle empty batches (cutana returns [] if all cutouts failed)
        if isinstance(batch_data, list):
            if len(batch_data) == 0:
                logger.warning(f"Batch {batch_idx} returned empty cutouts (list), skipping")
                continue
            # Convert list to numpy array if needed
            batch_data = np.array(batch_data)

        batch_size_actual = batch_data.shape[0]
        num_images += batch_size_actual

        batch_filenames = (source["source_id"] for source in loaded_batch["metadata"])
        filenames.extend(batch_filenames)

        # I/O and preprocessing in ThreadPool (returns numpy arrays)
        # CUDA operations are kept on main thread to prevent memory fragmentation
        batch_process_start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_args = [(batch_data[i], cfg) for i in range(batch_size_actual)]
            numpy_images = list(executor.map(load_and_preprocess_zarr, batch_args))

        # Tensor conversion on main thread (not in ThreadPool) to avoid CUDA context issues
        stack_start = time.time()
        batch_tensors = [transform(img).detach() for img in numpy_images]
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

    cutana_orchestrator.cleanup()

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
    parser.add_argument(
        "cutana_sources_path", type=str, help="Path to the directory to stream from"
    )
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

    logger.info("Setting batch size")
    batch_size = (
        estimate_batch_size(cfg) if cfg.N_batch_prediction is None else cfg.N_batch_prediction
    )
    logger.info(f"Batch size set to: {batch_size}")

    # Log key configuration parameters
    logger.debug("Configuration loaded with parameters:")
    logger.debug(f"  Save file: {cfg.save_file}")
    logger.debug(f"  Save path: {cfg.save_path}")
    logger.debug(f"  Model path: {cfg.model_path}")
    logger.debug(f"  Output directory: {cfg.output_dir}")
    logger.debug(f"  Image size: {cfg.normalisation.image_size}")

    # Log full configuration
    logger.debug("Full configuration:")
    logger.debug(f"{cfg.toDict()}")

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info(f"Streaming from directory: {args.cutana_sources_path}")

    try:
        evaluate_images_from_cutana(
            args.cutana_sources_path, cfg, batch_size=batch_size, top_n=args.top_n
        )
        elapsed_time = time.time() - start_time
        logger.success(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":

    # Configure logging
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Remove default handler and set up file logging
    logger.remove()
    script_logger_id = logger.add(
        os.path.join(logs_dir, "prediction_cutana_{time}.log"),
        rotation="1 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )
    logger.add(sys.stderr, level="INFO")
    main()
