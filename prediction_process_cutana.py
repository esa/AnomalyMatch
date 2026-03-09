#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import glob
import os
import time

import cutana
import numpy as np
import pandas as pd
import torch
from cutana.catalogue_preprocessor import extract_filter_name, parse_fits_file_paths
from dotmap import DotMap
from loguru import logger
from tqdm import tqdm

from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from prediction_utils import (
    clear_gpu_cache_if_needed,
    convert_cutana_cutout,
    create_cutana_format_cfg,
    load_model,
    load_prediction_config,
    process_batch_predictions,
    save_results,
    setup_prediction_logging,
)


def _resolve_filter_names_from_catalogue(catalogue_path, n_extensions):
    """Resolve filter/band names from the first source in a cutana catalogue.

    Cutana streaming predictions operate on large mosaic tiles (separate FITS
    files per band) referenced by the catalogue's ``fits_file_paths`` column.
    Filter names are extracted from file paths using Euclid naming conventions.

    Raises:
        ValueError: If filter names cannot be determined — this indicates the
            catalogue uses non-Euclid file naming.  In that case users must set
            ``cfg.normalisation.fits_extension`` to explicit filter name strings.
    """
    # catalogue_path may be a single file (buffer parquet) or a directory
    if os.path.isfile(catalogue_path):
        first_file = catalogue_path
    else:
        cat_files = sorted(glob.glob(os.path.join(catalogue_path, "*.parquet")))
        if not cat_files:
            cat_files = sorted(glob.glob(os.path.join(catalogue_path, "*.csv")))
        if not cat_files:
            raise FileNotFoundError(f"No catalogue files found in {catalogue_path}")
        first_file = cat_files[0]

    # Read just the first row to get fits_file_paths
    if first_file.endswith(".parquet"):
        df = pd.read_parquet(first_file, columns=["fits_file_paths"]).head(1)
    else:
        df = pd.read_csv(first_file, usecols=["fits_file_paths"], nrows=1)

    fits_paths = parse_fits_file_paths(df["fits_file_paths"].iloc[0])
    if len(fits_paths) != n_extensions:
        raise ValueError(
            f"Catalogue has {len(fits_paths)} FITS files per source but "
            f"cfg.normalisation.fits_extension specifies {n_extensions} extensions."
        )

    filter_names = [extract_filter_name(p) for p in fits_paths]
    unknown = [p for p, name in zip(fits_paths, filter_names) if name == "UNKNOWN"]
    if unknown:
        raise ValueError(
            f"Could not determine filter names from catalogue FITS file paths. "
            f"Cutana streaming predictions currently only support Euclid data with "
            f"standard file naming conventions (VIS, NIR-H, NIR-Y, NIR-J). "
            f"Unrecognised files: {unknown}. "
            f"If using non-Euclid data, set cfg.normalisation.fits_extension to "
            f"explicit filter name strings instead of integer indices."
        )

    logger.info(f"Resolved filter names from catalogue: {filter_names}")
    return filter_names


def evaluate_images_from_cutana(
    cutana_sources_path, cfg, top_n=1000, batch_size=1000, max_workers=4
):
    """Evaluate images provided by Cutana stream and return top N scores."""

    cutana_config = cutana.get_default_config()

    cutana_config.target_resolution = cfg.normalisation.image_size[0]
    cutana_config.source_catalogue = cutana_sources_path

    # Configure FITS extensions for cutana.
    #
    # AnomalyMatch's fits_extension uses integer HDU indices or cutout extension
    # names (e.g. CHANNEL_1) for reading multi-extension cutout files via
    # fitsbolt.  Cutana streaming operates on large mosaic tiles referenced in
    # the catalogue — each source has separate FITS files per band with a single
    # PRIMARY HDU.  These cutout-level extension values are NOT valid for the
    # source mosaics, so for multi-extension streaming we always resolve filter
    # names (e.g. "VIS", "NIR-H") from the catalogue file paths.
    fits_ext = cfg.normalisation.fits_extension
    if fits_ext is None:
        fits_ext = ["PRIMARY"]
    elif isinstance(fits_ext, (str, int)):
        fits_ext = [fits_ext]

    if len(fits_ext) > 1:
        # Multi-extension: resolve filter names from catalogue file paths.
        # fits_extension values (integer HDU indices like [1,2,3] or cutout
        # extension names like ['CHANNEL_1','CHANNEL_2','CHANNEL_3']) are only
        # meaningful for reading cutout files — they must not be passed to
        # cutana as-is because source mosaics have different HDU structure.
        extension_names = _resolve_filter_names_from_catalogue(
            cutana_sources_path, len(fits_ext)
        )
    else:
        extension_names = ["PRIMARY"]

    # Source mosaics have a single PRIMARY HDU per file.
    cutana_config.fits_extensions = ["PRIMARY"]

    selected_extensions = []
    for name in extension_names:
        selected_extensions.append({"name": name, "ext": "PRIMARY"})
    cutana_config.selected_extensions = selected_extensions

    # Build channel_weights dict for cutana from AM's channel configuration.
    # Cutana expects {"ext_name": [weight_per_output_channel, ...], ...}.
    # Channel combination must happen BEFORE normalisation (cutana's pipeline
    # ensures this) so that ZSCALE/ASINH see the same data shape as training.
    n_out = cfg.normalisation.n_output_channels
    # Prefer the original dict form (preserved by validate_config when the user
    # provides channel_combination as a dict).  Falls back to the numpy array.
    combo_dict = cfg.normalisation.channel_combination_dict
    # DotMap subclasses dict, so empty auto-created DotMaps pass isinstance check.
    # Only use combo_dict when it's a real non-empty dict set by validate_config.
    combo = combo_dict if isinstance(combo_dict, dict) and len(combo_dict) > 0 else cfg.normalisation.channel_combination
    if combo is not None and isinstance(combo, dict) and len(combo) > 0:
        # Dict form: keys are filter names, values are weight lists.
        # This is order-independent — no risk of channel jumbling when
        # the streaming catalogue has a different file order than training.
        missing = set(extension_names) - set(combo.keys())
        if missing:
            raise ValueError(
                f"channel_combination dict is missing keys for resolved filter names: "
                f"{missing}. Dict keys: {list(combo.keys())}, "
                f"resolved filters: {extension_names}"
            )
        channel_weights = {}
        for name in extension_names:
            weights = combo[name]
            channel_weights[name] = list(weights) if not isinstance(weights, list) else weights
        cutana_config.channel_weights = channel_weights
    elif combo is not None and hasattr(combo, "shape"):
        # Numpy array form: columns are positionally mapped to extension_names
        # (order depends on catalogue file path order — use dict form to avoid
        # ambiguity when streaming and training catalogues differ).
        channel_weights = {}
        for j, ext_name in enumerate(extension_names):
            channel_weights[str(ext_name)] = combo[:, j].tolist()
        cutana_config.channel_weights = channel_weights
    elif len(fits_ext) > 1:
        raise ValueError(
            "cfg.normalisation.channel_combination must be set when using multiple FITS extensions. "
            "This defines how extensions are combined into RGB channels."
        )
    else:
        # Single extension: replicate to n_output_channels (e.g. 1→3 for RGB)
        cutana_config.channel_weights = {str(extension_names[0]): [1.0] * n_out}

    # Verify channel configuration consistency
    if len(extension_names) > 1:
        if isinstance(combo, dict):
            n_in = len(combo)
        elif hasattr(combo, "shape"):
            n_in = combo.shape[1]
        else:
            n_in = len(extension_names)
        if len(extension_names) != n_in:
            raise ValueError(
                f"Number of resolved filter names ({len(extension_names)}) does not match "
                f"channel_combination input dimension ({n_in}). "
                f"Filter names: {extension_names}"
            )
        if isinstance(combo, dict):
            lengths = {k: len(v) for k, v in combo.items()}
            bad = {k: length for k, length in lengths.items() if length != n_out}
            if bad:
                raise ValueError(
                    f"channel_combination dict values must have length {n_out} "
                    f"(n_output_channels), but got: {bad}"
                )
        elif hasattr(combo, "shape") and combo.shape[0] != n_out:
            raise ValueError(
                f"channel_combination output dimension ({combo.shape[0]}) does not match "
                f"n_output_channels ({n_out})"
            )
        logger.info(
            f"Channel configuration: {len(extension_names)} inputs -> {n_out} outputs, "
            f"filter order: {extension_names}"
        )

    # Flux conversion: must match the training path setting
    cutana_config.apply_flux_conversion = cfg.normalisation.apply_flux_conversion

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
    transform = get_prediction_transforms(num_channels=n_out)

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

    # CONVERSION_ONLY config for format conversion (created once, reused per cutout)
    format_cfg = create_cutana_format_cfg(cfg)

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

        # Cutana already normalised the cutouts via external_fitsbolt_cfg.
        # Only format conversion is needed (CHW→HWC, dtype, channel replication).
        batch_process_start = time.time()
        numpy_images = [
            convert_cutana_cutout(batch_data[i], format_cfg) for i in range(batch_size_actual)
        ]

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

    cfg, batch_size = load_prediction_config(args.config_path)

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
    setup_prediction_logging("prediction_cutana")
    main()
