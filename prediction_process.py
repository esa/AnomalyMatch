#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import os
import pickle
import sys

from dotmap import DotMap
import torch
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

from anomaly_match.data_io.load_images import (
    load_and_process_single_wrapper,
)

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


def load_and_preprocess(args):
    """Load and preprocess a single image file.

    Note: Returns numpy array, not tensor. Tensor conversion is done on main
    thread to avoid CUDA context issues in ThreadPoolExecutor.
    """
    filepath, cfg = args
    image = load_and_process_single_wrapper(
        filepath,
        cfg,
        desc="image prediction process",
        show_progress=False,
        prediction=True,
    )
    return filepath, image


def evaluate_files(file_list, cfg, top_n=1000, batch_size=1000, max_workers=1):
    """Evaluate files in batches and return top N scores.
    file list is a list of cfg.prediction_search_dir+filename
    """
    logger.trace(f"{len(file_list)} unlabeled images remain.")

    # Load model first - this loads the fitsbolt config from the checkpoint
    model = load_model(cfg)
    model.eval()

    # Require fitsbolt config from model checkpoint for consistent predictions
    if not hasattr(cfg, "fitsbolt_cfg") or cfg.fitsbolt_cfg is None:
        raise ValueError(
            "Fitsbolt config not found in model checkpoint. "
            "Please retrain the model with the updated version to include normalisation settings."
        )
    logger.debug("Using fitsbolt config loaded from model checkpoint")

    transform = get_prediction_transforms()

    # I/O in ThreadPool (returns numpy arrays)
    args_list = [(filepath, cfg) for filepath in file_list]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(load_and_preprocess, args_list),
                desc="Loading images",
                total=len(file_list),
            )
        )

    # Process in batches
    scores_list = []
    filenames_list = []
    imgs_list = []

    for batch_idx, i in enumerate(range(0, len(results), batch_size)):
        batch = results[i : i + batch_size]  # noqa: E203
        batch_filenames = [item[0] for item in batch]
        numpy_images = [item[1] for item in batch]

        # Tensor conversion on main thread (not in ThreadPool)
        batch_tensors = [transform(img) for img in numpy_images]
        images = torch.stack(batch_tensors, dim=0)
        del numpy_images, batch_tensors  # Free memory before CUDA ops

        # CUDA inference with explicit cleanup
        batch_scores, batch_imgs = process_batch_predictions(model, images)
        del images  # Free CUDA tensor reference

        scores_list.append(batch_scores)
        filenames_list.extend(batch_filenames)
        imgs_list.append(batch_imgs)

        # Periodic GPU cache clearing to prevent fragmentation
        clear_gpu_cache_if_needed(batch_idx)

    # Concatenate results
    all_scores = np.concatenate(scores_list)
    all_imgs = np.concatenate(imgs_list)
    all_filenames = np.array(filenames_list)

    return save_results(cfg, all_scores, all_imgs, all_filenames, top_n)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument(
        "file_list_path",
        type=str,
        help="Path to file containing list of files to evaluate",
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

    logger.info(f"Loading file list from {args.file_list_path}")
    with open(args.file_list_path, "r") as f:
        group_list = [line.strip() for line in f]
    assert len(group_list) == 1, "Only one file list is allowed"
    with open(group_list[0], "r") as f:
        file_list = [line.strip() for line in f]
    logger.info(f"Found {len(file_list)} files to process")

    logger.info("Starting evaluation...")
    # evaluate_files calls save_results internally which handles accumulation
    # across multiple batches - no additional merging needed here
    scores, filenames, imgs = evaluate_files(
        file_list, cfg, batch_size=batch_size, top_n=args.top_n
    )
    logger.success(f"Evaluation complete. Top {len(scores)} scores returned")

    elapsed_time = time.time() - start_time
    logger.success(f"Script completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Configure logging
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.remove()
    logger.add(
        os.path.join(logs_dir, "prediction_thread_{time}.log"),
        rotation="1 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )
    main()
