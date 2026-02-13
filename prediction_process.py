#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from anomaly_match.data_io.load_images import (
    load_and_process_single_wrapper,
)
from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from prediction_utils import (
    clear_gpu_cache_if_needed,
    load_model,
    load_prediction_config,
    process_batch_predictions,
    save_results,
    setup_prediction_logging,
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
    if not file_list:
        raise FileNotFoundError(
            f"No files to evaluate. The prediction search directory "
            f"'{cfg.prediction_search_dir}' is empty or contains no supported image files."
        )

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

    cfg, batch_size = load_prediction_config(args.config_path)

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
    setup_prediction_logging("prediction_thread")
    main()
