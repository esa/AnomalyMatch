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
import pandas as pd
from tqdm import tqdm
import time

from prediction_utils import (
    load_model,
    save_results,
    process_batch_predictions,
)

from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from anomaly_match.data_io.load_images import read_and_resize_image

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


def load_and_preprocess(args):
    filename, transform, cfg = args
    image = read_and_resize_image(
        filename,
        cfg=cfg,
        convert_to_rgb=True,
    )
    image = transform(image)
    return filename, image


def evaluate_files(file_list, cfg, top_n=1000, batch_size=1000, max_workers=1):
    """Evaluate files in batches and return top N scores."""
    logger.trace(f"{len(file_list)} unlabeled images remain.")

    transform = get_prediction_transforms()
    args_list = [(filename, transform, cfg) for filename in file_list]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(load_and_preprocess, args_list),
                desc="Loading images",
                total=len(file_list),
            )
        )

    model = load_model(cfg)
    model.eval()

    # Process in batches
    scores_list = []
    filenames_list = []
    imgs_list = []

    for i in range(0, len(results), batch_size):
        batch = results[i : i + batch_size]  # noqa: E203
        batch_filenames = [item[0] for item in batch]
        batch_images = [item[1] for item in batch]

        # Stack images into a batch tensor
        images = torch.stack(batch_images, dim=0)
        batch_scores, batch_imgs = process_batch_predictions(model, images)

        scores_list.append(batch_scores)
        filenames_list.extend(batch_filenames)
        imgs_list.append(batch_imgs)

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

    logger.info(f"Loading file list from {args.file_list_path}")
    with open(args.file_list_path, "r") as f:
        group_list = [line.strip() for line in f]
    assert len(group_list) == 1, "Only one file list is allowed"
    with open(group_list[0], "r") as f:
        file_list = [line.strip() for line in f]
    logger.info(f"Found {len(file_list)} files to process")

    # Load existing results if they exist
    output_csv_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{args.top_n}.csv")
    output_npy_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{args.top_n}.npy")

    if os.path.exists(output_csv_path) and os.path.exists(output_npy_path):
        logger.info("Found existing results, loading...")
        existing_df = pd.read_csv(output_csv_path)
        existing_filenames = existing_df["Filename"].values
        existing_scores = existing_df["Score"].values

        existing_imgs = np.load(output_npy_path)
    else:
        existing_filenames = np.array([])
        existing_scores = np.array([])
        # Define image shape: (num_samples, channels, height, width)
        existing_imgs = np.empty((0, 3, cfg.size[0], cfg.size[1]), dtype=np.float32)

    logger.info("Starting evaluation...")
    scores, filenames, imgs = evaluate_files(file_list, cfg, top_n=args.top_n)
    logger.success(f"Evaluation complete. Computed {len(scores)} scores")

    # Merge new results with existing results
    all_filenames = np.concatenate([existing_filenames, filenames])
    all_scores = np.concatenate([existing_scores, scores])
    # Merge new results with existing results
    if existing_imgs.size == 0:
        all_imgs = imgs
    else:
        all_imgs = np.concatenate([existing_imgs, imgs])

    # Keep only top N results
    top_indices = np.argsort(all_scores)[::-1][: args.top_n]
    top_filenames = all_filenames[top_indices]
    top_scores = all_scores[top_indices]
    top_imgs = all_imgs[top_indices]

    logger.info(
        f"Score statistics - Min: {np.min(top_scores):.4f}, Max: {np.max(top_scores):.4f}"
        + f", Mean: {np.mean(top_scores):.4f}, Std: {np.std(top_scores):.4f}"
    )

    logger.info(f"Saving results to {output_csv_path} and {output_npy_path}")

    # Save merged results to CSV using pandas
    df = pd.DataFrame({"Filename": top_filenames, "Score": top_scores})
    df.to_csv(output_csv_path, index=False)

    # Save merged images using numpy
    np.save(output_npy_path, top_imgs)

    elapsed_time = time.time() - start_time
    logger.success(f"Script completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
