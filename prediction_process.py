#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import argparse
import os
import torch
import numpy as np
from loguru import logger
from dotmap import DotMap
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
import pandas as pd
from tqdm import tqdm
import time
import toml

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
logger.remove()
logger.add(
    os.path.join(logs_dir, "prediction_thread_{time}.log"),
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
)


def read_and_resize_image(filepath, size):
    """Read an image file and resize it."""
    try:
        # Handle different file formats
        file_ext = os.path.splitext(filepath.lower())[1]

        if file_ext in [".jpg", ".jpeg"]:
            # Use TurboJPEG for faster JPEG loading
            with open(filepath, "rb") as infile:
                jpeg_data = infile.read()
            # Decode JPEG image to array
            image = jpeg_decoder.decode(jpeg_data)
        else:
            # For other formats (png, tif, tiff), use imageio or PIL
            from PIL import Image

            image = np.array(Image.open(filepath))

        # If grayscale, convert to RGB
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = np.stack((image,) * 3, axis=-1)
        # Handle RGBA images by removing alpha channel
        elif len(image.shape) > 2 and image.shape[2] > 3:
            image = image[:, :, :3]

        # Resize if necessary
        if image.shape[:2] != tuple(size):
            image = resize(image, size, anti_aliasing=True, preserve_range=True)
            image = image.astype(np.uint8)

        return image
    except Exception as e:
        logger.error(f"Error reading image {filepath}: {e}")
        # Return blank image as fallback
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)


def load_and_preprocess(args):
    filename, size, transform = args
    image = read_and_resize_image(filename, size)
    image = transform(image)
    return filename, image


def evaluate_files(file_list, cfg, top_n=1000, batch_size=1000, max_workers=1):
    """Evaluate files in batches and return top N scores."""
    logger.trace(f"{len(file_list)} unlabeled images remain.")

    transform = get_transform()
    args_list = [(filename, cfg.size, transform) for filename in file_list]

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
        "file_list_path", type=str, help="Path to file containing list of files to evaluate"
    )
    parser.add_argument("top_n", type=int, default=1000, help="Number of top scores to keep")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config_path}")
    cfg = DotMap(toml.load(args.config_path))

    logger.info(f"Loading file list from {args.file_list_path}")
    with open(args.file_list_path, "r") as f:
        file_list = [line.strip() for line in f]
    logger.info(f"Found {len(file_list)} files to process")

    # Load existing results if they exist
    output_csv_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top1000.csv")
    output_npy_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top1000.npy")

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
