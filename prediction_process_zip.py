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
from tqdm import tqdm
import pandas as pd
import time
import toml
import zipfile
from PIL import Image
import io

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


def read_and_resize_image_zip(zip_file, file_name, size):
    """Read an image from zip file and resize it."""
    with zip_file.open(file_name) as img_file:
        file_ext = os.path.splitext(file_name.lower())[1]

        if file_ext in [".jpg", ".jpeg"]:
            # Use TurboJPEG for JPEG files
            jpeg_data = img_file.read()
            image = jpeg_decoder.decode(jpeg_data)
        else:
            # Use PIL for PNG, TIF, TIFF
            image_data = img_file.read()
            image = np.array(Image.open(io.BytesIO(image_data)))

    # If grayscale, convert to RGB
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = np.stack((image,) * 3, axis=-1)
    # Handle RGBA images by removing alpha channel
    elif len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    # Resize if necessary
    if image.shape[:2] != tuple(size):
        image = resize(image, size, anti_aliasing=True, preserve_range=True)
        image = image.astype(np.uint8)

    return image


def load_and_preprocess(args):
    zip_file, file_name, size, transform = args
    image = read_and_resize_image_zip(zip_file, file_name, size)
    image = transform(image)
    return file_name, image


def evaluate_files_in_zip(zip_path, cfg, top_n=1000, batch_size=1000, max_workers=1):
    """Evaluate images inside a zip file and return top N scores."""
    logger.info(f"Opening zip file {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        img_files = [
            name
            for name in zip_file.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]

        transform = get_transform()
        args_list = [(zip_file, file_name, cfg.size, transform) for file_name in img_files]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(load_and_preprocess, args_list),
                    desc="Loading images",
                    total=len(img_files),
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
    parser.add_argument("zip_path", type=str, help="Path to the zip file containing images")
    parser.add_argument("top_n", type=int, default=1000, help="Number of top scores to keep")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config_path}")
    cfg = DotMap(toml.load(args.config_path))

    logger.info(f"Processing images from zip file: {args.zip_path}")

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
    scores, filenames, imgs = evaluate_files_in_zip(args.zip_path, cfg, top_n=args.top_n)
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
