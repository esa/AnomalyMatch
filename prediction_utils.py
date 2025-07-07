#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Utility functions for the anomaly detection prediction process.

This module contains helper functions for loading models, processing predictions,
and saving results to disk. It handles conversion between different image formats
and provides functionality for accumulating results across multiple batch runs.
"""

import os
import torch
import numpy as np
from loguru import logger
import pandas as pd
from turbojpeg import TurboJPEG

# Initialize TurboJPEG
jpeg_decoder = TurboJPEG()


def load_model(cfg):
    """Initialize and load the anomaly detection model.

    Args:
        cfg: Configuration object containing model settings such as
             model path, network type, pretrained status, and GPU settings.

    Returns:
        torch.nn.Module: The loaded PyTorch model ready for inference.

    Raises:
        FileNotFoundError: If the model file doesn't exist at the specified path.
        KeyError: If the model checkpoint doesn't contain the expected 'eval_model' key.
    """
    logger.info("Loading model with following configuration:")
    logger.info(f"  Model path: {cfg.model_path}")
    model_path = cfg.model_path
    logger.info(f"Attempting to load model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    from anomaly_match.utils.get_net_builder import get_net_builder

    net_builder = get_net_builder(
        cfg.net,
        pretrained=cfg.pretrained,
        in_channels=cfg.num_channels,
    )
    model = net_builder(num_classes=2, in_channels=3)

    if torch.cuda.is_available():
        gpu_device = getattr(cfg, "gpu", 0)  # Default to 0 if not set
        torch.cuda.set_device(gpu_device)
        model = model.cuda()
        logger.info(f"Using GPU device {gpu_device}")
    else:
        logger.info("Using CPU for inference")

    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, weights_only=False)
    else:
        checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device("cpu"))

    if "eval_model" not in checkpoint:
        raise KeyError(
            f"Model checkpoint does not contain 'eval_model' key. Keys found: {checkpoint.keys()}"
        )

    model.load_state_dict(checkpoint["eval_model"])
    logger.success(f"Successfully loaded model from {model_path}")
    return model


def save_results(cfg, all_scores, all_imgs, all_filenames, top_n):
    """Save prediction results to files, including top-N anomalies and all predictions.

    This function handles accumulating results across multiple batch runs by loading
    existing predictions if they exist, and merging them with new predictions.
    It saves:
    1. A CSV file with top-N anomalies (filenames and scores)
    2. A NPY file with the actual images of top-N anomalies
    3. A NPZ file with all accumulated predictions (for further analysis)

    Args:
        cfg (DotMap): Configuration object containing output paths and save file naming.
        all_scores (np.ndarray): Array of anomaly scores for the current batch.
        all_imgs (np.ndarray): Array of images from the current batch.
        all_filenames (np.ndarray): Array of filenames corresponding to the images.
        top_n (int): Number of top anomalies to save.

    Returns:
        tuple: (top_scores, top_filenames, top_imgs) selected from the accumulated results.
    """
    logger.info(f"Saving results with {len(all_scores)} total predictions")

    # Get the paths for results
    output_csv_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{top_n}.csv")
    output_npy_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{top_n}.npy")
    predictions_file = os.path.join(cfg.output_dir, f"all_predictions_{cfg.save_file}.npz")

    # Load and merge existing predictions if they exist
    all_scores, all_filenames, existing_top_images = _load_existing_predictions(
        predictions_file, output_npy_path, all_scores, all_filenames
    )

    # Get top N results from combined data
    top_indices = np.argsort(all_scores)[::-1][:top_n]
    top_scores = all_scores[top_indices]
    top_filenames = all_filenames[top_indices]

    # Build the top images array
    top_imgs = _build_top_images_array(all_scores, all_imgs, top_indices, existing_top_images)

    # Ensure images are in consistent format
    top_imgs = _ensure_consistent_image_format(top_imgs)

    logger.debug(
        f"Top images shape: {top_imgs.shape}, dtype: {top_imgs.dtype}, range: [{top_imgs.min()}, {top_imgs.max()}]"
    )

    # Save top N results
    logger.info(f"Saving top {top_n} results:")
    logger.info(f"  CSV: {output_csv_path}")
    logger.info(f"  NPY: {output_npy_path}")

    # Save results to CSV using pandas
    df = pd.DataFrame({"Filename": top_filenames, "Score": top_scores})
    df.to_csv(output_csv_path, index=False)

    # Save top images using numpy
    np.save(output_npy_path, top_imgs)

    # Save all accumulated predictions
    logger.info(f"Saving {len(all_scores)} accumulated predictions to: {predictions_file}")
    np.savez_compressed(predictions_file, filenames=all_filenames, scores=all_scores)

    logger.info(
        f"Score statistics - Min: {np.min(all_scores):.4f}, Max: {np.max(all_scores):.4f}"
        + f", Mean: {np.mean(all_scores):.4f}, Std: {np.std(all_scores):.4f}"
    )

    return top_scores, top_filenames, top_imgs


def _load_existing_predictions(
    predictions_file, output_npy_path, current_scores, current_filenames
):
    """Load and merge existing predictions with current batch data.

    Args:
        predictions_file (str): Path to the NPZ file with all accumulated predictions.
        output_npy_path (str): Path to the NPY file with top images.
        current_scores (np.ndarray): Scores from the current batch.
        current_filenames (np.ndarray): Filenames from the current batch.

    Returns:
        tuple: (merged_scores, merged_filenames, existing_top_images)
    """
    existing_scores = []
    existing_filenames = []
    existing_top_images = None

    # Load existing predictions if available
    if os.path.exists(predictions_file):
        logger.info("Loading existing predictions for accumulation")
        with np.load(predictions_file, allow_pickle=True) as data:
            existing_scores = data["scores"]
            existing_filenames = data["filenames"]

        # Also load existing top images if they exist
        if os.path.exists(output_npy_path):
            logger.info("Loading existing top images for preservation")
            existing_top_images = np.load(output_npy_path)
            logger.debug(f"Loaded existing top images shape: {existing_top_images.shape}")

        # Combine existing and new predictions
        merged_scores = np.concatenate([existing_scores, current_scores])
        merged_filenames = np.concatenate([existing_filenames, current_filenames])
        logger.info(
            f"Combined {len(existing_scores)} existing and {len(current_scores)} new predictions"
        )
        return merged_scores, merged_filenames, existing_top_images

    return current_scores, current_filenames, existing_top_images


def _build_top_images_array(all_scores, current_batch_imgs, top_indices, existing_top_images):
    """Build an array of top images from current batch and existing images.

    This function handles the complex logic of selecting images either from the current batch
    or from previously saved top images, based on their ranking in the combined dataset.

    Args:
        all_scores (np.ndarray): Combined scores from all batches including current one.
        current_batch_imgs (np.ndarray): Images from the current batch only.
        top_indices (np.ndarray): Indices of top scoring images in the combined dataset.
        existing_top_images (np.ndarray or None): Previously saved top images.

    Returns:
        np.ndarray: Array of top images.
    """
    # Calculate indices for current batch in the combined scores array
    current_batch_start = len(all_scores) - len(current_batch_imgs)
    current_batch_global_indices = set(range(current_batch_start, len(all_scores)))

    # Collect images for each top index
    top_img_list = []

    for i, global_idx in enumerate(top_indices):
        # Case 1: This top result is from the current batch
        if global_idx in current_batch_global_indices:
            batch_idx = global_idx - current_batch_start
            top_img_list.append(current_batch_imgs[batch_idx])

        # Case 2: This top result is from a previous batch
        elif existing_top_images is not None and i < len(existing_top_images):
            # Use the existing top image at this position
            top_img_list.append(existing_top_images[i])

    # First check if all images have the same shape
    shapes = [img.shape for img in top_img_list]
    if len(set(shapes)) > 1:
        # If different shapes, resize all to the first image's shape
        logger.warning(f"Inconsistent image shapes detected: {set(shapes)}, standardizing")
        reference_shape = top_img_list[0].shape
        for i, img in enumerate(top_img_list):
            if img.shape != reference_shape:
                # Simple resize by zero-padding or cropping
                fixed_img = np.zeros(reference_shape, dtype=np.uint8)
                # Copy as much of the original image as will fit
                slices = tuple(slice(0, min(s, rs)) for s, rs in zip(img.shape, reference_shape))
                if len(reference_shape) == 3:  # For 3D arrays (HWC)
                    fixed_img[slices[0], slices[1], slices[2]] = img[slices]
                else:  # For other dimensions
                    fixed_img[slices] = img[slices]
                top_img_list[i] = fixed_img

    # Now convert to numpy array - each element is guaranteed to have the same shape
    return np.stack(top_img_list)


def _ensure_consistent_image_format(images):
    """Ensure images are in consistent format (uint8, HWC layout).

    Args:
        images (np.ndarray): Image array to normalize.

    Returns:
        np.ndarray: Normalized image array.
    """
    if len(images) == 0:
        return images

    # Convert images from tensor format [0,1] back to uint8 [0,255] for UI compatibility
    if images.dtype != np.uint8:
        # Handle different input ranges
        if images.max() <= 1.0:
            # Tensor format [0,1] -> uint8 [0,255]
            images = (images * 255.0).clip(0, 255).astype(np.uint8)
        else:
            # Already in uint8 range, just convert type
            images = images.clip(0, 255).astype(np.uint8)

    # Ensure images are in consistent HWC format (Height x Width x Channels)
    if len(images.shape) == 4:
        # If images are in CHW format (N, C, H, W), transpose to HWC format (N, H, W, C)
        if images.shape[1] <= 4 and images.shape[3] > 4:  # Likely CHW format
            logger.debug("Converting images from CHW to HWC format for consistent saving")
            images = images.transpose(0, 2, 3, 1)

    return images


def process_batch_predictions(model, images, original_images=None):
    """Process a batch of images through the model to get anomaly scores.

    This function handles running inference on a batch of images and extracting
    the anomaly probability scores from the model output. It can either return
    the original images (if provided) or convert the tensor images back to uint8
    format suitable for saving.

    Args:
        model (torch.nn.Module): The neural network model for anomaly detection.
        images (torch.Tensor): Preprocessed tensor images for model inference.
        original_images (np.ndarray, optional): Original uint8 images for saving.
            If None, the function will convert the input tensor back to uint8.

    Returns:
        tuple: (batch_scores, images_for_saving)
            - batch_scores (np.ndarray): Anomaly probability scores (0-1 range).
            - images_for_saving (np.ndarray): Images in uint8 format ready for saving.
    """
    if torch.cuda.is_available():
        images = images.cuda(non_blocking=True)

    with torch.no_grad():
        logits = model(images)
        batch_scores = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    # Return original uint8 images if provided, otherwise convert tensor back
    if original_images is not None:
        return batch_scores, original_images
    else:
        # Convert tensor images back to uint8 for saving
        images_np = images.cpu().numpy()
        if images_np.max() <= 1.0:
            # Tensor format [0,1] -> uint8 [0,255]
            images_uint8 = (images_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            # Assume already in correct range
            images_uint8 = images_np.clip(0, 255).astype(np.uint8)
        return batch_scores, images_uint8
