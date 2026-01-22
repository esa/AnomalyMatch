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
import pandas as pd

from loguru import logger
from turbojpeg import TurboJPEG


# Initialize TurboJPEG
jpeg_decoder = TurboJPEG()

# Memory model coefficients for batch size estimation
# These were derived from empirical measurements (R² > 0.9999)
# Formula: reserved_mb = a * batch_size * image_size² + b * batch_size + c
MEMORY_COEFFICIENTS = {
    "efficientnet-lite0": {"a": 0.000391, "b": 0.0637, "c": 31.67},
    "efficientnet-b1": {"a": 0.000513, "b": 0.0710, "c": 42.09},
    "efficientnet-b2": {"a": 0.000513, "b": 0.0739, "c": 47.10},
}

# GPU memory management constants
GPU_CACHE_CLEAR_INTERVAL = 5  # Clear GPU cache every N batches


def clear_gpu_cache_if_needed(batch_idx: int, interval: int = GPU_CACHE_CLEAR_INTERVAL):
    """Clear GPU cache periodically to prevent memory fragmentation.

    Args:
        batch_idx: Current batch index (0-based)
        interval: Clear cache every N batches
    """
    if torch.cuda.is_available() and (batch_idx + 1) % interval == 0:
        torch.cuda.empty_cache()


def estimate_batch_size(
    cfg,
    available_vram: float = None,
    safety_margin: float = 0.3,
) -> int:
    """Calculate optimal batch size based on available GPU VRAM and image dimensions.

    Uses empirically-derived memory consumption model to predict the maximum batch size
    that will fit in GPU memory. The model accounts for:
    - Input tensor memory (scales with batch_size × image_size²)
    - Intermediate activations (scales with batch_size × image_size²)
    - Model parameters (constant overhead)
    - CUDA memory allocator overhead (~1.65× peak allocation)

    The formula used is:
        reserved_mb = a × batch_size × image_size² + b × batch_size + c

    Args:
        available_vram: Available GPU VRAM in MB. If None, auto-detects
            from the current CUDA device.
        safety_margin: Fraction of VRAM to keep free (default: 0.2 = 20%).
            Higher values are safer but reduce batch size.
        model: Model architecture name. Supported values:
            - 'efficientnet-lite0' (default)
            - 'efficientnet-b1'
            - 'efficientnet-b2'

    Returns:
        int: Recommended batch size (minimum 1).

    Example:
        >>> # For a 16GB GPU with 64×64 images
        >>> # For a 16GB GPU with 64×64 images
        >>> batch_size = get_batch_size(image_size=64, available_vram=16384)
        >>> print(f"Recommended batch size: {batch_size}")
        Recommended batch size: 7852

        >>> # For 224×224 images with 20% safety margin
        >>> batch_size = get_batch_size(image_size=224, safety_margin=0.2)

    Notes:
        - The model was calibrated for EfficientNet architectures
        - For other architectures, efficientnet-lite0 coefficients provide
          a reasonable approximation
        - The safety_margin accounts for memory fragmentation and other
          processes using GPU memory
    """

    # Auto-detect available VRAM if not provided
    if available_vram is None:
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
            available_vram = device_props.total_memory / 1024**2  # Convert to MB
            logger.debug(f"Auto-detected GPU VRAM: {available_vram:.0f} MB")
        else:
            # Default to 4GB if no GPU detected (conservative estimate)
            available_vram = 4096
            logger.warning("No CUDA device detected, using default 4GB VRAM estimate")

    # Get coefficients for the specified model
    coef = MEMORY_COEFFICIENTS.get(cfg.net, MEMORY_COEFFICIENTS["efficientnet-lite0"])

    if cfg.net not in MEMORY_COEFFICIENTS:
        logger.warning(
            f"Unknown model '{cfg.net}', using efficientnet-lite0 coefficients. "
            f"Supported models: {list(MEMORY_COEFFICIENTS.keys())}"
        )

    # Calculate usable VRAM after safety margin
    usable_vram = available_vram * (1 - safety_margin)

    # Solve for batch_size:
    # usable_vram = a * B * S² + b * B + c
    # usable_vram - c = B * (a * S² + b)
    # B = (usable_vram - c) / (a * S² + b)
    S2 = cfg.normalisation.image_size[0] * cfg.normalisation.image_size[1]
    # Use num_channels if set, otherwise fall back to normalisation.n_output_channels
    num_channels = (
        cfg.num_channels
        if isinstance(cfg.num_channels, int)
        else cfg.normalisation.n_output_channels
    )
    denominator = coef["a"] * S2 * num_channels + coef["b"]

    if denominator <= 0:
        logger.warning("Invalid memory model parameters, returning minimum batch size")
        return 1

    batch_size = (usable_vram - coef["c"]) / denominator

    # Ensure batch size is at least 1
    batch_size = max(1, int(batch_size))

    logger.debug(
        f"Calculated batch size: {batch_size} "
        f"(image_size={cfg.normalisation.image_size[0]}, available_vram={available_vram:.0f}MB, "
        f"safety_margin={safety_margin}, model={cfg.net})"
    )

    return batch_size


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

    # Use num_channels if set, otherwise fall back to normalisation.n_output_channels
    num_channels = (
        cfg.num_channels
        if isinstance(cfg.num_channels, int)
        else cfg.normalisation.n_output_channels
    )
    net_builder = get_net_builder(
        cfg.net,
        pretrained=cfg.pretrained,
        in_channels=num_channels,
    )
    model = net_builder(num_classes=2, in_channels=num_channels)

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

    # Load fitsbolt config from checkpoint (DotMap pickles directly)
    if "fitsbolt_cfg" in checkpoint and checkpoint["fitsbolt_cfg"] is not None:
        cfg.fitsbolt_cfg = checkpoint["fitsbolt_cfg"]
        logger.info("Loaded fitsbolt config from model checkpoint")
    elif hasattr(cfg, "fitsbolt_cfg") and cfg.fitsbolt_cfg is not None:
        # Allow pre-set fitsbolt_cfg (for testing or advanced use cases)
        logger.info("Using fitsbolt config already present in cfg")
    else:
        raise ValueError(
            "Model checkpoint does not contain fitsbolt config. "
            "Please retrain the model with the updated version to include normalisation settings."
        )

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
    all_scores, all_filenames, existing_top_images, old_top_indices = _load_existing_predictions(
        predictions_file, output_npy_path, all_scores, all_filenames
    )

    # Get top N results from combined data
    top_indices = np.argsort(all_scores)[::-1][:top_n]
    top_scores = all_scores[top_indices]
    top_filenames = all_filenames[top_indices]

    # Ensure current batch images are in consistent HWC format BEFORE building top array
    all_imgs = _ensure_consistent_image_format(all_imgs)

    # Build the top images array
    top_imgs = _build_top_images_array(
        all_scores, all_imgs, top_indices, existing_top_images, old_top_indices
    )

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
        tuple: (merged_scores, merged_filenames, existing_top_images, old_top_indices)
    """
    existing_scores = []
    existing_filenames = []
    existing_top_images = None
    old_top_indices = None

    # Load existing predictions if available
    if os.path.exists(predictions_file):
        logger.info("Loading existing predictions for accumulation")
        with np.load(predictions_file, allow_pickle=True) as data:
            existing_scores = data["scores"]
            existing_filenames = data["filenames"]

        # Calculate the old top indices from existing scores
        old_top_indices = np.argsort(existing_scores)[::-1]
        logger.debug(f"Calculated old top indices shape: {old_top_indices.shape}")

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
        return merged_scores, merged_filenames, existing_top_images, old_top_indices

    return current_scores, current_filenames, existing_top_images, old_top_indices


def _build_top_images_array(
    all_scores, current_batch_imgs, top_indices, existing_top_images, old_top_indices
):
    """Build an array of top images from current batch and existing images.

    This function handles the complex logic of selecting images either from the current batch
    or from previously saved top images, based on their ranking in the combined dataset.

    Args:
        all_scores (np.ndarray): Combined scores from all batches including current one.
        current_batch_imgs (np.ndarray): Images from the current batch only.
        top_indices (np.ndarray): Indices of top scoring images in the combined dataset.
        existing_top_images (np.ndarray or None): Previously saved top images.
        old_top_indices (np.ndarray or None): Indices of old top results before merging.

    Returns:
        np.ndarray: Array of top images.
    """
    # Calculate indices for current batch in the combined scores array
    current_batch_start = len(all_scores) - len(current_batch_imgs)
    current_batch_global_indices = set(range(current_batch_start, len(all_scores)))

    # Create a mapping from old global index to position in existing_top_images
    old_idx_to_position = {}
    if old_top_indices is not None and existing_top_images is not None:
        for position, global_idx in enumerate(old_top_indices[: len(existing_top_images)]):
            old_idx_to_position[global_idx] = position

    # Collect images for each top index
    top_img_list = []
    missing_images = []

    for i, global_idx in enumerate(top_indices):
        # Case 1: This top result is from the current batch
        if global_idx in current_batch_global_indices:
            batch_idx = global_idx - current_batch_start
            top_img_list.append(current_batch_imgs[batch_idx])

        # Case 2: This top result is from a previous batch
        elif global_idx in old_idx_to_position:
            # Use the existing top image at the correct position
            old_position = old_idx_to_position[global_idx]
            top_img_list.append(existing_top_images[old_position])

        # Case 3: This image is from a previous batch but wasn't in the old top_N
        else:
            missing_images.append((i, global_idx))
            # Create a placeholder black image
            if len(top_img_list) > 0:
                placeholder = np.zeros_like(top_img_list[0])
            else:
                # Fallback: create a minimal placeholder
                placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
            top_img_list.append(placeholder)

    if missing_images:
        logger.warning(
            f"Could not retrieve {len(missing_images)} images from previous batches "
            f"(they were not in the previous top_N). Using placeholder images. "
            f"First few missing: {missing_images[:5]}"
        )

    # Verify we have the expected number of images
    if len(top_img_list) != len(top_indices):
        raise ValueError(
            f"Image count mismatch: expected {len(top_indices)} images, "
            f"but collected {len(top_img_list)}"
        )

    # Convert to numpy array - all images should have the same shape by now
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

    Note: Includes explicit CUDA tensor cleanup to prevent GPU memory fragmentation.

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

    # Explicit cleanup of CUDA tensors to prevent memory fragmentation
    del logits

    # Return original uint8 images if provided, otherwise convert tensor back
    if original_images is not None:
        # Clean up CUDA tensor before returning
        del images
        return batch_scores, original_images
    else:
        # Convert tensor images back to uint8 for saving with explicit cleanup
        images_np = images.detach().cpu().numpy()
        del images  # Free CUDA tensor

        if images_np.max() <= 1.0:
            # Tensor format [0,1] -> uint8 [0,255]
            images_uint8 = (images_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            # Assume already in correct range
            images_uint8 = images_np.clip(0, 255).astype(np.uint8)

        del images_np  # Free intermediate array
        return batch_scores, images_uint8
