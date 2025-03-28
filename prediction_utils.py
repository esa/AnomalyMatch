#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import os
import torch
import numpy as np
from loguru import logger
import pandas as pd
from torchvision import transforms
from turbojpeg import TurboJPEG

# Initialize TurboJPEG
jpeg_decoder = TurboJPEG()


def load_model(cfg):
    """Initialize and load the model."""
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
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda()
        logger.info(f"Using GPU device {cfg.gpu}")
    else:
        logger.info("Using CPU for inference")

    checkpoint = torch.load(model_path, map_location="cpu")
    if "eval_model" not in checkpoint:
        raise KeyError(
            f"Model checkpoint does not contain 'eval_model' key. Keys found: {checkpoint.keys()}"
        )

    model.load_state_dict(checkpoint["eval_model"])
    logger.success(f"Successfully loaded model from {model_path}")
    return model


def save_results(cfg, all_scores, all_imgs, all_filenames, top_n):
    """Save results to files."""
    logger.info(f"Saving results with {len(all_scores)} total predictions")

    # Get the paths for results
    output_csv_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{top_n}.csv")
    output_npy_path = os.path.join(cfg.output_dir, f"{cfg.save_file}_top{top_n}.npy")
    predictions_file = os.path.join(cfg.output_dir, f"all_predictions_{cfg.save_file}.npz")

    # Load existing predictions if they exist
    existing_scores = []
    existing_filenames = []
    if os.path.exists(predictions_file):
        logger.info("Loading existing predictions for accumulation")
        with np.load(predictions_file, allow_pickle=True) as data:
            existing_scores = data["scores"]
            existing_filenames = data["filenames"]

        # Combine existing and new predictions
        all_scores = np.concatenate([existing_scores, all_scores])
        all_filenames = np.concatenate([existing_filenames, all_filenames])
        logger.info(
            f"Combined {len(existing_scores)} existing and {len(all_scores) - len(existing_scores)} new predictions"
        )

    # Get top N results from combined data
    top_indices = np.argsort(all_scores)[::-1][:top_n]
    top_scores = all_scores[top_indices]
    top_filenames = all_filenames[top_indices]

    # For images, we only keep the current batch's top N
    current_batch_top_indices = np.argsort(all_scores[-len(all_imgs) :])[::-1][:top_n]
    top_imgs = all_imgs[current_batch_top_indices]

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


def get_transform():
    """Get the standard image transform."""
    return transforms.Compose([transforms.ToTensor()])


def process_batch_predictions(model, images):
    """Process a batch of images through the model."""
    if torch.cuda.is_available():
        images = images.cuda(non_blocking=True)

    with torch.no_grad():
        logits = model(images)
        batch_scores = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    return batch_scores, images.cpu().numpy()
