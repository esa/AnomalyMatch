#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Demo script to exercise PredictionProfiler against test data.

Generates example performance_report.json output for PR documentation.
Runs image and Zarr pipelines with profiler instrumentation.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger
from PIL import Image

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_process import evaluate_files
from prediction_process_zarr import evaluate_images_in_zarr

from anomaly_match.data_io.load_images import fitsbolt_channel_combination
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.utils.prediction_profiler import PredictionProfiler
from tests.test_data.generate_test_model import ensure_test_model

TEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "test_data", "test_model.safetensors")


def make_config(output_dir):
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [150, 150]
    cfg.normalisation.n_output_channels = 3
    cfg.net = "efficientnet-lite0"
    # load_model overwrites every weight from the checkpoint, so fetching
    # ImageNet weights first is pure download time.
    cfg.pretrained = False
    cfg.num_channels = 3
    cfg.model_path = TEST_MODEL_PATH
    cfg.gpu = 0
    cfg.output_dir = output_dir
    cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    cfg.log_level = "INFO"
    cfg.name = "profiler_demo"
    cfg.seed = 42
    cfg.test_ratio = 0.0
    cfg.save_dir = output_dir
    cfg.save_file = "demo"
    cfg.data_dir = "tests/test_data/grayscale/"
    cfg.N_batch_prediction = None

    cfg.fitsbolt_cfg = fb_create_cfg(
        output_dtype=np.uint8,
        size=cfg.normalisation.image_size,
        fits_extension=cfg.normalisation.fits_extension,
        interpolation_order=cfg.normalisation.interpolation_order,
        normalisation_method=cfg.normalisation.normalisation_method,
        channel_combination=fitsbolt_channel_combination(cfg),
        num_workers=cfg.num_workers,
        norm_maximum_value=cfg.normalisation.norm_maximum_value,
        norm_minimum_value=cfg.normalisation.norm_minimum_value,
        norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
        norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
        norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
        norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
    )
    return cfg


def create_sample_images(tmp_dir, n=20):
    img_dir = os.path.join(tmp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    paths = []
    for i in range(n):
        arr = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        path = os.path.join(img_dir, f"img_{i:04d}.jpg")
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def create_sample_zarr(tmp_dir, n=20):
    zarr_path = os.path.join(tmp_dir, "demo.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    imgs = np.random.randint(0, 255, (n, 150, 150, 3), dtype=np.uint8)
    root.create_array("images", data=imgs)
    return zarr_path


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    ensure_test_model(Path(TEST_MODEL_PATH))

    tmp_dir = tempfile.mkdtemp(prefix="profiler_demo_")
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir)

    logger.info(f"Demo output directory: {output_dir}")
    cfg = make_config(output_dir)

    # --- Process 0: Image files (auto-detected idx) ---
    logger.info("=== Running image file pipeline (process_idx=0) ===")
    image_paths = create_sample_images(tmp_dir, n=20)
    evaluate_files(image_paths, cfg, top_n=10, batch_size=8)

    # --- Process 1: Zarr (auto-detected idx) ---
    logger.info("=== Running Zarr pipeline (process_idx=1) ===")
    zarr_path = create_sample_zarr(tmp_dir, n=20)
    evaluate_images_in_zarr(zarr_path, cfg, top_n=10, batch_size=8)

    # --- Merge reports via Coordinator ---
    logger.info("=== Merging reports ===")
    coordinator = PredictionProfiler.Coordinator(output_dir)
    report = coordinator.finalize()

    profiling_dir = os.path.join(output_dir, PredictionProfiler.PROFILING_SUBDIR)
    report_path = os.path.join(profiling_dir, PredictionProfiler.MERGED_FILENAME)
    logger.info(f"Merged report at: {report_path}")

    # Pretty-print report
    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT (performance_report.json)")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    return report_path


if __name__ == "__main__":
    main()
