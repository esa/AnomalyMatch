#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import zarr
from loguru import logger
from tqdm import tqdm

from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from anomaly_match.prediction import AnomalyScoreDB, prediction_db_path
from anomaly_match.prediction.zarr_filenames import derive_filenames as _derive_zarr_filenames
from anomaly_match.utils.prediction_profiler import PredictionProfiler
from prediction_utils import (
    clear_gpu_cache_if_needed,
    filter_unprocessed_batch,
    gate_db_on_compatibility,
    install_shutdown_handler,
    load_and_preprocess_zarr,
    load_excluded_label_ids,
    load_model,
    load_prediction_config,
    log_resume_summary,
    model_inference_device,
    pipelined_batches,
    process_batch_predictions,
    setup_prediction_logging,
    shutdown_requested,
    warmup_inference,
)


def evaluate_images_in_zarr(zarr_path, cfg, top_n=1000, batch_size=1000, max_workers=4):
    """Evaluate images inside a Zarr file and return top N scores.

    Resume is per-batch: each batch asks the DB which of its candidate
    filenames are not yet scored and only reads / decodes those.  This
    avoids the O(N) full-set fetch that scaled to ~8 minutes per chunk
    subprocess once the DB grew past ~10 M rows (#435).
    """
    logger.info(f"Opening Zarr file {zarr_path}")

    zarr_path = Path(zarr_path)

    # Open Zarr store
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
    except Exception as e:
        logger.error(f"Failed to open Zarr store: {e}")
        raise

    if "images" not in root:
        raise ValueError(f"No 'images' array found in Zarr store {zarr_path}")

    images_array = root["images"]
    num_images = images_array.shape[0]
    logger.info(f"Found {num_images} images in the Zarr file")
    logger.info(f"Image array shape: {images_array.shape}")
    logger.info(f"Image array dtype: {images_array.dtype}")

    filenames = _derive_zarr_filenames(zarr_path)
    if len(filenames) != num_images:
        raise ValueError(
            f"derive_filenames returned {len(filenames)} names for a zarr "
            f"with {num_images} images at {zarr_path}"
        )

    model = load_model(cfg)
    model.eval()
    model_device = model_inference_device(model)
    transform = get_prediction_transforms()

    # Front-load cuDNN autotuning (and torch.compile when enabled) so the first
    # real batch isn't a latency spike and any compile failure surfaces now.
    warmup_inference(model, cfg, batch_size, model_device)

    profiler = PredictionProfiler(output_dir=cfg.output_dir, script="prediction_zarr")

    scored_count = 0
    resume_skipped = 0
    label_excluded = 0
    n_batches = (num_images + batch_size - 1) // batch_size

    # Incremental DB writes for live UI polling. The path may be relocated to
    # local scratch (NFS sessions); ensure its dir exists before opening.
    db_path = prediction_db_path(cfg)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with AnomalyScoreDB(db_path) as db:
        gate_db_on_compatibility(db, cfg)

        # Zarr derived filenames share the label CSV's id space, so the labelled
        # set can be excluded directly, keeping training data out of the gallery.
        excluded_ids = load_excluded_label_ids(cfg)

        def prepare(batch_idx):
            """Resume-filter a chunk on the consumer thread (keeps sqlite single-threaded).

            Returns ``(keep_indices, keep_filenames)``, or ``None`` to skip an
            all-already-scored chunk.
            """
            nonlocal resume_skipped, label_excluded
            chunk_start = batch_idx * batch_size
            chunk_end = min(chunk_start + batch_size, num_images)
            chunk_indices = list(range(chunk_start, chunk_end))
            chunk_filenames = [filenames[i] for i in chunk_indices]
            keep, n_skipped, n_excluded = filter_unprocessed_batch(
                chunk_filenames, db, exclude=excluded_ids
            )
            resume_skipped += n_skipped
            label_excluded += n_excluded
            keep_indices = [idx for idx, k in zip(chunk_indices, keep) if k]
            keep_filenames = [fn for fn, k in zip(chunk_filenames, keep) if k]
            if not keep_indices:
                return None
            return keep_indices, keep_filenames

        def load(plan):
            """Read the kept Zarr rows + preprocess on the prefetch worker (no DB, no CUDA).

            Contiguous runs are the fast path; oindex covers the sparse resume
            case without a separate code path.
            """
            keep_indices, keep_filenames = plan
            n_kept = len(keep_indices)
            if keep_indices[-1] - keep_indices[0] == n_kept - 1:
                batch_data = images_array[keep_indices[0] : keep_indices[-1] + 1]
            else:
                batch_data = images_array.oindex[keep_indices]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                numpy_images = list(
                    executor.map(
                        load_and_preprocess_zarr, [(batch_data[i], cfg) for i in range(n_kept)]
                    )
                )
            return keep_filenames, numpy_images

        # One batch ahead: the worker reads/decodes Zarr rows for batch N+1 while
        # the GPU scores batch N. Tensor creation stays on this thread.
        batch_stream = pipelined_batches(n_batches, prepare, load, should_stop=shutdown_requested)
        with tqdm(total=n_batches, desc="Processing batches") as progress:
            while True:
                if shutdown_requested():
                    logger.warning("Shutdown requested — stopping.")
                    break
                with profiler.stage("io_load"):
                    try:
                        batch_idx, (keep_filenames, numpy_images) = next(batch_stream)
                    except StopIteration:
                        break
                progress.update(batch_idx + 1 - progress.n)
                n_kept = len(keep_filenames)

                with profiler.stage("preprocess"):
                    batch_tensors = [transform(img) for img in numpy_images]
                    images = torch.stack(batch_tensors, dim=0)
                    del numpy_images, batch_tensors  # Free memory before CUDA ops

                with profiler.stage("inference"):
                    # return_images=False: scores are all we keep, so skip the
                    # per-batch GPU→host image copy + uint8 conversion.
                    batch_scores, _ = process_batch_predictions(model, images, return_images=False)
                    del images  # Free CUDA tensor reference

                # db_write inside the batch window (before end_batch) so the commit
                # + WAL checkpoint cost is attributed instead of hiding in the gap.
                with profiler.stage("db_write"):
                    db.store_results(
                        list(zip([str(fn) for fn in keep_filenames], batch_scores.tolist()))
                    )

                profiler.end_batch(batch_size=n_kept)
                scored_count += n_kept
                clear_gpu_cache_if_needed(batch_idx)
        batch_stream.close()

    log_resume_summary(scored_count, resume_skipped, num_images, kind="zarr entries")
    if label_excluded:
        logger.info(
            "Skipped {:,} already-labelled entry(ies) — kept out of the score gallery",
            label_excluded,
        )

    profiler.record_finalization()
    profiler.save_partial_report()


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing images")
    parser.add_argument("top_n", type=int, default=1000, help="Number of top scores to keep")
    args = parser.parse_args()

    cfg, batch_size = load_prediction_config(args.config_path)

    logger.info(f"Processing Zarr file: {args.zarr_path}")

    try:
        evaluate_images_in_zarr(args.zarr_path, cfg, batch_size=batch_size, top_n=args.top_n)
        elapsed_time = time.time() - start_time
        logger.success(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    setup_prediction_logging("prediction_zarr")
    install_shutdown_handler()
    main()
