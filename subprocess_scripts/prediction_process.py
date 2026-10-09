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

import torch
from loguru import logger
from tqdm import tqdm

from anomaly_match.data_io.load_images import (
    load_and_process_single_wrapper,
)
from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
)
from anomaly_match.prediction import AnomalyScoreDB, prediction_db_path
from anomaly_match.utils.prediction_profiler import PredictionProfiler
from prediction_utils import (
    basename_exclusions,
    clear_gpu_cache_if_needed,
    filter_unprocessed_batch,
    gate_db_on_compatibility,
    install_shutdown_handler,
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

    Resume is per-batch: each chunk asks the DB which of its candidate
    filenames are not yet scored and only loads those.  This avoids the
    O(N) full-set fetch that scaled to ~8 minutes per chunk subprocess
    once the DB grew past ~10 M rows (#435).
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
    model_device = model_inference_device(model)

    # Require fitsbolt config from model checkpoint for consistent predictions
    if not hasattr(cfg, "fitsbolt_cfg") or cfg.fitsbolt_cfg is None:
        raise ValueError(
            "Fitsbolt config not found in model checkpoint. "
            "Please retrain the model with the updated version to include normalisation settings."
        )
    logger.debug("Using fitsbolt config loaded from model checkpoint")

    # Front-load cuDNN autotuning (and torch.compile when enabled) so the first
    # real batch isn't a latency spike and any compile failure surfaces now.
    warmup_inference(model, cfg, batch_size, model_device)

    transform = get_prediction_transforms()

    profiler = PredictionProfiler(output_dir=cfg.output_dir, script="prediction_thread")

    scored_count = 0
    resume_skipped = 0
    label_excluded = 0
    n_batches = (len(file_list) + batch_size - 1) // batch_size

    # Incremental DB writes for live UI polling. The path may be relocated to
    # local scratch (NFS sessions); ensure its dir exists before opening.
    db_path = prediction_db_path(cfg)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with AnomalyScoreDB(db_path) as db:
        gate_db_on_compatibility(db, cfg)

        # Label ids for folder sources are image basenames, but the DB / resume
        # keys are full paths; map labelled basenames into path space per chunk
        # so already-labelled images are never re-scored into the gallery.
        excluded_basenames = load_excluded_label_ids(cfg)

        def prepare(batch_idx):
            """Resume-filter a chunk on the consumer thread (keeps sqlite single-threaded).

            Returns the kept file paths, or ``None`` to skip a chunk whose files
            are all already scored or labelled.
            """
            nonlocal resume_skipped, label_excluded
            chunk = file_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            chunk_keys = [str(fp) for fp in chunk]
            exclude = basename_exclusions(chunk_keys, excluded_basenames)
            keep, n_skipped, n_excluded = filter_unprocessed_batch(chunk_keys, db, exclude=exclude)
            resume_skipped += n_skipped
            label_excluded += n_excluded
            return [fp for fp, k in zip(chunk, keep) if k] or None

        def load(chunk_kept):
            """Read + preprocess the kept files on the prefetch worker (no DB, no CUDA)."""
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(load_and_preprocess, [(fp, cfg) for fp in chunk_kept]))
            return [item[0] for item in results], [item[1] for item in results]

        # One batch ahead: the worker reads/decodes batch N+1 while the GPU scores
        # batch N. Tensor creation stays on this thread (the worker returns numpy),
        # matching load_and_preprocess's no-CUDA-in-threads contract.
        batch_stream = pipelined_batches(n_batches, prepare, load, should_stop=shutdown_requested)
        with tqdm(total=n_batches, desc="Processing batches") as progress:
            while True:
                if shutdown_requested():
                    logger.warning("Shutdown requested — stopping.")
                    break
                with profiler.stage("io_load"):
                    try:
                        batch_idx, (batch_filenames, numpy_images) = next(batch_stream)
                    except StopIteration:
                        break
                progress.update(batch_idx + 1 - progress.n)

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
                        list(zip([str(fn) for fn in batch_filenames], batch_scores.tolist()))
                    )

                profiler.end_batch(batch_size=len(batch_filenames))
                scored_count += len(batch_filenames)
                clear_gpu_cache_if_needed(batch_idx)
        batch_stream.close()

    log_resume_summary(scored_count, resume_skipped, len(file_list), kind="files")
    if label_excluded:
        logger.info(
            "Skipped {:,} already-labelled image(s) — kept out of the score gallery",
            label_excluded,
        )

    profiler.record_finalization()
    profiler.save_partial_report()


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
    evaluate_files(file_list, cfg, batch_size=batch_size, top_n=args.top_n)
    logger.success("Evaluation complete. Results written to predictions.db")

    elapsed_time = time.time() - start_time
    logger.success(f"Script completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    setup_prediction_logging("prediction_thread")
    install_shutdown_handler()
    main()
