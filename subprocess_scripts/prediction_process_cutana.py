#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import argparse
import os
import time

import cutana
import numpy as np
import pandas as pd
from cutana.catalogue_preprocessor import parse_fits_file_paths
from dotmap import DotMap
from loguru import logger
from tqdm import tqdm

from anomaly_match.prediction import AnomalyScoreDB, prediction_db_path
from anomaly_match.utils.prediction_profiler import PredictionProfiler
from prediction_utils import (
    clear_gpu_cache_if_needed,
    cutana_batch_to_model_tensor_gpu,
    filter_unprocessed_batch,
    gate_db_on_compatibility,
    install_shutdown_handler,
    load_excluded_label_ids,
    load_model,
    load_prediction_config,
    model_inference_device,
    pipelined_batches,
    process_batch_predictions,
    setup_prediction_logging,
    shutdown_requested,
    warmup_inference,
)


def _diagnose_unreachable_fits(catalogue_path: str) -> str:
    """Inspect a Cutana catalogue parquet and return a brief reachability summary.

    Called when every batch returns empty cutouts — the most common cause
    is that the FITS tiles referenced by the catalogue have become
    unreachable (e.g. a network volume detached mid-run).  Reads the
    ``fits_file_paths`` column with Cutana's canonical parser and counts
    how many distinct paths do/don't exist on disk.

    Trusts the catalogue format: a malformed parquet or unparseable cell
    raises out of this function so the caller can decide how to surface
    it.  The caller (the empty-batch branch in
    :func:`evaluate_images_from_cutana`) wraps this in a broad
    ``except`` so a diagnostic failure can't escalate an already-failed
    chunk into a hard subprocess crash.

    Args:
        catalogue_path: Path to the cutana buffer parquet that drove the
            empty-cutout chunk.

    Returns:
        Single-line summary like ``"6/8 referenced FITS files unreachable
        (e.g. '/data/.../tile.fits')"``.
    """
    df = pd.read_parquet(catalogue_path, columns=["fits_file_paths"])
    paths: set[str] = set()
    for raw in df["fits_file_paths"]:
        paths.update(parse_fits_file_paths(raw))

    missing = [p for p in paths if not os.path.isfile(p)]
    if not missing:
        return f"all {len(paths)} referenced FITS file(s) exist on disk"
    return f"{len(missing)}/{len(paths)} referenced FITS file(s) unreachable (e.g. {missing[0]!r})"


def evaluate_images_from_cutana(cutana_sources_path, cfg, top_n=1000, batch_size=1000):
    """Evaluate images provided by a Cutana stream and write scores to the DB.

    Uses the shared :func:`build_cutana_orchestrator_config` to configure
    the orchestrator with identity channel weights, then applies the user's
    ``channel_combination`` via :func:`cutana_batch_to_model_tensor` on each
    batch — the same post-processing path used by training and the cache
    read path.
    """
    # Lazy import: cutana_source pulls in the AnomalyMatch datasets package,
    # which touches PyTorch. Importing it at module load in this subprocess
    # can hang on CUDA initialisation before we've set the right env vars.
    from anomaly_match.datasets.cutana_source import (  # noqa: PLC0415
        build_cutana_orchestrator_config,
    )

    # Load the model FIRST: load_model syncs cfg.fitsbolt_cfg *and*
    # cfg.normalisation from the checkpoint, which is the single source of truth
    # for resolution and channel mixing.  build_cutana_orchestrator_config reads
    # cfg.normalisation.image_size for target_resolution and cfg.fitsbolt_cfg for
    # the per-band normalisation, so it MUST run after this sync — otherwise the
    # orchestrator produces cutouts at a stale resolution/normalisation that the
    # model was never trained on (silent accuracy loss).
    model = load_model(cfg)
    model.eval()
    model_device = model_inference_device(model)

    # load_model already raises when the checkpoint has no fitsbolt config; this
    # guards the remaining case — an externally pre-set cfg.fitsbolt_cfg (the
    # advanced/testing path in load_model) that is malformed (missing 'size').
    # Building the orchestrator from such a config would silently mis-resolve
    # the cutout resolution.  Note: DotMap auto-creates empty child maps on
    # missing-key access, so check 'size' membership, not truthiness.
    fitsbolt_cfg = cfg.fitsbolt_cfg
    if fitsbolt_cfg is None or (isinstance(fitsbolt_cfg, DotMap) and "size" not in fitsbolt_cfg):
        raise ValueError(
            "cfg.fitsbolt_cfg is missing or malformed (no 'size'); cannot build the "
            "Cutana orchestrator. Models must be saved with a fitsbolt config for prediction."
        )
    logger.debug("Using fitsbolt config loaded from model checkpoint")

    cutana_config = build_cutana_orchestrator_config(
        cutana_sources_path, cfg, prune_unused_bands=True
    )

    try:
        logger.info(f"Creating Cutana orchestrator, streaming from {cutana_sources_path}")
        logger.debug(
            f"Cutana config: target_resolution={cutana_config.target_resolution}, "
            f"fits_extensions={cutana_config.fits_extensions}, "
            f"selected_extensions={cutana_config.selected_extensions}"
        )

        cutana_orchestrator = cutana.StreamingOrchestrator(cutana_config)

        cutana_orchestrator.init_streaming(
            batch_size=batch_size,
            write_to_disk=False,
            min_workers=cfg.cutana_min_workers,
            max_workers=cfg.cutana_max_workers,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Cutana orchestrator: {e}")
        raise

    logger.info("Cutana orchestrator streaming mode initialized")

    logger.info(f"Available batches in cutana: {cutana_orchestrator.get_batch_count()}")

    # Warm up the GPU fast paths on a throwaway batch before the timed loop:
    # front-loads cuDNN autotuning (and torch.compile when enabled) so the first
    # *real* batch isn't a latency spike, and surfaces any compile-toolchain
    # failure here — before we start consuming the stream — rather than mid-run.
    # The profiler also benefits (compile cost isn't misattributed to inference),
    # but the smoother first-batch latency is the production reason.
    # Announce it: warmup is several silent GPU-bound seconds the UI would
    # otherwise show as a frozen "Starting scoring..." (#scoring-startup).
    logger.info("Warming up GPU for inference (cuDNN autotuning)...")
    warmup_inference(model, cfg, batch_size, model_device)

    profiler = PredictionProfiler(output_dir=cfg.output_dir, script="prediction_cutana")

    # Incremental DB writes for live UI polling. The path may be relocated to
    # local scratch (NFS sessions); ensure its dir exists before opening.
    db_path = prediction_db_path(cfg)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with AnomalyScoreDB(db_path) as db:
        gate_db_on_compatibility(db, cfg)

        batches_count = cutana_orchestrator.get_batch_count()
        scored_count = 0
        resume_skipped = 0
        label_excluded = 0
        # Cutana source ids match the label CSV's id column directly, so the
        # labelled set can be intersected against batch source ids as-is.
        excluded_ids = load_excluded_label_ids(cfg)

        # Prefetch the next batch on a single background worker so the GPU isn't
        # idle while Cutana streams/assembles cutouts. pipelined_batches uses one
        # worker, satisfying the orchestrator's single-reader requirement; the
        # resume filter stays on this (the consumer) thread. A streaming failure
        # is re-raised from next() and reaches main()'s handler and the log.
        batch_stream = pipelined_batches(
            batches_count,
            prepare=lambda _index: True,  # no main-thread prep; filtering is post-load
            load=lambda _plan: cutana_orchestrator.next_batch(),
            should_stop=shutdown_requested,
            # Depth-2: Cutana's NFS tile reads are bursty, so keep the worker
            # producing ahead to absorb a slow batch instead of stalling the GPU.
            lookahead=2,
        )

        # The first batch can block for tens of seconds while Cutana streams
        # and assembles cutouts from FITS tiles over NFS, before "Processing
        # batches" can tick — say so rather than sit silent.
        logger.info("Streaming first cutout batch (reading FITS tiles over the network)...")
        for batch_idx in tqdm(range(batches_count), desc="Processing batches"):
            if shutdown_requested():
                logger.warning("Shutdown requested — stopping before batch {}.", batch_idx)
                break

            # io_load measures the wait on the prefetcher: ~0 when streaming keeps
            # ahead of the GPU, exposing how much is truly un-overlapped.
            with profiler.stage("io_load"):
                try:
                    _idx, loaded_batch = next(batch_stream)
                except StopIteration:
                    # The stream should yield exactly batches_count items (Cutana
                    # prepare never skips), so early exhaustion means the producer
                    # ended short — surface it instead of silently scoring fewer.
                    if not shutdown_requested() and batch_idx < batches_count - 1:
                        logger.warning(
                            "Cutana stream exhausted after {} of {} batches — "
                            "some sources were not scored.",
                            batch_idx,
                            batches_count,
                        )
                    break
            batch_data = loaded_batch["cutouts"]

            logger.debug(
                f"Batch {batch_idx}: cutouts type={type(batch_data).__name__}, "
                f"metadata count={len(loaded_batch.get('metadata', []))}"
            )

            # Handle empty batches (cutana returns [] if all cutouts failed)
            if isinstance(batch_data, list):
                if len(batch_data) == 0:
                    logger.warning(
                        f"Batch {batch_idx} returned empty cutouts (list), skipping. "
                        f"Please ensure that number of extensions in Cutana source "
                        f"catalogues matches those of the training set."
                    )
                    continue
                batch_data = np.array(batch_data)

            batch_source_ids = [str(source["source_id"]) for source in loaded_batch["metadata"]]
            # db_read: the resume filter is a SELECT against the score DB; on a
            # bloated NFS WAL it grew large, so measure it explicitly.
            with profiler.stage("db_read"):
                keep, n_skipped, n_excluded = filter_unprocessed_batch(
                    batch_source_ids, db, exclude=excluded_ids
                )
            resume_skipped += n_skipped
            label_excluded += n_excluded
            n_dropped = n_skipped + n_excluded
            if n_dropped:
                # Stream-driven path: cutana already produced the cutouts,
                # so we drop already-scored / already-labelled entries from
                # the in-memory batch instead of seeking past them.
                if n_dropped == len(batch_source_ids):
                    logger.debug(
                        f"Batch {batch_idx}: all {len(batch_source_ids)} sources "
                        "already scored or labelled — skipping batch."
                    )
                    continue
                mask = np.asarray(keep, dtype=bool)
                batch_data = batch_data[mask]
                batch_source_ids = [sid for sid, k in zip(batch_source_ids, keep) if k]

            batch_size_actual = batch_data.shape[0]

            with profiler.stage("preprocess"):
                # Channel-combination + scaling on the GPU: upload the per-band
                # uint8 batch as-is and combine/scale on-device (issue #458).
                images = cutana_batch_to_model_tensor_gpu(batch_data, cfg, device=model_device)

            with profiler.stage("inference"):
                # return_images=False: the streaming path discards the cutouts, so
                # skip the per-batch GPU→host image copy + uint8 conversion.
                batch_scores, _ = process_batch_predictions(model, images, return_images=False)
                del images  # Free CUDA tensor reference

            # db_write inside the batch window (before end_batch) so the commit +
            # WAL checkpoint cost is attributed instead of hiding in the gap.
            with profiler.stage("db_write"):
                db.store_results(list(zip(batch_source_ids, batch_scores.tolist())))

            profiler.end_batch(batch_size=batch_size_actual)
            scored_count += len(batch_source_ids)

            # Periodic GPU cache clearing to prevent fragmentation
            clear_gpu_cache_if_needed(batch_idx)

        # Stop the prefetch worker (no-op if the stream is already exhausted).
        batch_stream.close()

        # Every batch came back empty.  Common cause: the data volume
        # disconnected mid-run, so Cutana workers can't reach the FITS
        # tiles this catalogue chunk references.  Don't raise — that
        # would kill the whole multi-hour scoring loop on a transient
        # platform issue.  Log a clear warning with a reachability
        # diagnostic so the parent's chunk-level "rows_added == 0"
        # check can mark this chunk as skipped and continue.  Resume
        # and shutdown-requested exits don't fit this pattern (the
        # former scores nothing because everything's already scored;
        # the latter exits intentionally), so don't even diagnose
        # them — only the genuine empty-data case lands here.
        if label_excluded:
            logger.info(
                "Skipped {:,} already-labelled source(s) — kept out of the score gallery",
                label_excluded,
            )

        if (
            batches_count > 0
            and scored_count == 0
            and resume_skipped == 0
            and label_excluded == 0
            and not shutdown_requested()
        ):
            try:
                diagnosis = _diagnose_unreachable_fits(cutana_sources_path)
            except Exception as exc:
                # Diagnostic failure must not escalate an already-failed
                # chunk into a hard crash — log and fall back to the
                # generic message so the parent's skip path still kicks in.
                logger.warning(
                    "Could not diagnose empty-cutout cause from catalogue {}: {}",
                    cutana_sources_path,
                    exc,
                )
                diagnosis = "Could not diagnose underlying cause from catalogue."
            logger.warning(
                "Cutana prediction completed {} batch(es) but scored zero sources — "
                "every batch returned empty cutouts. {} The parent process will "
                "skip this chunk and continue.",
                batches_count,
                diagnosis,
            )

    cutana_orchestrator.cleanup()

    profiler.record_finalization()
    profiler.save_partial_report()


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument(
        "cutana_sources_path", type=str, help="Path to the directory to stream from"
    )
    parser.add_argument("top_n", type=int, default=1000, help="Number of top scores to keep")
    args = parser.parse_args()

    cfg, batch_size = load_prediction_config(args.config_path)

    logger.info(f"Streaming from directory: {args.cutana_sources_path}")

    try:
        evaluate_images_from_cutana(
            args.cutana_sources_path, cfg, batch_size=batch_size, top_n=args.top_n
        )
        elapsed_time = time.time() - start_time
        logger.success(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    setup_prediction_logging("prediction_cutana")
    install_shutdown_handler()
    main()
