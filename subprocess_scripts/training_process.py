#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Training subprocess — trains FixMatch for N iterations and saves the checkpoint.

Launched by :meth:`BackendInterface.launch_training_subprocess` to keep the
UI responsive and memory-isolated.  Reads a pickled config and a labels CSV,
creates datasets, trains the model, writes progress to a JSON-lines file,
and saves the model checkpoint.

Usage::

    python subprocess_scripts/training_process.py <config_pickle> <labels_csv> <progress_file>
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from dotmap import DotMap
from loguru import logger

from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.datasets.cutana_source import _size_bin_edges
from anomaly_match.datasets.SSL_Dataset import SSL_Dataset
from anomaly_match.datasets.training_data_source import (
    TrainingDataSource,
    create_training_data_source,
)
from anomaly_match.models.FixMatch import FixMatch
from anomaly_match.utils.get_net_builder import get_net_builder
from anomaly_match.utils.get_optimizer import get_optimizer
from anomaly_match.utils.validate_config import validate_config
from prediction_utils import load_prediction_config, setup_prediction_logging


def _write_progress(path: str, data: dict) -> None:
    """Append a JSON-lines entry to the progress file."""
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def _emit_size_histogram(progress_file: str, data_source: TrainingDataSource, cfg: DotMap) -> None:
    """Emit the tile-population vs. sampled-pool source-size distributions.

    Only Cutana sources expose a per-source diameter; for others
    ``get_last_unlabeled_sizes`` returns ``None`` and we emit nothing.  Two count
    arrays go over the JSON-lines channel — the candidate population of the drawn
    tiles and the sampled unlabeled pool — sharing one set of bin edges so the UI
    can overlay them.  Comparing the two is what shows whether size stratification
    flattened the pool (the population is small-skewed; a stratified sample is
    ~flat in log size).  Rendering stays in the UI to keep matplotlib out of the
    subprocess.

    The bins are **log-spaced** and span the population's full size range (fixed
    width up to ``max_px``, continued past it to cover the heavy tail) because the
    diameter distribution is heavy-tailed — linear bins would collapse the bulk
    into one bar.  They share the sampler's edges (``cutana_size_stratify_bins`` /
    ``_max_px``) so the sampled overlay lines up with the flattened target: a flat
    plateau up to ~``max_px``, then a thinning at the large-size end where the
    catalogue simply runs out of large sources (every one is kept, but there are
    few).  The UI draws them on a log x-axis.

    Args:
        progress_file: Path to the JSON-lines progress file.
        data_source: The training data source just used to build the datasets.
        cfg: Active configuration (for the stratification flag + bin count).
    """
    sampled = data_source.get_last_unlabeled_sizes()
    if sampled is None or len(sampled) == 0:
        return
    population = data_source.get_last_unlabeled_population_sizes()
    if population is None or len(population) == 0:
        # The Cutana samplers always record the tile population alongside the
        # sample, so this only trips if an unsized catalogue slipped through.
        # Skip the overlay rather than draw a meaningless population==sample plot.
        logger.debug("No candidate-population sizes recorded — skipping size overlay")
        return
    # Bin over the population's full size range — the same edges the sampler uses,
    # so the sampled overlay and the population share a scale and the tail shows.
    edges = _size_bin_edges(
        population, cfg.cutana_size_stratify_bins, cfg.cutana_size_stratify_max_px
    )
    if edges is None:
        return
    # Clip to the shared edges before histogramming so the extreme endpoints land
    # in the end bins rather than being dropped by np.histogram's half-open range.
    sampled_counts, _ = np.histogram(np.clip(sampled, edges[0], edges[-1]), bins=edges)
    population_counts, _ = np.histogram(np.clip(population, edges[0], edges[-1]), bins=edges)
    unit = data_source.get_last_unlabeled_size_unit() or "pixel"
    _write_progress(
        progress_file,
        {
            "status": "size_histogram",
            "edges": edges.tolist(),
            "population_counts": population_counts.tolist(),
            "sampled_counts": sampled_counts.tolist(),
            "stratified": bool(cfg.cutana_stratify_source_size),
            "unit": unit,
        },
    )
    logger.info(
        "Unlabeled pool size distribution: {} sampled / {} candidate sources, "
        "sampled diameter [{:.0f}, {:.0f}] {}",
        len(sampled),
        len(population),
        float(np.min(sampled)),
        float(np.max(sampled)),
        unit,
    )


def main() -> None:
    """Run the training subprocess."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description="AnomalyMatch training subprocess")
    parser.add_argument("config_path", help="Path to pickled config")
    parser.add_argument("labels_csv", help="Path to labelled_data CSV")
    parser.add_argument("progress_file", help="Path to write JSON-lines progress")
    parser.add_argument(
        "--labeled-cache", default=None, help="Path to labeled data cache directory"
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    cfg, _batch_size = load_prediction_config(args.config_path)

    # Add iteration-specific log so all training output lands next to the
    # model checkpoint (e.g. .../session/iteration_0/training.log).
    iter_dir = os.path.dirname(cfg.model_path) if cfg.model_path else cfg.output_dir
    if iter_dir:
        os.makedirs(iter_dir, exist_ok=True)
        logger.add(
            os.path.join(iter_dir, "training.log"),
            rotation="10 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
        )

    # For small datasets, force single-process data loading: this script
    # already runs as a child subprocess, so spawning DataLoader workers
    # causes hangs on Windows and unnecessary overhead for small pools.
    # For larger datasets, keep the configured value for throughput.
    original_num_workers = cfg.num_workers

    # Compute N_unlabeled: enough to fill all training iterations, capped
    # to avoid excessive memory usage with high-resolution images.
    image_size = cfg.normalisation.image_size
    hires_threshold = cfg.unlabeled_pool_hires_threshold
    cap = (
        cfg.unlabeled_pool_cap_hires
        if max(image_size) > hires_threshold
        else cfg.unlabeled_pool_cap
    )
    n_unlabeled = min(cfg.batch_size * cfg.uratio * cfg.num_train_iter, cap)
    cfg.N_to_load = int(n_unlabeled)
    logger.info("N_unlabeled = {} (cap={})", cfg.N_to_load, cap)

    # Ensure label_file points to our CSV
    cfg.label_file = args.labels_csv

    # Rebuild fitsbolt_cfg from current normalisation settings so that
    # image_size changes made in the UI take effect (the pickled config
    # may carry a stale fitsbolt_cfg from a previous model checkpoint).
    cfg = get_fitsbolt_config(cfg)

    validate_config(cfg)

    # Seed the training subprocess for reproducibility. cfg.seed was stored but
    # never applied, so labeled/unlabeled splits, samplers and weight/head
    # initialisation were unseeded — a source of large run-to-run variance.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Build datasets ───────────────────────────────────────────────
    _write_progress(args.progress_file, {"status": "loading", "message": "Loading datasets..."})
    use_lazy = args.labeled_cache is not None
    data_source = create_training_data_source(cfg, lazy=use_lazy)
    # Emit the size histogram the moment the unlabeled pool's sizes are recorded
    # (before the minutes-long cutout streaming), so the plot is visible while
    # cutouts stream rather than only after. Cutana is the only source that fires
    # this; others never record sizes so it simply never runs.
    data_source.unlabeled_sizes_callback = lambda: _emit_size_histogram(
        args.progress_file, data_source, cfg
    )
    train_dset = SSL_Dataset(
        cfg=cfg,
        train=True,
        data_source=data_source,
        labeled_cache_path=args.labeled_cache,
    )
    labeled_dataset, unlabeled_dataset = train_dset.get_ssl_dset()
    cfg.num_classes = train_dset.num_classes
    cfg.num_channels = train_dset.num_channels

    test_dataset = None
    if cfg.test_ratio > 0:
        test_dataset = SSL_Dataset(cfg=cfg, train=False, data_source=data_source).get_dset()

    logger.info(
        "Datasets ready — {} labeled, {} unlabeled", len(labeled_dataset), len(unlabeled_dataset)
    )

    # Decide num_workers now that dataset sizes are known
    small_pool = len(labeled_dataset) < 1000 and len(unlabeled_dataset) < 1000
    if small_pool:
        cfg.num_workers = 0
        logger.info("Small dataset pool — forcing num_workers=0 (was {})", original_num_workers)
    else:
        logger.info("Using configured num_workers={}", cfg.num_workers)

    # ── Build model ──────────────────────────────────────────────────
    _write_progress(args.progress_file, {"status": "loading", "message": "Building model..."})
    net_builder = get_net_builder(cfg.net, pretrained=cfg.pretrained, in_channels=cfg.num_channels)
    model = FixMatch(
        net_builder,
        cfg.num_classes,
        cfg.num_channels,
        cfg.ema_m,
        T=cfg.temperature,
        p_cutoff=cfg.p_cutoff,
        lambda_u=cfg.ulb_loss_ratio,
        logger=logger,
    )

    # Apply the configured BatchNorm momentum. cfg.bn_momentum (= 1 - ema_m) is
    # the intended running-statistics momentum, but nothing applied it to the
    # model, so BN silently ran at PyTorch/timm's 0.1 default — ~10x too fast for
    # batch_size 16, which destabilises the running stats and degrades the EMA
    # eval model. Set it on every BatchNorm layer of both models (timm's fused
    # BatchNormAct2d subclasses _BatchNorm, so it is covered too).
    for submodel in (model.train_model, model.eval_model):
        for module in submodel.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = cfg.bn_momentum

    optimizer = get_optimizer(model.train_model, cfg.opt, cfg.lr, cfg.momentum, cfg.weight_decay)
    model.set_optimizer(optimizer)

    if torch.cuda.is_available():
        cfg.gpu = 0
        torch.cuda.set_device(cfg.gpu)
        model.train_model = model.train_model.cuda(cfg.gpu)
        model.eval_model = model.eval_model.cuda(cfg.gpu)

    # Set data loaders
    model.set_data_loader(cfg, labeled_dataset, unlabeled_dataset, test_dataset)

    # ── Train ────────────────────────────────────────────────────────
    logger.info("Training for {} iterations...", cfg.num_train_iter)
    _write_progress(
        args.progress_file,
        {"status": "training", "iteration": 0, "total": cfg.num_train_iter},
    )

    def progress_callback(iteration: int, total_iterations: int) -> None:
        _write_progress(
            args.progress_file,
            {
                "status": "training",
                "iteration": iteration,
                "total": total_iterations,
            },
        )

    model.train(cfg, progress_callback=progress_callback)
    logger.info("Training complete.")

    # ── Save model ───────────────────────────────────────────────────
    _write_progress(args.progress_file, {"status": "saving", "message": "Saving model..."})

    # Ensure fitsbolt config is set for prediction consistency
    cfg = get_fitsbolt_config(cfg)

    # Save into the iteration subdirectory prepared by BackendInterface
    # (e.g. .../session/iteration_0/).  Falls back to output_dir if
    # model_path wasn't set (e.g. in tests).
    if not cfg.model_path:
        cfg.model_path = os.path.join(cfg.output_dir, "model.safetensors")
    iter_dir = os.path.dirname(cfg.model_path)
    os.makedirs(iter_dir, exist_ok=True)

    session_io = SessionIOHandler()
    model_path = session_io.save_model(model, cfg)
    logger.info("Model saved to: {}", model_path)

    # Save labels CSV next to the model checkpoint
    labels_df = pd.read_csv(args.labels_csv)
    labels_out = os.path.join(iter_dir, "labelled_data.csv")
    labels_df.to_csv(labels_out, index=False)
    logger.info("Labels saved to: {}", labels_out)

    elapsed = time.time() - start_time
    _write_progress(
        args.progress_file,
        {
            "status": "done",
            "model_path": str(model_path),
            "elapsed": elapsed,
        },
    )
    logger.success("Training subprocess completed in {:.1f}s", elapsed)


if __name__ == "__main__":
    # Skip session-level log: training adds its own handler to the
    # iteration subdirectory (e.g. iteration_0/training.log) in main().
    setup_prediction_logging("training", session_log=False)
    main()
