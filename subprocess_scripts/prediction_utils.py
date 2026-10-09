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

import contextlib
import os
import pickle
import signal
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from dotmap import DotMap
from loguru import logger

from anomaly_match.data_io.checkpoint_io import (
    load_checkpoint,
    sync_normalisation_from_checkpoint,
)
from anomaly_match.data_io.container_loaders import decode_zarr_image
from anomaly_match.data_io.load_images import (
    _drop_unused_combination_columns,
    _get_channel_combination_array,
)
from anomaly_match.data_io.source_scanning import read_label_map
from anomaly_match.prediction.anomaly_score_db import build_compat_metadata
from anomaly_match.utils.get_net_builder import get_net_builder

# Memory model coefficients for batch size estimation
# Derived from empirical GPU profiling on NVIDIA L40S (R² > 0.999)
# Formula: reserved_mb = a * BS * S² + b * BS * nch + d
#   a: spatial activation memory per pixel (dominant, channel-independent)
#   b: input tensor memory per channel per sample
#   d: constant model parameter overhead
MEMORY_COEFFICIENTS = {
    "efficientnet-lite0": {"a": 0.000378, "b": 0.1753, "d": 112.6},
    "efficientnet-b1": {"a": 0.000496, "b": 0.1753, "d": 150.0},
    "efficientnet-b2": {"a": 0.000496, "b": 0.1753, "d": 168.0},
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
    - Intermediate activations (scales with batch_size × image_size², channel-independent)
    - Input tensor memory (scales with batch_size × num_channels)
    - Model parameters (constant overhead)
    - CUDA memory allocator overhead

    The formula used is:
        reserved_mb = a × BS × S² + b × BS × nch + d

    where S² = image_width × image_height and nch = num_channels.

    Args:
        cfg: Configuration with ``net``, ``num_channels``, and
            ``normalisation.image_size``.
        available_vram: Available GPU VRAM in MB. If None, auto-detects
            from the current CUDA device.
        safety_margin: Fraction of VRAM to keep free (default: 0.3 = 30%).
            Higher values are safer but reduce batch size.

    Returns:
        int: Recommended batch size (minimum 1).

    Notes:
        - Coefficients were calibrated on NVIDIA L40S (45 GB) with
          EfficientNet architectures; the safety margin accounts for
          driver overhead, memory fragmentation, and other GPU consumers.
    """

    # Auto-detect available VRAM if not provided
    if available_vram is None:
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
            available_vram = device_props.total_memory / 1024**2  # Convert to MB
            logger.debug("Auto-detected GPU VRAM: {:.0f} MB", available_vram)
        else:
            # Default to 4GB if no GPU detected (conservative estimate)
            available_vram = 4096
            logger.warning("No CUDA device detected, using default 4GB VRAM estimate")

    # Get coefficients for the specified model
    coef = MEMORY_COEFFICIENTS.get(cfg.net, MEMORY_COEFFICIENTS["efficientnet-lite0"])

    if cfg.net not in MEMORY_COEFFICIENTS:
        logger.warning(
            "Unknown model '{}', using efficientnet-lite0 coefficients. Supported models: {}",
            cfg.net,
            list(MEMORY_COEFFICIENTS.keys()),
        )

    # Calculate usable VRAM after safety margin
    usable_vram = available_vram * (1 - safety_margin)

    # Solve for batch_size:
    # usable_vram = a * B * S² + b * B * nch + d
    # usable_vram - d = B * (a * S² + b * nch)
    # B = (usable_vram - d) / (a * S² + b * nch)
    S2 = cfg.normalisation.image_size[0] * cfg.normalisation.image_size[1]
    denominator = coef["a"] * S2 + coef["b"] * cfg.num_channels

    if denominator <= 0:
        logger.warning("Invalid memory model parameters, returning minimum batch size")
        return 1

    batch_size = (usable_vram - coef["d"]) / denominator

    # Ensure batch size is at least 1
    batch_size = max(1, int(batch_size))

    logger.debug(
        "Calculated batch size: {} "
        "(image_size={}, available_vram={:.0f}MB, "
        "safety_margin={}, model={})",
        batch_size,
        cfg.normalisation.image_size[0],
        available_vram,
        safety_margin,
        cfg.net,
    )

    return batch_size


def configure_torch_backends() -> None:
    """Enable inference-time GPU fast paths (TF32 + cuDNN autotuning).

    Process-wide switches, no-ops on CPU. ``set_float32_matmul_precision("high")``
    selects TF32 for fp32 matmuls (and cuDNN convolution TF32 is on by default),
    trading a few mantissa bits of the fp32 accumulation for Ada/Ampere
    tensor-core throughput.  Scores are persisted fp32 in
    :class:`AnomalyScoreDB`, so the store no longer masks that loss: it is
    reproducible across reruns of the same model and input, but it perturbs
    each sample independently at the ~1e-3 level, so genuinely close sources
    can still swap rank.  Turning autocast off is the remaining lever if that
    matters.  ``cudnn.benchmark`` lets cuDNN pick the fastest convolution
    algorithm for our fixed cutout resolution after a one-batch warmup (the
    cutout size is constant across a run).
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def optimize_model_for_inference(model, cfg):
    """Apply memory-format and (optionally) graph-compilation speedups.

    Always converts to ``channels_last`` (EfficientNet is depthwise-convolution
    heavy and benefits strongly on tensor-core GPUs). When ``cfg.compile_model``
    is set it additionally wraps the module with :func:`torch.compile`.

    Compilation is lazy — the graph is only built on the first forward — so we run
    a tiny probe forward here to force it and *validate* the inductor/triton
    toolchain. If compilation can't run (e.g. the CUDA dev headers are missing so
    the generated kernels won't build), we fall back to the eager
    ``channels_last`` model: compile is an optimisation, never a correctness
    requirement, and must not crash a multi-hour scoring run mid-flight.

    Inductor caches compiled kernels on disk; we point ``TORCHINDUCTOR_CACHE_DIR``
    at a persistent location so the build cost is paid once and reused across the
    per-chunk subprocess respawns (and across sessions), rather than recompiled on
    every spawn.

    Args:
        model: The loaded eval model (already on the target device).
        cfg: Config providing ``compile_model``, ``num_channels`` and
            ``normalisation.image_size`` for the probe forward.

    Returns:
        The optimised model (a validated ``torch.compile`` wrapper when enabled
        and the toolchain works, else the eager ``channels_last`` model).
    """
    if not torch.cuda.is_available():
        return model
    model = model.to(memory_format=torch.channels_last)
    if not cfg.compile_model:
        return model

    # Persistent kernel cache so repeated subprocess spawns reuse the build.
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.expanduser("~/.cache/anomaly_match/torchinductor"),
    )
    compiled = torch.compile(model)
    try:
        device = model_inference_device(model)
        width, height = cfg.normalisation.image_size
        probe = torch.zeros((2, cfg.num_channels, height, width), device=device).to(
            memory_format=torch.channels_last
        )
        process_batch_predictions(compiled, probe, return_images=False)
        # CUDA kernels (and the lazy inductor/triton build the probe triggers) run
        # asynchronously, so a build failure would otherwise surface later, off this
        # try/except. Block here to force the compile to finish and make any failure
        # land in the except below — not mid-scoring.
        # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
        torch.cuda.synchronize()
        return compiled
    except Exception as exc:
        logger.warning("torch.compile unusable — running eager: {}", exc)
        return model


def warmup_inference(model, cfg, batch_size: int, device) -> None:
    """Run one throwaway forward at the real batch shape to prime per-shape caches.

    Both fast paths specialise on the *input shape*: ``cudnn.benchmark`` benchmarks
    convolution algorithms the first time it sees a given (shape, dtype) and caches
    the winner, and a ``torch.compile`` graph recompiles for a new batch size. This
    is distinct from the probe in :func:`optimize_model_for_inference`, which runs
    only when ``cfg.compile_model`` is set and only to *validate* the compile
    toolchain at a fixed size-2 shape. Without this warmup the first *real* batch
    would still pay the cuDNN autotune (and, with compile on, a recompile) as a
    one-off latency spike — so we move that cost off the first scored batch (and
    out of the profiler's ``inference`` stage) by forwarding once at the production
    ``batch_size`` here. Note the cuDNN benefit applies even on the default
    eager path, where the compile probe never runs. Best-effort: a warmup failure
    must not abort scoring.

    Args:
        model: The optimised eval model.
        cfg: Config providing ``num_channels`` and ``normalisation.image_size``.
        batch_size: The per-forward image count used at steady state.
        device: Target CUDA device.
    """
    if not torch.cuda.is_available():
        return
    try:
        width, height = cfg.normalisation.image_size
        dummy = torch.zeros((batch_size, cfg.num_channels, height, width), device=device).to(
            memory_format=torch.channels_last
        )
        process_batch_predictions(model, dummy, return_images=False)
        torch.cuda.synchronize()
    except Exception as exc:
        logger.warning("Inference warmup failed — continuing: {}", exc)


def pipelined_batches(n_batches, prepare, load, *, should_stop=lambda: False, lookahead=1):
    """Yield loaded batches up to ``lookahead`` indices ahead so I/O overlaps consumption.

    A single background worker runs ``load`` for the upcoming batches while the
    caller consumes the current one (typically GPU inference), hiding
    streaming/decode I/O behind compute. Shared by every prediction process so
    the prefetch loop isn't re-implemented per loader.

    ``lookahead`` keeps that many batches in flight on the worker. The default
    (``1``) prefetches a single batch ahead — the worker idles between finishing
    batch *N+1* and the consumer requesting *N+2*, keeping peak residency at two
    batches (current + one in flight). Callers with **bursty** load latency pass
    ``lookahead=2`` so the worker keeps producing continuously and a slow batch is
    absorbed by the buffer instead of stalling the GPU; the Cutana path does this
    because its NFS tile reads spike. Raising it costs one extra resident
    decoded batch per step, so leave it at 1 unless a path is shown to benefit.

    The ``prepare``/``load`` split keeps thread-affinity correct:

    * ``prepare(index)`` runs on the **caller's** thread. Put anything that must
      stay single-threaded here — notably the resume DB filter, since the sqlite
      connection can't be shared across threads. Return an opaque "plan" for
      :func:`load`, or ``None`` to skip the batch (e.g. all already scored).
    * ``load(plan)`` runs on the **background worker**. Do the pure I/O +
      preprocessing here (no DB, no shared mutable state).

    Exactly one worker thread is used, so loaders that require a single reader
    (the Cutana orchestrator) stay correct. A ``load`` exception is re-raised
    from the ``yield`` (carried by the future), so it reaches the process-level
    handler — and the log — like any other failure.

    Args:
        n_batches: Total number of batch indices to iterate.
        prepare: ``index -> plan | None``, run on the caller thread.
        load: ``plan -> payload``, run on the background worker thread.
        should_stop: Optional predicate; when it returns ``True`` no further
            batches are prepared (cooperative shutdown).
        lookahead: Number of batches to keep in flight on the worker (>= 1).

    Yields:
        ``(index, payload)`` for each non-skipped batch, in index order.
    """
    if lookahead < 1:
        raise ValueError(f"lookahead must be >= 1, got {lookahead}")

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as executor:
        # Futures for batches currently loading on the single worker, oldest first.
        # Only real loads live here — skipped batches (prepare -> None) never enter,
        # so the consumer loop below never has to special-case them.
        inflight = deque()
        next_index = 0

        def fill():
            """Submit batches until ``lookahead`` are loading (caller thread; holds the DB)."""
            nonlocal next_index
            while len(inflight) < lookahead and next_index < n_batches and not should_stop():
                index = next_index
                next_index += 1
                plan = prepare(index)
                if plan is not None:  # None => batch fully skipped (e.g. all already scored)
                    inflight.append((index, executor.submit(load, plan)))

        fill()
        while inflight:
            index, future = inflight.popleft()
            fill()  # refill the worker before blocking on this result
            yield index, future.result()


def model_inference_device(model):
    """Return the device of the model's first parameter, or a sensible default.

    Test mock models can expose no parameters, so fall back to CUDA when
    available (else CPU) instead of letting ``next()`` raise ``StopIteration``.

    Args:
        model: The loaded eval model.

    Returns:
        torch.device: Device to place inference-time tensors on.
    """
    param = next(model.parameters(), None)
    if param is not None:
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        ValueError: If the checkpoint has no embedded fitsbolt normalisation config;
            prediction must not fall back to a default linear stretch.
    """
    logger.info("Loading model with following configuration:")
    logger.info("  Model path: {}", cfg.model_path)
    model_path = cfg.model_path
    logger.info("Attempting to load model from: {}", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load checkpoint first to read architecture metadata
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_torch_backends()
    checkpoint = load_checkpoint(model_path, device=device)

    # Use architecture from checkpoint if available (overrides config default)
    if checkpoint.get("net") is not None:
        cfg.net = checkpoint["net"]
        logger.info("  Architecture from checkpoint: {}", cfg.net)

    # Use channel count from checkpoint — training sets this from the dataset,
    # but the pickled prediction config may still carry the default (3).
    cfg.num_channels = checkpoint["num_channels"]
    logger.info("  Channels from checkpoint: {}", cfg.num_channels)

    net_builder = get_net_builder(
        cfg.net,
        pretrained=cfg.pretrained,
        in_channels=cfg.num_channels,
    )
    model = net_builder(num_classes=2, in_channels=cfg.num_channels)

    if torch.cuda.is_available():
        gpu_device = cfg.gpu
        torch.cuda.set_device(gpu_device)
        model = model.cuda()
        logger.info("Using GPU device {}", gpu_device)
    else:
        logger.info("Using CPU for inference")

    if "eval_model" not in checkpoint:
        raise KeyError(
            f"Model checkpoint does not contain 'eval_model' key. Keys found: {checkpoint.keys()}"
        )

    model.load_state_dict(checkpoint["eval_model"])
    model = optimize_model_for_inference(model, cfg)

    # Make the checkpoint the single source of truth for normalisation: this
    # overwrites cfg.fitsbolt_cfg (inference decode), cfg.normalisation
    # (Cutana orchestrator resolution) and cfg.normalisation.channel_combination
    # (the GPU band-mixing matmul) so the model is fed images produced exactly as
    # in training — see sync_normalisation_from_checkpoint.
    # load_checkpoint always populates both keys, so access them directly
    # (fail hard if a loader ever stops setting them).
    #
    # A checkpoint without an embedded fitsbolt config (sync returns False) is
    # rejected outright: there is no safe fallback.  Reusing whatever
    # cfg.fitsbolt_cfg happens to be present (e.g. the UI default, which is a
    # linear CONVERSION_ONLY stretch) would silently feed the model images
    # normalised differently from its training data — which defeats this PR's
    # whole point of making the checkpoint the single source of truth.  Such
    # legacy checkpoints must be retrained.
    if not sync_normalisation_from_checkpoint(
        cfg, checkpoint["fitsbolt_cfg"], checkpoint["channel_combination"]
    ):
        raise ValueError(
            "Model checkpoint does not contain an embedded fitsbolt normalisation "
            "config. Prediction cannot proceed without it — falling back to the "
            "default linear (CONVERSION_ONLY) stretch would feed the model images "
            "it was never trained on. Retrain the model with the current version so "
            "the checkpoint embeds its normalisation settings."
        )
    logger.info(
        "Synced normalisation from model checkpoint: image_size={}, method={}, channels={}",
        cfg.normalisation.image_size,
        cfg.normalisation.normalisation_method,
        cfg.normalisation.n_output_channels,
    )

    logger.success("Successfully loaded model from {}", model_path)
    return model


def setup_prediction_logging(log_name, *, session_log=True):
    """Set up logging for prediction scripts.

    Configures file logging with rotation and stderr output. Also adds
    session-specific logging if a config path is available in sys.argv.

    Args:
        log_name: Name used for the log file (e.g. "prediction_thread",
            "prediction_zarr", "prediction_cutana", "training").
        session_log: If True (default), add a log handler next to the
            checkpoint being scored (``dirname(cfg.model_path)``).  Set to
            False when the caller adds its own per-iteration handler (e.g.
            training writes ``iteration_N/training.log`` itself).
    """
    # Peek at the pickled config so rotated subprocess logs land inside the
    # session tree instead of polluting subprocess_scripts/logs/.
    output_dir = None
    iter_dir = None
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], "rb") as _f:
                _pre_cfg = DotMap(pickle.load(_f))
            if _pre_cfg.output_dir:
                output_dir = _pre_cfg.output_dir
            if _pre_cfg.model_path:
                iter_dir = os.path.dirname(_pre_cfg.model_path)
        except Exception as exc:
            print(f"setup_prediction_logging: could not read config: {exc}", file=sys.stderr)

    if output_dir:
        rotated_dir = os.path.join(output_dir, "subprocess_logs")
    else:
        rotated_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "logs")
    os.makedirs(rotated_dir, exist_ok=True)

    logger.remove()
    logger.add(
        os.path.join(rotated_dir, f"{log_name}_{{time}}.log"),
        rotation="1 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )
    logger.add(sys.stderr, level="INFO")

    if session_log and iter_dir:
        try:
            os.makedirs(iter_dir, exist_ok=True)
            logger.add(
                os.path.join(iter_dir, f"{log_name}.log"),
                rotation="10 MB",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                level="DEBUG",
            )
        except Exception as exc:
            logger.warning("Failed to set up per-iteration prediction log: {}", exc)


def load_prediction_config(config_path):
    """Load prediction config from pickle file and compute batch size.

    Args:
        config_path: Path to the pickled config file.

    Returns:
        tuple: (cfg, batch_size) where cfg is a DotMap config object
            and batch_size is the computed or configured batch size.
    """
    logger.info("Loading config from {}", config_path)
    try:
        with open(config_path, "rb") as f:
            cfg = pickle.load(f)
            cfg = DotMap(cfg)
    except Exception as e:
        logger.error("Failed to load config from {}: {}", config_path, e)
        sys.exit(1)

    logger.info("Setting batch size")
    batch_size = (
        estimate_batch_size(cfg) if cfg.N_batch_prediction is None else cfg.N_batch_prediction
    )
    logger.info("Batch size set to: {}", batch_size)

    # Log key configuration parameters
    logger.debug("Configuration loaded with parameters:")
    logger.debug("  Save file: {}", cfg.save_file)
    logger.debug("  Save path: {}", cfg.save_path)
    logger.debug("  Model path: {}", cfg.model_path)
    logger.debug("  Output directory: {}", cfg.output_dir)
    logger.debug("  Image size: {}", cfg.normalisation.image_size)

    # Log full configuration
    logger.debug("Full configuration:")
    logger.debug("{}", cfg.toDict())

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)

    return cfg, batch_size


def read_and_preprocess_image_from_zarr(image_data, cfg):
    """Read and preprocess raw image data from a Zarr array.

    Delegates to the centralised implementation in container_loaders.
    """
    return decode_zarr_image(image_data, cfg)


def load_and_preprocess_zarr(args):
    """Load and preprocess a single image from Zarr.

    Note: Returns numpy array, not tensor. Tensor conversion is done on main
    thread to avoid CUDA context issues in ThreadPoolExecutor.
    """
    image_data, cfg = args
    return read_and_preprocess_image_from_zarr(image_data, cfg)


def cutana_batch_to_model_tensor_gpu(batch_data, cfg, device=None) -> torch.Tensor:
    """Build the model-input tensor from a Cutana batch entirely on the GPU.

    Uploads the per-band ``(N, H, W, C_in)`` cutout batch as-is (uint8 stays
    uint8 — a quarter the bytes of the float32 the CPU path would copy), then
    does the ``channel_combination`` matmul and the uint8→float ``/255`` scaling
    on-device.  This removes the per-batch CPU channel-combination, which
    profiling found to be ~89% of multi-band prediction wall time (issue #458).

    Numerically matches the CPU/fitsbolt combination path
    (``apply_channel_combination_to_cutana_batch`` + uint8→float scaling)
    bit-for-bit up to float rounding: for an integer ``output_dtype`` the
    combined values are **clipped** to ``[0, 255]`` and **truncated** toward zero
    (fitsbolt's ``np.clip(...).astype(uint8)``), preserving the exact quantised
    input the model was trained on; float output passes through unscaled.

    Args:
        batch_data: ``(N, H, W, C_in)`` Cutana batch (uint8 or float).
        cfg: Configuration with ``normalisation.channel_combination`` and
            ``normalisation.n_output_channels``.
        device: Target device; defaults to the current CUDA device (or CPU when
            CUDA is unavailable).

    Returns:
        ``(N, C_out, H, W)`` float32 tensor on *device* — in ``[0, 1]`` for a
        uint8 input.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_data = np.ascontiguousarray(batch_data)
    is_uint8 = batch_data.dtype == np.uint8

    # A plain host->device copy: torch.from_numpy shares the pageable numpy
    # buffer, so non_blocking would be a silent no-op (it only overlaps for
    # pinned memory) — the copy is synchronous regardless, so don't imply async.
    tensor = torch.from_numpy(batch_data).to(device).float()  # (N,H,W,Cin)

    channel_combination = _get_channel_combination_array(cfg)
    if (
        channel_combination is None
        and tensor.ndim == 4
        and tensor.shape[-1] == 1
        and cfg.normalisation.n_output_channels > 1
    ):
        # Mirror the single-band broadcast ("replicate VIS to RGB").  This is an
        # implicit reshape of the input, so log it at debug to make the silent
        # band expansion visible when diagnosing a channel-count surprise.
        logger.debug(
            "Broadcasting single-band cutout to {} output channels (no channel_combination set)",
            cfg.normalisation.n_output_channels,
        )
        channel_combination = np.ones((cfg.normalisation.n_output_channels, 1), dtype=np.float32)

    n_bands = int(tensor.shape[-1])
    # Streaming extraction skips bands whose channel_combination column is
    # all-zero, so the matrix can be wider than the bands Cutana produced;
    # realign before the shape guard below (a no-op when they already match).
    if channel_combination is not None and channel_combination.shape[1] != n_bands:
        channel_combination = _drop_unused_combination_columns(channel_combination, n_bands)
    if channel_combination is None and n_bands != cfg.normalisation.n_output_channels:
        # No matrix to reduce the bands and they don't already match the model's
        # input channels — the model would be fed the wrong channel count.  This
        # happens for a catalogue/Cutana model (``fits_extension=None``) whose
        # band-mixing matrix was never persisted: fitsbolt only embeds it in
        # fitsbolt_cfg when fits_extension is set, and the standalone checkpoint
        # field predates this model.  Fail clearly here instead of crashing deep
        # in the conv stem.
        raise ValueError(
            f"Cutana produced {n_bands} bands but the model expects "
            f"{cfg.normalisation.n_output_channels} input channels and no "
            "channel_combination is available to reduce them. The model checkpoint "
            "has no embedded band-mixing matrix — retrain the model so its "
            "channel_combination is saved in the checkpoint."
        )

    if channel_combination is not None:
        matrix = torch.as_tensor(
            np.asarray(channel_combination), dtype=torch.float32, device=device
        )  # (Cout, Cin)
        # The matrix must map the cutout's bands to exactly the model's input
        # channels: rows = n_output_channels, cols = the bands Cutana produced.
        # A wrong-shaped matrix (e.g. a 2-row matrix left over from stale UI
        # state while the model wants 3 channels) would silently feed the model
        # the wrong channel count via the einsum below — fail hard instead.
        if matrix.ndim != 2 or matrix.shape != (cfg.normalisation.n_output_channels, n_bands):
            raise ValueError(
                f"channel_combination has shape {tuple(matrix.shape)} but must be "
                f"(n_output_channels={cfg.normalisation.n_output_channels}, "
                f"n_bands={n_bands}). The model would be fed the wrong channel count. "
                "This indicates a corrupted/stale band-mixing matrix; retrain or "
                "re-select the model so its channel_combination matches its channels."
            )
        tensor = torch.einsum("nhwc,oc->nhwo", tensor, matrix)
        if is_uint8:
            # Match fitsbolt's integer dtype conversion: clip to range, then
            # truncate toward zero (astype, not round) — keeps the GPU result
            # identical to the quantised inputs the model trained on.
            tensor = tensor.clamp_(0.0, 255.0).trunc_()

    # channels_last keeps the NHWC physical layout the GPU upload already has
    # (no transpose copy) and is the fast path for the depthwise-heavy backbone
    # on tensor-core GPUs.
    tensor = tensor.permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last
    )  # (N, Cout, H, W)
    if is_uint8:
        tensor = tensor.div_(255.0)
    return tensor


def process_batch_predictions(model, images, original_images=None, return_images=True):
    """Process a batch of images through the model to get anomaly scores.

    Runs inference on a batch and extracts the anomaly probability. The forward
    pass uses bfloat16 autocast on CUDA (the model's tensor-core fast path; the
    softmax is taken in fp32 for a stable probability) under ``inference_mode``.
    ``bfloat16`` needs no loss scaling.  Since scores became fp32 on disk the
    forward pass, not the store, sets the precision floor — and bf16 carries
    fewer mantissa bits (8) than the fp16 store it replaced, so it is now the
    dominant source of near-ties.  The error is reproducible across reruns but
    perturbs each sample independently, so close sources can still swap rank;
    switching autocast off is the lever, at the cost of the tensor-core path.

    Note: Includes explicit CUDA tensor cleanup to prevent GPU memory fragmentation.

    Args:
        model (torch.nn.Module): The neural network model for anomaly detection.
        images (torch.Tensor): Preprocessed tensor images for model inference.
        original_images (np.ndarray, optional): Original uint8 images for saving.
            If None, the function will convert the input tensor back to uint8.
        return_images (bool): When False, skip building ``images_for_saving``
            entirely and return ``None`` for it. Callers that discard the images
            (e.g. the streaming Cutana path) avoid a full per-batch GPU→host copy
            and uint8 conversion that would otherwise dominate the inference stage.

    Returns:
        tuple: (batch_scores, images_for_saving)
            - batch_scores (np.ndarray): Anomaly probability scores (0-1 range).
            - images_for_saving (np.ndarray | None): Images in uint8 format ready
              for saving, ``original_images`` passthrough, or ``None`` when
              ``return_images`` is False.
    """
    if torch.cuda.is_available():
        images = images.cuda(non_blocking=True)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if images.is_cuda
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        logits = model(images)
        batch_scores = torch.nn.functional.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

    # Explicit cleanup of CUDA tensors to prevent memory fragmentation
    del logits

    # Skip the host copy when the caller discards the images (streaming path).
    if not return_images:
        del images
        return batch_scores, None

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


def gate_db_on_compatibility(db, cfg: DotMap) -> None:
    """Validate the prediction DB against the current run and store metadata.

    Shared entry point for all three prediction subprocesses: raise if
    the DB already holds metadata for an incompatible run, otherwise
    persist the cfg's metadata so subsequent resumes share the same
    handshake.  Both the build and the validate live inside the DB
    layer (see :func:`build_compat_metadata`), so a future metadata-key
    addition only touches that module.

    Args:
        db: Open ``AnomalyScoreDB`` instance.
        cfg: Prediction config.

    Raises:
        RuntimeError: If the existing DB metadata is incompatible with
            *cfg*.  The message names the offending field; callers print
            it as-is.
    """
    ok, msg = db.validate_compatibility(cfg)
    if not ok:
        raise RuntimeError(
            f"Predictions DB at {db.path} is incompatible with the current run: {msg}. "
            "Delete the DB file and re-run, or point the output folder somewhere else."
        )
    db.set_metadata_batch(build_compat_metadata(cfg))


def filter_unprocessed_batch(
    keys: list[str], db, exclude: set[str] | None = None
) -> tuple[list[bool], int, int]:
    """Compute a per-batch keep mask against the predictions DB and exclusions.

    Wraps the indexed ``db.get_unprocessed(keys)`` query so all three
    prediction subprocesses share one resume code path.  Cost scales with
    the batch size, not the cumulative DB size — this is the per-chunk
    fix for #435 (the old ``get_processed_filenames`` startup fetch took
    ~8 min/chunk once the DB grew past ~10 M rows).

    ``exclude`` lets callers drop keys that must never be scored regardless of
    DB state — in particular the already-labelled training sources, which would
    otherwise be re-scored and resurface at the top of the gallery once the
    model has trained on them (the active-learning reappearance behind this
    fix).  Keys in ``exclude`` are dropped using the same mask the caller
    already applies to its in-memory batch, so no separate plumbing is needed.

    Resume skips and label exclusions are returned as separate counts: a key
    already in the DB counts as a resume skip, while a key absent from the DB
    but present in ``exclude`` counts as a label exclusion.  This keeps the
    end-of-run resume summary honest — a fresh (non-resume) run that excludes
    labelled sources must not report them as "skipped on resume".

    Args:
        keys: Candidate keys for the current batch (filenames or source
            IDs, already stringified).
        db: Open ``AnomalyScoreDB`` instance.
        exclude: Optional set of keys (same key space as ``keys``) to drop in
            addition to already-scored entries.  ``None`` keeps the pure
            resume behaviour.

    Returns:
        Tuple ``(keep, n_resume_skipped, n_label_excluded)`` where ``keep[i]``
        is ``True`` iff ``keys[i]`` is *not* yet in the DB and not in
        ``exclude``.  ``n_resume_skipped`` counts keys already in the DB;
        ``n_label_excluded`` counts not-yet-scored keys dropped only because
        they are in ``exclude``.
    """
    unprocessed_set = set(db.get_unprocessed(keys))
    exclude = exclude or set()
    keep = []
    n_resume_skipped = 0
    n_label_excluded = 0
    for k in keys:
        if k not in unprocessed_set:
            n_resume_skipped += 1
            keep.append(False)
        elif k in exclude:
            n_label_excluded += 1
            keep.append(False)
        else:
            keep.append(True)
    return keep, n_resume_skipped, n_label_excluded


def basename_exclusions(keys: list[str], excluded_basenames: set[str]) -> set[str]:
    """Map labelled image basenames into a chunk's full-path key space.

    Folder prediction keys (and the DB / resume keys) are full paths, but the
    label CSV ``id`` for folder sources is a bare image basename.  This selects
    the chunk keys whose basename is labelled so they can be passed to
    :func:`filter_unprocessed_batch` as ``exclude`` in the same key space.

    Folder prediction lists a single flat directory (``os.listdir``), so
    basenames are unique within a chunk and the basename match is unambiguous.

    Args:
        keys: Full-path chunk keys (already stringified).
        excluded_basenames: Labelled image basenames from the label CSV.

    Returns:
        The subset of ``keys`` whose basename is in ``excluded_basenames``, or
        an empty set when there is nothing to exclude.
    """
    if not excluded_basenames:
        return set()
    return {k for k in keys if os.path.basename(k) in excluded_basenames}


def load_excluded_label_ids(cfg: DotMap) -> set[str]:
    """Return the set of already-labelled source ids to skip during scoring.

    Labelled training samples must not reappear in the score gallery: scoring
    them wastes compute and, once the model has trained on them, they surface at
    the top of the gallery as fresh anomalies the user already labelled.  This
    mirrors the training side, which already excludes labelled ids from the
    unlabeled pool (``TrainingDataSource.get_unlabeled_batch``).

    The ids are the label CSV's ``id`` column (Cutana ``SourceID`` for the
    streaming path, image basenames for folder sources, derived names for
    Zarr).  Each subprocess intersects these against its own key space before
    passing them to :func:`filter_unprocessed_batch`.

    Args:
        cfg: Prediction config carrying ``label_file``.

    Returns:
        Stringified label ids, or an empty set when no label file is configured
        — a pure prediction run legitimately has none to exclude.

    Raises:
        FileNotFoundError: If ``label_file`` is set but does not resolve to an
            existing file.  Silently returning an empty set there would
            reintroduce the very bug this guards against (a stale/typo'd path
            would let labelled sources score again with no error), so a
            configured-but-missing label file fails hard via ``read_label_map``.
    """
    label_file = cfg.label_file
    if not label_file:
        return set()
    excluded = set(read_label_map(label_file).keys())
    logger.info("Excluding {:,} already-labelled source(s) from scoring", len(excluded))
    return excluded


def log_resume_summary(scored: int, skipped: int, total: int, kind: str) -> None:
    """Emit the end-of-run resume log lines shared by every subprocess.

    Args:
        scored: Number of items the model actually scored this run.
        skipped: Number of items skipped because they were already in the DB.
        total: Total number of items the run started with.
        kind: Human-readable plural label (``"files"``, ``"zarr entries"``,
            ``"sources"``) — used in both log lines so users see source-
            specific phrasing while the implementation stays shared.
    """
    if skipped:
        logger.info("Resume: skipped {}/{} already-scored {}", skipped, total, kind)
    if scored == 0 and skipped == total and total > 0:
        logger.success("All {} already scored — nothing to do.", kind)


# Process-local Event that the SIGTERM/SIGINT handler sets when a
# graceful shutdown has been requested.  ``threading.Event`` is the
# canonical signal-to-loop primitive in Python: ``set()`` from the
# handler, ``is_set()`` from the batch loop.  It lives at module scope
# (not on a Session) because each prediction subprocess is its own
# process — the three subprocess scripts share this module's
# ``install_shutdown_handler`` / ``shutdown_requested`` pair, but each
# subprocess has its own private Event instance.  No cross-process
# coordination is intended or needed.
_shutdown_event = threading.Event()


def install_shutdown_handler() -> None:
    """Trap SIGTERM and SIGINT and flag a graceful shutdown.

    The handler does not abort the process; it sets
    :data:`_shutdown_event`, which :func:`shutdown_requested` reports.
    The subprocess batch loops check between batches and exit cleanly —
    scores from the in-flight batch commit first, the DB checkpoints,
    then the process exits.

    Safe to call more than once (re-installing the same handler is a
    no-op).
    """

    def _handler(signum: int, _frame: object) -> None:
        if not _shutdown_event.is_set():
            logger.warning(
                "Shutdown signal {} received — finishing current batch and exiting cleanly.",
                signum,
            )
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def shutdown_requested() -> bool:
    """Return ``True`` once a SIGTERM/SIGINT has been observed."""
    return _shutdown_event.is_set()
