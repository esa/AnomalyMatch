#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Default configuration factory for AnomalyMatch."""

from __future__ import annotations

import os

import numpy as np
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from .create_model_string import create_model_string


def get_default_cfg() -> DotMap:
    """Returns the default configuration.

    Returns:
        DotMap: the default configuration
    """
    cfg = DotMap(_dynamic=False)

    # General settings
    cfg.name = "MyRun"
    cfg.log_level = "INFO"

    # Resolve paths relative to the repo/package root so defaults work
    # regardless of the notebook's working directory (e.g. on datalabs).
    _pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _test_data = os.path.join(_pkg_root, "tests", "test_data", "grayscale")
    cfg.save_dir = "anomaly_match_results/sessions/"
    cfg.data_dir = _test_data
    cfg.output_dir = "anomaly_match_results/sessions/"
    cfg.label_file = os.path.join(_test_data, "labeled_data.csv")
    cfg.metadata_file = None  # Path to the metadata CSV file
    cfg.training_data_source = None  # None = auto-detect; or "image_folder", "zarr", "cutana"
    cfg.labeled_cache_path = None  # Path to LabeledDataCache for container sources
    cfg.prediction_search_dir = None
    # Directory holding the live predictions.db during scoring. None keeps it in
    # output_dir (default). Set to a local-disk path to escape NFS WAL bloat when
    # output_dir is on a network filesystem; the DB is snapshotted back to
    # output_dir on completion. See anomaly_match.prediction.db_location.
    cfg.prediction_db_dir = None
    cfg.save_path = os.path.join(cfg.save_dir)
    cfg.save_file = create_model_string(cfg) + ".safetensors"
    cfg.model_path = None  # Will be set by SessionIOHandler when session is active
    cfg.N_batch_prediction = None  # User specified batch size for evaluating a directory, if None: determined automatically
    # Sources packed into each intermediate buffer file handed to a prediction subprocess.
    # Each chunk spawns a fresh subprocess (and, for Cutana sources, a StreamingOrchestrator with
    # pool init + worker spawn), so smaller chunks pay that startup cost repeatedly. Larger chunks
    # amortise the per-chunk respawn overhead at the price of higher peak memory per buffer.
    cfg.subprocess_buffer_size = 500_000

    cfg.seed = 42
    cfg.test_ratio = 0.0

    # DataLoader settings
    cfg.N_to_load = 1000
    cfg.pin_memory = True
    cfg.oversample = True

    cfg.gpu = 0
    cfg.num_workers = 4
    cfg.fitsbolt_cfg = None  # Set at runtime from model checkpoint or normalisation settings
    # normalisation settings for fitsbolt settings
    cfg.normalisation = DotMap(_dynamic=False)
    cfg.normalisation.output_dtype = np.uint8  # output dtype of the images
    cfg.normalisation.image_size = [150, 150]  # default resolution (width, height)
    cfg.normalisation.n_output_channels = 3  # number of output channels (e.g. 3 for RGB)
    cfg.num_channels = cfg.normalisation.n_output_channels  # set from dataset at runtime

    # FITS file handling settings
    # fits_extension: Extension(s) to use when loading FITS files
    # (can be int, string, or list of int/string, or list of lists of int/string)
    cfg.normalisation.fits_extension = None

    # channel_combination: (np.array) combine FITS extensions into n_output (3 = RGB) channels, shape n_out x n_input = len
    # cfg.normalisation.fits_extension, or None if only one extension is used or n_out=n_input
    cfg.normalisation.channel_combination = None

    # further interpolation and normalisation settings
    cfg.normalisation.interpolation_order = (
        1  # order of interpolation for resizing with skimage, 0-5
    )
    cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    # settings for normalisation:
    cfg.normalisation.norm_maximum_value = None  # None or float
    cfg.normalisation.norm_minimum_value = None  # None or float
    cfg.normalisation.norm_crop_for_maximum_value = None  # None or integer tuple (height, width)
    # Bool, if False assumes min value to be 0 or cfg.normalisation.norm_minimum_value if not None
    cfg.normalisation.norm_log_calculate_minimum_value = False
    # only used if cfg.normalisation.normalisation_method == NormalisationMethod.ASINH: asinh_scale list of n_output_channel -
    # floats > 0, defining the scale for each channel (lower = higher stretch):
    cfg.normalisation.norm_asinh_scale = [
        0.7,
        0.7,
        0.7,
    ]
    # norm_asinh_clip: asinh_clip list of n_output_channel floats in ]0.,100.], defining the clip for each channel:
    cfg.normalisation.norm_asinh_clip = [
        99.8,
        99.8,
        99.8,
    ]
    # norm_asinh_n_samples: pixels per channel subsampled when estimating the asinh percentile
    # bounds (fitsbolt). AnomalyMatch sets this aggressively low — the scores are robust to a
    # small bright-tail bias, and the streaming cutout production is the throughput bottleneck.
    cfg.normalisation.norm_asinh_n_samples = 2000
    # end of fitsbolt settings

    # Cutana cutout padding factor: multiplies source diameter to control
    # how much sky context is included (1.0 = match diameter, 2.0 = 2x).
    cfg.normalisation.cutout_padding_factor = 1.0

    # Flux conversion (Euclid): convert pixel values to flux density in Jansky
    # using the AB zeropoint (MAGZERO) from FITS headers.  Defaulted on because
    # Cutana currently targets Euclid data, where per-tile MAGZERO drift plus
    # non-scale-invariant normalisation (asinh/log) means inference without
    # conversion gives tile-dependent scores.  The UI widget is hidden while
    # this is the only supported mission — see issue to re-expose when
    # Cutana gains non-Euclid support.
    cfg.normalisation.apply_flux_conversion = True
    cfg.normalisation.flux_conversion_zeropoint_keyword = "MAGZERO"
    cfg.normalisation.cutout_padding_factor = 1.0

    # FixMatch settings
    cfg.ema_m = 0.99
    cfg.hard_label = True
    cfg.temperature = 0.5
    cfg.ulb_loss_ratio = 1.0
    cfg.p_cutoff = 0.95
    cfg.uratio = 5

    # Training settings
    cfg.batch_size = 16
    cfg.lr = 0.0075
    cfg.weight_decay = 7.5e-4
    cfg.opt = "SGD"
    cfg.momentum = 0.9
    cfg.bn_momentum = 1.0 - cfg.ema_m
    cfg.num_train_iter = 200
    cfg.eval_batch_size = 500
    cfg.num_eval_iter = -1  # -1 means no evaluation
    cfg.top_N = 5000  # amount of top files that are actively tracked

    # Training subprocess — unlabeled pool size caps
    cfg.unlabeled_pool_cap = 20_000  # max unlabeled images for low-res (≤ 200px)
    cfg.unlabeled_pool_cap_hires = 10_000  # max unlabeled images for high-res (> 200px)
    cfg.unlabeled_pool_hires_threshold = 200  # image size threshold in pixels
    cfg.cutana_streaming_batch_size = 1000  # max cutouts per Cutana streaming batch
    # Max FITS tile sets to sample unlabeled data from. Size stratification needs
    # enough tiles to supply the rare large sources: the 100–200px bins hold only
    # ~70–90 sources per tile, so 16 tiles give ~1000+/bin to fill a flat pool up
    # to 200px without depleting the large end (Q1 catalogue analysis).
    cfg.cutana_max_unlabeled_tiles = 16
    # When True, sample unlabeled Cutana cutouts with an equal per-bin quota over
    # log-spaced size bins so the pool's source-diameter distribution is ~uniform
    # instead of dominated by the small-source bulk (Euclid Q1/DR1 sources skew
    # heavily small). Off preserves the random tile sampler. Only meaningful for
    # the Cutana source; ignored otherwise.
    cfg.cutana_stratify_source_size = False
    # Binning for size stratification. Euclid Q1 diameters are heavy-tailed (median
    # ~12px, p99 ~66px, max ~2400px). ``max_px`` sets where the pool is flattened to
    # ~uniform in log size; above it the sampler keeps every large source and the
    # distribution tapers along the real tail (log bins continue past max_px at the
    # same width, so the giants get their own bins rather than one lump). ``bins``
    # sets the flattened region's resolution — more bins = finer flattening, but the
    # sparse large bins deplete past ~40 bins at 16 tiles; ~30 balances the two, and
    # bins are coupled to tile count (more bins need more tiles). Q1 analysis: 30
    # bins / 16 tiles / 200px gives a flat plateau to ~200px then a tapering tail.
    cfg.cutana_size_stratify_bins = 30  # log-spaced bins spanning [min, max_px]
    cfg.cutana_size_stratify_max_px = 200  # top of the flattened (uniform) range, in pixels
    cfg.cutana_min_workers = 4  # min parallel workers for Cutana StreamingOrchestrator
    # Cutana cutout production on scattered FITS tiles is I/O-bound (workers sit in NFS
    # read-wait, not on CPU), so this is intentionally *oversubscribed* past the pod's core
    # count: each worker is a separate NFS stream and a single stream is round-trip-limited
    # (~1 Gbit/s at the repository's rsize=64KB), so more concurrent streams aggregate more
    # bandwidth up to the link ceiling. Keep production ahead of the GPU so inference never
    # starves. Tune against available pod memory (each worker holds tile data while extracting)
    # and the measured link ceiling; raising it past the point where the NFS link saturates
    # yields nothing. Avoid capping it purely for UI responsiveness — throttle the per-poll DB
    # aggregation instead.
    cfg.cutana_max_workers = 16  # max parallel workers for Cutana StreamingOrchestrator

    # Backbone settings
    cfg.pretrained = True
    cfg.net = "efficientnet-lite0"
    # torch.compile the eval model for prediction. Off by default: eager
    # bf16 + TF32 + channels_last already saturates efficientnet-lite0 (compile's
    # Triton convs lose to cuDNN at this size and add per-subprocess build
    # latency). Enable for larger backbones where Inductor wins — it needs CUDA
    # dev headers (cuda.h) in the env, and compiled kernels are cached on disk so
    # the build is paid once and reused across subprocess respawns.
    cfg.compile_model = False

    # Use lightweight defaults when data_dir points to the bundled test data
    # so that start_ui() works out of the box without long training times.
    if cfg.data_dir == _test_data:
        cfg.num_train_iter = 32
        cfg.normalisation.image_size = [64, 64]

    return cfg


def is_shipped_default_path(key: str, value: str | None) -> bool:
    """Return whether *value* is the path ``get_default_cfg()`` ships for *key*.

    The setup screens let an explicitly configured file outrank the user's
    remembered chooser folder, but the bundled defaults (e.g. the test-data
    ``labeled_data.csv``) always exist, so without this check they would win on
    every fresh kernel and replace what the user last picked.

    Args:
        key: Top-level config key, e.g. ``"label_file"``.
        value: The configured path to compare, or ``None``.

    Returns:
        ``True`` when *value* resolves to the same path as the shipped default.
    """
    default = get_default_cfg()[key]
    if not isinstance(value, str) or not isinstance(default, str):
        return False
    return os.path.abspath(value) == os.path.abspath(default)
