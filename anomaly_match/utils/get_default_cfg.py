#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import os

import numpy as np
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from .create_model_string import create_model_string


def get_default_cfg():
    """Returns the default configuration.

    Returns:
        DotMap: the default configuration
    """
    cfg = DotMap(_dynamic=False)

    # General settings
    cfg.name = "MyRun"
    cfg.log_level = "INFO"

    cfg.save_dir = "anomaly_match_results/sessions/"
    cfg.data_dir = "tests/test_data/"
    cfg.output_dir = "anomaly_match_results/sessions/"
    cfg.label_file = "tests/test_data/labeled_data.csv"
    cfg.metadata_file = None  # Path to the metadata CSV file
    cfg.prediction_search_dir = None
    cfg.save_path = os.path.join(cfg.save_dir)
    cfg.save_file = create_model_string(cfg) + ".safetensors"
    cfg.model_path = None  # Will be set by SessionIOHandler when session is active
    cfg.N_batch_prediction = None  # User specified batch size for evaluating a directory, if None: determined automatically
    cfg.subprocess_buffer_size = (
        100_000  # Number of sources packed into intermediate files for subprocesses
    )

    cfg.seed = 42
    cfg.test_ratio = 0.0

    # DataLoader settings
    cfg.N_to_load = 1000
    cfg.pin_memory = True
    cfg.oversample = True

    cfg.num_workers = 4
    # normalisation settings for fitsbolt settings
    cfg.normalisation = DotMap()
    cfg.normalisation.output_dtype = np.uint8  # output dtype of the images
    # NOTE: image_size has no default - user must explicitly set it
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
    # end of fitsbolt settings

    # Flux conversion (Euclid): convert pixel values to flux density in Jansky
    # using the AB zeropoint (MAGZERO) from FITS headers.  When True, must be
    # applied in both training (load_and_process_wrapper) and prediction (cutana) paths.
    cfg.normalisation.apply_flux_conversion = False
    cfg.normalisation.flux_conversion_zeropoint_keyword = "MAGZERO"

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

    # Backbone settings
    cfg.pretrained = True
    cfg.net = "efficientnet-lite0"

    return cfg
