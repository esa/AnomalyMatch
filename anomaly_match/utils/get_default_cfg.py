#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from dotmap import DotMap

import os

from .create_model_string import create_model_string
from anomaly_match.image_processing.NormalisationMethod import NormalisationMethod


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
    cfg.save_file = create_model_string(cfg) + ".pth"
    cfg.model_path = None  # Will be set by SessionIOHandler when session is active

    cfg.seed = 42
    cfg.test_ratio = 0.0

    # DataLoader settings
    cfg.N_to_load = 1000
    cfg.size = [224, 224]
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.oversample = True
    cfg.interpolation_order = 1  # order of interpolation for resizing with skimage, 0-5
    # Normalisation settings
    cfg.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    # Optional normalisation settings
    cfg.normalisation = DotMap()
    cfg.normalisation.maximum_value = None  # None or float
    cfg.normalisation.minimum_value = None  # None or float
    cfg.normalisation.crop_for_maximum_value = None  # None or integer tuple (height, width)
    # Bool, if False assumes min value to be 0 or cfg.normalisation.minimum_value if not None
    cfg.normalisation.log_calculate_minimum_value = False
    # only used if cfg.normalisation_method == NormalisationMethod.ASINH:
    # asinh_scale list of 3 floats > 0, defining the scale for each channel (lower = higher stretch):
    cfg.normalisation.asinh_scale = [0.7, 0.7, 0.7]
    # asinh_clip list of 3 floats in ]0.,100.], defining the clip for each channel:
    cfg.normalisation.asinh_clip = [99.8, 99.8, 99.8]

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

    # FITS file handling settings
    cfg.fits_extension = None  # Extension(s) to use when loading FITS files (can be int, string, or list of int/string)

    return cfg
