#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from dotmap import DotMap

import os

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

    cfg.save_dir = "anomaly_match_results/saved_models/"
    cfg.data_dir = "tests/test_data/"
    cfg.output_dir = "anomaly_match_results/out/"
    cfg.label_file = "tests/test_data/labeled_data.csv"
    cfg.search_dir = None
    cfg.save_path = os.path.join(cfg.save_dir)
    cfg.save_file = create_model_string(cfg) + ".pth"
    cfg.model_path = os.path.join(cfg.save_path, cfg.save_file)

    cfg.seed = 42
    cfg.test_ratio = 0.0

    # DataLoader settings
    cfg.N_to_load = 1000
    cfg.size = [224, 224]
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.oversample = True

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

    # Backbone settings
    cfg.pretrained = True
    cfg.net = "efficientnet-lite0"

    # Prediction settings
    cfg.prediction_file_type = "hdf5"  # can be 'zip', 'hdf5', or 'image'

    return cfg
