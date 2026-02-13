#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Shared fixtures for unit tests."""

import ipywidgets as widgets
import pytest

import anomaly_match as am


@pytest.fixture(scope="module")
def base_config():
    """Base configuration for unit tests (no training, no output dir)."""
    out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid white", height="400px", background_color="black", overflow="auto"
        ),
    )

    cfg = am.get_default_cfg()
    am.set_log_level("debug", cfg)
    cfg.data_dir = "tests/test_data/"
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.net = "test-cnn"
    cfg.pretrained = False
    cfg.num_train_iter = 2
    cfg.num_workers = 0
    cfg.test_ratio = 0.5
    cfg.output_dir = "tests/test_output"
    return cfg, out
