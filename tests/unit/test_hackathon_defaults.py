#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""The Euclid Hackathon App's defaults apply inside the App, and only there."""

import numpy as np
from fitsbolt import NormalisationMethod

from anomaly_match.utils.get_default_cfg import HACKATHON_DATA_DIR_ENV, get_default_cfg


def test_upstream_defaults_stay_outside_the_app(monkeypatch, tmp_path):
    monkeypatch.setenv(HACKATHON_DATA_DIR_ENV, str(tmp_path / "absent"))
    cfg = get_default_cfg()
    assert cfg.normalisation.channel_combination is None
    assert cfg.normalisation.image_size == [64, 64]


def test_the_hackathon_defaults_inside_the_app(monkeypatch, tmp_path):
    (tmp_path / "q1_search_catalogue").mkdir()
    monkeypatch.setenv(HACKATHON_DATA_DIR_ENV, str(tmp_path))
    cfg = get_default_cfg()
    assert cfg.data_dir == str(tmp_path / "q1_search_catalogue")
    assert cfg.prediction_search_dir == cfg.data_dir
    assert cfg.label_file == str(tmp_path / "labelled_data.csv")
    assert cfg.output_dir.startswith(str(tmp_path / "anomaly_match_results"))
    assert cfg.save_dir == cfg.output_dir == cfg.save_path
    assert cfg.num_train_iter == 200
    assert cfg.normalisation.image_size == [192, 192]
    assert cfg.normalisation.cutout_padding_factor == 3.0
    assert cfg.normalisation.normalisation_method == NormalisationMethod.ASINH
    combination = cfg.normalisation.channel_combination
    assert combination.shape == (3, len(cfg.normalisation.fits_extension))
    assert np.all(combination.sum(axis=1) <= 1)
