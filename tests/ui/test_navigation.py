#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from unittest.mock import MagicMock, patch

import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match_ui.app import AnomalyMatchApp

pytestmark = pytest.mark.ui


@pytest.fixture()
def mock_session():
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "/fake/data"
    session.cfg.net = "test-cnn"
    session.cfg.N_to_load = 1000
    session.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    session.cfg.normalisation.image_size = [64, 64]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.num_channels = 3
    session.cfg.test_ratio = 0.5
    session.cfg.top_N = 10
    session.cfg.num_eval_iter = 10
    session.cfg.metadata_file = None
    session.cfg.model_path = None
    session.cfg.prediction_search_dir = None
    session.cfg.output_dir = "/fake/output"
    return session


@patch("IPython.display.display")
def test_unknown_screen_raises(mock_display, mock_session):
    app = AnomalyMatchApp(mock_session)
    with pytest.raises(ValueError, match="Unknown screen"):
        app.navigate_to("nonexistent")
