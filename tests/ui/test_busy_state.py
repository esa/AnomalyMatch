#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from unittest.mock import MagicMock, patch

import pytest

from anomaly_match_ui.app import AnomalyMatchApp
from anomaly_match_ui.screens.training_screen import TrainingScreen
from anomaly_match_ui.utils.backend_interface import BackendInterface

pytestmark = pytest.mark.ui


@pytest.fixture()
def mock_training_screen():
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "/fake/data"
    session.cfg.net = "test-cnn"
    session.cfg.N_to_load = 1000
    session.cfg.normalisation.normalisation_method = "CONVERSION_ONLY"
    session.cfg.normalisation.image_size = [64, 64]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.num_channels = 3
    session.cfg.test_ratio = 0.5
    session.cfg.top_N = 10
    session.cfg.num_eval_iter = 10
    session.cfg.metadata_file = None

    BackendInterface.set_session(session)
    with patch("IPython.display.display"):
        app = AnomalyMatchApp(session)
        screen = TrainingScreen(app)
        _ = screen.widget
    return screen


def test_set_busy_disables_buttons(mock_training_screen):
    screen = mock_training_screen
    # All buttons should start enabled
    for btn in screen._action_buttons:
        assert not btn.disabled

    screen._set_busy(True)
    for btn in screen._action_buttons:
        assert btn.disabled

    screen._set_busy(False)
    for btn in screen._action_buttons:
        assert not btn.disabled
