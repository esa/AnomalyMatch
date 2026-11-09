#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""The training setup screen opens on the Euclid Hackathon's imaging defaults."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.utils.get_default_cfg import HACKATHON_DATA_DIR_ENV, get_default_cfg
from anomaly_match_ui.app import AnomalyMatchApp

pytestmark = pytest.mark.ui

BANDS = ["VIS", "NIR-H", "NIR-J", "NIR-Y"]


@pytest.fixture()
def hackathon_session(monkeypatch, tmp_path):
    data = tmp_path / "euclid_hackathon_data"
    (data / "q1_search_catalogue").mkdir(parents=True)
    monkeypatch.setenv(HACKATHON_DATA_DIR_ENV, str(data))
    session = MagicMock()
    session.cfg = get_default_cfg()
    return session


@patch("IPython.display.display")
def test_the_screen_keeps_the_hackathon_imaging_after_detecting_the_bands(_disp, hackathon_session):
    expected = hackathon_session.cfg.normalisation.channel_combination.copy()
    app = AnomalyMatchApp(hackathon_session)
    app.navigate_to("training_setup")
    screen = app._current_screen
    with patch(
        "anomaly_match_ui.screens.training_setup_screen.detect_cutana_filter_names",
        return_value=BANDS,
    ):
        screen._apply_cutana_band_info("/fake/cutana")

    widget = screen._norm_widget.get_normalisation_config()
    assert screen._norm_widget.extensions == BANDS
    assert widget["normalisation_method"] == NormalisationMethod.ASINH
    assert widget["cutout_padding_factor"] == 3.0
    assert widget["image_size"] == [192, 192]
    assert widget["n_output_channels"] == 3
    assert np.allclose(widget["channel_combination"], expected)
