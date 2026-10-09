#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from unittest.mock import MagicMock

import ipywidgets as widgets
import pytest

from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.widgets.header_widget import HeaderWidget

pytestmark = pytest.mark.ui


@pytest.fixture(autouse=True)
def _mock_backend():
    session = MagicMock()
    session.cfg = MagicMock()
    BackendInterface.set_session(session)
    yield
    BackendInterface._session = None


def test_header_widget_returns_vbox():
    app = MagicMock()
    out = widgets.Output()
    header = HeaderWidget(app, out)
    assert isinstance(header.widget, widgets.VBox)
    # Should contain at least the header bar
    assert len(header.widget.children) >= 1


def test_header_widget_main_menu_mode():
    app = MagicMock()
    header = HeaderWidget(app)
    assert isinstance(header.widget, widgets.VBox)


def test_header_widget_with_back_button():
    app = MagicMock()
    header = HeaderWidget(app, show_back_button=True)
    assert isinstance(header.widget, widgets.VBox)


def test_set_back_to_retargets_navigation():
    """``set_back_to`` updates which screen the back-click navigates to.

    The image-detail screen relies on this — it serves training and
    prediction galleries, and the back-button target depends on which
    one navigated here (#427 follow-up).
    """
    app = MagicMock()
    header = HeaderWidget(app, show_back_button=True, back_to="prediction")

    header.set_back_to("training")
    header._back_btn.click()

    app.navigate_to.assert_called_with("training")
