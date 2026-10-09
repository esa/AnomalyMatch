#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from unittest.mock import MagicMock

import ipywidgets as widgets
import pytest

from anomaly_match_ui.screens.base_screen import BaseScreen

pytestmark = pytest.mark.ui


class ConcreteScreen(BaseScreen):
    """Minimal concrete implementation for testing."""

    def __init__(self, app):
        super().__init__(app)
        self.entered = False
        self.left = False

    def build(self):
        return widgets.Label(value="test screen")

    def on_enter(self):
        self.entered = True

    def on_leave(self):
        self.left = True


def test_lazy_build():
    app = MagicMock()
    screen = ConcreteScreen(app)
    assert screen._widget is None
    w = screen.widget
    assert isinstance(w, widgets.Label)
    assert screen._widget is w
    # Second access returns the same widget
    assert screen.widget is w


def test_on_enter_on_leave():
    app = MagicMock()
    screen = ConcreteScreen(app)
    assert not screen.entered
    screen.on_enter()
    assert screen.entered
    assert not screen.left
    screen.on_leave()
    assert screen.left


def test_navigate_to_delegates():
    app = MagicMock()
    screen = ConcreteScreen(app)
    screen.navigate_to("training", some_kwarg=42)
    app.navigate_to.assert_called_once_with("training", some_kwarg=42)
