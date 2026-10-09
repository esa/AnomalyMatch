#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Base screen class for the modular UI architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets

if TYPE_CHECKING:  # Avoid circular import: app → screens → base_screen → app
    from anomaly_match_ui.app import AnomalyMatchApp


class BaseScreen(ABC):
    """Abstract base class for all UI screens.

    Each screen owns a widget tree built lazily on first access.
    Navigation between screens is handled via the parent ``app`` reference.

    Args:
        app: The parent application instance used for navigation.
    """

    def __init__(self, app: AnomalyMatchApp) -> None:
        self.app = app
        self._widget: widgets.Widget | None = None

    @abstractmethod
    def build(self) -> widgets.Widget:
        """Build and return the screen's root widget.

        Returns:
            The root ipywidget for this screen.
        """

    def on_enter(self) -> None:
        """Called when the screen becomes the active screen."""

    def on_leave(self) -> None:
        """Called when the screen is being navigated away from."""

    def navigate_to(self, screen_name: str, **kwargs: Any) -> None:
        """Navigate to another screen via the app.

        Args:
            screen_name: The registered name of the target screen.
            **kwargs: Extra keyword arguments forwarded to the screen factory.
        """
        self.app.navigate_to(screen_name, **kwargs)

    @property
    def widget(self) -> widgets.Widget:
        """Lazy-built root widget for this screen."""
        if self._widget is None:
            self._widget = self.build()
        return self._widget
