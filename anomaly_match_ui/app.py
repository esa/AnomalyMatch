#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Entry point for the AnomalyMatch UI with screen-based navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
from IPython.display import display
from loguru import logger

from anomaly_match.pipeline.session import Session
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match_ui.styles import BG_COLOR, COMMON_STYLES, set_ui_scale
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.scrolling_log_sink import install_auto_scroll
from anomaly_match_ui.widgets.loading_widget import LoadingWidget

if TYPE_CHECKING:
    from anomaly_match_ui.screens.base_screen import BaseScreen


class AnomalyMatchApp:
    """Top-level application with screen-based navigation.

    Creates a container widget and manages screen transitions. Each screen
    is lazily instantiated on first navigation and re-used on subsequent
    visits.

    Args:
        session: The backend Session instance.
        ui_scale: UI scale factor (1.0 = default).
    """

    def __init__(self, session: Session, ui_scale: float = 1.0) -> None:
        BackendInterface.set_session(session)
        set_ui_scale(ui_scale)

        self.container = widgets.VBox(layout=widgets.Layout(background_color=BG_COLOR))
        self._screens: dict[str, BaseScreen] = {}
        self._current_screen: BaseScreen | None = None
        self._loading = LoadingWidget()
        self._detail_context: dict | None = None

    # ========== Navigation ==========

    def navigate_to(self, screen_name: str, **kwargs: Any) -> None:
        """Navigate to a named screen.

        If the current screen implements ``on_leave`` it is called first.
        The target screen is lazily created if it doesn't exist yet, then
        ``on_enter`` is called.

        Args:
            screen_name: Registered name of the target screen.
            **kwargs: Extra keyword arguments forwarded to the screen factory.
        """
        if self._current_screen is not None:
            self._current_screen.on_leave()

        # Show loading spinner (with live log tail) while the new screen builds
        self._loading.start()
        self.container.children = [self._loading.widget]

        try:
            screen = self._get_or_create_screen(screen_name, **kwargs)
            self._current_screen = screen
            # Access .widget first to ensure build() has run before on_enter()
            w = screen.widget
            screen.on_enter()
            self.container.children = [w]
        finally:
            self._loading.stop()

    def _get_or_create_screen(self, name: str, **kwargs: Any) -> BaseScreen:
        """Return an existing screen or create a new one.

        Args:
            name: Screen identifier.
            **kwargs: Forwarded to the screen constructor.

        Returns:
            The screen instance.
        """
        if name not in self._screens:
            self._screens[name] = self._create_screen(name, **kwargs)
        return self._screens[name]

    def _create_screen(self, name: str, **kwargs: Any) -> BaseScreen:
        """Instantiate a screen by name.

        Args:
            name: Screen identifier.
            **kwargs: Forwarded to the screen constructor.

        Returns:
            A new screen instance.

        Raises:
            ValueError: If *name* is not a known screen identifier.
        """
        # lazy: avoid circular import — screens reference the app
        if name == "main_menu":
            from anomaly_match_ui.screens.main_menu_screen import MainMenuScreen  # noqa: PLC0415

            return MainMenuScreen(self, **kwargs)
        if name == "training_setup":
            from anomaly_match_ui.screens.training_setup_screen import (  # noqa: PLC0415
                TrainingSetupScreen,
            )

            return TrainingSetupScreen(self, **kwargs)
        if name == "training":
            from anomaly_match_ui.screens.training_screen import TrainingScreen  # noqa: PLC0415

            return TrainingScreen(self, **kwargs)
        if name == "prediction_setup":
            from anomaly_match_ui.screens.prediction_setup_screen import (  # noqa: PLC0415
                PredictionSetupScreen,
            )

            return PredictionSetupScreen(self, **kwargs)
        if name == "prediction":
            from anomaly_match_ui.screens.prediction_screen import PredictionScreen  # noqa: PLC0415

            return PredictionScreen(self, **kwargs)
        if name == "image_detail":
            from anomaly_match_ui.screens.image_detail_screen import (  # noqa: PLC0415
                ImageDetailScreen,
            )

            return ImageDetailScreen(self, **kwargs)
        raise ValueError(f"Unknown screen: {name!r}")

    # ========== Display ==========

    def show(self) -> None:
        """Display the app in the current Jupyter notebook output cell."""
        display(COMMON_STYLES)
        # Inject the per-page MutationObserver that pins every
        # ``am-log-sink`` Output widget to its bottom (#430).  It must
        # land here because ``display`` only reaches the front-end from
        # a notebook display context, not from the loguru dispatcher
        # thread that builds each screen's sink.
        install_auto_scroll()
        display(self.container)
        self.navigate_to("main_menu")


def start_ui(session: Session | None = None, *, ui_scale: float = 1.0) -> AnomalyMatchApp:
    """Start the AnomalyMatch UI.

    This is the main entry point for launching the UI. When called without
    arguments it creates a default session automatically — all configuration
    (data paths, normalisation, model) is then done interactively via the
    setup screens.

    Args:
        session: An AnomalyMatch Session instance.  When ``None`` (the
            default), a session is created from :func:`get_default_cfg`.
        ui_scale: UI scale factor (1.0 = default).

    Returns:
        The created AnomalyMatchApp instance.

    Example:
        >>> from anomaly_match_ui import start_ui
        >>> app = start_ui()
    """
    if session is None:
        session = Session(get_default_cfg())

    logger.debug("Starting AnomalyMatch UI...")

    app = AnomalyMatchApp(session, ui_scale=ui_scale)
    app.show()

    logger.debug("AnomalyMatch UI started successfully.")
    return app
