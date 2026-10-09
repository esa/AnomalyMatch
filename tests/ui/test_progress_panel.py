#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the ProgressPanel waiting spinner."""

import pytest

from anomaly_match_ui.styles import ESA_GREEN, ESA_RED
from anomaly_match_ui.widgets.progress_panel import ProgressPanel

pytestmark = pytest.mark.ui


def test_start_spins_until_the_bar_first_moves():
    panel = ProgressPanel()

    panel.start("Starting scoring...")
    assert "am-spinner" in panel.spinner.value
    assert panel.status_label.value == "Starting scoring..."

    # A status-only update at 0 (e.g. "0 / N" while polling) keeps spinning.
    panel.update(0.0, "0 / 1,000")
    assert "am-spinner" in panel.spinner.value

    # Screens also write the bar directly; the first real tick ends the wait.
    panel.progress_bar.value = 0.01
    assert panel.spinner.value == ""


def test_stop_ends_the_wait_without_progress():
    """A run that fails or is stopped before any progress must not spin forever."""
    panel = ProgressPanel()
    panel.start("Starting prediction...")

    panel.update(0.0, "No results")
    panel.set_color(ESA_RED)
    assert "am-spinner" in panel.spinner.value, "recolouring alone must not end the wait"

    panel.stop()
    assert panel.spinner.value == ""


def test_new_run_after_completion_spins_again():
    panel = ProgressPanel()
    panel.start("Starting training...")
    panel.update(1.0, "Done")
    panel.set_color(ESA_GREEN)
    panel.stop()

    panel.start("Starting scoring...")

    assert panel.progress_bar.value == 0.0
    assert "am-spinner" in panel.spinner.value


def test_idle_panel_does_not_spin():
    assert ProgressPanel().spinner.value == ""
