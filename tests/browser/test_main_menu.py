#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Browser tests for the main menu screen and navigation."""

import pytest

pytestmark = pytest.mark.browser


def test_main_menu_renders(page):
    """Verify the main menu screen renders with navigation cards."""
    # At least one AnomalyMatch logo should be visible
    page.locator("img[alt='AnomalyMatch']").first.wait_for(state="visible", timeout=15_000)

    # Both navigation buttons must be present
    page.locator("button", has_text="Open Training").wait_for(state="visible", timeout=10_000)
    page.locator("button", has_text="Open Prediction").wait_for(state="visible", timeout=10_000)


def test_navigate_to_training_setup(page):
    """Click 'Open Training' and verify training setup screen appears."""
    page.locator("button", has_text="Open Training").click()

    # Training setup screen shows folder selection widgets
    page.locator("text=Source folder").wait_for(state="visible", timeout=15_000)


def test_navigate_to_prediction_setup(page):
    """Click 'Open Prediction' and verify prediction setup screen appears."""
    page.locator("button", has_text="Open Prediction").click()

    # Prediction setup screen shows model checkpoint chooser
    page.locator("text=Model checkpoint").wait_for(state="visible", timeout=15_000)
