#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end browser test for the training workflow.

Exercises: training_setup → Start Training → subprocess training completes.
Scoring is tested separately via prediction E2E.
"""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_training_completes(page):
    """Start training from setup, verify subprocess training finishes."""
    # Navigate to training setup
    page.locator("button", has_text="Open Training").click()
    page.locator("text=/Found \\d+ images/").wait_for(state="visible", timeout=30_000)

    # Click Start Training — navigates to training screen, auto-starts training
    page.locator("button", has_text="Start Training").click()
    page.locator("text=Source folder").wait_for(state="hidden", timeout=15_000)

    # Wait for training subprocess to complete (progress bar shows "Training complete")
    page.locator("text=/Training complete/").first.wait_for(state="visible", timeout=120_000)
