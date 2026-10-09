#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end browser test for the full prediction workflow.

Exercises: prediction_setup → Start Prediction → prediction subprocess →
gallery with scored results.

Uses the shared test_model.safetensors checkpoint built by the
``test_model_path`` fixture.
"""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_full_prediction_cycle(page):
    """Start prediction from setup, wait for completion, verify results."""
    # Navigate to prediction setup
    page.locator("button", has_text="Open Prediction").click()
    page.locator("text=Model checkpoint").wait_for(state="visible", timeout=15_000)

    # Wait for auto-validation to pass (model + search folder)
    page.locator("text=/Model:.*test_model\\.safetensors/").wait_for(
        state="visible", timeout=15_000
    )
    page.locator("text=/Found \\d+ images/").wait_for(state="visible", timeout=30_000)

    # Click Start Prediction — navigates to prediction screen
    page.locator("button", has_text="Start Prediction").click()

    # Prediction screen loads — wait for "Start Prediction" button on prediction screen
    page.locator("text=Model checkpoint").wait_for(state="hidden", timeout=15_000)
    page.locator("button", has_text="Start Prediction").wait_for(state="visible", timeout=30_000)

    # Click Start Prediction on the prediction screen
    page.locator("button", has_text="Start Prediction").click()

    # Wait for prediction to complete — "results" in status text
    page.locator("text=/results/").first.wait_for(state="visible", timeout=180_000)
