#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Browser tests for the prediction setup screen.

All tests share a single page/kernel to reduce Voila overhead.
"""

import pytest

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def prediction_setup_page(browser, voila_server):
    """Navigate to prediction setup once and share the page across tests."""
    context = browser.new_context()
    pg = context.new_page()
    pg.goto(voila_server, wait_until="domcontentloaded", timeout=60_000)
    pg.wait_for_selector(".widget-vbox", timeout=60_000)
    pg.locator("button", has_text="Open Prediction").click()
    pg.locator("text=Model checkpoint").wait_for(state="visible", timeout=15_000)
    # Wait for auto-validation to fully complete
    pg.locator("text=/Model:.*test_model\\.safetensors/").wait_for(state="visible", timeout=15_000)
    pg.locator("text=/Found \\d+ images/").wait_for(state="visible", timeout=30_000)
    yield pg
    pg.close()
    context.close()


def test_file_choosers_render(prediction_setup_page):
    """All three file chooser sections are visible."""
    page = prediction_setup_page
    assert page.locator("text=Model checkpoint").is_visible()
    assert page.locator("text=Search folder (images to scan)").is_visible()
    assert page.locator("text=Output folder (results destination)").is_visible()


def test_auto_validation_model(prediction_setup_page):
    """Pre-configured model_path shows model name in validation."""
    page = prediction_setup_page
    assert page.locator("text=/Model:.*test_model\\.safetensors/").is_visible()


def test_auto_validation_search_folder(prediction_setup_page):
    """Pre-configured prediction_search_dir triggers source counting."""
    page = prediction_setup_page
    assert page.locator("text=/Found \\d+ images/").is_visible()


def test_start_prediction_enabled(prediction_setup_page):
    """Start Prediction button is enabled once all validations pass."""
    page = prediction_setup_page
    btn = page.locator("button", has_text="Start Prediction")
    assert btn.is_visible()
    assert btn.is_enabled()


def test_preview_gallery_renders(prediction_setup_page):
    """Source counting generates a preview gallery with sample images."""
    page = prediction_setup_page
    # Preview gallery is rendered as a matplotlib figure (base64 PNG)
    assert page.locator("img[src*='data:image/png']").first.is_visible()
