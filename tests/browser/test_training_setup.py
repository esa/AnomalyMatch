#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Browser tests for the training setup screen.

All tests share a single page/kernel to reduce Voila overhead.
"""

import pytest

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def training_setup_page(browser, voila_server):
    """Navigate to training setup once and share the page across tests."""
    context = browser.new_context()
    pg = context.new_page()
    pg.goto(voila_server, wait_until="domcontentloaded", timeout=60_000)
    pg.wait_for_selector(".widget-vbox", timeout=60_000)
    pg.locator("button", has_text="Open Training").click()
    pg.locator("text=Source folder").wait_for(state="visible", timeout=15_000)
    # Wait for auto-validation to fully complete before running tests
    pg.locator("text=/Found \\d+ images/").wait_for(state="visible", timeout=30_000)
    # Both the label-CSV summary and the found/missing line match this
    # pattern now (the found line added a per-class breakdown) — first
    # visible is enough to confirm auto-validation finished.
    pg.locator("text=/\\d+ anomaly, \\d+ normal/").first.wait_for(state="visible", timeout=30_000)
    yield pg
    pg.close()
    context.close()


def test_file_choosers_render(training_setup_page):
    """File chooser sections are visible (model checkpoint removed)."""
    page = training_setup_page
    assert page.locator("text=Source folder").is_visible()
    assert page.locator("text=Labelled data CSV").is_visible()
    assert page.locator("text=Metadata CSV (optional)").is_visible()


def test_auto_validation_shows_image_count(training_setup_page):
    """Pre-configured data_dir triggers auto-validation showing image count."""
    page = training_setup_page
    assert page.locator("text=/Found \\d+ images/").is_visible()


def test_auto_validation_shows_label_summary(training_setup_page):
    """Pre-configured label_file triggers auto-validation showing label breakdown."""
    page = training_setup_page
    assert page.locator("text=/\\d+ anomaly, \\d+ normal/").first.is_visible()


def test_image_probe_shows_resolution(training_setup_page):
    """Auto-validation probes the first image and shows resolution info."""
    page = training_setup_page
    assert page.locator("text=/\\d+.\\d+ px/").is_visible()


def test_preview_shows_labels(training_setup_page):
    """Preview gallery exposes Anomaly / Normal label toggle buttons.

    The setup screen now renders cutouts via the labelable gallery
    instead of static caption tags (#427), so the cell label state is
    available through the toggle buttons rather than ``[anomaly]`` /
    ``[normal]`` HTML badges.  The test waits for the gallery cells to
    decode (background thread) by looking for the toggle buttons.
    """
    page = training_setup_page
    # Preview loads on a background thread — wait for fitsbolt processing
    page.locator("button", has_text="Anomaly").first.wait_for(state="visible", timeout=30_000)
    page.locator("button", has_text="Normal").first.wait_for(state="visible", timeout=10_000)


def test_start_training_button_enabled(training_setup_page):
    """Start Training button is visible and enabled when validation passes."""
    page = training_setup_page
    btn = page.locator("button", has_text="Start Training")
    assert btn.is_visible()
    assert btn.is_enabled()


def test_stratify_checkbox_renders_and_disabled_for_image_folder(training_setup_page):
    """The 'Stratify source size per tile' checkbox is present but disabled for
    an image-folder source (it's Cutana-only — no per-source diameter)."""
    page = training_setup_page
    checkbox = page.locator(".widget-checkbox", has_text="Stratify source size per tile")
    assert checkbox.is_visible()
    assert checkbox.locator("input[type=checkbox]").is_disabled()
