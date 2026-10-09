#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Browser test: normalisation config carries over from setup to training screen."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def _method_dropdown(page):
    """Locate the normalisation method dropdown by its 'Method' label."""
    return page.locator(".widget-dropdown").filter(has_text="Method").locator("select")


def test_normalisation_widget_visible_on_setup(page):
    """Training setup screen shows the normalisation configuration widget."""
    page.locator("button", has_text="Open Training").click()
    page.locator("text=/Found \\d+ images/").wait_for(state="visible", timeout=30_000)

    # Both normalisation sections should be visible
    page.locator("text=Normalisation Configuration").wait_for(state="visible", timeout=10_000)
    page.locator("text=Channel Combination").wait_for(state="visible", timeout=10_000)


def test_normalisation_method_carries_over(page):
    """Changing normalisation method on setup persists on the training screen."""
    # Navigate to training setup
    page.locator("button", has_text="Open Training").click()
    page.locator("text=/Found \\d+ images/").wait_for(state="visible", timeout=30_000)

    # The normalisation widget should be visible
    page.locator("text=Normalisation Configuration").wait_for(state="visible", timeout=10_000)

    # Change the normalisation method from default (ConversionOnly) to LogStretch.
    # ipywidgets enum dropdowns: use index since option labels lack HTML label attrs.
    # Index 1 = LogStretch per NormalisationMethod.get_options() order.
    setup_dropdown = _method_dropdown(page)
    setup_dropdown.select_option(index=1)

    # Click Start Training — navigates to training screen
    page.locator("button", has_text="Start Training").click()
    page.locator("text=Source folder").wait_for(state="hidden", timeout=15_000)

    # On the training screen, verify the normalisation widget carried over LogStretch
    page.locator("text=Normalisation Configuration").wait_for(state="visible", timeout=15_000)
    training_dropdown = _method_dropdown(page)
    selected_text = training_dropdown.evaluate("el => el.options[el.selectedIndex].textContent")
    assert selected_text == "LogStretch"


def test_normalisation_method_remembered_across_kernels(page, voila_server):
    """The method a run started with is the default on the next session.

    Reloading the Voila page spawns a fresh Jupyter kernel with a fresh
    Session, so the setup screen is rebuilt from the notebook's cfg defaults
    plus whatever was persisted — exactly the "restart AnomalyMatch" case.
    """
    page.locator("button", has_text="Open Training").click()
    page.locator("text=/Found \\d+ images/").wait_for(state="visible", timeout=30_000)
    page.locator("text=Normalisation Configuration").wait_for(state="visible", timeout=10_000)

    # Index 3 = Asinh per NormalisationMethod.get_options() order.
    _method_dropdown(page).select_option(index=3)
    page.locator("button", has_text="Start Training").click()
    page.locator("text=Source folder").wait_for(state="hidden", timeout=15_000)

    # Restart: fresh kernel, cfg back to the notebook's defaults.
    page.goto(voila_server, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_selector(".widget-vbox", timeout=60_000)
    page.locator("button", has_text="Open Training").click()
    page.locator("text=Normalisation Configuration").wait_for(state="visible", timeout=30_000)

    selected_text = _method_dropdown(page).evaluate(
        "el => el.options[el.selectedIndex].textContent"
    )
    assert selected_text == "Asinh"
