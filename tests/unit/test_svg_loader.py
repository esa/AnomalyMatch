#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import importlib.resources
from pathlib import Path

import anomaly_match_ui
from anomaly_match_ui.utils.svg_loader import (
    _ASSETS_PACKAGE,
    load_am_logo_base64,
    load_am_logo_only_base64,
    load_esa_logo,
    load_icon_svg,
)


def test_assets_resolve_inside_anomaly_match_ui():
    """Assets must come from our own package, never a same-named one elsewhere.

    A top-level ``assets`` package was shadowed by the one cutana installs, which has
    no AnomalyMatch logos, so the UI crashed on start with an editable install.
    """
    assets_dir = Path(str(importlib.resources.files(_ASSETS_PACKAGE)))
    assert assets_dir.parent == Path(anomaly_match_ui.__file__).parent


def test_every_logo_and_icon_loads():
    """Each asset the UI embeds must be packaged and readable."""
    assert load_am_logo_base64().startswith("data:image/png;base64,")
    assert load_am_logo_only_base64().startswith("data:image/png;base64,")
    assert "<svg" in load_esa_logo(height=40)
    assert "<svg" in load_icon_svg("training_icon.svg")
    assert "<svg" in load_icon_svg("prediction_icon.svg")
