#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import pytest

from anomaly_match_ui.styles import (
    COMMON_STYLES,
    ESA_BLUE_BRIGHT,
    ESA_BLUE_DEEP,
    ESA_GREEN,
    ESA_RED,
    get_ui_scale,
    scale_px,
    set_ui_scale,
)

pytestmark = pytest.mark.ui


def test_color_constants_are_hex():
    for color in [ESA_BLUE_DEEP, ESA_BLUE_BRIGHT, ESA_GREEN, ESA_RED]:
        assert color.startswith("#")
        assert len(color) == 7


def test_common_styles_contains_css():
    assert ".widget-label" in COMMON_STYLES.data
    assert "am-header" in COMMON_STYLES.data


def test_scale_px_default():
    set_ui_scale(1.0)
    assert scale_px(14) == "14px"


def test_scale_px_scaled():
    set_ui_scale(1.5)
    assert scale_px(10) == "15px"
    set_ui_scale(1.0)


def test_get_ui_scale():
    set_ui_scale(2.0)
    assert get_ui_scale() == 2.0
    set_ui_scale(1.0)
