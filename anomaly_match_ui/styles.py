#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""ESA-branded styles, color constants, and CSS for the AnomalyMatch UI."""

from IPython.display import HTML

# ========== ESA Color Palette ==========
ESA_BLUE_DEEP = "#003247"
ESA_BLUE_BRIGHT = "#00a1de"
ESA_GREEN = "#28a745"
ESA_RED = "#dc3545"
ESA_ORANGE = "#fd7e14"
TEXT_COLOR = "#ffffff"
BG_COLOR = "#000000"
NEUTRAL_GREY = "#555555"  # Neutral / unlabelled state
CONTROL_BG = "#335E6E"  # Blue-grey tint for form controls (Cutana-style)

# ========== UI Scale ==========
_ui_scale = 1.0


def set_ui_scale(scale: float) -> None:
    """Set the global UI scale factor.

    Args:
        scale: The scale factor (1.0 = default size).
    """
    global _ui_scale  # noqa: PLW0603
    _ui_scale = scale


def get_ui_scale() -> float:
    """Get the current UI scale factor.

    Returns:
        The current scale factor.
    """
    return _ui_scale


def scale_px(base_px: int) -> str:
    """Scale a pixel value by the current UI scale and return as CSS string.

    Args:
        base_px: The base pixel value at scale 1.0.

    Returns:
        CSS pixel string like ``"14px"``.
    """
    return f"{int(base_px * _ui_scale)}px"


# ========== Common CSS ==========
COMMON_STYLES = HTML(
    """
    <style>
        .widget-label, .widget-slider {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        /* Checkbox descriptions render outside .widget-label, so colour them
           explicitly or they stay dark-on-dark and read as an unlabelled box. */
        .widget-checkbox, .widget-checkbox label, .widget-checkbox span {
            color: #ffffff !important;
        }
        .widget-output {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #4a6a7a !important;
        }
        .widget-box {
            background-color: #000000 !important;
        }
        .widget-hbox, .widget-vbox {
            background-color: #000000 !important;
        }
        .widget-slider .slider {
            background-color: #000000 !important;
        }
        .widget-slider .widget-readout {
            color: #ffffff !important;
        }
        .widget-text {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .widget-button[button_style="primary"] {
            background-color: #00a1de !important;
            color: #ffffff !important;
        }
        .widget-button[button_style="info"] {
            background-color: #17a2b8 !important;
            color: #000000 !important;
        }
        .widget-button[button_style="warning"] {
            background-color: #ffc107 !important;
            color: #000000 !important;
        }
        .widget-button[button_style="success"] {
            background-color: #28a745 !important;
            color: #000000 !important;
        }
        .widget-output pre {
            color: #d5d5d5 !important;
            background-color: #000000 !important;
        }
        .am-header {
            background-color: #003247 !important;
            padding: 8px 16px;
            border-bottom: 2px solid #00a1de;
        }
        .am-screen {
            background-color: #000000 !important;
        }
        @keyframes am-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .am-spinner {
            border: 3px solid rgba(255,255,255,0.15);
            border-top: 3px solid #00a1de;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            animation: am-spin 0.8s linear infinite;
            margin: 0 auto 12px auto;
        }
    </style>
    """
)


# ========== Shared Widget Layout Helpers ==========
# Reusable layout / style dicts for ipywidgets form controls.
def inline_spinner_html(size_px: int = 14, margin: str = "0") -> str:
    """Return a small inline ``.am-spinner`` for status lines and captions.

    The ``.am-spinner`` class is styled as a large centred block for loading
    panels; the inline overrides here shrink it and reset its margin.

    Args:
        size_px: Diameter in pixels.
        margin: CSS margin, e.g. ``"0 6px 0 0"`` to space it from following text.

    Returns:
        HTML for the spinner.
    """
    return (
        '<span class="am-spinner" style="display:inline-block;'
        f" width:{size_px}px; height:{size_px}px; border-width:2px;"
        f' vertical-align:middle; margin:{margin};"></span>'
    )


WIDGET_STYLE = {"description_width": "initial"}
WIDGET_LAYOUT_KWARGS = {"background_color": BG_COLOR}
WIDGET_WIDE_LAYOUT_KWARGS = {"background_color": BG_COLOR, "width": "300px"}

# Themed form controls (dropdowns, number inputs, sliders, checkboxes).
# Raw CSS string — wrap in ``ipywidgets.HTML(FORM_CONTROL_CSS)`` when
# embedding as a widget child. Uses the ``.am-form-themed`` CSS class
# as scope — add it to the container via ``widget.add_class("am-form-themed")``.
FORM_CONTROL_CSS = f"""
    <style>
    /* Dropdown styling */
    .am-form-themed .widget-dropdown select {{
        background: {CONTROL_BG} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {CONTROL_BG} !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
    }}
    .am-form-themed .widget-dropdown select:focus {{
        border-color: {ESA_BLUE_BRIGHT} !important;
        box-shadow: 0 0 0 2px rgba(0, 161, 222, 0.2) !important;
        outline: none !important;
    }}
    .am-form-themed .widget-dropdown select option {{
        background: {CONTROL_BG} !important;
        color: {TEXT_COLOR} !important;
    }}
    /* Number / text input styling */
    .am-form-themed .widget-text input[type="number"],
    .am-form-themed .widget-text input[type="text"] {{
        background: {CONTROL_BG} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {CONTROL_BG} !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
    }}
    .am-form-themed .widget-text input:focus {{
        border-color: {ESA_BLUE_BRIGHT} !important;
        box-shadow: 0 0 0 2px rgba(0, 161, 222, 0.2) !important;
        outline: none !important;
    }}
    /* Slider readout */
    .am-form-themed .widget-readout {{
        color: {ESA_BLUE_BRIGHT} !important;
        font-weight: 600 !important;
    }}
    /* Checkbox label */
    .am-form-themed .widget-checkbox label,
    .am-form-themed .widget-checkbox .widget-label {{
        color: {TEXT_COLOR} !important;
    }}
    </style>
    """
