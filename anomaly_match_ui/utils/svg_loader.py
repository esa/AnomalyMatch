#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Logo loaders for ESA and AnomalyMatch branding in the UI."""

from __future__ import annotations

import base64
import importlib.resources


def _load_png_base64(filename: str) -> str:
    """Load a PNG from the assets package as a base64 data URI.

    Args:
        filename: Name of the PNG file inside the ``assets`` package.

    Returns:
        A ``data:image/png;base64,...`` URI string for embedding in HTML.
    """
    ref = importlib.resources.files("assets").joinpath(filename)
    data = ref.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_esa_logo(height: int) -> str:
    """Load the ESA logo SVG content from the assets package.

    Args:
        height: Height of the logo in pixels.

    Returns:
        The SVG markup as a string.
    """
    esa_native_width, esa_native_height = 107.38, 38.975
    esa_svg = load_icon_svg("ESA_logo.svg")
    esa_svg = esa_svg.replace(f'width="{esa_native_width}"', "", 1)
    esa_svg = esa_svg.replace(f'height="{esa_native_height}"', "", 1)
    esa_svg = esa_svg.replace(
        "<svg ",
        f'<svg viewBox="0 0 {esa_native_width} {esa_native_height}" '
        f'style="height:{height}px; width:auto; display:block" ',
        1,
    )
    return esa_svg


def load_am_logo_base64() -> str:
    """Load the full AnomalyMatch logo (white variant) as a base64 data URI.

    Returns:
        A ``data:image/png;base64,...`` URI string.
    """
    return _load_png_base64("am_logo_white.png")


def load_am_logo_only_base64() -> str:
    """Load the AnomalyMatch icon-only logo as a base64 data URI.

    Returns:
        A ``data:image/png;base64,...`` URI string.
    """
    return _load_png_base64("am_logo_only.png")


def load_icon_svg(filename: str) -> str:
    """Load an SVG icon from the assets package.

    Args:
        filename: Name of the SVG file inside the ``assets`` package.

    Returns:
        The SVG markup as a string.
    """
    ref = importlib.resources.files("assets").joinpath(filename)
    return ref.read_text(encoding="utf-8")


def get_dual_logo_html(height: int = 32) -> str:
    """Return inline HTML with both ESA and AnomalyMatch logos side by side.

    Uses the icon-only AM logo for compact display (e.g. headers).

    Args:
        height: Height of each logo in pixels.

    Returns:
        HTML string with both logos suitable for ``ipywidgets.HTML``.
    """
    am_src = load_am_logo_only_base64()
    return (
        f'<div style="display:flex; align-items:center; justify-content:center; gap:12px;">'
        f"{load_esa_logo(height)}"
        f'<img src="{am_src}" style="height:{height}px; width:auto; display:block;" '
        f'alt="AnomalyMatch"/>'
        f"</div>"
    )
