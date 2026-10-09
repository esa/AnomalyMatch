#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the PreviewGrid widget."""

import pytest

from anomaly_match_ui.widgets.preview_grid import _N_CELLS, PreviewGrid

pytestmark = pytest.mark.ui


def test_grid_pins_two_row_layout():
    """The preview grid is intentionally clipped to two rows of cells.

    Pages always show exactly two rows independent of window width;
    pagination (in the GalleryWidget) compensates when the user wants
    more.  An earlier fix attempt (#432) removed these rules, but the
    multi-row gallery that produced broke the rest of the screen
    geometry — keep the constraint pinned and add this test so a
    well-meaning future cleanup doesn't drop it again.
    """
    grid = PreviewGrid()
    layout = grid._grid.layout

    assert layout.grid_template_rows == "repeat(2, auto)"
    assert layout.grid_auto_rows == "0px"
    assert layout.overflow == "hidden"


def test_all_cells_present_in_dom():
    """All ``_N_CELLS`` cells are children of the grid widget."""
    grid = PreviewGrid()
    assert len(grid._grid.children) == _N_CELLS


def test_update_with_more_items_than_cells_caps_at_n_cells():
    """``update`` only fills the first ``_N_CELLS`` items."""
    grid = PreviewGrid()
    items = [(f"img_{i}", b"") for i in range(_N_CELLS + 5)]
    grid.update(items)
    visible = [cell for cell in grid._cells if cell.layout.display != "none"]
    assert len(visible) == _N_CELLS


def test_update_with_fewer_items_hides_extra_cells():
    """``update`` hides cells past the supplied items."""
    grid = PreviewGrid()
    items = [(f"img_{i}", b"") for i in range(5)]
    grid.update(items)
    visible = [cell for cell in grid._cells if cell.layout.display != "none"]
    assert len(visible) == 5
