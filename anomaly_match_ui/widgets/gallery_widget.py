#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Paginated thumbnail gallery with sort controls."""

from __future__ import annotations

from collections.abc import Callable

import ipywidgets as widgets

from anomaly_match_ui.styles import BG_COLOR, ESA_BLUE_BRIGHT, ESA_BLUE_DEEP
from anomaly_match_ui.widgets.thumbnail_cell import ThumbnailCell

# Column-count range the user picks from the size slider.  The slider
# sets the number of grid columns *directly* rather than a cell pixel
# width, so the on-screen grid and the pagination math share a single
# source of truth — they can never disagree about how many cells a page
# holds.  That disagreement (a guessed column count vs. CSS auto-fill
# against the real panel width) is what previously clipped labeled
# cutouts past row 2 while pagination collapsed to one page (#481).
# Fewer columns => larger thumbnails; the real panel width only changes
# cell *size*, never the count, so no image is ever skipped regardless
# of the user's Jupyter layout width.
_COLUMNS_MIN = 2
_COLUMNS_MAX = 10
_COLUMNS_DEFAULT = 5

# Rows shown per page.  Two rows keep the gallery at a fixed, compact
# height so the surrounding screen geometry stays stable — an earlier
# attempt to let the grid grow to many rows broke the layout other
# panels depend on (#432).  A page holds exactly ``_ROWS * columns``
# cells, so both visible rows are always full and pagination reaches
# every cell.
_ROWS = 2

# Thumbnail cells kept in the DOM.  Exactly the largest possible page
# (``_ROWS * _COLUMNS_MAX``); at the widest column count every cell is
# shown, and at narrower counts the surplus sits in clipped overflow
# rows (see the grid's two-row layout) until a page needs it.
PAGE_SIZE = _ROWS * _COLUMNS_MAX


def page_size_for_columns(n_columns: int) -> int:
    """Return the page size for a grid *n_columns* wide.

    Args:
        n_columns: Number of thumbnail columns the user selected.

    Returns:
        Cells per page — exactly ``_ROWS * n_columns`` — so the two
        visible rows are always full and pagination never skips a cell.
    """
    return _ROWS * n_columns


_SORT_MODES = {
    "Highest Score": "score_desc",
    "Lowest Score": "score_asc",
    "Closest to Mean": "score_mean_dist",
    "Closest to Median": "score_median_dist",
    "Most Recent": "updated_desc",
    "Random": "random",
}


class GalleryWidget:
    """Paginated thumbnail grid with sort controls.

    Args:
        on_magnify: Callback invoked with filename when a thumbnail's magnify button is clicked.
        on_star: Callback invoked with filename when a thumbnail's star button is clicked.
        cell_factory: Optional callable to create custom cell widgets. Called
            with ``on_magnify`` and ``on_star`` keyword arguments and must
            return an object with the same API as :class:`ThumbnailCell`.
        setup_mode: When ``True`` the sort bar is hidden — the score-based
            sort modes have no meaning before any prediction has run, so the
            controls would only confuse the setup-screen flow (#427).  The
            scale slider, pagination, and cell factory still work normally.
    """

    def __init__(
        self,
        on_magnify: Callable[[str], None] | None = None,
        on_star: Callable[[str], None] | None = None,
        cell_factory: Callable[..., object] | None = None,
        setup_mode: bool = False,
    ) -> None:
        self._sort_mode = "score_desc"
        self._on_sort_change: Callable[[], None] | None = None
        self._on_page_change: Callable[[int], None] | None = None
        self._on_scale_change_callback: Callable[[], None] | None = None
        self._current_page = 0
        self._total_pages = 0
        self._setup_mode = setup_mode

        # Sort controls
        self._sort_buttons: dict[str, widgets.Button] = {}
        sort_children = []
        for label, mode in _SORT_MODES.items():
            btn = widgets.Button(
                description=label,
                layout=widgets.Layout(height="28px"),
                style={"button_color": ESA_BLUE_DEEP},
            )
            btn.on_click(lambda _b, m=mode: self._set_sort(m))
            self._sort_buttons[mode] = btn
            sort_children.append(btn)
        self._highlight_active_sort()

        sort_bar = widgets.HBox(
            sort_children,
            layout=widgets.Layout(gap="4px", padding="4px 0", justify_content="center"),
        )
        if setup_mode:
            sort_bar.layout.display = "none"

        # Column-count slider.  Sets how many thumbnails sit in each row;
        # fewer columns make every thumbnail larger (each cell is a
        # ``1fr`` share of the panel), which is how the user gets more
        # detail without the count ever depending on screen width.
        self._scale_slider = widgets.IntSlider(
            value=_COLUMNS_DEFAULT,
            min=_COLUMNS_MIN,
            max=_COLUMNS_MAX,
            step=1,
            description="Columns",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "initial", "handle_color": "white"},
        )
        self._scale_slider.observe(self._on_scale_change, names="value")

        controls_bar = widgets.HBox(
            [sort_bar, self._scale_slider],
            layout=widgets.Layout(
                justify_content="space-between",
                align_items="center",
                padding="0 4px",
            ),
        )

        # Grid of thumbnail cells
        factory = cell_factory or (lambda **kw: ThumbnailCell(**kw))
        self._cells = [factory(on_magnify=on_magnify, on_star=on_star) for _ in range(PAGE_SIZE)]
        # The column count is fixed (`repeat(N, 1fr)`), set from the
        # slider — not derived from the panel width via `auto-fill`.  A
        # page holds exactly `_ROWS * N` cells, so the two-row viewport
        # (`grid_template_rows="repeat(2, auto)"` + `grid_auto_rows="0px"`
        # + `overflow="hidden"`) is always exactly full and never clips a
        # *populated* cell — only the leftover DOM cells past the current
        # page land in the clipped overflow rows.  Keeping the height
        # fixed to two rows preserves the surrounding screen geometry;
        # an earlier attempt to grow into many rows broke it (#432).
        self._grid = widgets.GridBox(
            [cell.widget for cell in self._cells],
            layout=widgets.Layout(
                grid_template_columns=f"repeat({_COLUMNS_DEFAULT}, 1fr)",
                grid_template_rows=f"repeat({_ROWS}, auto)",
                grid_auto_rows="0px",
                grid_gap="8px",
                padding="8px 0",
                width="100%",
                overflow="hidden",
            ),
        )

        # Pagination
        self._first_btn = widgets.Button(
            description="\u21e4 First",
            layout=widgets.Layout(width="100px", height="28px"),
            style={"button_color": ESA_BLUE_BRIGHT},
        )
        self._first_btn.on_click(lambda _: self._jump_to_page(0))

        self._prev_btn = widgets.Button(
            description="\u2190 Previous",
            layout=widgets.Layout(width="100px", height="28px"),
            style={"button_color": ESA_BLUE_BRIGHT},
        )
        self._prev_btn.on_click(lambda _: self._change_page(-1))

        self._page_label = widgets.HTML(
            value=self._page_text(0, 0, 0),
        )

        self._next_btn = widgets.Button(
            description="Next \u2192",
            layout=widgets.Layout(width="100px", height="28px"),
            style={"button_color": ESA_BLUE_BRIGHT},
        )
        self._next_btn.on_click(lambda _: self._change_page(1))

        pagination = widgets.HBox(
            [self._first_btn, self._prev_btn, self._page_label, self._next_btn],
            layout=widgets.Layout(
                justify_content="center",
                align_items="center",
                gap="8px",
            ),
        )

        # Assemble
        self.widget = widgets.VBox(
            [controls_bar, self._grid, pagination],
            layout=widgets.Layout(background_color=BG_COLOR),
        )

    def set_pagination(self, page: int, total_pages: int, total_count: int) -> None:
        """Update only the pagination label and button states.

        Cheap enough to call on the UI thread — touches no cells and issues
        no DB query — so a background poll refresh can keep the page counter
        current without disturbing (or re-spinning) the visible thumbnails.

        Args:
            page: Current zero-based page index.
            total_pages: Total number of pages.
            total_count: Total number of results.
        """
        self._current_page = page
        self._total_pages = total_pages
        self._page_label.value = self._page_text(page, total_pages, total_count)
        self._first_btn.disabled = page <= 0
        self._prev_btn.disabled = page <= 0
        self._next_btn.disabled = page >= total_pages - 1

    def begin_page(self, page: int, total_pages: int, total_count: int, n_loading: int) -> None:
        """Flip to *page* instantly, showing spinners while images load.

        Called on the UI thread the moment the user navigates so the page
        turns without waiting on the DB query or image decode: the first
        *n_loading* cells show a loading spinner and the rest are cleared.
        Filenames, scores, and images are filled in afterwards from a
        background thread. This is what makes ``Next``/``Previous``
        non-blocking.

        Args:
            page: Current zero-based page index.
            total_pages: Total number of pages.
            total_count: Total number of results.
            n_loading: Number of leading cells to put in the loading state.
        """
        self.set_pagination(page, total_pages, total_count)
        for i, cell in enumerate(self._cells):
            if i < n_loading:
                cell.mark_loading()
                cell.widget.layout.display = ""
            else:
                cell.clear()
                if self._setup_mode:
                    cell.widget.layout.display = "none"

    def update_page(
        self,
        results: list[dict],
        page: int,
        total_pages: int,
        total_count: int,
        image_loader: Callable[[str], bytes | None] | None = None,
    ) -> None:
        """Fill the grid cells from a page of DB results.

        Args:
            results: List of dicts with keys ``filename``, ``score``, and
                an optional ``display`` caption to show instead of
                ``filename`` (defaults to ``filename``).
            page: Current zero-based page index.
            total_pages: Total number of pages.
            total_count: Total number of results.
            image_loader: Callable that returns PNG bytes for a filename,
                or None if unavailable.
        """
        self._current_page = page
        self._total_pages = total_pages

        for i, cell in enumerate(self._cells):
            if i < len(results):
                r = results[i]
                img_bytes = b""
                if image_loader:
                    loaded = image_loader(r["filename"])
                    if loaded:
                        img_bytes = loaded
                cell.update(r["filename"], r["score"], img_bytes, display_name=r.get("display"))
                cell.widget.layout.display = ""
            else:
                cell.clear()
                # In setup_mode the gallery shows a finite candidate
                # list; rendering "Waiting for results…" placeholders
                # in the leftover slots is just noise (#427 follow-up).
                # Score-mode galleries leave them visible — a short
                # last page next to a populated previous one reads
                # better as placeholders than as empty space.
                if self._setup_mode:
                    cell.widget.layout.display = "none"

        self._page_label.value = self._page_text(page, total_pages, total_count)
        self._first_btn.disabled = page <= 0
        self._prev_btn.disabled = page <= 0
        self._next_btn.disabled = page >= total_pages - 1

    def get_sort_mode(self) -> str:
        """Return the current sort key.

        Returns:
            Sort key string (e.g. ``"score_desc"``).
        """
        return self._sort_mode

    def set_on_sort_change(self, callback: Callable[[], None]) -> None:
        """Register a callback for sort mode changes.

        Args:
            callback: Called when the user clicks a different sort button.
        """
        self._on_sort_change = callback

    def set_on_page_change(self, callback: Callable[[int], None]) -> None:
        """Register a callback for page changes.

        Args:
            callback: Called with the new page index.
        """
        self._on_page_change = callback

    def set_on_scale_change(self, callback: Callable[[], None]) -> None:
        """Register a callback for slider changes.

        The consumer should re-render the current page using the new
        :attr:`effective_page_size` so paging tracks what's visible.

        Args:
            callback: Zero-arg callable invoked after the slider's CSS
                update has been applied.
        """
        self._on_scale_change_callback = callback

    @property
    def effective_page_size(self) -> int:
        """Cells shown per page at the current column count.

        Equal to ``_ROWS * columns`` — the grid shows exactly this many
        cells, so pagination consumers (gallery_screen_base,
        training_setup_screen) slice their data with this rather than the
        static :data:`PAGE_SIZE` and every cell stays reachable by paging.
        """
        return page_size_for_columns(self._scale_slider.value)

    # ── Internal ─────────────────────────────────────────────────────

    def _on_scale_change(self, change: dict) -> None:
        """Update the grid's column count from the slider.

        Cell width follows from ``1fr`` against the real panel width, so
        a wider Jupyter layout simply renders larger thumbnails; the
        column count (and therefore the page size) stays screen-
        independent.  The callback re-pages so ``total_pages`` tracks the
        new page size.
        """
        n_columns = change["new"]
        self._grid.layout.grid_template_columns = f"repeat({n_columns}, 1fr)"
        if self._on_scale_change_callback:
            self._on_scale_change_callback()

    def _set_sort(self, mode: str) -> None:
        if mode == self._sort_mode:
            return
        self._sort_mode = mode
        self._current_page = 0
        self._highlight_active_sort()
        if self._on_sort_change:
            self._on_sort_change()

    def _highlight_active_sort(self) -> None:
        for mode, btn in self._sort_buttons.items():
            btn.style.button_color = ESA_BLUE_BRIGHT if mode == self._sort_mode else ESA_BLUE_DEEP

    def _change_page(self, delta: int) -> None:
        new_page = self._current_page + delta
        if 0 <= new_page < max(self._total_pages, 1):
            self._current_page = new_page
            if self._on_page_change:
                self._on_page_change(new_page)

    def _jump_to_page(self, page: int) -> None:
        """Jump straight to an absolute page (used by the First button).

        Unlike :meth:`_change_page`, which steps relative to the current page,
        this seeks directly so "First" lands on page 0 from anywhere. No-op if
        already on the target page so we don't trigger a redundant DB refresh.

        Args:
            page: Zero-based page index to jump to.
        """
        target = max(0, min(page, max(self._total_pages, 1) - 1))
        if target != self._current_page:
            self._current_page = target
            if self._on_page_change:
                self._on_page_change(target)

    @staticmethod
    def _page_text(page: int, total_pages: int, total_count: int) -> str:
        if total_count == 0:
            return '<span style="color:#888; font-size:12px;">No results yet</span>'
        return (
            f'<span style="color:white; font-size:12px;">'
            f"Page {page + 1} of {total_pages} ({total_count:,} results)"
            f"</span>"
        )
