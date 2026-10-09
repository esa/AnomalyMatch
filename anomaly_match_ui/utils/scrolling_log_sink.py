#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Capped, auto-scrolling loguru sink for ipywidgets ``Output`` panes.

Four screens (training / training-setup / prediction / prediction-setup)
mirror the loguru stream into an :class:`ipywidgets.Output` so the user
can watch subprocess INFO lines from the UI.  Two problems on long runs
(#430):

1. ``Output.append_stdout`` retains every line for the lifetime of the
   widget, so a multi-day run accumulates tens of thousands of entries
   and slows the kernel / browser.
2. The Output widget's ``overflow="auto"`` does *not* auto-scroll — new
   lines appear below the viewport unless the user scrolls.

:class:`ScreenLogSink` wraps the sink-creation logic and the
``logger.add`` / ``logger.remove`` lifecycle.  The sink maintains a
bounded ``deque`` of lines and rewrites the widget's ``outputs``
payload in place, so memory stays flat.  A one-shot
``MutationObserver`` keeps any widget tagged with the ``am-log-sink``
CSS class pinned to its bottom.  The on-disk ``session.log`` continues
to hold the unabridged history.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import ipywidgets as widgets
from IPython.display import Javascript, display
from loguru import logger

DEFAULT_MAX_LINES = 250

_LOG_SINK_CSS_CLASS = "am-log-sink"

# JS observer that pins every existing and future ``am-log-sink`` widget
# to its bottom on each DOM mutation.  A single global observer beats
# per-message ``IPython.display`` calls (those flood the browser console
# and slow rendering).  Injected once from
# :meth:`AnomalyMatchApp.show` because that is where ``IPython.display``
# actually publishes to the Voila / Lab front-end — calls from inside
# the sink-creation path don't have a notebook display context to land in.
_AUTO_SCROLL_JS = """
(function() {
    if (window.__amLogScrollObserverInstalled) { return; }
    window.__amLogScrollObserverInstalled = true;

    var stickToBottom = function(el) {
        el.scrollTop = el.scrollHeight;
    };
    var watch = function(el) {
        if (el.__amLogObserved) { return; }
        el.__amLogObserved = true;
        stickToBottom(el);
        new MutationObserver(function() { stickToBottom(el); }).observe(
            el, { childList: true, subtree: true, characterData: true }
        );
    };
    var scan = function() {
        document.querySelectorAll('.am-log-sink').forEach(watch);
    };
    scan();
    new MutationObserver(scan).observe(
        document.body, { childList: true, subtree: true }
    );
})();
"""


def install_auto_scroll() -> None:
    """Inject the global auto-scroll observer into the front-end.

    Call once from a notebook display context (e.g. the app's ``show()``
    method).  Subsequent calls are harmless: the JS itself is idempotent
    via ``window.__amLogScrollObserverInstalled``.  Calling from inside
    a sink (i.e. from a loguru dispatcher thread) does *not* publish to
    the front-end, which is why this is a separate entry point.
    """
    display(Javascript(_AUTO_SCROLL_JS))


def _seed_buffer_from_widget(out: widgets.Output, buffer: deque[str]) -> None:
    """Seed *buffer* with the existing widget content's tail.

    Without this, navigating away from a screen and back (which
    re-creates the sink) would wipe the visible log on the first new
    message — the new sink replaces ``out.outputs`` entirely with its
    own buffer, which would start empty.
    """
    text_chunks: list[str] = []
    for entry in out.outputs:
        if entry.get("output_type") == "stream":
            text_chunks.append(entry.get("text", ""))
    if not text_chunks:
        return
    for line in "".join(text_chunks).splitlines(keepends=True):
        buffer.append(line)


def _make_scrolling_log_sink(
    out: widgets.Output, max_lines: int = DEFAULT_MAX_LINES
) -> Callable[[object], None]:
    """Return a loguru sink that caps *out* to *max_lines* and auto-scrolls.

    Module-internal helper invoked from :meth:`ScreenLogSink.attach`.
    Kept separate from the lifecycle wrapper so the cap / seed / replace
    behaviour can be unit-tested as a pure callable instead of having to
    drive the global loguru dispatcher.

    Args:
        out: The ipywidgets ``Output`` to render into.  The widget gets
            tagged with the ``am-log-sink`` CSS class on first call;
            the global :data:`_AUTO_SCROLL_JS` observer then keeps it
            pinned to its bottom.
        max_lines: Maximum number of lines to retain in the widget.
            Older lines roll off the top; the on-disk ``session.log``
            holds the unabridged history.

    Returns:
        A loguru-compatible sink callable ``sink(message)``.  Loguru
        will dispatch to it from its own thread; ipywidgets attribute
        writes are thread-safe (they marshal through the comm channel).
    """
    out.add_class(_LOG_SINK_CSS_CLASS)

    buffer: deque[str] = deque(maxlen=max_lines)
    _seed_buffer_from_widget(out, buffer)

    def sink(message: object) -> None:
        # ``str(message)`` already includes the trailing newline because
        # loguru renders the configured ``"{message}\n"`` format.  Join
        # with empty separator so multi-line entries stay intact.
        buffer.append(str(message))
        out.outputs = ({"output_type": "stream", "name": "stdout", "text": "".join(buffer)},)

    return sink


class ScreenLogSink:
    """Lifecycle wrapper for a screen-bound, capped, auto-scrolling sink.

    Each of the four UI screens (training / training_setup / prediction /
    prediction_setup) mirrors the loguru stream into its own ``Output``
    widget via the same attach/detach dance: tear down any prior sink,
    build a fresh capped sink, ``logger.add``; on screen leave,
    ``logger.remove`` while tolerating the ``ValueError`` that
    :meth:`Session.set_terminal_out` raises after a global sink clear.

    This class consolidates that lifecycle so the per-screen wrappers
    don't have to redefine the same method twice.

    Args:
        screen_label: Human-readable name used in the debug log when a
            sink turns out to have already been removed (e.g.
            ``"TrainingScreen"``).
    """

    def __init__(self, screen_label: str) -> None:
        self._screen_label = screen_label
        self._sink_id: int | None = None

    def attach(self, out: widgets.Output) -> None:
        """Bind a capped, auto-scrolling sink to *out*.

        Idempotent — any previously-attached sink is detached first.
        Bypasses ipykernel's ``OutStream`` (via the underlying
        :func:`_make_scrolling_log_sink`) to dodge ``parent_header``-walk
        hangs on thread-ident reuse.
        """
        self.detach()
        sink = _make_scrolling_log_sink(out)
        self._sink_id = logger.add(sink, level="INFO", format="{message}")

    def detach(self) -> None:
        """Remove this screen's sink, tolerating session-level clears.

        :meth:`Session.set_terminal_out` may have already removed all
        sinks between attach and detach, in which case ``logger.remove``
        raises ``ValueError``.  Log at DEBUG so genuine bugs don't hide
        here.
        """
        if self._sink_id is None:
            return
        try:
            logger.remove(self._sink_id)
        except ValueError:
            logger.debug("{} log sink {} was already removed", self._screen_label, self._sink_id)
        self._sink_id = None
