#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for :mod:`anomaly_match_ui.utils.scrolling_log_sink`."""

from __future__ import annotations

import ipywidgets as widgets
import pytest
from loguru import logger

from anomaly_match_ui.utils import scrolling_log_sink

pytestmark = pytest.mark.ui


def _stream_text(out: widgets.Output) -> str:
    return "".join(
        entry.get("text", "") for entry in out.outputs if entry.get("output_type") == "stream"
    )


def test_sink_caps_at_max_lines():
    out = widgets.Output()
    sink = scrolling_log_sink._make_scrolling_log_sink(out, max_lines=3)

    for i in range(5):
        sink(f"line {i}\n")

    text = _stream_text(out)
    assert text == "line 2\nline 3\nline 4\n"


def test_sink_tags_widget_with_css_class():
    out = widgets.Output()
    scrolling_log_sink._make_scrolling_log_sink(out)
    assert "am-log-sink" in out._dom_classes


def test_sink_seeds_buffer_from_existing_widget_content():
    out = widgets.Output()
    out.outputs = ({"output_type": "stream", "name": "stdout", "text": "old A\nold B\n"},)

    sink = scrolling_log_sink._make_scrolling_log_sink(out, max_lines=10)
    sink("new C\n")

    assert _stream_text(out) == "old A\nold B\nnew C\n"


def test_sink_seed_respects_max_lines():
    out = widgets.Output()
    out.outputs = ({"output_type": "stream", "name": "stdout", "text": "a\nb\nc\nd\n"},)

    sink = scrolling_log_sink._make_scrolling_log_sink(out, max_lines=2)
    sink("e\n")

    assert _stream_text(out) == "d\ne\n"


def test_sink_replaces_outputs_each_call():
    """Each call rewrites the single stream entry instead of appending."""
    out = widgets.Output()
    sink = scrolling_log_sink._make_scrolling_log_sink(out, max_lines=10)

    sink("first\n")
    sink("second\n")

    assert len(out.outputs) == 1
    assert out.outputs[0]["text"] == "first\nsecond\n"


def test_two_sinks_on_different_widgets_have_independent_buffers():
    out_a = widgets.Output()
    out_b = widgets.Output()
    sink_a = scrolling_log_sink._make_scrolling_log_sink(out_a, max_lines=5)
    sink_b = scrolling_log_sink._make_scrolling_log_sink(out_b, max_lines=5)

    sink_a("only-a\n")
    sink_b("only-b\n")

    assert _stream_text(out_a) == "only-a\n"
    assert _stream_text(out_b) == "only-b\n"


# ── ScreenLogSink lifecycle ──────────────────────────────────────


def test_screen_log_sink_attach_then_detach_returns_to_idle():
    out = widgets.Output()
    sink = scrolling_log_sink.ScreenLogSink("TestScreen")

    sink.attach(out)
    assert sink._sink_id is not None
    sink.detach()
    assert sink._sink_id is None


def test_screen_log_sink_attach_is_idempotent():
    """Re-attaching detaches the previous sink first instead of leaking ids."""
    out = widgets.Output()
    sink = scrolling_log_sink.ScreenLogSink("TestScreen")

    sink.attach(out)
    first_id = sink._sink_id
    sink.attach(out)
    assert sink._sink_id != first_id


def test_screen_log_sink_detach_tolerates_external_clear():
    """If ``Session.set_terminal_out`` cleared all sinks, detach must not raise."""
    out = widgets.Output()
    sink = scrolling_log_sink.ScreenLogSink("TestScreen")
    sink.attach(out)

    logger.remove()  # external clear — drops the sink id we remember

    sink.detach()  # must not raise
    assert sink._sink_id is None


def test_screen_log_sink_detach_without_attach_is_noop():
    sink = scrolling_log_sink.ScreenLogSink("TestScreen")
    sink.detach()  # must not raise
    assert sink._sink_id is None
