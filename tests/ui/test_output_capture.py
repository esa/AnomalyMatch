#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Exceptions must escape an Output widget that would swallow them in a kernel."""

import pytest

from anomaly_match_ui.utils.output_capture import run_in_output

pytestmark = pytest.mark.ui


class SwallowingOutput:
    """Behaves like ``ipywidgets.Output`` inside a kernel: it suppresses exceptions."""

    def __init__(self):
        self.entered = 0
        self.exited_with = []

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, etype, evalue, tb):
        self.exited_with.append(etype)
        return True


def test_returns_value_inside_the_output():
    out = SwallowingOutput()
    assert run_in_output(out, lambda: 42) == 42
    assert out.entered == 1
    assert out.exited_with == [None]


def test_exception_escapes_a_swallowing_output():
    out = SwallowingOutput()

    def _fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_in_output(out, _fail)


def test_keyboard_interrupt_escapes_a_swallowing_output():
    def _stop():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_in_output(SwallowingOutput(), _stop)
