#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Run a callable inside an ipywidgets ``Output`` without losing its exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

T = TypeVar("T")


def run_in_output(out: AbstractContextManager, func: Callable[[], T]) -> T:
    """Call *func* with its output captured by *out*, and let its exceptions escape.

    Inside a running IPython kernel, ``ipywidgets.Output.__exit__`` shows an exception
    in the widget and then suppresses it. A caller's ``except`` clauses therefore never
    run, and a failed backend call looks like one that finished with no results. The
    exception is caught inside the widget context and raised again, unchanged, once
    that context can no longer swallow it (``KeyboardInterrupt`` included).

    Args:
        out: The ``Output`` widget (or any context manager) that captures display output.
        func: The work to run.

    Returns:
        Whatever *func* returns.
    """
    caught: BaseException
    with out:
        try:
            return func()
        except BaseException as exc:  # raised again below, outside the widget
            caught = exc
    raise caught
