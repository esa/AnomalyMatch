#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Graceful-shutdown wiring for prediction subprocesses."""

from __future__ import annotations

import signal

import pytest

from prediction_utils import (
    _shutdown_event,
    install_shutdown_handler,
    shutdown_requested,
)


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """Snapshot and restore the global signal handlers around every test.

    ``install_shutdown_handler`` mutates process-wide state; without a
    restore pytest itself would start catching SIGTERM/SIGINT.  The
    module-level ``_shutdown_event`` is cleared so each test starts
    from the unsignalled state — production code has no business
    resetting it, so a public helper would be dead code.
    """
    original_term = signal.getsignal(signal.SIGTERM)
    original_int = signal.getsignal(signal.SIGINT)
    _shutdown_event.clear()
    yield
    _shutdown_event.clear()
    signal.signal(signal.SIGTERM, original_term)
    signal.signal(signal.SIGINT, original_int)


class TestShutdownHandler:
    """Verify SIGTERM / SIGINT install the flag-setting handler."""

    def test_flag_false_by_default(self):
        assert shutdown_requested() is False

    def test_sigterm_flips_flag(self):
        install_shutdown_handler()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert shutdown_requested() is True

    def test_sigint_flips_flag(self):
        install_shutdown_handler()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert shutdown_requested() is True

    def test_repeated_signals_stay_latched(self):
        """A second signal must not un-set the flag or crash the handler."""
        install_shutdown_handler()
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        handler(signal.SIGTERM, None)
        assert shutdown_requested() is True

    def test_reinstall_is_idempotent(self):
        install_shutdown_handler()
        install_shutdown_handler()
        assert callable(signal.getsignal(signal.SIGTERM))
        assert callable(signal.getsignal(signal.SIGINT))
