#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import pytest
import asyncio
from anomaly_match.ui.memory_monitor import MemoryMonitor


@pytest.mark.asyncio
async def test_memory_monitor_basic():
    monitor = MemoryMonitor(update_interval=0.1)

    # Check initial state
    assert "Memory: --" in monitor.memory_text.value
    assert not monitor._running
    assert monitor._task is None

    # Start monitoring
    monitor.start()
    assert monitor._running
    # Task might be None initially but should be set after a brief wait
    await asyncio.sleep(0.1)
    assert monitor._task is not None

    # Wait for at least one update
    await asyncio.sleep(0.2)

    # Check that memory value has been updated
    assert "Memory:" in monitor.memory_text.value
    assert "GB" in monitor.memory_text.value
    assert "Memory: --" not in monitor.memory_text.value

    # Stop monitoring
    monitor.stop()
    assert not monitor._running

    # Wait for task to be fully cancelled
    await asyncio.sleep(0.1)
