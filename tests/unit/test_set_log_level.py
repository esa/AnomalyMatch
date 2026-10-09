#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for set_log_level sink management."""

from loguru import logger

from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.utils.set_log_level import set_log_level


def test_preserves_external_sinks_across_calls(tmp_path):
    """A later set_log_level must not drop externally-added sinks.

    ``set_log_level`` runs after every prediction subprocess, so if it did a
    blanket ``logger.remove()`` it would orphan the per-session ``session.log``
    (added by SessionIOHandler) the moment the first scoring chunk finished —
    freezing it for the rest of a multi-hour run.  It must remove only the
    sinks it added itself.
    """
    cfg = get_default_cfg()
    # First configure — establishes this module's own console sink.
    set_log_level("INFO", cfg, log_to_file=False)

    # Stand-in for the session.log sink an external owner registers.
    external = tmp_path / "session.log"
    ext_id = logger.add(str(external), format="{message}", level="DEBUG")
    try:
        logger.info("line-before")
        # A subsequent reconfigure (as happens after each subprocess) must keep
        # the external sink alive.
        set_log_level("INFO", cfg, log_to_file=False)
        logger.info("line-after")
    finally:
        logger.remove(ext_id)

    content = external.read_text()
    assert "line-before" in content
    assert "line-after" in content  # survived the second set_log_level call
