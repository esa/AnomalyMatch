#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for ``anomaly_match.utils.tqdm_logging``.

These guard the fix for the frozen ``Decoding labeled cache: 0/N`` counter.
The oracle deliberately replays meter lines through the *real* production
parse path — the parent's ``[subprocess]`` relay feeding
:class:`LogProgressTap` — so the parsed ``desc`` the UI status label renders is
asserted exactly, not just that lines happen to parse.
"""

import io

from loguru import logger
from tqdm import tqdm

from anomaly_match.utils.tqdm_logging import _NewlineTqdmFile, tqdm_logging_file
from anomaly_match_ui.utils.progress_tap import LogProgressTap


def _run_meter(desc: str, total: int) -> list[str]:
    """Drive a tqdm meter through the wrapper and return the lines it emitted.

    Args:
        desc: tqdm description.
        total: Number of iterations.

    Returns:
        The newline-terminated lines the wrapper wrote to its stream.
    """
    fake_stderr = io.StringIO()
    # mininterval=0 + miniters=1 force a refresh per iteration so the test
    # doesn't depend on wall-clock timing.
    for _ in tqdm(
        range(total),
        desc=desc,
        unit="img",
        file=tqdm_logging_file(fake_stderr),
        mininterval=0,
        miniters=1,
    ):
        pass
    return [ln for ln in fake_stderr.getvalue().splitlines() if ln.strip()]


class TestNewlineTqdmFile:
    def test_blank_redraws_are_dropped(self):
        """tqdm clears its line with bare ``\\r`` / whitespace writes; those
        carry no progress and must not become empty lines."""
        stream = io.StringIO()
        sink = _NewlineTqdmFile(stream)
        sink.write("\r")
        sink.write("   \n")
        sink.write("")
        assert stream.getvalue() == ""

    def test_refresh_becomes_one_newline_terminated_line(self):
        stream = io.StringIO()
        sink = _NewlineTqdmFile(stream)
        sink.write("\rDecoding labeled cache:  50%|#####| 2/4 [00:01<00:01,  2.00img/s]")
        assert stream.getvalue() == (
            "Decoding labeled cache:  50%|#####| 2/4 [00:01<00:01,  2.00img/s]\n"
        )


class TestProductionParsePath:
    def test_desc_is_clean_and_counter_advances(self):
        """Replays the meter through the real relay + tap.

        Writing the bare meter to stderr (rather than via loguru) is what
        keeps ``desc`` free of a ``module:function:line`` header — so the UI
        label reads ``Decoding labeled cache``, not an internal path."""
        lines = _run_meter("Decoding labeled cache", 4)
        assert lines, "wrapper emitted no lines"

        events = []
        tap = LogProgressTap(on_tqdm=events.append)
        tap.install()
        try:
            # Reproduce the parent's stderr relay verbatim.
            for line in lines:
                logger.info("[subprocess] {}", line)
        finally:
            tap.uninstall()

        assert events, "no meter lines reached the progress tap"
        assert all(e["desc"] == "Decoding labeled cache" for e in events)
        assert all(e["total"] == 4 for e in events)
        currents = [e["current"] for e in events]
        assert max(currents) == 4
        assert max(currents) > min(currents)
